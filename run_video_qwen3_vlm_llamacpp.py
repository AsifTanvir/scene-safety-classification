#!/usr/bin/env python3
"""
Safety classification script for Videos using Qwen3-VL-2B via llama-server backend.
"""
import argparse
import base64
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
import urllib.error
import io
import atexit
from pathlib import Path
from PIL import Image, ImageFilter
import cv2

# -------------------------------------------------------------------------
SAFETY_PROMPT = """You are a security camera analyst reviewing a video feed captured as {num_frames} sequential frames.

Treat these frames as a single continuous video. Do NOT reference individual frame numbers in your response.
Describe the overall scene and events as a unified sequence.

Look carefully at every person, object, and action.
Identify ANY of the following threats:
- WEAPONS: Firearms, guns, rifles, handguns, knives held by a person
- VIOLENCE: Physical assault, fighting, threatening gestures
- HAZARDS: Fire, smoke, explosions
- FALLS/ACCIDENTS: Person lying on the ground, collapsed, fallen off a bicycle, injured

If you detect EVEN A TINY HINT of a weapon, hazard, fall, or accident, classify as UNSAFE.
If the environment is completely normal and peaceful, classify as SAFE.

IMPORTANT: Your REASON must be consistent with your CLASSIFICATION. Do NOT say UNSAFE and then explain why it is safe, or vice versa.

Reply in EXACTLY this format:
CLASSIFICATION: [SAFE or UNSAFE]
REASON: [1-2 sentences describing what happens in the video overall.]"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch Qwen3-VL Video inference via HTTP API.")
    parser.add_argument("--videos-dir", default="videos")
    parser.add_argument("--frames", type=int, default=6,
                        help="Number of evenly spaced frames to extract.")
    parser.add_argument("--server-url", default="http://127.0.0.1:8080")
    parser.add_argument("--max-new-tokens", "-n", type=int, default=512,
                        help="Max tokens to generate.")
    parser.add_argument("--max-image-size", type=int, default=560,
                        help="Max frame dimension.")
    parser.add_argument("--jpeg-quality", type=int, default=85,
                        help="JPEG encoding quality (higher = more detail preserved).")
    return parser.parse_args()


def collect_video_paths(videos_dir: Path) -> list[Path]:
    supported = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
    return sorted(f for f in videos_dir.iterdir()
                  if f.is_file() and f.suffix.lower() in supported)


def parse_classification(text: str) -> tuple[str, str]:
    """Return (label, reason) extracted from model output."""
    label = "UNKNOWN"
    reason = text.strip()

    match_label = re.search(r"CLASSIFICATION\s*:\s*(SAFE|UNSAFE)", text, re.IGNORECASE)
    match_reason = re.search(r"REASON\s*:\s*(.+)", text, re.IGNORECASE | re.DOTALL)

    if match_label:
        label = match_label.group(1).upper()
    if match_reason:
        reason = match_reason.group(1).strip().replace('\n', ' ')

    return label, reason


def extract_frames(video_path: Path, num_frames: int) -> list[Image.Image]:
    """Extract N evenly spaced frames from a video using OpenCV."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Failed to open video {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total == 0:
        cap.release()
        raise ValueError(f"Video has 0 frames")

    num_frames = min(8, num_frames)

    if num_frames == 1:
        indices = [total // 2]
    else:
        # IMPROVED: Use linspace for perfectly even distribution
        import numpy as np
        indices = [int(x) for x in np.linspace(0, total - 1, num_frames)]

    images = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if ok:
            images.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))

    cap.release()
    if not images:
        raise ValueError(f"Could not read any frames from {video_path}")
    return images


def check_server_running(server_url: str) -> bool:
    try:
        with urllib.request.urlopen(f"{server_url.rstrip('/')}/health", timeout=2) as r:
            return r.status == 200
    except Exception:
        return False


def start_server_if_needed(server_url: str) -> subprocess.Popen:
    if check_server_running(server_url):
        print("  (llama-server is already running...)")
        return None

    print("  (Starting llama-server automatically... ~5-10s)")
    script_path = Path(__file__).parent / "run_llama_server.sh"
    if not script_path.exists():
        print(f"[ERROR] {script_path} not found")
        sys.exit(1)

    proc = subprocess.Popen(["/bin/bash", str(script_path)],
                             stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL,
                             start_new_session=True)
    for _ in range(30):
        if check_server_running(server_url):
            print("  (Server ready!)")
            return proc
        time.sleep(1)

    print("[ERROR] Server failed to start")
    proc.terminate()
    sys.exit(1)


def encode_frame(img: Image.Image, max_size: int, jpeg_quality: int) -> tuple[str, int, int]:
    """Resize, snap to multiples of 56, sharpen, encode to base64. Returns (b64_str, w, h)."""
    if max(img.width, img.height) > max_size:
        img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

    # Snap to multiples of 56 (Qwen-VL 2x2 patch-merge requirement)
    new_w = max(56, (img.width // 56) * 56)
    new_h = max(56, (img.height // 56) * 56)
    if new_w != img.width or new_h != img.height:
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    if img.mode != "RGB":
        img = img.convert("RGB")

    # Sharpen to make small object edges (weapons, knives) crisper for the ViT
    img = img.filter(ImageFilter.SHARPEN)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=jpeg_quality)
    b64 = base64.b64encode(buf.getvalue()).decode()
    buf.close()
    return b64, new_w, new_h


def run_video(video_path: Path, server_url: str, num_frames: int,
              max_size: int, max_tokens: int, jpeg_quality: int) -> str:
    start_t = time.time()

    try:
        print(f"  (Extracting {num_frames} frames...)")
        imgs = extract_frames(video_path, num_frames)
    except Exception as e:
        return f"Error extracting frames: {e}"

    # IMPROVED: Label each frame temporally so the model understands sequence
    message_content = []
    first_dims = None

    for i, img in enumerate(imgs):
        b64, w, h = encode_frame(img, max_size, jpeg_quality)
        if first_dims is None:
            first_dims = (w, h)
            print(f"  (Frames normalized to {w}x{h}, JPEG quality={jpeg_quality})")

        # IMPROVED: Add a text label before each image for temporal grounding
        message_content.append({
            "type": "text",
            "text": f"[Frame {i+1} of {len(imgs)}]"
        })
        message_content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
        })

    # Inject actual frame count into prompt
    prompt_text = SAFETY_PROMPT.format(num_frames=len(imgs))
    message_content.append({"type": "text", "text": prompt_text})

    print(f"  (Sending {len(imgs)} labeled frames to llama-server...)")

    url = f"{server_url.rstrip('/')}/v1/chat/completions"
    data = {
        "model": "qwen3vl",
        "messages": [{"role": "user", "content": message_content}],
        "temperature": 0.1,       # tiny temp helps description quality
        "max_tokens": max_tokens,
        # enable thinking for Qwen3 reasoning 
        "thinking": {"type": "enabled", "budget_tokens": 512}
    }

    req = urllib.request.Request(
        url,
        data=json.dumps(data).encode(),
        headers={'Content-Type': 'application/json'}
    )

    try:
        with urllib.request.urlopen(req, timeout=120) as resp:  # IMPROVED: explicit timeout
            resp_data = json.loads(resp.read().decode())
            raw_text = resp_data['choices'][0]['message']['content']
            print(f"  [Time: {time.time() - start_t:.1f}s]")
            return raw_text
    except urllib.error.HTTPError as e:
        body = e.read().decode() if e.fp else ""
        print(f"  [HTTP ERROR] {e}\n  {body}")
        # IMPROVED: retry without thinking mode if server rejects it
        if "thinking" in body.lower() or e.code == 400:
            print("  (Retrying without thinking mode...)")
            data.pop("thinking", None)
            req2 = urllib.request.Request(url, data=json.dumps(data).encode(),
                                           headers={'Content-Type': 'application/json'})
            try:
                with urllib.request.urlopen(req2, timeout=120) as r2:
                    rd = json.loads(r2.read().decode())
                    return rd['choices'][0]['message']['content']
            except Exception as e2:
                return f"Error on retry: {e2}"
        return f"HTTP Error: {e}"
    except Exception as e:
        print(f"  [ERROR] {e}")
        return f"Error: {e}"


def main() -> None:
    args = parse_args()
    videos_dir = Path(args.videos_dir)

    if not videos_dir.is_dir():
        videos_dir.mkdir(parents=True, exist_ok=True)
        print(f"[WARN] Created missing directory '{videos_dir}'. Add videos and re-run.")
        sys.exit(0)

    video_paths = collect_video_paths(videos_dir)
    if not video_paths:
        print(f"[ERROR] No video files found in {videos_dir}")
        sys.exit(1)

    print(f"Server : {args.server_url}")
    print(f"Videos : {len(video_paths)} found in '{videos_dir}'")
    print(f"Frames : {args.frames} per video")
    print(f"Max Res: {args.max_image_size}px  |  JPEG quality: {args.jpeg_quality}")
    print()

    server_proc = start_server_if_needed(args.server_url)
    atexit.register(lambda: server_proc and server_proc.terminate())

    results = []
    for video_path in video_paths:
        print(f"{'─'*60}")
        print(f"▶  {video_path.name}")
        print(f"{'─'*60}")
        sys.stdout.flush()

        raw_text = run_video(
            video_path, args.server_url, args.frames,
            args.max_image_size, args.max_new_tokens, args.jpeg_quality
        )
        label, reason = parse_classification(raw_text)

        out_file = videos_dir / f"{video_path.stem}_result.txt"
        out_file.write_text(
            f"Video         : {video_path.name}\n"
            f"Frames Used   : {args.frames}\n"
            f"Classification: {label}\n"
            f"Reason        : {reason}\n"
        )

        icon = "✅" if label == "SAFE" else "⚠️ "
        print(f"\n  {icon}  {label}  —  {reason}")
        print(f"  Saved → {out_file.name}\n")
        results.append((video_path.name, label, reason))

    print(f"\n{'='*60}")
    print(f"{'SUMMARY':^60}")
    print(f"{'='*60}")
    for name, label, reason in results:
        icon = "✅" if label == "SAFE" else "⚠️ "
        short = (reason[:55] + "...") if len(reason) > 55 else reason
        print(f"{icon} [{label:6}]  {name:<20}  {short}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()