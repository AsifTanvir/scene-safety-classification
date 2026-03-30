#!/usr/bin/env python3
"""
Safety classification script for Qwen3-VL-2B via llama-server backend.

Runs batch image processing dramatically faster by querying a persistent
llama-server running on localhost, saving image encoding and LLM reload time.
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

# -------------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------------

SAFETY_PROMPT = """You are an elite security AI analyst reviewing a single image from a security camera.

Look carefully at every person, object, and action in the scene.
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
REASON: [1-2 sentences describing what you observed.]"""

# -------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch Qwen3-VL inference via HTTP API.")
    parser.add_argument(
        "--images-dir", default="images",
        help="Directory containing local image files.",
    )
    parser.add_argument(
        "--server-url", default="http://127.0.0.1:8080",
        help="URL of the running llama-server (default: http://127.0.0.1:8080)",
    )
    parser.add_argument(
        "--max-new-tokens", "-n", type=int, default=256,
        help="Max tokens to generate.",
    )
    parser.add_argument(
        "--max-image-size", type=int, default=560,
        help="Max image dimension. Larger images are downscaled preserving aspect ratio.",
    )
    return parser.parse_args()


def collect_image_paths(images_dir: Path) -> list[Path]:
    supported = {".jpg", ".jpeg", ".png", ".webp"}
    paths = []
    for f in images_dir.iterdir():
        if f.is_file() and f.suffix.lower() in supported:
            paths.append(f)
    return sorted(paths)


def parse_classification(text: str) -> tuple[str, str]:
    """Return (label, reason) extracted from the model output."""
    label = "UNKNOWN"
    reason = text.strip()

    match_label = re.search(r"CLASSIFICATION\s*:\s*(SAFE|UNSAFE)", text, re.IGNORECASE)
    match_reason = re.search(r"REASON\s*:\s*(.+)", text, re.IGNORECASE)

    if match_label:
        label = match_label.group(1).upper()
    if match_reason:
        reason = match_reason.group(1).strip()

    return label, reason


def check_server_running(server_url: str) -> bool:
    """Check if llama-server is responding to HTTP GET /health."""
    url = f"{server_url.rstrip('/')}/health"
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=2) as resp:
            return resp.status == 200
    except (urllib.error.URLError, ConnectionResetError):
        return False


def start_server_if_needed(server_url: str) -> subprocess.Popen:
    """Start run_llama_server.sh if the server is not already running."""
    if check_server_running(server_url):
        print("  (llama-server is already running...)")
        return None

    print(f"  (Starting llama-server automatically... taking ~5-10s to load the models into GPU)")
    script_path = Path(__file__).parent / "run_llama_server.sh"
    
    if not script_path.exists():
        print(f"[ERROR] Auto-start script not found at {script_path}")
        sys.exit(1)
    
    proc = subprocess.Popen(
        [str(script_path)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True  # so it can be killed completely
    )

    # Wait up to 30 seconds for the model to load into VRAM
    for _ in range(30):
        if check_server_running(server_url):
            print("  (Server initialized properly!)")
            return proc
        time.sleep(1)
        
    print("[ERROR] llama-server failed to start or timed out!")
    proc.terminate()
    sys.exit(1)


def run_image(image_path: Path, server_url: str, max_size: int, max_tokens: int) -> str:
    """Send image via REST API to llama-server and return generated text."""
    try:
        img = Image.open(image_path)
    except Exception as e:
        return f"Error opening image: {e}"

    # Downscale dynamically right before encoding
    if max(img.width, img.height) > max_size:
        print(f"  (Resizing {img.width}x{img.height} → max {max_size}px...)")
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
        
    # Base64 encode the final resized image directly from a memory buffer
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=85)
    img_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    buffer.close()

    print("  (Sending to llama-server via HTTP...)")
    start_t = time.time()
    
    url = f"{server_url.rstrip('/')}/v1/chat/completions"
    data = {
        "model": "qwen3vl",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}
                    },
                    {
                        "type": "text",
                        "text": SAFETY_PROMPT
                    }
                ]
            }
        ],
        "max_tokens": max_tokens,
        "temperature": 0.0
    }
    
    req = urllib.request.Request(
        url, 
        data=json.dumps(data).encode('utf-8'), 
        headers={"Content-Type": "application/json"}
    )
    
    raw_text = ""
    try:
        with urllib.request.urlopen(req) as resp:
            res = json.loads(resp.read().decode('utf-8'))
            raw_text = res["choices"][0]["message"]["content"]
    except urllib.error.URLError as e:
        raw_text = f"Error communicating with server: {e}"
        print(f"  [ERROR] {raw_text}")
        try:
            # Try to read the error body if it exists
            err_body = e.read().decode('utf-8')
            print(f"  [ERROR BODY] {err_body}")
        except:
            pass

    elapsed = time.time() - start_t
    print(f"  [Time elapsed: {elapsed:.1f}s]")
    return raw_text


def main() -> None:
    args = parse_args()

    images_dir = Path(args.images_dir)
    if not images_dir.is_dir():
        print(f"[ERROR] Images directory not found: {images_dir}")
        sys.exit(1)

    image_paths = collect_image_paths(images_dir)
    if not image_paths:
        print(f"[ERROR] No image files found in {images_dir}")
        sys.exit(1)

    print(f"Server : {args.server_url}")
    print(f"Images : {len(image_paths)} found in '{images_dir}'")
    print(f"Max Res: {args.max_image_size}px")
    print()

    server_proc = start_server_if_needed(args.server_url)
    
    def cleanup():
        if server_proc:
            print("\n  (Shutting down automatic llama-server...)")
            server_proc.terminate()
            
    atexit.register(cleanup)

    results = []

    for image_path in image_paths:
        print(f"{'─'*60}")
        print(f"▶  {image_path.name}")
        print(f"{'─'*60}")
        sys.stdout.flush()

        raw_text = run_image(image_path, args.server_url, args.max_image_size, args.max_new_tokens)
        label, reason = parse_classification(raw_text)

        # Write result file into the images directory
        out_file = images_dir / f"{image_path.stem}_text.txt"
        out_file.write_text(
            f"Image      : {image_path.name}\n"
            f"Classification: {label}\n"
            f"Reason     : {reason}\n"
        )

        icon = "✅" if label == "SAFE" else "⚠️ "
        print(f"\n  {icon}  {label}  —  {reason}")
        print(f"  Saved → {out_file.name}\n")
        results.append((image_path.name, label, reason))

    # Print summary table
    print(f"\n{'='*60}")
    print(f"{'SUMMARY':^60}")
    print(f"{'='*60}")
    for name, label, reason in results:
        icon = "✅" if label == "SAFE" else "⚠️ "
        # Print a shortened reason for the summary table
        short_reason = (reason[:55] + "...") if len(reason) > 55 else reason
        print(f"{icon} [{label:6}]  {name:<15} {short_reason}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
