#!/usr/bin/env python3
"""
Safety classification from an OAK-D Lite live RGB stream using Qwen3-VL-2B via llama-server.
  - Captures fixed-length chunks from the camera (default 5 s), samples N evenly-spaced frames
  - Classifies environment with Places365 ResNet-18 and passes it to the VLM
  - UNSAFE → opens a cv2 alert window + logs event to CSV; then repeats forever
  - Run with --view-log to review all logged events with full explanations

Requires: pip install depthai
"""
import argparse
import base64
import csv
import datetime
import hashlib
import io
import json
import re
import subprocess
import sys
import time
import urllib.request
import urllib.error
import atexit
from pathlib import Path
from typing import Any, Optional

try:
    import depthai as dai
except ImportError:
    dai = None  # type: ignore

import numpy as np
from PIL import Image, ImageFilter
import cv2
import torch
import torchvision.models as models
import torchvision.transforms as transforms


# ═══════════════════════════════════════════════════════════════════════
#  Places365 Environment Classifier
# ═══════════════════════════════════════════════════════════════════════

ENV_CLASSES = ["classroom", "home", "office"]

ENV_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def load_env_model(weights_path: str, device: torch.device) -> torch.nn.Module:
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, len(ENV_CLASSES))
    state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device).eval()
    return model


def classify_environment(
    frames: list,
    model: torch.nn.Module,
    device: torch.device,
) -> tuple[str, float]:
    """Majority-vote environment classification across all extracted frames."""
    votes: dict[str, list[float]] = {c: [] for c in ENV_CLASSES}
    with torch.no_grad():
        for img in frames:
            tensor = ENV_TRANSFORM(img.convert("RGB")).unsqueeze(0).to(device)
            probs = torch.nn.functional.softmax(model(tensor), dim=1).squeeze()
            idx = probs.argmax().item()
            votes[ENV_CLASSES[idx]].append(probs[idx].item())
    best = max(votes, key=lambda c: len(votes[c]))
    avg_conf = sum(votes[best]) / len(votes[best]) if votes[best] else 0.0
    return best, avg_conf


# ═══════════════════════════════════════════════════════════════════════
#  Alert Logic  (adapted from alert_Generator notebook)
# ═══════════════════════════════════════════════════════════════════════

ALERT_DIR   = Path("./alert_output")
FRAMES_DIR  = ALERT_DIR / "frames"
CSV_LOG     = ALERT_DIR / "safety_events.csv"

CSV_FIELDS  = [
    "event_id", "timestamp", "video_source", "frames_used",
    "environment", "env_confidence", "event_label",
    "classification", "reason", "threat_reasoning", "frame_path",
]

ENVIRONMENT_RULES = {
    "home"     : ["gun", "knife", "fire", "flames", "smoke", "burning", "person falling", "fall"],
    "office"   : ["gun", "fire", "flames", "smoke", "person falling", "fall"],
    "classroom": ["gun", "knife", "fire", "flames", "smoke", "burning", "person falling", "fall"],
}
EVENT_LABEL_MAP = {
    "gun": "Gun Detection",    "knife": "Knife Detection",
    "fire": "Fire Detection",  "flames": "Fire Detection",
    "smoke": "Fire Detection", "burning": "Fire Detection",
    "person falling": "Fall Detection", "fall": "Fall Detection",
}

BORDER = "=" * 70
THIN   = "-" * 70


def infer_event_label(reason: str, environment: str) -> str:
    rl = reason.lower()
    for trigger in ENVIRONMENT_RULES.get(environment.lower(), []):
        if trigger in rl:
            return EVENT_LABEL_MAP.get(trigger, "Unsafe Event")
    for kw, lbl in EVENT_LABEL_MAP.items():
        if kw in rl:
            return lbl
    return "Unsafe Event"


def build_threat_reasoning(environment: str, event_label: str, reason: str) -> str:
    rule = {
        "Gun Detection"  : f"Firearms are never permitted in a {environment.upper()} environment.",
        "Knife Detection": (
            "Knife detected outside kitchen context — potential threat."
            if environment == "home"
            else f"Bladed weapons are not permitted in a {environment.upper()} environment."
        ),
        "Fire Detection" : f"Fire, flames, or smoke detected in {environment.upper()} — immediate hazard.",
        "Fall Detection" : "A person appears to have fallen and may be injured.",
    }.get(event_label, f"Unsafe condition detected in {environment.upper()} environment.")
    return (
        f"Environment : {environment.upper()}\n"
        f"Threat type : {event_label}\n"
        f"Rule applied: {rule}\n"
        f"Scene context: {reason.strip()}"
    )


def log_event_csv(event_id, timestamp, video_source, frames_used,
                  environment, env_confidence, event_label,
                  classification, reason, threat_reasoning, frame_path) -> None:
    """Append one row to the CSV log, creating headers if the file is new."""
    is_new = not CSV_LOG.exists()
    with open(CSV_LOG, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if is_new:
            writer.writeheader()
        writer.writerow({
            "event_id"       : event_id,
            "timestamp"      : timestamp,
            "video_source"   : video_source,
            "frames_used"    : frames_used,
            "environment"    : environment,
            "env_confidence" : f"{env_confidence:.0%}",
            "event_label"    : event_label,
            "classification" : classification,
            "reason"         : reason,
            "threat_reasoning": threat_reasoning.replace("\n", " | "),
            "frame_path"     : frame_path or "",
        })


def annotate_frame(bgr_frame, event_label: str, environment: str,
                   env_conf: float, event_id: str) -> Optional[str]:
    """Burn alert banner onto frame and save JPEG. Returns saved path."""
    ann = bgr_frame.copy()
    h, w = ann.shape[:2]
    banner = (f"UNSAFE | {event_label} | ENV: {environment.upper()} "
              f"| CONF: {env_conf:.0%} | ID: {event_id}")
    cv2.rectangle(ann, (0, h - 50), (w, h), (0, 0, 180), -1)
    cv2.putText(ann, banner, (8, h - 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1)
    save_path = str(FRAMES_DIR / f"{event_id}.jpg")
    cv2.imwrite(save_path, ann, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return save_path


# ── Alert window script run in a subprocess ──────────────────────────
# cv2 GUI (imshow / waitKey / destroyWindow) MUST run on the OS main
# thread (Qt/GTK event-loop requirement on Linux & macOS).
# Running it on a background thread deadlocks the event loop.
# Solution: spawn a fresh Python subprocess per alert — that process
# has its own main thread so cv2 works perfectly, and the parent loop
# is never blocked.
_ALERT_SCRIPT = r"""
import sys, json, cv2, time
args = json.loads(sys.argv[1])
img = cv2.imread(args['frame_path'])
if img is None:
    sys.exit(0)
h, w = img.shape[:2]
overlay = img.copy()
cv2.rectangle(overlay, (0, 0), (w, h // 3), (0, 0, 160), -1)
cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)
cv2.putText(img, '! UNSAFE SCENE DETECTED',
            (20, 60), cv2.FONT_HERSHEY_DUPLEX, 1.1, (0, 0, 255), 2)
cv2.putText(img, 'Threat : ' + args['event_label'],
            (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
cv2.putText(img, 'Env    : ' + args['environment'].upper(),
            (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
cv2.putText(img, 'Source : ' + args['video_name'],
            (20, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 1)
cv2.rectangle(img, (0, h - 40), (w, h), (30, 30, 30), -1)
win = 'UNSAFE ALERT'
cv2.namedWindow(win, cv2.WINDOW_NORMAL)
cv2.resizeWindow(win, min(w, 900), min(h, 600))
dur = args['duration_s']
deadline = time.time() + dur
while time.time() < deadline:
    remaining = max(0.0, deadline - time.time())
    fs = img.copy()
    cv2.rectangle(fs, (0, h - 40), (int(w * remaining / dur), h), (0, 0, 200), -1)
    cv2.putText(fs, f'Closing in {remaining:.1f}s  (press any key to dismiss)',
                (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    cv2.imshow(win, fs)
    if cv2.waitKey(50) != -1:
        break
cv2.destroyAllWindows()
"""


def show_alert_async(frame_path: str, event_label: str, environment: str,
                     video_name: str, duration_s: int = 3) -> subprocess.Popen:
    """
    Launch the alert window in a separate Python subprocess (not a thread).
    Returns the Popen object immediately — caller is never blocked.
    Each subprocess owns its own main thread so cv2 GUI works correctly.
    """
    args_json = json.dumps({
        "frame_path" : frame_path,
        "event_label": event_label,
        "environment": environment,
        "video_name" : video_name,
        "duration_s" : duration_s,
    })
    try:
        proc = subprocess.Popen(
            [sys.executable, "-c", _ALERT_SCRIPT, args_json],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return proc
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════════════
#  VLM Prompt & Frame Processing  (original N-frame batch logic)
# ═══════════════════════════════════════════════════════════════════════

SAFETY_PROMPT = """\
You are a security camera analyst. You are shown {num_frames} stills sampled at evenly spaced \
moments along **one continuous video** from a **{environment}** environment \
(classifier confidence: {env_confidence:.0%}).

The stills are in time order (beginning of the clip → end). Treat them as one recording: \
describe flowing action and how the situation changes, not a checklist of separate images.

In your REASON only: do **not** mention frame numbers, image numbers, or phrases like \
"first frame", "frame 7", or "by frame 9". Use natural time language instead, for example: \
at the start of the clip, early on, as the scene continues, later, toward the end, by the end.

Identify ANY of the following threats:
- WEAPONS: firearms, guns, knives held by a person
- VIOLENCE: assault, fighting, threatening gestures
- HAZARDS: fire, smoke, explosions
- FALLS/ACCIDENTS: person collapsed, lying on the ground, injured

Classify as UNSAFE if you see even a hint of a threat. Classify as SAFE otherwise.

Alignment rule (mandatory): Your CLASSIFICATION and REASON must agree. \
If you choose UNSAFE, both sentences must describe the threat you are flagging. \
If you choose SAFE, both sentences must describe a benign scene and must NOT describe \
weapons, fire/smoke, violence, or a person falling or injured. \
Do not output SAFE with a REASON that mentions a threat, or UNSAFE with a REASON that says no threat is visible.

Reply in EXACTLY this format — no extra text, no repetition:
CLASSIFICATION: SAFE or UNSAFE
REASON: Exactly 2 sentences forming one continuous narrative (like explaining the video to a colleague). \
Sentence 1: what is happening at the beginning of the clip. \
Sentence 2: how things develop by the end of the clip — outcome or person state; \
build on sentence 1, do not re-label the same event with conflicting certainty \
(e.g. if sentence 1 states a fall occurred, do not call it a "possible" or "potential" fall in sentence 2). \
Do NOT repeat the same phrase twice. Stop after 2 sentences."""


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Live OAK-D Lite Qwen3-VL safety monitor with alert generator.")
    ap.add_argument("--chunk-seconds", type=float, default=5.0,
                    help="Length of each camera segment to classify (default: 5).")
    ap.add_argument("--frames", type=int, default=6,
                    help="Number of evenly-spaced frames per chunk (capped by frames captured).")
    ap.add_argument("--oak-preview-width", type=int, default=1280,
                    help="RGB preview width (must match camera capabilities).")
    ap.add_argument("--oak-preview-height", type=int, default=720,
                    help="RGB preview height.")
    ap.add_argument("--server-url", default="http://127.0.0.1:8080")
    ap.add_argument("--max-new-tokens", "-n", type=int, default=512)
    ap.add_argument("--max-image-size", type=int, default=560)
    ap.add_argument("--jpeg-quality", type=int, default=85)
    ap.add_argument("--env-model", default="places365_environment_model_new.pth")
    ap.add_argument("--session-id", default="session_001")
    ap.add_argument("--alert-duration", type=int, default=3,
                    help="Seconds the alert window stays open.")
    ap.add_argument("--view-log", action="store_true",
                    help="Print the full event log with explanations and exit.")
    return ap.parse_args()


def sample_frames_evenly_from_buffer(
    bgr_frames: list,
    num_frames: int,
) -> list[tuple[int, np.ndarray]]:
    """Pick N evenly-spaced frames from an in-memory list (same logic as file-based sampling)."""
    total = len(bgr_frames)
    if total == 0:
        raise ValueError("No frames in buffer")
    num_frames = min(total, max(1, num_frames))
    indices = (
        [total // 2] if num_frames == 1
        else [int(x) for x in np.linspace(0, total - 1, num_frames)]
    )
    out: list[tuple[int, np.ndarray]] = []
    for idx in indices:
        out.append((idx, bgr_frames[idx]))
    return out


def build_oak_rgb_pipeline(preview_w: int, preview_h: int) -> Any:
    """DepthAI pipeline: ColorCamera preview → host (BGR)."""
    if dai is None:
        raise RuntimeError("depthai is not installed")
    pipeline = dai.Pipeline()
    cam_rgb = pipeline.create(dai.node.ColorCamera)
    xout = pipeline.create(dai.node.XLinkOut)
    xout.setStreamName("rgb")
    cam_rgb.setPreviewSize(preview_w, preview_h)
    cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam_rgb.setInterleaved(False)
    cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    cam_rgb.preview.link(xout.input)
    return pipeline


def capture_oak_chunk_seconds(queue: Any, seconds: float) -> list:
    """Block-read preview frames from the OAK RGB queue for `seconds` wall-clock time."""
    frames: list = []
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        pkt = queue.get()
        frames.append(pkt.getCvFrame())
    return frames


def encode_frame(img: Image.Image, max_size: int, jpeg_quality: int) -> str:
    if max(img.width, img.height) > max_size:
        img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    new_w = max(56, (img.width // 56) * 56)
    new_h = max(56, (img.height // 56) * 56)
    if new_w != img.width or new_h != img.height:
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    if img.mode != "RGB":
        img = img.convert("RGB")
    img = img.filter(ImageFilter.SHARPEN)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=jpeg_quality)
    b64 = base64.b64encode(buf.getvalue()).decode()
    buf.close()
    return b64


def deduplicate_reason(text: str, max_sentences: int = 3) -> str:
    """
    Remove repeated sentences and cap at max_sentences.
    Handles the LLM repetition-loop failure mode where the same sentence
    is emitted dozens of times.
    """
    # Split on sentence-ending punctuation
    raw_sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    seen: list[str] = []
    for s in raw_sentences:
        s = s.strip()
        if not s:
            continue
        # Normalise for duplicate check (lowercase, collapse whitespace)
        key = re.sub(r'\s+', ' ', s.lower())
        # Skip if this sentence (or a very similar one) was already added
        if any(key == re.sub(r'\s+', ' ', x.lower()) for x in seen):
            continue
        seen.append(s)
        if len(seen) >= max_sentences:
            break
    return ' '.join(seen)


def parse_classification(text: str) -> tuple[str, str]:
    """
    Extract CLASSIFICATION and REASON. Handles either field order; REASON never
    includes a trailing CLASSIFICATION: line (models sometimes emit REASON first).
    """
    text = text.strip()
    label = "UNKNOWN"
    reason = text

    m_cls = re.search(r"CLASSIFICATION\s*:\s*(SAFE|UNSAFE)\b", text, re.IGNORECASE)
    if m_cls:
        label = m_cls.group(1).upper()

    m_reas = re.search(r"REASON\s*:\s*", text, re.IGNORECASE)
    if m_reas:
        body = text[m_reas.end():]
        m_cut = re.search(r"^\s*CLASSIFICATION\s*:\s*", body, re.MULTILINE | re.IGNORECASE)
        if m_cut:
            body = body[: m_cut.start()]
        raw_reason = body.strip().replace("\n", " ")
        reason = deduplicate_reason(raw_reason)
    elif m_cls:
        tail = text[m_cls.end():].lstrip()
        tail = re.sub(r"^REASON\s*:\s*", "", tail, count=1, flags=re.IGNORECASE)
        if tail:
            reason = deduplicate_reason(tail.replace("\n", " ").strip())
        else:
            reason = ""

    # Remove any echoed CLASSIFICATION line from free-form reason text
    reason = re.sub(
        r"\s*CLASSIFICATION\s*:\s*(?:SAFE|UNSAFE)\b\s*",
        " ",
        reason,
        flags=re.IGNORECASE,
    ).strip()

    if not reason and label != "UNKNOWN":
        reason = "(No reason provided.)"
    return label, reason


def check_server_running(server_url: str) -> bool:
    try:
        with urllib.request.urlopen(f"{server_url.rstrip('/')}/health", timeout=2) as r:
            return r.status == 200
    except Exception:
        return False


def _find_bash() -> list[str]:
    """Return a shell command prefix that can run a .sh script on any OS."""
    import platform, shutil
    if platform.system() == "Windows":
        # Prefer Git Bash, then WSL bash
        for candidate in [
            r"C:\Program Files\Git\bin\bash.exe",
            r"C:\Program Files (x86)\Git\bin\bash.exe",
            shutil.which("bash"),   # e.g. Git Bash on PATH
        ]:
            if candidate and Path(candidate).exists():
                return [candidate]
        # Fall back to WSL
        wsl = shutil.which("wsl")
        if wsl:
            return [wsl, "bash"]
        print("[ERROR] No bash found on Windows. Install Git for Windows or WSL.")
        sys.exit(1)
    return ["/bin/bash"]


def start_server_if_needed(server_url: str) -> Optional[subprocess.Popen]:
    if check_server_running(server_url):
        print("  (llama-server is already running...)")
        return None
    print("  (Starting llama-server automatically... ~5-10s)")
    script_path = Path(__file__).parent / "run_llama_server.sh"
    if not script_path.exists():
        print(f"[ERROR] {script_path} not found")
        sys.exit(1)
    bash_cmd = _find_bash()
    proc = subprocess.Popen(bash_cmd + [str(script_path)],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                             start_new_session=True)
    for _ in range(30):
        if check_server_running(server_url):
            print("  (Server ready!)")
            return proc
        time.sleep(1)
    print("[ERROR] Server failed to start")
    proc.terminate()
    sys.exit(1)


def _vlm_timeline_caption(i: int, n: int) -> str:
    """Short chronological hint without 'frame' wording (reduces frame-by-frame replies)."""
    if n <= 1:
        return "[Moment from the clip]"
    t = i / (n - 1)
    if t < 0.2:
        qual = "Near the start"
    elif t < 0.45:
        qual = "Early in the clip"
    elif t < 0.55:
        qual = "Around the middle"
    elif t < 0.8:
        qual = "Later on"
    else:
        qual = "Toward the end"
    return f"[{qual}]"


def run_video(video_path: Path, pil_imgs: list, bgr_frames: list,
              env_label: str, env_conf: float,
              server_url: str, max_tokens: int,
              jpeg_quality: int, max_size: int) -> tuple[str, str]:
    """Send all frames to VLM with env context. Returns (label, reason)."""
    prompt = SAFETY_PROMPT.format(
        num_frames=len(pil_imgs),
        environment=env_label,
        env_confidence=env_conf,
    )
    # Instructions first so the model reads constraints before the image sequence.
    message_content: list = [{"type": "text", "text": prompt}]
    for i, img in enumerate(pil_imgs):
        message_content.append({"type": "text", "text": _vlm_timeline_caption(i, len(pil_imgs))})
        b64 = encode_frame(img, max_size, jpeg_quality)
        message_content.append({"type": "image_url",
                                 "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})

    url = f"{server_url.rstrip('/')}/v1/chat/completions"
    data = {
        "model"             : "qwen3vl",
        "messages"          : [{"role": "user", "content": message_content}],
        "temperature"       : 0.1,
        "max_tokens"        : min(max_tokens, 200),  # 2 sentences don't need more
        "repetition_penalty": 1.3,   # penalise repeating the same tokens
        "frequency_penalty" : 0.5,   # further penalise high-frequency tokens
        "thinking"          : {"type": "enabled", "budget_tokens": 512},
    }
    req = urllib.request.Request(url, data=json.dumps(data).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            raw = json.loads(resp.read().decode())["choices"][0]["message"]["content"]
            return parse_classification(raw)
    except urllib.error.HTTPError as e:
        body = e.read().decode() if e.fp else ""
        if "thinking" in body.lower() or e.code == 400:
            data.pop("thinking", None)
            req2 = urllib.request.Request(url, data=json.dumps(data).encode(),
                                          headers={"Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req2, timeout=120) as r2:
                    raw = json.loads(r2.read().decode())["choices"][0]["message"]["content"]
                    return parse_classification(raw)
            except Exception as e2:
                return "UNKNOWN", f"Error on retry: {e2}"
        return "UNKNOWN", f"HTTP Error: {e}"
    except Exception as e:
        return "UNKNOWN", f"Error: {e}"


# ═══════════════════════════════════════════════════════════════════════
#  One clip: env + VLM + optional alert / CSV
# ═══════════════════════════════════════════════════════════════════════

def process_safety_segment(
    *,
    segment_idx: int,
    video_name: str,
    frame_pairs: list[tuple[int, np.ndarray]],
    args: argparse.Namespace,
    env_model: Optional[torch.nn.Module],
    device: torch.device,
    result_out: Path,
    alert_procs: list,
) -> str:
    """Run classification and logging for one sampled clip. Returns VLM label (SAFE/UNSAFE/UNKNOWN)."""
    bgr_frames = [f for _, f in frame_pairs]
    pil_imgs = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
                for f in bgr_frames]

    print(f"  (Using {len(pil_imgs)} sampled frames from buffer)")

    env_label, env_conf = "unknown", 0.0
    if env_model is not None:
        env_label, env_conf = classify_environment(pil_imgs, env_model, device)
        print(f"  (Environment: {env_label.upper()} @ {env_conf:.0%})")

    print(f"  (Sending {len(pil_imgs)} frames to VLM...)")
    start_t = time.time()
    label, reason = run_video(
        Path(video_name), pil_imgs, bgr_frames,
        env_label, env_conf,
        args.server_url, args.max_new_tokens,
        args.jpeg_quality, args.max_image_size,
    )
    elapsed = time.time() - start_t
    print(f"  [VLM time: {elapsed:.1f}s]")

    result_out.parent.mkdir(parents=True, exist_ok=True)
    result_out.write_text(
        f"Source        : {video_name}\n"
        f"Segment       : {segment_idx}\n"
        f"Frames Used   : {len(pil_imgs)}\n"
        f"Environment   : {env_label.upper()} ({env_conf:.0%})\n"
        f"Classification: {label}\n"
        f"Reason        : {reason}\n"
    )

    icon = "✅" if label == "SAFE" else "⚠️ "
    print(f"\n  {icon}  {label}  |  ENV: {env_label.upper()} ({env_conf:.0%})")
    for chunk in [reason[i:i+70] for i in range(0, len(reason), 70)]:
        print(f"  {chunk}")
    print(f"  Saved → {result_out.name}\n")

    if label == "UNSAFE":
        event_label = infer_event_label(reason, env_label)
        threat_reasoning = build_threat_reasoning(env_label, event_label, reason)
        timestamp = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S") + "Z"
        event_id = hashlib.md5(
            f"{args.session_id}-oak-{segment_idx}-{time.time()}".encode()).hexdigest()[:12]

        mid = len(bgr_frames) // 2
        frame_path = annotate_frame(
            bgr_frames[mid], event_label, env_label,
            env_conf, event_id)

        log_event_csv(
            event_id, timestamp, video_name, len(pil_imgs),
            env_label, env_conf, event_label,
            label, reason, threat_reasoning, frame_path)

        print(f"  Event logged → {CSV_LOG.name}  (ID: {event_id})")
        print(f"  Annotated frame → {frame_path}")

        proc = show_alert_async(
            frame_path, event_label, env_label,
            video_name, args.alert_duration)
        if proc:
            alert_procs.append(proc)
            print(f"  Alert window launched (pid: {proc.pid})")

    return label


# ═══════════════════════════════════════════════════════════════════════
#  Log Viewer
# ═══════════════════════════════════════════════════════════════════════

def view_log() -> None:
    if not CSV_LOG.exists():
        print(f"No event log found at {CSV_LOG}")
        return

    with open(CSV_LOG, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        print("Event log is empty.")
        return

    print(f"\n{BORDER}")
    print(f"  EVENT LOG — {len(rows)} event(s)   [{CSV_LOG}]")
    print(BORDER)
    print(f"  {'#':<4} {'TIMESTAMP':<22} {'VIDEO':<30} {'ENV':<12} {'LABEL':<18} CONF")
    print(f"  {THIN}")

    for i, row in enumerate(rows, 1):
        print(f"  {i:<4} {row['timestamp'][:19]:<22} {row['video_source']:<30} "
              f"{row['environment'].upper():<12} {row['event_label']:<18} "
              f"{row['env_confidence']}")

    print(f"\n{THIN}")
    while True:
        choice = input(
            "\n  Enter event # to see full explanation  "
            "(or press Enter to exit): ").strip()
        if not choice:
            break
        try:
            row = rows[int(choice) - 1]
        except (ValueError, IndexError):
            print("  Invalid number.")
            continue

        reasoning_lines = row["threat_reasoning"].split(" | ")
        print(f"\n{BORDER}")
        print(f"  EVENT #{choice}  —  {row['event_label'].upper()}")
        print(BORDER)
        print(f"  Event ID    : {row['event_id']}")
        print(f"  Timestamp   : {row['timestamp']}")
        print(f"  Video       : {row['video_source']}")
        print(f"  Frames Used : {row['frames_used']}")
        print(f"  Environment : {row['environment'].upper()} ({row['env_confidence']})")
        print(f"  Threat type : {row['event_label']}")
        print(THIN)
        print("  THREAT REASONING:")
        for line in reasoning_lines:
            print(f"    {line}")
        print(THIN)
        print("  SCENE DESCRIPTION (VLM):")
        desc = row["reason"]
        for i in range(0, max(len(desc), 1), 65):
            print(f"    {desc[i:i+65]}")
        print(THIN)
        if row["frame_path"]:
            print(f"  Annotated frame : {row['frame_path']}")
        print(BORDER)


# ═══════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════

def main() -> None:
    args = parse_args()

    if args.view_log:
        view_log()
        return

    if dai is None:
        print("[ERROR] DepthAI is required for the OAK-D Lite camera. Install with:")
        print("        pip install depthai")
        sys.exit(1)

    # ── Load environment classifier ──
    env_model: Optional[torch.nn.Module] = None
    device = torch.device("cpu")
    env_weights = Path(args.env_model)
    if env_weights.exists():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Env model  : {env_weights}  (device: {device})")
        env_model = load_env_model(str(env_weights), device)
        print(f"Env classes: {ENV_CLASSES}")
    else:
        print(f"[WARN] Env model not found at '{env_weights}' — skipping env classification.")

    ALERT_DIR.mkdir(parents=True, exist_ok=True)
    FRAMES_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Server       : {args.server_url}")
    print(f"Camera       : OAK-D Lite (RGB preview {args.oak_preview_width}x{args.oak_preview_height})")
    print(f"Chunk length : {args.chunk_seconds} s  →  alert / classify each chunk")
    print(f"Frames/chunk : {args.frames} (evenly sampled from captured buffer)")
    print(f"Max Res      : {args.max_image_size}px  |  JPEG quality: {args.jpeg_quality}")
    print(f"CSV log      : {CSV_LOG}")
    print(f"\nCtrl+C to stop.\n")

    server_proc = start_server_if_needed(args.server_url)
    atexit.register(lambda: server_proc and server_proc.terminate())

    alert_procs: list = []
    segment = 0
    safe_count = 0
    unsafe_count = 0

    pipeline = build_oak_rgb_pipeline(args.oak_preview_width, args.oak_preview_height)

    # ── Connect to OAK device with retry ──
    # On Windows with native USB, the OAK-D Lite may show as X_LINK_UNBOOTED
    # on the first attempt. The DepthAI library boots it automatically when
    # a Device is created; we just need to give it enough time and retries.
    import platform
    is_windows = platform.system() == "Windows"
    device_oak = None
    max_attempts = 15
    for attempt in range(1, max_attempts + 1):
        print(f"  (Connecting to OAK-D Lite... attempt {attempt}/{max_attempts})")
        try:
            # List available devices for diagnostics
            found = dai.Device.getAllAvailableDevices()
            if found:
                for d in found:
                    print(f"    Found device: {d.getMxId()} state={d.state.name}")
            else:
                print("    No DepthAI devices detected.")
                if is_windows:
                    print("    Ensure the OAK-D Lite is plugged into a USB3 port and")
                    print("    Windows has installed the WinUSB driver (use Zadig if needed).")

            # On Windows, give the USB device extra time for boot sequence
            open_timeout = 15.0 if is_windows else 10.0
            device_oak = dai.Device(pipeline, maxUsbSpeed=dai.UsbSpeed.SUPER)
            break
        except Exception as e:
            print(f"  (Connection error: {e})")
            if attempt < max_attempts:
                wait = 5 if is_windows else 3
                print(f"  (Waiting {wait}s before retry...)")
                time.sleep(wait)
    if device_oak is None:
        print("[ERROR] Could not connect to OAK-D Lite after all attempts.")
        if is_windows:
            print("        Troubleshooting:")
            print("          1. Unplug and replug the OAK-D Lite")
            print("          2. Try a different USB3 port")
            print("          3. Install WinUSB driver via Zadig (https://zadig.akeo.ie)")
            print("          4. Run: python -c \"import depthai; print(depthai.Device.getAllAvailableDevices())\"")
        sys.exit(1)

    try:
        q_rgb = device_oak.getOutputQueue(name="rgb", maxSize=60, blocking=True)
        print("OAK-D Lite connected. Starting capture loop.\n")

        while True:
            segment += 1
            print(f"{'─'*60}")
            print(f"▶  Segment {segment}  —  capturing {args.chunk_seconds} s")
            print(f"{'─'*60}")
            sys.stdout.flush()

            try:
                raw_bgr = capture_oak_chunk_seconds(q_rgb, args.chunk_seconds)
            except Exception as e:
                print(f"  [ERROR] Camera read failed: {e}")
                continue

            if not raw_bgr:
                print("  [WARN] No frames in chunk; skipping.")
                continue

            try:
                frame_pairs = sample_frames_evenly_from_buffer(raw_bgr, args.frames)
            except Exception as e:
                print(f"  [ERROR] Frame sampling: {e}")
                continue

            video_name = f"OAK-D-Lite live (seg.{segment}, {args.chunk_seconds}s)"
            result_path = ALERT_DIR / f"oak_seg_{segment:05d}_result.txt"

            label = process_safety_segment(
                segment_idx=segment,
                video_name=video_name,
                frame_pairs=frame_pairs,
                args=args,
                env_model=env_model,
                device=device,
                result_out=result_path,
                alert_procs=alert_procs,
            )
            if label == "SAFE":
                safe_count += 1
            elif label == "UNSAFE":
                unsafe_count += 1

            print(f"  Running totals — SAFE: {safe_count}  UNSAFE: {unsafe_count}\n")

    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        try:
            device_oak.close()
        except Exception:
            pass

    open_alerts = [p for p in alert_procs if p and p.poll() is None]
    if open_alerts:
        print(f"\n  Waiting for {len(open_alerts)} alert window(s) to close...")
        for p in open_alerts:
            try:
                p.wait(timeout=args.alert_duration + 2)
            except subprocess.TimeoutExpired:
                p.kill()

    if unsafe_count:
        print(f"\n  {unsafe_count} UNSAFE event(s) logged to {CSV_LOG}")
        print("  To review with full explanation run:")
        print('    python "run_video_qwen3_vlm_llamacpp full pipeline.py" --view-log')
    print()


if __name__ == "__main__":
    main()
