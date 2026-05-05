#!/usr/bin/env python3
import argparse
import csv
import importlib.util
import sys
import time
import atexit
from pathlib import Path

import cv2
import torch
from PIL import Image

# ═══════════════════════════════════════════════════════════════════════
#  Dynamic Module Loader (Handles spaces in filenames)
# ═══════════════════════════════════════════════════════════════════════
def load_original_script(file_path: str):
    """Dynamically load the original python script as a module."""
    script_path = Path(file_path)
    if not script_path.exists():
        print(f"[ERROR] Original script not found at {script_path}")
        sys.exit(1)
        
    spec = importlib.util.spec_from_file_location("vlm_pipeline", str(script_path))
    vlm_pipeline = importlib.util.module_from_spec(spec)
    sys.modules["vlm_pipeline"] = vlm_pipeline
    spec.loader.exec_module(vlm_pipeline)
    return vlm_pipeline


# ═══════════════════════════════════════════════════════════════════════
#  Evaluation Logic
# ═══════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="Evaluate Safety and Environment Classifiers")
    parser.add_argument("--script-path", default="run video with environment classifier.py", help="Path to your original script")
    parser.add_argument("--videos-dir", default="videos", help="Directory with videos")
    parser.add_argument("--eval-csv", required=True, help="Path to ground truth CSV")
    parser.add_argument("--env-model", default="places365_environment_model_new.pth", help="Path to env model weights")
    parser.add_argument("--server-url", default="http://127.0.0.1:8080", help="llama-server URL")
    parser.add_argument("--frames", type=int, default=6)
    parser.add_argument("--max-image-size", type=int, default=336)
    args = parser.parse_args()

    # Load the original script's functions
    vlm_pipeline = load_original_script(args.script_path)

    # 1. Load Ground Truth CSV
    eval_csv_path = Path(args.eval_csv)
    if not eval_csv_path.exists():
        print(f"[ERROR] Ground truth CSV not found at {eval_csv_path}")
        return

    ground_truth = {}
    with open(eval_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ground_truth[row["video_name"].strip()] = {
                "env": row["true_environment"].strip().lower(),
                "safety": row["true_safety"].strip().upper()
            }

    # 2. Setup Device & Models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env_weights = Path(args.env_model)
    env_model = None
    if env_weights.exists():
        env_model = vlm_pipeline.load_env_model(str(env_weights), device)
    else:
        print(f"[ERROR] Environment model not found at {env_weights}.")
        return

    # 3. Ensure Llama Server is running
    server_proc = vlm_pipeline.start_server_if_needed(args.server_url)
    if server_proc:
        atexit.register(lambda: server_proc.terminate())

    # 4. Collect Videos
    videos_dir = Path(args.videos_dir)
    video_paths = vlm_pipeline.collect_video_paths(videos_dir)
    
    # 5. Tracking Metrics
    env_correct = 0
    safety_correct = 0
    total_latency = 0.0
    total_env_latency = 0.0
    total_safety_latency = 0.0
    total_evaluated = 0

    print(f"\n{'='*70}")
    print(f"{'STARTING EVALUATION':^70}")
    print(f"{'='*70}")

    for video_path in video_paths:
        if video_path.name not in ground_truth:
            print(f"[WARN] Skipping {video_path.name} — no labels found in CSV.")
            continue

        gt = ground_truth[video_path.name]
        print(f"\n▶ Evaluating {video_path.name}")
        print(f"  Ground Truth -> Env: {gt['env'].upper()} | Safety: {gt['safety']}")

        start_time = time.time()

        try:
            # Reusing the extraction logic from your original file
            frame_pairs = vlm_pipeline.extract_frames(video_path, args.frames)
            bgr_frames = [f for _, f in frame_pairs]
            pil_imgs = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in bgr_frames]

            # Environment Classification
            env_start_time = time.time()
            env_label, env_conf = vlm_pipeline.classify_environment(pil_imgs, env_model, device)
            env_latency = time.time() - env_start_time

            # Safety Classification
            safety_start_time = time.time()
            safety_label, reason = vlm_pipeline.run_video(
                video_path=video_path,
                pil_imgs=pil_imgs,
                bgr_frames=bgr_frames,
                env_label=env_label,
                env_conf=env_conf,
                server_url=args.server_url,
                max_tokens=512,
                jpeg_quality=85,
                max_size=args.max_image_size
            )
            safety_latency = time.time() - safety_start_time

            latency = time.time() - start_time

            # Update Metrics
            env_match = (env_label.strip().lower() == gt["env"])
            safety_match = (safety_label.strip().upper() == gt["safety"])

            if env_match: env_correct += 1
            if safety_match: safety_correct += 1
            total_latency += latency
            total_env_latency += env_latency
            total_safety_latency += safety_latency
            total_evaluated += 1

            # Print individual result
            print(f"  Prediction   -> Env: {env_label.upper():<10} [{'Pass ✅' if env_match else 'Fail ❌'}]")
            print(f"  Prediction   -> Saf: {safety_label.upper():<10} [{'Pass ✅' if safety_match else 'Fail❌'}]")
            print(f"  Latency      -> Env: {env_latency:.2f}s | Saf: {safety_latency:.2f}s | Total: {latency:.2f}s")

        except Exception as e:
            print(f"  [ERROR] Processing {video_path.name} failed: {e}")

    # 6. Final Evaluation Output
    if total_evaluated > 0:
        print(f"\n{'='*70}")
        print(f"{'EVALUATION RESULTS':^70}")
        print(f"{'='*70}")
        print(f"  Total Evaluated      : {total_evaluated} videos")
        print(f"  Environment Accuracy : {env_correct / total_evaluated:.1%} ({env_correct}/{total_evaluated})")
        print(f"  Safety Accuracy      : {safety_correct / total_evaluated:.1%} ({safety_correct}/{total_evaluated})")
        print(f"  Avg Env Latency      : {total_env_latency / total_evaluated:.2f} seconds/video")
        print(f"  Avg Safety Latency   : {total_safety_latency / total_evaluated:.2f} seconds/video")
        print(f"  Avg E2E Latency      : {total_latency / total_evaluated:.2f} seconds/video")
        print(f"{'='*70}\n")
    else:
        print("\n[INFO] No videos were evaluated. Ensure the filenames in the CSV exactly match the files in the directory.")

if __name__ == "__main__":
    main()
