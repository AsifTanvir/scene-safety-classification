# Scene Safety Classification — Live OAK-D Lite Pipeline

Real-time safety monitoring pipeline using an **OAK-D Lite** RGB camera and **Qwen3-VL-2B** multimodal LLM (served locally via `llama.cpp`). Captures live video in fixed-length chunks, classifies the environment with a fine-tuned **Places365 ResNet-18** model, then sends sampled frames to the VLM to detect weapons, violence, fire, and falls. Unsafe events are logged to CSV and trigger an on-screen alert window.

---

## Table of Contents

1. [System Requirements](#1-system-requirements)
2. [Install CUDA Toolkit & Drivers](#2-install-cuda-toolkit--drivers)
3. [Install Git for Windows (Git Bash)](#3-install-git-for-windows-git-bash)
4. [Install Miniconda](#4-install-miniconda)
5. [Create the Conda Environment](#5-create-the-conda-environment)
6. [Install Python Dependencies](#6-install-python-dependencies)
7. [Install OAK-D Lite USB Driver](#7-install-oak-d-lite-usb-driver)
8. [Download llama.cpp Pre-built Binaries](#8-download-llamacpp-pre-built-binaries)
9. [Download Model Weights](#9-download-model-weights)
10. [Download the Environment Classifier Weights](#10-download-the-environment-classifier-weights)
11. [Configure the Llama Server Script](#11-configure-the-llama-server-script)
12. [Run the Pipeline](#12-run-the-pipeline)
13. [View the Event Log](#13-view-the-event-log)
14. [Run the Evaluation Script](#14-run-the-evaluation-script)
15. [Common Errors & Fixes](#15-common-errors--fixes)

---

## 1. System Requirements

| Component | Minimum | Recommended |
|---|---|---|
| OS | Windows 10 (64-bit) | Windows 11 (64-bit) |
| GPU | NVIDIA GPU, 6 GB VRAM | RTX 4500 / RTX 3090 (8 GB+) |
| CUDA | 12.8 | 12.8+ |
| RAM | 16 GB | 32 GB |
| Python | 3.12 | 3.12 |
| Camera | OAK-D Lite (USB3) | OAK-D Lite |
| Storage | ~6 GB free (model files) | 10 GB+ |
| Shell | Git for Windows (Git Bash) | Git for Windows |

> **Note:** This pipeline is CUDA-only. CPU inference is not supported for real-time operation.
> The `.sh` server script requires **Git Bash** (included with Git for Windows) to run on Windows.

---

## 2. Install CUDA Toolkit & Drivers

### Step 2a — Verify your GPU and driver

Open **Command Prompt** or **PowerShell** and run:

```cmd
nvidia-smi
```

You should see your GPU listed with a driver version ≥ 527. If not, download and install the latest driver from:
👉 https://www.nvidia.com/drivers

### Step 2b — Install CUDA Toolkit 12.8 (or higher)

1. Go to: https://developer.nvidia.com/cuda-downloads
2. Select: **Windows → x86_64 → 11 (or 10) → exe (local)**
3. Download and run the installer
4. Choose **Custom install** and select at minimum:
   - CUDA Toolkit
   - CUDA Development Tools (nvcc)
5. After install, open a **new** Command Prompt and verify:

```cmd
nvcc --version
```

Expected output:
```
Cuda compilation tools, release 12.8, V12.8.xxx (or higher)
```

> **Tip:** If `nvcc` is not found after a restart, add `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin` to your system `PATH` manually via **System Properties → Environment Variables**.

---

## 3. Install Git for Windows (Git Bash)

Git Bash is required to run `run_llama_server.sh` on Windows.

1. Download the installer from: https://git-scm.com/download/win
2. Run the installer — use all default settings
3. When asked about **default terminal emulator**, select **"Use MinTTY"**
4. Open a new **Git Bash** window and verify:

```bash
git --version
```

---

## 4. Install Miniconda

Miniconda provides conda for managing isolated Python environments.

### Step 4a — Download and install Miniconda

1. Go to: https://docs.conda.io/en/latest/miniconda.html
2. Download **Miniconda3 Windows 64-bit** (`.exe` installer)
3. Run the installer with these settings:
   - ✅ Install for **Just Me** (recommended)
   - ✅ **Add Miniconda3 to my PATH environment variable** ← important
   - ✅ Register Miniconda3 as my default Python

### Step 4b — Verify conda is available

Open a **new** Anaconda Prompt (or Git Bash) and run:

```cmd
conda --version
```

Expected: `conda 24.x.x` (or higher)

### Step 4c — Initialize conda for Git Bash (optional but recommended)

If you want to use conda inside Git Bash:

```bash
conda init bash
```

Close and reopen Git Bash for the change to take effect.

---

## 5. Create the Conda Environment

Open **Anaconda Prompt** (or Git Bash with conda initialized) and run:

```cmd
conda create -n scene-safety python=3.12 -y
conda activate scene-safety
```

> All subsequent commands in Steps 6–12 must be run inside this activated environment.
> To confirm the environment is active, your prompt should show `(scene-safety)` at the beginning.

---

## 6. Install Python Dependencies

### Step 6a — Install all dependencies

`requirements.txt` already includes the PyTorch CUDA 12.8 index URL and all packages, so a single command installs everything:

```cmd
pip install -r requirements.txt
```

### Step 6b — Verify GPU is available in PyTorch

```cmd
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

Expected output:
```
True
NVIDIA GeForce RTX XXXX
```

If `False` is printed, PyTorch was installed without CUDA support. Reinstall:
```cmd
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

---

## 7. Install OAK-D Lite USB Driver

On Windows, the OAK-D Lite requires the **WinUSB** driver installed via Zadig.

### Step 7a — Install Zadig

1. Download Zadig from: https://zadig.akeo.ie
2. Plug the OAK-D Lite into a **USB 3** port (blue connector)
3. Open Zadig
4. Go to **Options → List All Devices**
5. Select **Movidius MyriadX** (or similar OAK-D entry) from the dropdown
6. Set the target driver to **WinUSB**
7. Click **Install Driver** (or **Replace Driver** if one is already present)

### Step 7b — Verify the camera is detected

```cmd
python -c "import depthai; print(depthai.Device.getAllAvailableDevices())"
```

You should see your OAK-D Lite listed. If the output is empty, unplug and replug the device and retry.

---

## 8. Download llama.cpp Pre-built Binaries

Instead of compiling from source, download the official pre-built Windows CUDA binary from the llama.cpp GitHub Releases page.

### Step 8a — Download the release zip

1. Go to: https://github.com/ggerganov/llama.cpp/releases
2. Find the **latest release** (top of the list)
3. Under **Assets**, download the file matching your CUDA version, for example:
   ```
   llama-bXXXX-bin-win-cuda-cu12.2.0-x64.zip
   ```
   > Pick the `cu12.x` zip that matches your installed CUDA version (12.1 → `cu12.1`, 12.4 → `cu12.4`, etc.)

### Step 8b — Extract and place the binary

1. Extract the zip file
2. Inside the extracted folder, locate `llama-server.exe`
3. In your **project root**, create the following folder structure:

   ```
   scene-safety-classification\
   └── llama.cpp\
       └── build\
           └── bin\
               └── llama-server.exe   ← place it here
   ```

   You can do this in Command Prompt:

   ```cmd
   mkdir llama.cpp\build\bin
   copy C:\path\to\extracted\llama-server.exe llama.cpp\build\bin\
   ```

4. Also copy all `.dll` files from the extracted zip into the same `llama.cpp\build\bin\` folder — they are required for the server to start:

   ```cmd
   copy C:\path\to\extracted\*.dll llama.cpp\build\bin\
   ```

### Step 8c — Verify the binary works

Open **Git Bash** in the project root and run:

```bash
./llama.cpp/build/bin/llama-server.exe --version
```

Expected: `version: 1.x.x (some commit hash)`

---

## 9. Download Model Weights

Two GGUF files are required for the Qwen3-VL-2B model:

| File | Size | Purpose |
|---|---|---|
| `Qwen_Qwen3-VL-2B-Instruct-bf16.gguf` | ~3.4 GB | Language model weights |
| `mmproj-Qwen_Qwen3-VL-2B-Instruct-bf16.gguf` | ~1.0 GB | Vision projector weights |

### Download from Hugging Face

In your activated conda environment:

```cmd
pip install hf

hf download bartowski/Qwen_Qwen3-VL-2B-Instruct-GGUF Qwen_Qwen3-VL-2B-Instruct-bf16.gguf --local-dir .

hf download bartowski/Qwen_Qwen3-VL-2B-Instruct-GGUF mmproj-Qwen_Qwen3-VL-2B-Instruct-bf16.gguf --local-dir .
```

> **Note:** Adjust the repository name/filenames to match the exact Qwen3-VL-2B GGUF release you are using. Both `.gguf` files must sit in the **project root** (same directory as `run_llama_server.sh`).

---

## 10. Download the Environment Classifier Weights

The pipeline uses a fine-tuned ResNet-18 (trained on Places365 with 3 classes: `classroom`, `home`, `office`) to provide environment context to the VLM.

Place the weights file in the project root:

```
places365_environment_model_new.pth
```

If you have trained your own model, point to it with the `--env-model` flag at runtime (see Step 12). If the file is missing, the pipeline will skip environment classification and still run — with a warning.

---

## 11. Configure the Llama Server Script

Open `run_llama_server.sh` in a text editor and verify these variables:

```bash
MODEL="Qwen_Qwen3-VL-2B-Instruct-bf16.gguf"           # must be in project root
MMPROJ="mmproj-Qwen_Qwen3-VL-2B-Instruct-bf16.gguf"   # must be in project root
BIN="llama.cpp/build/bin/llama-server"  # path to llama-server.exe (no .exe needed in Git Bash)
```

Key server flags (edit if needed):

| Flag | Default | Description |
|---|---|---|
| `-ngl 99` | 99 | GPU layers — 99 = all layers on GPU |
| `-c 4096` | 4096 | Context window size |
| `--port 8080` | 8080 | HTTP port the server listens on |
| `--host 127.0.0.1` | localhost | Bind address |

> **VRAM tip:** If you get out-of-memory errors, reduce `-ngl` (e.g., `-ngl 60`) to keep some layers in RAM.

---

## 12. Run the Pipeline

### Step 12a — Start the llama server (if not using auto-start)

Open **Git Bash** in the project root and run:

```bash
bash run_llama_server.sh
```

Wait until you see `llama server listening` in the output (~5–10 seconds on a CUDA build). Leave this terminal open.

> The pipeline can also auto-start `llama-server` if it isn't already running — the Python code detects Windows and uses Git Bash automatically, so Step 12a is optional.

### Step 12b — Run the full OAK-D Lite pipeline (live camera)

> ⚠️ **Important:** This mode requires an **OAK-D Lite camera** physically connected via USB 3. The code depends on the `depthai` library to capture live video from the OAK-D Lite sensor. If you do not have an OAK-D Lite camera, use the **video-file version** in Step 12c instead.

Open **Anaconda Prompt** (or a terminal with `scene-safety` activated):

```cmd
:: Basic run — 3-second chunks, 6 sampled frames each
python "run full pipeline.py"

:: Custom chunk length and frame count
python "run full pipeline.py" --chunk-seconds 8 --frames 8

:: Use a custom environment model path
python "run full pipeline.py" --env-model C:\path\to\your_model.pth

:: All available options
python "run full pipeline.py" --help
```

### Step 12c — Run the video-file version (no camera required)

Use this mode to classify pre-recorded video files without needing an OAK-D Lite camera.

**1. Place your videos in the `videos/` folder** (or any folder you specify with `--videos-dir`):

```
scene-safety-classification\
└── videos\
    ├── my_video_1.mp4
    ├── my_video_2.avi
    └── another_clip.mkv
```

Supported formats: `.mp4`, `.mov`, `.avi`, `.mkv`, `.webm`

**2. Run the classifier on all videos in the folder:**

```cmd
python "run video with environment classifier.py"
```

**3. Or specify a custom videos directory:**

```cmd
python "run video with environment classifier.py" --videos-dir C:\path\to\my_videos
```

**4. All available options:**

```cmd
python "run video with environment classifier.py" --help
```

Each video will be classified as **SAFE** or **UNSAFE**. Unsafe events are logged to `alert_output\safety_events.csv` with annotated frames saved to `alert_output\frames\`.

---

## 13. View the Event Log

All unsafe events are saved to `alert_output\safety_events.csv`. To review them interactively with full VLM explanations:

```cmd
python "run full pipeline.py" --view-log
```

This opens an interactive numbered list. Enter an event number to see:
- Threat type & timestamp
- Environment classification
- VLM scene description
- Path to the annotated frame image

---

## 14. Run the Evaluation Script

The evaluation script (`full_evaluation.py`) benchmarks both the environment classifier and VLM safety classifier against a ground truth CSV file. It reports per-video pass/fail results plus aggregate accuracy and latency metrics.

### Step 14a — Prepare the ground truth CSV

Create a CSV file (e.g., `videos/ground_truth.csv`) with the following columns:

| Column | Description | Example Values |
|---|---|---|
| `video_name` | Exact filename (must match the file in `--videos-dir`) | `Home fire unsafe.mp4` |
| `true_environment` | Expected environment label | `home`, `office`, `classroom` |
| `true_safety` | Expected safety classification | `SAFE`, `UNSAFE` |

Example `ground_truth.csv`:

```csv
video_name,true_environment,true_safety
Home fire unsafe.mp4,home,UNSAFE
Office Gun unsafe.mp4,office,UNSAFE
school Gun unsafe-Elephant.mp4,classroom,UNSAFE
```

> **Note:** Videos in the `--videos-dir` folder that are not listed in the CSV will be skipped with a warning.

### Step 14b — Run the evaluation

```cmd
python full_evaluation.py --eval-csv ./videos/ground_truth.csv --videos-dir videos
```

### Step 14c — Available options

| Argument | Default | Description |
|---|---|---|
| `--eval-csv` | *(required)* | Path to the ground truth CSV file |
| `--videos-dir` | `videos` | Directory containing the video files |
| `--script-path` | `run video with environment classifier.py` | Path to the VLM pipeline script |
| `--env-model` | `places365_environment_model_new.pth` | Path to environment classifier weights |
| `--server-url` | `http://127.0.0.1:8080` | llama-server URL |
| `--frames` | `6` | Number of evenly-spaced frames per video |
| `--max-image-size` | `560` | Max image dimension sent to the VLM (px) |

### Step 14d — Example output

```
▶ Evaluating Home fire unsafe.mp4
  Ground Truth -> Env: HOME | Safety: UNSAFE
  Prediction   -> Env: HOME       [Pass ✅]
  Prediction   -> Saf: UNSAFE     [Pass ✅]
  Latency      -> Env: 0.08s | Saf: 4.45s | Total: 4.86s

======================================================================
                          EVALUATION RESULTS
======================================================================
  Total Evaluated      : 21 videos
  Environment Accuracy : 76.2% (16/21)
  Safety Accuracy      : 76.2% (16/21)
  Avg Env Latency      : 0.09 seconds/video
  Avg Safety Latency   : 4.90 seconds/video
  Avg E2E Latency      : 5.35 seconds/video
======================================================================
```

### Step 14e — Add your own videos to the evaluation

You can extend the evaluation with your own video clips:

1. **Copy your video files** into the `videos/` folder (or your custom `--videos-dir`):

   ```
   videos\
   ├── existing_video.mp4
   ├── my_new_test_clip.mp4        ← add here
   └── ground_truth.csv
   ```

2. **Update `ground_truth.csv`** — add a new row for each video with the correct environment and safety label:

   ```csv
   video_name,true_environment,true_safety
   Home fire unsafe.mp4,home,UNSAFE
   Office Gun unsafe.mp4,office,UNSAFE
   my_new_test_clip.mp4,home,SAFE
   ```

   - `video_name` must **exactly match** the filename (including spaces and capitalization)
   - `true_environment` must be one of: `home`, `office`, `classroom`
   - `true_safety` must be: `SAFE` or `UNSAFE`

3. **Re-run the evaluation:**

   ```cmd
   python full_evaluation.py --eval-csv ./videos/ground_truth.csv --videos-dir videos
   ```

   Your new videos will appear in the per-video results and be included in the accuracy metrics.

> **Tip:** Videos in the folder that are not listed in the CSV are skipped with a `[WARN]` message. Videos listed in the CSV but missing from the folder will cause an error — make sure filenames match exactly.

---

## 15. Common Errors & Fixes

### `CUDA out of memory`
Reduce `-ngl` in `run_llama_server.sh` (e.g., `-ngl 60`) to offload fewer layers to GPU.

### `llama-server.exe not found` or `No such file or directory`
Verify the binary is at `llama.cpp\build\bin\llama-server.exe` and all `.dll` files from the release zip are in the same folder. In Git Bash:
```bash
ls llama.cpp/build/bin/llama-server.exe
```

### `Error: qwen3vl-2b-fp16.gguf not found`
Both GGUF files must be in the **project root** (same folder as `run_llama_server.sh`). Move them there or update the `MODEL` variable in the script.

### `No DepthAI devices detected`
- Ensure the OAK-D Lite is plugged into a **USB 3** port (blue connector)
- Confirm Zadig installed the **WinUSB** driver (Step 7a)
- Unplug and replug the device, then retry
- Verify with: `python -c "import depthai; print(depthai.Device.getAllAvailableDevices())"`

### `torch.cuda.is_available()` returns `False`
PyTorch was installed without CUDA support. Reinstall:
```cmd
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

### `bash: run_llama_server.sh: Permission denied`
Run it explicitly with bash — no chmod needed on Windows:
```bash
bash run_llama_server.sh
```

### `conda` not recognized in Git Bash
Run `conda init bash` in Anaconda Prompt, then close and reopen Git Bash. If still missing, ensure Miniconda was installed with **"Add to PATH"** checked (Step 4a).

### Server returns HTTP 400 with `"thinking"` in error body
Your llama.cpp binary doesn't support the thinking budget parameter. The pipeline automatically retries without it — this is expected behavior on older builds.

---

## License

Apache 2.0 — see [LICENSE.txt](LICENSE.txt)
