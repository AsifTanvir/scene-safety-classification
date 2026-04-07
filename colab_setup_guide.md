# Google Colab Setup Guide — Qwen3-VL Safety Classification

This guide walks you through running the `run_video_qwen3_llamacpp.py` and
`run_qwen3_vlm_llamacpp.py` (image) scripts on **Google Colab** using a free
T4 GPU.

---

## Prerequisites

- A Google account with access to [colab.research.google.com](https://colab.research.google.com)
- Your pipeline scripts from this repo
- Video/image files you want to classify

---

## Step 1 — Enable GPU Runtime

1. Open a **new notebook** in Google Colab.
2. Go to **Runtime → Change runtime type**.
3. Set **Hardware accelerator** to **T4 GPU**.
4. Click **Save**.

Verify the GPU is active by running in a cell:

```python
!nvidia-smi
```

You should see a Tesla T4 listed.

---

## Step 2 — Build `llama.cpp` with CUDA

Colab ships with CUDA and build tools pre-installed. Run the following in a
code cell to compile the server binary:

```bash
%%bash
# Clone the repo
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp

# Build with CUDA support — uses all available CPU cores
cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j$(nproc)
```

> **Note:** Compilation takes 3–5 minutes. The server binary will be at
> `llama.cpp/build/bin/llama-server`.

---

## Step 3 — Download the GGUF Model Files

Install the Hugging Face CLI, then download both model files:

```bash
!pip install -q "huggingface_hub[cli]"
```

```bash
%%bash
# Language model (Q4_K_M quantized, ~1.1 GB)
huggingface-cli download bartowski/Qwen2.5-VL-3B-Instruct-GGUF \
  "Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf" \
  --local-dir . --local-dir-use-symlinks False

# Vision encoder / mmproj (FP16, ~782 MB)
# Source: same repo or the official Qwen GGUF repo
huggingface-cli download bartowski/Qwen2.5-VL-3B-Instruct-GGUF \
  "mmproj-Qwen2.5-VL-3B-Instruct-f16.gguf" \
  --local-dir . --local-dir-use-symlinks False
```

> **Tip:** If you already quantized your mmproj to Q5_K_M on the Jetson and
> want to reuse it, upload that file instead and adjust the filename below.

---

## Step 4 — Install Python Dependencies

```bash
!pip install -q Pillow opencv-python-headless numpy
```

> Use `opencv-python-headless` on Colab — it avoids display-related errors
> that occur with the standard `opencv-python` package in a headless environment.

---

## Step 5 — Upload Scripts and Media

In the left sidebar, click the **📁 Folder icon** to open the File Explorer.

Upload the following files to `/content/`:

| File | Description |
|---|---|
| `run_video_qwen3_llamacpp.py` | Video classification script |
| `run_qwen3_vlm_llamacpp.py` | Image classification script |
| `run_llama_server.sh` | Server launcher script |

Then create two folders and upload your media:

```bash
!mkdir -p /content/videos /content/images
```

Upload your `.mp4` video files into `/content/videos/` and `.jpg`/`.png`
images into `/content/images/` using the sidebar.

---

## Step 6 — Patch `run_llama_server.sh` for Colab

The Jetson version uses hardcoded absolute paths. Patch the script to work with
Colab's `/content/` working directory:

```bash
%%bash
cat > /content/run_llama_server.sh << 'EOF'
#!/bin/bash
cd /content || exit 1

export LD_LIBRARY_PATH="/content/llama.cpp/build/bin"

MODEL="Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf"
MMPROJ="mmproj-Qwen2.5-VL-3B-Instruct-f16.gguf"
BIN="llama.cpp/build/bin/llama-server"

if [ ! -f "$MODEL" ]; then echo "Error: $MODEL not found!"; exit 1; fi
if [ ! -f "$MMPROJ" ]; then echo "Error: $MMPROJ not found!"; exit 1; fi

echo "Starting llama-server for $MODEL on port 8080..."

exec $BIN \
    -m "$MODEL" \
    --mmproj "$MMPROJ" \
    --port 8080 \
    --host 127.0.0.1 \
    -ngl 99 \
    -c 4096 \
    -b 2048 \
    -ub 512 \
    -np 1 \
    --no-mmap
EOF

chmod +x /content/run_llama_server.sh
echo "Server script patched!"
```

> **Important:** Update `MODEL` and `MMPROJ` filenames above to exactly match
> the files you downloaded in Step 3.

---

## Step 7 — Run the Classification Pipeline

Make sure your working directory is `/content/` before running:

```bash
%cd /content
```

---

### 🎬 Video Script — `run_video_qwen3_vlm_llamacpp.py`

#### Parameters

| Parameter | Jetson Default | Colab T4 Recommended | Description |
|---|---|---|---|
| `--videos-dir` | `videos` | `videos` | Folder containing `.mp4`, `.mov`, `.avi`, `.mkv`, `.webm` files |
| `--frames` | `6` | `6` | Number of evenly-spaced frames to extract from each video (max 8) |
| `--max-image-size` | `560` | **`672`** | Max width/height in pixels per frame — T4 has extra VRAM headroom |
| `--max-new-tokens` | `512` | `512` | Max tokens the model can generate in its response |
| `--jpeg-quality` | `85` | `85` | JPEG compression quality for frames (1–95) |
| `--server-url` | `http://127.0.0.1:8080` | `http://127.0.0.1:8080` | llama-server endpoint |

#### Usage Examples

**Default run (recommended starting point):**
```bash
!python run_video_qwen3_vlm_llamacpp.py --videos-dir videos
```

**Higher resolution for better small-object detection (Colab T4 has headroom):**
```bash
!python run_video_qwen3_vlm_llamacpp.py --videos-dir videos \
  --max-image-size 672 \
  --frames 6
```

**More frames for longer or complex videos:**
```bash
!python run_video_qwen3_vlm_llamacpp.py --videos-dir videos \
  --frames 8 \
  --max-image-size 448
```

**Fast/lightweight run to save time:**
```bash
!python run_video_qwen3_vlm_llamacpp.py --videos-dir videos \
  --frames 4 \
  --max-image-size 336 \
  --max-new-tokens 128
```

#### Sample Terminal Output

```
Server : http://127.0.0.1:8080
Videos : 3 found in 'videos'
Frames : 6 per video
Max Res: 560px  |  JPEG quality: 85

  (Starting llama-server automatically... ~5-10s)
  (Server ready!)
────────────────────────────────────────────────────────────
▶  nashville_shooting.mp4
────────────────────────────────────────────────────────────
  (Extracting 6 frames...)
  (Frames normalized to 560x315, JPEG quality=85)
  (Sending 6 labeled frames to llama-server...)
  [Time: 7.2s]

  ⚠️  UNSAFE  —  The video shows a person walking through a corridor carrying a long firearm, moving in a deliberate and threatening manner.
  Saved → nashville_shooting_result.txt

────────────────────────────────────────────────────────────
▶  office_hallway.mp4
────────────────────────────────────────────────────────────
  (Extracting 6 frames...)
  (Frames normalized to 560x315, JPEG quality=85)
  (Sending 6 labeled frames to llama-server...)
  [Time: 4.8s]

  ✅  SAFE  —  The video shows a normal office hallway with people walking between rooms, no weapons or hazards visible.
  Saved → office_hallway_result.txt

============================================================
                         SUMMARY
============================================================
⚠️  [UNSAFE]  nashville_shooting.mp4  The video shows a person walking through ...
✅  [SAFE  ]  office_hallway.mp4      The video shows a normal office hallway wi...
============================================================
```

---

### 🖼️ Image Script — `run_image_qwen3_vlm_llamacpp.py`

#### Parameters

| Parameter | Jetson Default | Colab T4 Recommended | Description |
|---|---|---|---|
| `--images-dir` | `images` | `images` | Folder containing `.jpg`, `.jpeg`, `.png`, `.webp` files |
| `--max-image-size` | `560` | **`672`** | Max width/height in pixels — T4 has extra VRAM headroom |
| `--max-new-tokens` | `256` | `256` | Max tokens the model can generate in its response |
| `--server-url` | `http://127.0.0.1:8080` | `http://127.0.0.1:8080` | llama-server endpoint |

#### Usage Examples

**Default run:**
```bash
!python run_image_qwen3_vlm_llamacpp.py --images-dir images
```

**Higher resolution on Colab T4 (more VRAM available):**
```bash
!python run_image_qwen3_vlm_llamacpp.py --images-dir images \
  --max-image-size 672
```

**Custom subfolder and longer descriptions:**
```bash
!python run_image_qwen3_vlm_llamacpp.py \
  --images-dir /content/my_cctv_frames \
  --max-image-size 560 \
  --max-new-tokens 512
```

#### Sample Terminal Output

```
Server : http://127.0.0.1:8080
Images : 4 found in 'images'
Max Res: 560px

  (llama-server is already running...)
────────────────────────────────────────────────────────────
▶  fall1.jpg
────────────────────────────────────────────────────────────
  (Resizing 1200x628 → max 560px...)
  (Sending to llama-server via HTTP...)
  [Time elapsed: 5.9s]

  ⚠️  UNSAFE  —  An elderly man is lying on the floor next to a wooden cane, indicating he has fallen and is in distress.
  Saved → fall1_text.txt

────────────────────────────────────────────────────────────
▶  gun1.jpg
────────────────────────────────────────────────────────────
  (Resizing 1280x975 → max 560px...)
  (Sending to llama-server via HTTP...)
  [Time elapsed: 1.8s]

  ⚠️  UNSAFE  —  Multiple individuals are holding rifles with visible magazines, constituting a clear armed threat.
  Saved → gun1_text.txt

────────────────────────────────────────────────────────────
▶  celebration.jpg
────────────────────────────────────────────────────────────
  (Sending to llama-server via HTTP...)
  [Time elapsed: 1.4s]

  ✅  SAFE  —  The image shows people gathered at an outdoor celebration with no visible weapons, violence, or hazards.
  Saved → celebration_text.txt

============================================================
                         SUMMARY
============================================================
⚠️  [UNSAFE]  fall1.jpg        An elderly man is lying on the floor next to a ...
⚠️  [UNSAFE]  gun1.jpg         Multiple individuals are holding rifles with vis...
✅  [SAFE  ]  celebration.jpg  The image shows people gathered at an outdoor ce...
============================================================
```

> [!NOTE]
> Result files are saved next to each input file with `_result.txt` (video) or `_text.txt` (image) suffix. They contain the filename, classification, and reason.

---

## Step 8 — Download Results

After classification finishes, download your result files from the sidebar, or
zip and download them all at once:

```bash
!zip -r /content/results.zip /content/videos/*.txt /content/images/*.txt
```

Then click the zip file in the sidebar to download it.

---

## Troubleshooting

| Error | Fix |
|---|---|
| `CUDA out of memory` | Lower `--max-image-size` to `336` or reduce `--frames` to `4` |
| `Model not found` | Double-check filenames in `run_llama_server.sh` match downloaded files |
| `Connection refused` | The server may still be loading — wait 20s and re-run the script |
| `HTTP 500 Internal Server Error` | The image resolution may be hitting a patch-alignment edge case — the scripts already handle this with 56px snapping |
| `opencv-python` headless error | Make sure you installed `opencv-python-headless` not `opencv-python` |

---

## Performance Expectations on Colab T4

| Metric | Jetson Orin Nano (8GB) | Colab T4 (16GB) |
|---|---|---|
| Model load time | ~10s | ~8s |
| Per-image inference | ~2–6s | ~1–3s |
| Per-video inference (6 frames) | ~5–12s | ~3–7s |
| Max stable resolution | 560px | 672px (more headroom) |

> **Tip:** On Colab T4 you have 16GB VRAM vs 8GB on the Jetson, so you can
> safely push `--max-image-size` up to `672` for even better detection quality.

