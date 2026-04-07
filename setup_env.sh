#!/usr/bin/env bash

set -euo pipefail

ENV_NAME="${1:-smolvlm}"
CONDA_BASE="/home/jetson/miniconda3"
CONDA_SH="$CONDA_BASE/etc/profile.d/conda.sh"

if [[ ! -f "$CONDA_SH" ]]; then
  echo "Conda init script not found at $CONDA_SH"
  exit 1
fi

source "$CONDA_SH"

echo "Creating conda environment: $ENV_NAME"
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "Environment $ENV_NAME already exists, reusing it."
else
  conda create -n "$ENV_NAME" python=3.10 -y
fi

conda activate "$ENV_NAME"

echo "Upgrading pip and installing SmolVLM dependencies"
python -m pip install --upgrade pip
python -m pip install \
  numpy==1.26.4 \
  datasets \
  pillow \
  sentencepiece \
  transformers==4.48.3 \
  accelerate==0.34.2 \
  safetensors \
  huggingface_hub

echo "Checking for PyTorch in $ENV_NAME"
if python -c "import torch; print(torch.__version__)" >/dev/null 2>&1; then
  TORCH_VERSION="$(python -c "import torch; print(torch.__version__)")"
  echo "PyTorch is installed: $TORCH_VERSION"
  python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
else
  cat <<'EOF'
PyTorch is not installed in this environment.

On Jetson, do not install a generic pip torch wheel.
Install the NVIDIA Jetson PyTorch build that matches your JetPack version, then rerun:

  conda activate smolvlm
  python -c "import torch; print(torch.__version__, torch.cuda.is_available())"

After PyTorch is working, run:

  python /home/jetson/Projects/JetsonVLM/run_smolvlm.py --image /path/to/image.jpg
  python /home/jetson/Projects/JetsonVLM/run_smolvlm_video.py --video /path/to/video.mp4
EOF
fi

echo "Setup complete for conda env: $ENV_NAME"
