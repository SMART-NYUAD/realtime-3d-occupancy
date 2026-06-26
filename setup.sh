#!/usr/bin/env bash
# Setup for people_tracker_3d on Jetson AGX Thor (JetPack R39).
set -e
cd "$(dirname "$0")"

echo "==> Creating virtualenv (.venv)"
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

# ---- PyTorch with CUDA for Jetson --------------------------------------------
# Thor needs the NVIDIA-built torch wheels for GPU support. The generic PyPI
# wheel is usually CPU-only on aarch64. Try the NVIDIA Jetson index first.
# Official PyTorch cu130 index ships CUDA-enabled aarch64 (sbsa) wheels that
# run on Thor (verified: torch 2.12.1+cu130, torch.cuda.is_available()==True).
echo "==> Installing PyTorch (CUDA 13 build)"
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130

echo "==> Installing remaining requirements"
pip install -r requirements.txt

echo
echo "==> Verifying"
python - <<'PY'
import torch
print("torch", torch.__version__, "CUDA available:", torch.cuda.is_available())
import cv2; print("opencv", cv2.__version__)
import ultralytics; print("ultralytics", ultralytics.__version__)
PY

echo
echo "Done. Next:"
echo "  source .venv/bin/activate"
echo "  python calibrate.py        # set up floor coordinates"
echo "  python track.py            # run real-time tracking"
