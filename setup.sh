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

# ---- Expose system PyGObject (gi) + GStreamer to the venv --------------------
# NTP-synchronized RTSP capture (src/gst_stream.py) needs PyGObject, which ships
# only as a system apt package (no working pip wheel on Jetson). A .pth makes the
# venv import it; the path is APPENDED, so the venv's own numpy/opencv still win.
# If the import check fails, install the system packages it names, then re-run.
echo "==> Linking system gi/GStreamer into the venv"
SITE=$(python -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
echo "/usr/lib/python3/dist-packages" > "$SITE/system_gi.pth"
python - <<'PY' || echo "  MISSING: sudo apt install -y python3-gi gstreamer1.0-libav \
gstreamer1.0-plugins-base gstreamer1.0-plugins-good gstreamer1.0-plugins-bad"
import gi; gi.require_version("Gst", "1.0")
from gi.repository import Gst; Gst.init(None)
print("   gi/GStreamer OK:", Gst.version_string())
PY

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
