# people_tracker_3d

Real-time people tracking with **real-world floor coordinates (meters)** on an
NVIDIA Jetson AGX Thor, using a single RGB camera (USB / CSI / video file).

## How it works

A normal camera can't measure depth. But people stand on a **flat floor**, so
the bottom-center of each person's bounding box (their feet) lies on the ground
plane. A one-time **homography** calibration maps floor pixels → real meters.

```
  camera frame ─▶ YOLO detect (people) ─▶ ByteTrack (stable IDs)
                                              │
                            feet = bottom-center of each box
                                              │
                          homography H : image px ─▶ floor (X,Y) m
                                              │
                         ┌────────────────────┴───────────────────┐
                    annotated video                    top-down floor map
```

Each tracked person gets: a persistent **ID**, a screen bounding box, and an
**(X, Y) position in meters** on your floor plan.

## Setup

```bash
cd ~/people_tracker_3d
./setup.sh
source .venv/bin/activate
```

`setup.sh` creates a venv and installs CUDA PyTorch (Jetson wheel) + Ultralytics
YOLO + OpenCV.

The **world frame is the DXF floorplan** (`SMART-floorplans.dxf`), in meters,
origin at the plan's bottom-left corner. Every camera calibrates into this same
frame, so positions are consistent across all 6 cameras.

## 1. Configure the camera

Edit `config.yaml` → `camera.source`:
- `rtsp://192.168.50.72/ch0_0.h264` → RTSP IP camera (Yi Home)
- `usb:0` → USB webcam · `csi:0` → Jetson CSI · `/path/video.mp4` → file

Set `geometry.homography_file` to a per-camera name (e.g. `yi01_homography.npy`).

## 2. Calibrate (once per camera placement)

```bash
python calibrate.py
```
Two windows open — the **camera** and the **floorplan**:
- Press **SPACE** to freeze a camera frame.
- Click a **floor** point in the camera, then click the **same physical spot**
  on the floorplan. Repeat for **≥4 well-spread points** (room/bench corners,
  door edges, floor marks).
- **ENTER** to finish. Saves the homography and prints reprojection error (< 0.3 m good).

No tape-measure needed — the floorplan's known scale supplies the meters.

## 3. Run real-time tracking

```bash
python track.py
```
Opens two windows: the annotated **camera** view and the **top-down floor map**.
Press `q` to quit. Set `output.show_window: false` in `config.yaml` to run
headless and stream coordinates to stdout instead.

## Accuracy notes (monocular limits)

- Floor (X, Y) is solid when feet are visible and the floor is flat.
- If feet are occluded (behind a desk), the estimate drifts — the box bottom is
  no longer the real foot point.
- True height/Z is not measured; only a rough estimate is possible.
- For centimeter-accurate or multi-floor 3D, use a depth camera (RealSense /
  ZED / OAK-D) — the detector/tracker stay the same, only `geometry.py` changes
  to read per-pixel depth instead of a homography.

## Files

| file | role |
|------|------|
| `config.yaml` | all settings |
| `calibrate.py` | camera↔floorplan point matching → homography |
| `track.py` | main real-time loop |
| `src/capture.py` | RTSP / USB / CSI / file camera capture |
| `src/detector.py` | YOLO person detect + ByteTrack |
| `src/geometry.py` | image→world homography math |
| `src/floorplan.py` | DXF floorplan render + world↔pixel mapping |
| `src/visualizer.py` | annotated camera-view overlay |
