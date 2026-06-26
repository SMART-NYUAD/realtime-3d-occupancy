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

## The world frame (top-down map)

Every camera calibrates into ONE shared **top-down map**, in meters, origin at
the map's bottom-left corner — so positions are consistent across all cameras.

Two sources can provide that map (`floorplan.source` in `config.yaml`):

- **`pointcloud`** (default) — a top-down view rendered from the lab scan
  `smart_lab.las`. The scan is Y-up (floor = X-Z plane), so it's projected
  straight down onto its floor plane and shaded by height. This is the
  up-to-date reference.
- **`dxf`** — the legacy BIM plan `SMART-floorplans.dxf`. Kept as a fallback;
  being retired because it's out of date.

Preview the map (and warm its render cache) before calibrating:

```bash
python topdown.py --show              # writes topdown.png
python topdown.py --mode rgb --show   # true-colour instead of height shading
```

> ⚠️ The two sources are **different frames** (origin, orientation and up-axis
> all differ), so a homography or reference point made against one does **not**
> transfer to the other. After switching `source`, **recalibrate every camera**
> (and start a fresh `reference_points.json`).

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

No tape-measure needed — the map's known scale (point cloud or DXF) supplies the
meters. On the point-cloud map, pick spots you can also identify in the camera:
floor/wall corners, bench ends, equipment, the ceiling-beam grid.

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
| `config.yaml` | all settings (incl. `floorplan.source`) |
| `calibrate.py` | camera↔map point matching → homography |
| `track.py` | main real-time loop |
| `topdown.py` | preview/export the top-down map (PNG) |
| `src/capture.py` | RTSP / USB / CSI / file camera capture |
| `src/detector.py` | YOLO person detect + ByteTrack |
| `src/geometry.py` | image→world homography math |
| `src/plan.py` | shared world↔pixel frame + `load_plan(cfg)` factory |
| `src/pointcloud_plan.py` | top-down map rendered from the `.las` scan |
| `src/floorplan.py` | DXF floorplan render (legacy `source: dxf`) |
| `src/visualizer.py` | annotated camera-view overlay |
