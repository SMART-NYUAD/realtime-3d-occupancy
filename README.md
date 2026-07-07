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
  straight down onto its floor plane in its **true scan colour** (a clean
  mean-colour orthophoto; `color_mode: height` shades by height instead). This
  is the up-to-date reference.
- **`dxf`** — the legacy BIM plan `SMART-floorplans.dxf`. Kept as a fallback;
  being retired because it's out of date.

Preview the map (and warm its render cache) before calibrating:

```bash
python topdown.py --show               # writes topdown.png (true colour)
python topdown.py --mode height --show # height-shaded instead of true colour
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

To publish positions to MQTT, set `output.mqtt.enabled: true` in `config.yaml`,
fill in the broker settings, and copy `.env.example` to `.env` with
`MQTT_USERNAME` / `MQTT_PASSWORD`. The tracker publishes TAC-B-style `position`
messages to `output.mqtt.topic`, under `smx/device/...` by default, so existing
subscribers on `smx/device/#` can consume them.

### Face blur in previews (privacy)

The preview camera windows (and `--headless` snapshots) can blur every face so a
shoulder-surfer or a saved screenshot never exposes identities. This is
**preview-only**: the frames fed to the person detector, homography and fusion
are untouched, so tracking accuracy is unchanged. Faces are detected with the
[`deface`](https://github.com/ORB-HD/deface) library's CenterFace model
(`pip install deface`, in `requirements.txt`; the model ships bundled).

Configure it under `output.blur_faces` in `config.yaml`:

- `method` — `blur` | `pixelate` | `solid`
- `threshold` — face-detection confidence (default `0.2`; kept low because
  small/far faces score low)
- `det_size` — detector resolution; **`0` = full frame** (best for small faces),
  a positive value caps the long side for speed
- `ellipse`, `mask_scale`, `mosaicsize` — blur shape/size

If `deface` isn't installed, face blur simply stays off and tracking runs
normally. Full-resolution detection is the biggest CPU cost of the preview — if
previews get slow, set a `det_size` (e.g. `960`) or turn `blur_faces` off.

## Synchronisation (NTP) — why one person isn't two

Cameras have different end-to-end latency (measured here: ~120 ms on one camera,
~430 ms on two others). If you fuse "whatever each camera last sent", a moving
person is at different places in each camera's *stale* frame; when those differ
by more than `fusion.merge_distance_m` they stop merging and you get **two dots
for one person**.

Fix: the cameras are **chrony/NTP clients of this host**, so every frame carries
an RTCP capture timestamp on a clock shared across cameras. With `sync.enabled:
true`, `track.py` buffers a second of timestamped frames per camera and, each
step, lines them all up to a **common capture instant** (the measured ~300 ms
skew drops to ~20 ms) before detecting and fusing.

- RTSP/H.264 sources only; tune in the `sync:` block of `config.yaml`.
- Needs system **PyGObject + GStreamer** (`setup.sh` links them into the venv;
  it prints the `apt` packages to install if they're missing).
- `decoder: sw` (openh264) is the portable default; `hw`/`nvdec` use Jetson NVDEC.
- `protocol: udp` keeps latency *live*: over `tcp`, packet loss makes the jitter
  buffer ratchet up and never drain, so one camera slowly drifts ~1s behind the
  rest. `drop_on_latency`/`retransmission: false` bound it further. Use `tcp` only
  if a stream won't stay connected over udp.
- A camera that still lags more than `lag_budget_ms` behind the most-live one is
  dropped from the aligned set rather than dragging every camera back to its time
  (and freezing them once the gap exceeds `buffer_sec`).
- Verify the live skew yourself: `python src/gst_stream.py config.yaml <decoder> <protocol>`
  (e.g. `... config.yaml hw udp`), or watch the `[sync] spread=…` line `track.py`
  now prints each second.

This addresses lag-induced duplicates. Duplicates from a camera that *can't see
your feet* (it places you wrong, not late) are a separate, foot-point problem —
see **Ground-plane blob intersection** below.

## Ground-plane blob intersection — why one person isn't two (the other half)

The single foot point (bbox bottom-center through the homography) is exact only
when the feet are visible. When they're occluded (behind a desk) or clipped
(below the frame), the box bottom sits *above* the real feet, so that camera
places the person too far away. If the gap exceeds `fusion.merge_distance_m`, the
person stops merging and you get **two dots** — even with perfect sync.

The fix (`fusion.use_blob: true`, on by default) projects each detection's whole
bounding box to a **floor trapezoid of foot-point uncertainty** instead of one
point, and puts the person where the cameras' trapezoids **overlap**:

```
        camera that SEES the feet           camera that CAN'T (occluded/clipped)
        ─────────────────────────           ────────────────────────────────────
        short trapezoid at the feet         long trapezoid reaching to the camera
                    ╲   ╱                        the feet are "somewhere in here"
                     ╲ ╱   ── intersection ──▶  ◀ collapses onto the true feet
```

Because occlusion only ever makes the foot look *too far*, the trapezoid reaches
mostly toward the camera (tunable in the `fusion:` block: `blob_near_frac`,
`blob_body_aspect`, `blob_clip_inflate`, `blob_short_aspect`). A partial
torso-only box (short relative to its width, bottom edge at the waist) reaches
even more aggressively, scaled by how far its aspect ratio falls below
`blob_short_aspect`. Two detections merge when their
trapezoids overlap — not only when their foot points are close — which is what
actually rejoins the split. Set `fusion.show_blobs: true` to draw the trapezoids
and their intersection on the top-down map. Trapezoids that don't overlap (e.g.
calibration error) fall back to the old point-mean, so it never does harm.

Limits: very heavy occlusion *near the horizon* (a few visible pixels mapping to
many meters) can drift beyond `blob_merge_gate_m` and still split — no worse than
before. See `src/groundblob.py`.

## Accuracy notes (monocular limits)

- Floor (X, Y) is solid when feet are visible and the floor is flat.
- If feet are occluded (behind a desk), the single-point estimate drifts — the
  box bottom is no longer the real foot point. Multi-camera **blob intersection**
  (above) recovers it when another camera sees the feet.
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
| `src/capture.py` | RTSP / USB / CSI / file camera capture (cv2) |
| `src/gst_stream.py` | NTP-synced RTSP capture (GStreamer, timestamped frames) |
| `src/sync.py` | align cameras to a common capture instant |
| `src/detector.py` | YOLO person detect + ByteTrack |
| `src/geometry.py` | image→world homography math |
| `src/groundblob.py` | bbox→floor trapezoid + cross-camera blob intersection |
| `src/plan.py` | shared world↔pixel frame + `load_plan(cfg)` factory |
| `src/pointcloud_plan.py` | top-down map rendered from the `.las` scan |
| `src/floorplan.py` | DXF floorplan render (legacy `source: dxf`) |
| `src/visualizer.py` | annotated camera-view overlay |
| `src/face_blur.py` | `deface`/CenterFace face blur for privacy-preserving previews |
