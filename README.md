# people_tracker_3d

Real-time multi-camera people tracking in **real-world floor coordinates
(meters)** on an NVIDIA Jetson AGX Thor, using ordinary RGB IP cameras.

## How it works

A normal camera can't measure depth, but people stand on a **flat floor**, so
their feet lie on the ground plane. A one-time **homography** calibration per
camera maps floor pixels → meters on one shared top-down map.

```
 3 RTSP cameras ─▶ NVDEC decode + NTP time-alignment (one frame per camera, same instant)
                          │
                ONE batched YOLO11-pose pass (TensorRT FP16)
                          │
          per-camera ByteTrack ─▶ foot point (ankles, or extrapolated + sigma)
                          │
             homography H: image px ─▶ floor (X, Y) m
                          │
        cross-camera fusion (inverse-variance + ground-blob intersection)
                          │
          ┌───────────────┼──────────────────┐
     floor map        camera mosaic        MQTT
                   (heads masked)
```

Each person gets a persistent **ID** and an **(X, Y) position in meters**.

## Setup

```bash
cd ~/people_tracker_3d
./setup.sh                 # venv, CUDA PyTorch, deps, GStreamer link, TensorRT engine
source .venv/bin/activate
```

The detector runs as a **TensorRT FP16 engine** (`yolo11m-pose.engine`). `setup.sh`
builds it; rebuild it after upgrading Ultralytics/TensorRT or changing
`detector.imgsz`/`max_batch`:

```bash
python tools/export_engine.py
```

If the engine is missing, `track.py` falls back to the `.pt` weights (≈3× slower).

## Repository layout

| path | role |
|------|------|
| `track.py` | main real-time loop |
| `calibrate.py` | click camera↔map point pairs → homography |
| `recalibrate.py` | live tuner: drag points while watching your tracked dot |
| `config.yaml` / `bytetrack.yaml` | all settings / tracker thresholds |
| `tracker3d/` | the library (below) |
| `tools/` | `export_engine.py` (TensorRT), `check_sync.py` (live skew), `topdown.py` (map PNG) |
| `tests/test_core.py` | geometry / fusion / tracker-glue / privacy tests |
| `calib/` | per-camera `<name>_homography.npy` + `.json` sidecar (image size, clicked points) |
| `data/` | `smart_lab.las` (world map scan), `reference_points.json`, `gs_lod2.sog` (splat, unused yet) |
| `docs/REVIEW.md` | code review: bugs found/fixed, open issues, roadmap |

`tracker3d/`: `config` (config/env), `capture` (USB/file/FFMPEG), `gst_stream` +
`sync` (NTP-synced RTSP), `detector` (batched TensorRT + ByteTrack), `geometry`
(homography, foot point), `localize` (detection → floor), `groundblob` +
`fusion` (cross-camera merge), `plan` (point-cloud world map), `privacy` (head
masking), `render` (preview), `mqtt_output`.

## The world frame (top-down map)

Every camera calibrates into ONE shared top-down map rendered from the lab
point-cloud scan `data/smart_lab.las` (projected straight down, true scan
colour; `color_mode: height` shades by height). Meters, origin at the scan
footprint's min corner. Preview it (and warm its render cache):

```bash
python tools/topdown.py --show
```

> ⚠️ `flip_x`/`flip_y`/`clip_percentile`/`up_axis` move the frame itself — change
> them and every camera must be recalibrated.

## 1. Configure the cameras

`config.yaml` → `cameras:` — one entry per camera with its `source`
(`rtsp://…`, `usb:0` or a video file) and `homography_file`
(`calib/<name>_homography.npy`).

## 2. Calibrate (once per camera placement)

```bash
python calibrate.py --camera yi01
```
Two windows open — the **camera** and the **map**:
- Press **SPACE** to freeze a camera frame.
- Click a point on the **map** (a gray shared reference point from another
  camera, or empty floor to create one), then the **same physical spot** in the
  camera. Repeat for **≥5 well-spread floor points** (room/bench corners, floor
  tape marks), ideally including shared points in the overlap with other cameras.
- **ENTER** to finish. Saves `calib/<name>_homography.npy` + `.json` (image size
  and the clicked points) and prints two errors:
  - *fit error* on the clicked points (always ≈0 with exactly 4 points — meaningless),
  - *leave-one-out error* (needs ≥5 points): how far a NEW floor point will land.
    < 0.3 m is good.

Fine-tune all cameras live afterwards with `python recalibrate.py` (drag points
until everyone's dot lands on the true spot and the cameras agree; `s` saves).

The calibration records the image size, so a stream delivered at a different
resolution is rescaled automatically instead of silently mis-projecting.

## 3. Run real-time tracking

```bash
python track.py                      # GUI: "floor map" + "cameras" windows, q quits
python track.py --headless 60        # no GUI for 60 s, snapshots to /tmp every second
```
Set `output.show_window: false` to run without a GUI (positions printed each
second + MQTT). It prints `FPS | detect ms | draw ms` and a `[sync]` health line
every second.

MQTT: set `output.mqtt.enabled: true`, fill in the broker, and copy `.env.example`
to `.env` with `MQTT_USERNAME` / `MQTT_PASSWORD`. Positions are published as
TAC-B-style `position` messages to `output.mqtt.topic`.

### Privacy (masked heads in previews)

Every detected person's head is masked (pixelate / blur / solid) in the preview
windows and snapshots. The head is located from the **pose keypoints the detector
already computes** (nose/eyes/ears, else the top of the box), so masking costs
~0 ms and is never stale. It's preview-only: the frames fed to the detector are
never modified. Coverage follows the detector: any detection above
`detector.conf` (0.15) is masked, tracked or not; a person the detector misses
entirely is not. Configure under `output.privacy`.

(This replaced a separate CenterFace face detector that took ~600 ms per frame on
the CPU and capped the whole tracker at ~4 FPS.)

## Performance on the Jetson AGX Thor

| | before | now |
|---|---|---|
| detector | 3× PyTorch FP32 calls, ~46 ms | 1 batched TensorRT FP16 call, ~15 ms |
| face privacy | CenterFace on CPU, ~600 ms/frame (3 threads, ~8 cores) | pose keypoints, ~0 ms |
| preview drawing | full-res draw then resize | draw on the 640-px tile |
| loop rate (3 cams, headless) | 3–5 FPS | ~20–30 FPS (≈ camera rate) |
| cross-camera skew | 330–450 ms | 40–150 ms |

Also run the board in `MAXN` (`sudo nvpmodel -m 0 && sudo jetson_clocks`), and
keep other CPU-heavy services off the Thor — timings above vary with background
load.

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
- `decoder: hw` (Jetson NVDEC) is the default; `sw` (openh264) is the portable fallback.
- `protocol: udp` keeps latency *live*: over `tcp`, packet loss makes the jitter
  buffer ratchet up and never drain, so one camera slowly drifts ~1s behind the
  rest. `drop_on_latency`/`retransmission: false` bound it further. Use `tcp` only
  if a stream won't stay connected over udp.
- A camera that still lags more than `lag_budget_ms` behind the most-live one is
  dropped from the aligned set rather than dragging every camera back to its time
  (and freezing them once the gap exceeds `buffer_sec`).
- Verify the live skew yourself: `python tools/check_sync.py`
  or watch the `[sync] spread=…` line `track.py`
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
and their intersection on the top-down map (debug). Trapezoids that don't overlap (e.g.
calibration error) fall back to the old point-mean, so it never does harm.

Limits: very heavy occlusion *near the horizon* (a few visible pixels mapping to
many meters) can drift beyond `blob_merge_gate_m` and still split — no worse than
before. See `tracker3d/groundblob.py`.

## Accuracy notes (monocular limits)

- Floor (X, Y) is solid when feet are visible and the floor is flat.
- **Lens distortion is not modelled yet.** The Yi cameras are wide-angle with
  visible barrel distortion, so a single homography is least accurate near the
  image edges (see `docs/REVIEW.md`, roadmap).
- If feet are occluded, the pose model extrapolates them (flagged uncertain);
  multi-camera fusion and blob intersection recover them when another camera
  sees the feet.
- True height/Z is not measured.

## Tests

```bash
python tests/test_core.py
```
