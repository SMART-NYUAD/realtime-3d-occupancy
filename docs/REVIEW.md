# Code review — people_tracker_3d

Review of the tracker as of commit `8ac30e9` (main), and what the
`refactor/thor-cleanup-optimize` branch changed. Status: **Fixed** (on this
branch), **Open** (known, not done), **Roadmap** (planned larger work).

## 1. Real-time capture / tracking bugs

| # | Issue | Impact | Status |
|---|-------|--------|--------|
| 1 | **Loop froze when no aligned frames arrived.** `track.py` did `continue` before `cv2.waitKey` and before the headless timer check. | If cameras dropped out, the GUI stopped responding (`q` dead) and `--headless N` never exited. | Fixed — UI/stats/exit run every iteration. |
| 2 | **Annotations drawn into the shared capture buffer.** `frame_at()` returns the buffered array; the face blur returned the same array when it found no face; `draw_camera_view` drew on it in place. | A frame re-served on the next step was fed to YOLO with boxes/text burned in. | Fixed — everything is drawn on a downscaled copy. |
| 3 | **The same frame was tracked twice.** No record of which capture time each camera had processed. | ByteTrack's Kalman filter stepped on duplicates; wasted inference. | Fixed — frames are keyed by capture time (synced) / sequence number (plain); a re-served frame reuses its last result. |
| 4 | **`trails` grew forever.** Appended every frame even with `draw_track_trails: false`; keys never removed. | Slow memory leak on long runs. | Fixed — per-camera trails removed. |
| 5 | **ByteTrack low-score recovery never ran.** With `fuse_score: True`, Ultralytics also fuses in the second (low-score) stage: cost `1 − IoU·score` against a 0.5 gate, so any box below `track_high_thresh` (0.35) can never match. | The config's premise ("low conf lets ByteTrack ride out dips") was false; tracks dropped on confidence dips. | Fixed — `fuse_score: False`. |
| 6 | **Keypoints attached to the wrong person** (Ultralytics bug, latent until #5 was fixed). ByteTrack numbers the low-score subset from 0, so a track matched in stage 2 reports an index into that subset and `result[idx]` picks another detection's keypoints/confidence. | Wrong foot point → wrong floor position for tracks kept alive by low-score boxes. | Fixed — `_ByteTracker` re-stamps global indices; covered by a test. |
| 7 | **Mixed clocks at stream start.** Frames before the first RTCP sender report are stamped with arrival time, then NTP time takes over; both sat in the buffer for ~1 s. | Mis-ordered buffer, bad alignment right after (re)connect. | Fixed — buffer is cleared on the first NTP-stamped frame. |
| 8 | **Fusion used wall-clock time** instead of the aligned capture time. | Speed cap / aging used the wrong dt (off by the pipeline latency jitter). | Fixed — `fuser.update(..., now=capture_t)`. |
| 9 | **Foot points above the horizon were projected.** yi04's horizon crosses the top of its frame (v≈80–140 px); `image_to_floor` ignored the homogeneous sign. | A point above it mapped to a mirrored location behind the camera. | Fixed — `FloorProjector` rejects the far side of the horizon. |
| 10 | **Homography resolution not recorded.** `.npy` had no image size. | Any change of stream resolution (e.g. the old `rtsp_hw_decode` path resized to 1280×720) silently broke the mapping. | Fixed — JSON sidecar stores the size; H is rescaled automatically. |
| 11 | **Headless mode wrote 3–4 PNGs every step** and printed every step. | Big I/O cost in headless runs. | Fixed — one snapshot + one status line per second. |
| 12 | `bytetrack.yaml` `track_buffer: 75` was documented as "~2.5 s @30 fps", but it counts processed frames (the old loop ran at ~4 FPS → ~19 s). | Ghost IDs lingered far longer than intended. | Fixed — 50 frames ≈ 2.5 s at the new ~20 FPS. |
| 13 | Sync defaults differed between `track.py` fallbacks (`tcp`, retransmission on) and `config.yaml` (`udp`, off). | Surprising behaviour if a key was missing. | Fixed — one set of defaults in `SyncGroup`. |
| 14 | MQTT speed spike after a broker reconnect (last position from before the outage). | One bogus `speed` value per person. | Fixed — history cleared on connect. |
| 15 | `test_groundblob.py` assumed 1280×720 frames; the cameras and homographies are 1920×1080. | Tests exercised the wrong geometry (clip detection, box sizes). | Fixed — tests use the calibrated size. |

## 2. Calibration issues

| # | Issue | Status |
|---|-------|--------|
| C1 | **Lens distortion is ignored.** The Yi cameras are wide-angle with clear barrel distortion; one homography can't fit the whole image, so error grows toward the edges — exactly where cameras overlap and fusion needs agreement. Likely the largest remaining error source. | **Fixed in code, pending lab calibration** — `tracker3d/lens.py` + `tools/calib_intrinsics.py`: points are undistorted before the homography once each camera has an `intrinsics_file`. |
| C2 | `cv2.findHomography(..., RANSAC)` used the default 3.0 threshold, in **meters** here → no outlier rejection. | Fixed — least-squares below 6 points, RANSAC at 0.25 m above. |
| C3 | The reported "reprojection error" was in-sample; with exactly 4 points it is always ≈0. | Fixed — `calibrate.py` also reports leave-one-out error (≥5 points). |
| C4 | `recalibrate.py` placed the dot from the box bottom while `track.py` used pose ankles, so tuning targeted a different point than tracking used. | Fixed — same pose foot estimate. |
| C5 | `recalibrate.py` wrote `<cam>_calib.json` into the working directory. | Fixed — points live in the calibration sidecar in `calib/`. |

## 3. Removed (unused, non-functional, or superseded)

- **DXF/BIM floorplan** (`src/floorplan.py`, `SMART-floorplans.dxf/.svg`, `ezdxf`) and
  `measure.py` (its corner snapping only worked on the DXF). The README already
  called the DXF out of date; the point-cloud map is the world frame.
- **deface/CenterFace face blur** → pose-keypoint head masking (see §4).
- **`rtsp_hw_decode` and CSI capture** — both needed OpenCV's GStreamer backend,
  which the pip `opencv-python` build doesn't have (`GStreamer: NO`), so they could
  never open. RTSP hardware decode lives in the synced GStreamer path.
- **Legacy `conf_weighting` / `sigma_weighting` switches** — inverse-variance
  (sigma) weighting is always on; `conf_drop_ratio` is kept as the secondary gate.
- **Per-camera track trails** (off by default, leaked memory — #4).
- `PlanBase.render()` / `snap()` (unused).

## 4. Jetson Thor performance

Measured with 3 × 1080p cameras at ~20 fps each:

| stage | before | after |
|---|---|---|
| detection | 3 serial PyTorch calls ≈ 46 ms | 1 batched TensorRT FP16 call ≈ 15 ms |
| face privacy | CenterFace on CPU ≈ 620 ms/frame; 3 async threads used ~8 cores | pose keypoints, ≈ 0 ms |
| preview draw | full-res draw + resize ≈ 20 ms | 640-px tile ≈ 2–3 ms |
| loop | **3.3–4.6 FPS** | **~20–30 FPS** (headless) |
| cross-camera skew | 330–450 ms | 40–150 ms |

The skew/latency drop is a side effect: the face-blur threads were starving the
GStreamer decode threads, which is why yi05 looked like a ~500 ms camera.

**External load:** at the time of testing another service
(`python -m smart_lab_agent.main`, root) used ~1400 % CPU (all 14 cores) and an
Ollama `llama-server` shared the GPU. The detect-time jitter (15–40 ms) comes from
that contention; the tracker would run steadier with those moved or niced.

## 5. Open issues (not fixed)

- **O1 — Pacing to the slowest camera.** `SyncGroup` aligns to the newest instant
  *all* live cameras cover, so output latency = the slowest camera's latency
  (plus `lag_budget_ms` tolerance). Fine now (~250 ms), but one slow camera slows
  everything.
- **O2 — Greedy, order-dependent clustering** in `Fusion._cluster`; a global
  assignment (Hungarian on pairwise cost) would be more stable with crowds.
- **O3 — `dup_suppress_m` also blocks births next to *coasting* tracks**, so a new
  person walking through a ghost's spot appears up to `max_age_s` late.
- **O4 — No velocity model in fusion** (constant-position EMA + speed cap); a
  constant-velocity Kalman per global track would reduce lag on fast walkers.
- **O5 — `valid_area: auto` is the whole scan footprint** (any rendered pixel,
  including furniture tops), not the walkable floor.
- **O6 — Privacy coverage follows the detector.** A person YOLO misses entirely
  is not masked (low risk at `conf: 0.15`, but not zero).
- **O7 — Frame conversion on CPU.** NVDEC output is converted BGRx→BGR by
  `videoconvert` on the CPU (3 × 1080p × 20 fps). Could stay on the GPU with
  NVMM + a CUDA preprocessing path if CPU becomes the limit again.

## 6. Roadmap: calibration

**R1 — Intrinsics + distortion, once per camera (do first).** Lens parameters
don't change when a camera is moved or bumped. A ~30 s checkerboard session
(board shown on a tablet/laptop screen — nothing on the floor), or plumb-line
self-calibration from straight room edges. Then undistort the foot point before
the homography. Fixes C1 regardless of how extrinsics are obtained.

**R2 — Detect and correct camera movement (reference-frame registration).**
Store a reference frame with each calibration; every few seconds match features
(ORB/SuperPoint) on static background between live and reference; if the camera
moved, estimate the image-to-image transform and update `H_new = H_old · H_img`.
Handles bumps/re-aims automatically, and tells you when a camera moved.

**R3 — People as the calibration target (markerless, for big moves / new
cameras).** With NTP sync, a person seen by the moved camera and a still-calibrated
one at the same instant is a free correspondence; a few minutes of normal traffic
gives hundreds → robust homography refit. Single-camera fallback: head-to-foot
lines of standing people give the vertical vanishing point (tilt/roll, focal
length) and an assumed ~1.7 m height gives scale (~5–10 % error; bootstrap only).

**R4 — Anchor to the map from walked paths.** Align the accumulated walked-area
footprint to the scan's free floor (walls/furniture are never walked through).
Ambiguous in symmetric rooms — use as a sanity check.

**R5 — Optional: register to the Gaussian splat / scan** (`data/gs_lod2.sog`).
Render views, match learned features (LightGlue/LoFTR/MASt3R), PnP → full pose +
intrinsics. It recomputes pose from scratch, so it survives camera moves (re-run
it); it breaks when the room changes vs the scan. Most work; kept as a
"recalibrate everything" tool.

Suggested order: R1 → R2 → R3, with R4/R5 as checks/fallbacks, and
`recalibrate.py` as the manual override.

## 7. Literature check: expected accuracy of the calibration roadmap

Sensitivity for this room (our own arithmetic): camera ~2.7 m high, f ≈ 670 px
at 1920 px / 110°. A 1° error in camera tilt moves a floor point ~10 cm at 3 m
and ~30 cm at 6 m. **Rotation accuracy decides floor accuracy**: staying under
~10 cm across the room needs ≲0.3°. Distortion must be removed first, because a
homography is only exact on an undistorted (pinhole) image.

| Method | Expected floor error here | Data / time | Effort | Role |
|---|---|---|---|---|
| R1 Intrinsics (handheld checkerboard, or refine inside R5) | prerequisite | 30 s per camera, once | small | do first |
| R2 Reference-frame registration, rotation-only `K R K⁻¹` after undistortion | **< 3–5 cm** for bumps/pan/tilt; degrades with translation (parallax) | 1 frame, ms | 1–2 days | always-on tamper detection + correction |
| R5 Register to splat/scan (render from the current pose → MASt3R/LightGlue → PnP) | **~5–20 cm** (0.3–0.7°), only if splat renders are good from the ceiling viewpoints | seconds per camera | 1–2 weeks | re-anchor after big moves; replaces the clicked homographies |
| R3a Pedestrians across synced cameras | ~5–15 cm relative to the reference camera (inherits its error) | < 1 min of one walker; minutes with crowds | 3–5 days | continuous consistency monitor + refinement |
| R3b Single-camera from people's heights | 3–5 % of distance (15–40 cm at 6 m); **cannot give x, y, yaw on the map** | minutes–hours | medium | sanity check of tilt/height only |
| R4 Walked-area ↔ map alignment | ~20–50 cm (no published benchmark), occasional gross failures | days | medium | weak prior / plausibility check |
| Deep single-image calibration (GeoCalib) | pitch error ~0.9–1.9° → 30–60 cm at 6 m | 1 image | small | too coarse for extrinsics |

Key evidence:
- SuperPoint+LightGlue registers planar views to ~1 px; a bump that is pure
  rotation is exactly a homography, so this is the most accurate way to correct one.
- GS-CPR (ICLR 2025): 0.8 cm / 0.25° and GSplatLoc (2024): 1.4 cm / 0.37° on
  7-Scenes, but with hand-held, walking-height queries. The closest match to
  our case (YOWO 2025, ceiling cameras against a walking-height model):
  **1.2 m average error without a pose prior, 0.21–0.26 m / 0.6–0.7° with one.**
  We always have a prior (the current calibration), which is why R5 is viable.
- Pedestrian-based extrinsics (Truong et al., Sensors 2019): 1.3–3 cm
  triangulation error in controlled/kitchen scenes from < 1 min of walking;
  much worse with small, far-away pedestrians. NTP jitter on consumer cameras
  (≈50–200 ms, not measured here) is 7–28 cm at walking speed, so use
  interpolated tracks or moments when people stand still.
- Single-view people methods: Brouwers 2016 reports ≤ 3.7 % metric error on
  real data; CasCalib 2024 ~11 % focal-length error; the classic
  vanishing-point baselines in Xu et al. 2020 fail badly (19–89°) with few people.
- Yi cameras switch to IR/greyscale at night: keep separate day and IR
  reference frames for R2.

Revised order: **R1 → R2 (always on) → R5 (re-anchor, validate splat renders
at each camera pose first) → R3a (monitor: re-calibrate when the same person's
position differs by more than ~20–25 cm between cameras)**. R3b and R4 are
checks only.

References: GeoCalib https://arxiv.org/abs/2409.06704 · Truong et al. 2019
https://pmc.ncbi.nlm.nih.gov/articles/PMC6891296/ · GS-CPR https://arxiv.org/abs/2408.11085 ·
GSplatLoc https://arxiv.org/abs/2409.16502 · YOWO https://arxiv.org/abs/2511.16521 ·
CasCalib https://arxiv.org/abs/2405.06845 · Xu/Roy/Kitani 2020 https://arxiv.org/abs/1912.05758 ·
Brouwers 2016 https://mlanthology.org/eccv/2016/brouwers2016eccv-automatic/ ·
LightGlue https://arxiv.org/abs/2306.13643 · Reloc3r https://arxiv.org/abs/2412.08376
