# Plan: automatic multi-camera calibration

Status: **steps 1–4 implemented; lab acceptance pending.** `autocalib.py` records
a walk, jointly solves camera poses, anchors the rig, validates a separate walk,
and exports a candidate tracker configuration. Current manual calibrations remain
usable. Continuous maintenance (step 5) and automatic scan/splat registration
are future work. Background: `docs/REVIEW.md` §6–7.

## Implemented workflow

Install the updated `requirements.txt` in the tracker environment (SciPy is the
new solver dependency). Offline solve/anchor/validate/export require neither
YOLO nor GStreamer. Run commands from the repository root; relative config
paths retain the existing repository convention.

```bash
# One person, all cameras NTP-synced, existing per-camera lens files required.
python autocalib.py record --out calib/walks/run01 --seconds 120
python autocalib.py solve --session calib/walks/run01 --out calib/rig01.json
python autocalib.py anchor --rig calib/rig01.json --out calib/anchored01.json

# A separate walk with measured checkpoints tests generalization.
python autocalib.py record --out calib/walks/check01 --seconds 120
python autocalib.py validate --rig calib/anchored01.json \
  --session calib/walks/check01 --checkpoints checkpoints.json --out calib/report01.json
python autocalib.py export --rig calib/anchored01.json --report calib/report01.json \
  --out calib/candidates/run01
python track.py --config calib/candidates/run01/config.yaml
```

All stages support `--config`; output files/directories must be new. Export
does not replace `config.yaml` or existing homographies. The candidate includes
its own lens snapshots, homographies, provenance and validation report; MQTT is
disabled for the initial trial. Return to `python track.py` to use the previous
configuration. Stop and restart the tracker when changing calibration sets.

### Walking and recording

Walk through every overlap in several directions. Pause **upright for 1–2 s at
at least 12 well-spread locations**, with both ankles visible and your face
visible in overlapping cameras at several locations. Avoid one straight line.
The solver uses quiet samples, so a continuous fast walk is insufficient.
Every camera must connect to the rest through a useful overlap; all cameras do
not have to see every pause.

Press **SPACE** during a pause to name it `M1`, `M2`, etc.; remain still for
another second. These markers identify recognizable floor spots for anchoring
or checkpoints. **Q** finishes. With `--headless`, identify pauses using capture
timestamps in a points JSON instead. A session stores complete lens snapshots,
raw keypoints/confidences, per-frame NTP provenance and reconnect generations,
not video. Pre-NTP arrival timestamps are rejected by the solver. Preview heads
are masked. Aborted/crashed recordings remain marked incomplete; inspect them
before using `solve --allow-incomplete` to recover intact records.

### Joint geometry and observation assumptions

Pairwise normalized-image homographies seed a connected pose graph. The solver
tests physical decomposition branches, propagates camera-to-plane distance,
then jointly optimizes rotations, camera centres, separate left/right ankle
positions and a shared nose height with robust sparse least squares. Nose
reprojection is a soft upright constraint. Per-camera IDs are local only;
single-person observations are matched in time, with no interpolation across
missing detections, excessive gaps, track changes or reconnects.

Robustness to real pose-detector output: pause detection compares median ankle
positions over 0.25 s either side of each sample, so normal keypoint jitter
(several pixels) is not mistaken for walking. Left/right ankle labels, which pose
models occasionally exchange, are aligned between interpolated frames and across
cameras before and after the joint solve; agreement statistics compare foot
positions regardless of labels. The pairwise seed uses a 12 px RANSAC threshold.
Reaching the optimizer's evaluation limit (default 1500) is recorded in the
result (`optimization.converged`) rather than treated as failure; held-out and
independent validation remain the acceptance gate. A synthetic stress run with
4 px keypoint noise, 5% ankle swaps, a nose 8 cm ahead of the feet, ankles 3 cm
above the assumed height, 5 ms timestamp jitter and 1.5 cm sway still validated,
with checkpoint error at most 2.6 cm (synthetic, not a lab measurement).

The rig is right-handed and Z-up. The root camera is at `(0,0,1)` with fixed
heading; its tilt is optimized. Units are its height until map anchoring supplies
meters. `X_camera = R @ (X_rig - C)`. Exported physical-floor homographies are
`inverse(K_rect @ [r1 r2 -R@C])`, followed by the common map similarity.

Anatomical ankles are above the floor. `--ankle-height-ratio` defaults to **0.05
of nose height**, a documented anthropometric assumption, not a measurement.
Both ankles share this plane during quiet upright observations. The physical
camera poses and floor homography account for the offset. A second homography
in each sidecar projects observed ankles onto their plane; the runtime projects
each visible ankle individually, then averages their floor XY coordinates.
Other localization fallbacks and floor blobs use the physical-floor homography.
Posture, shoes, ankle offset and head movement remain sources of error to test
against measured checkpoints. This does not measure body height independently.

Board intrinsics/distortion stay fixed by default. `solve --refine-focal` tries
a per-camera focal multiplier bounded to ±3%, with a 1% prior, keeping distortion
fixed. It is retained only if withheld agreement improves; new lens snapshots
and rectification fingerprints are exported together. Contiguous two-second
blocks are withheld from fitting. A separate validation recording is still
required for export, including after focal-model selection.

### Anchoring and acceptance

The GUI selects a named recorded pause (`n`/`p`), then its matching floor location
on the map. `c` toggles anchor/checkpoint, `u` removes a pick, ENTER saves.
Two anchor correspondences are the minimum; use three or more where possible,
spanning at least **2 m**. Two arbitrary map clicks without matched rig positions
do not define an anchor. Additional checkpoints are excluded from the similarity
fit. Reflection parity is fixed by scan axis order and configured flips, rather
than freely chosen by the fit; these settings are part of map identity.

For repeatable/headless use, `anchor --points anchors.json` accepts:

```json
[
  {"marker": "M1", "world": [1.0, 1.0], "role": "anchor"},
  {"marker": "M2", "world": [4.0, 1.0], "role": "anchor"},
  {"marker": "M3", "world": [2.0, 3.0], "role": "check"}
]
```

Coordinates above are illustrative; supply measured map meters for the actual
pauses. Anchoring prints the walker's reconstructed nose height in meters as an
independent check of the anchor scale (about 10 cm below standing height) and
warns outside 1.2–2.0 m. `{"t": 1234.5, "world": [x,y]}` can replace a named marker. The position
is the median of shared reconstructed observations within ±0.5 s. Moving or
unobserved pauses are refused. `validate --checkpoints checkpoints.json` uses the
same list format, with markers/timestamps belonging to the **validation walk**;
every supplied point is a checkpoint. Use several checkpoints spanning overlaps,
image edges and distances, with the person's feet centred on the measured spot.

Export requires independent validation with median camera disagreement ≤10 cm,
95th percentile ≤25 cm, and measured checkpoint median error ≤20 cm / maximum
≤40 cm. These are acceptance targets, not measured lab performance. Reports
contain camera/pair errors, rejection counts, before-fusion agreement, checkpoint
errors, and the current manual calibration baseline when available. Failure
writes a report, exits nonzero, and prevents export. Hashes bind report, rig,
lenses and map. Changing scan contents, up axis, flips or footprint clipping
requires re-anchoring; the candidate tracker checks map identity at startup.

The manual tuner refuses joint results because there are no original clicked
pairs to edit. `calibrate.py --manual-override` explicitly replaces an individual
camera calibration and loses its shared-rig guarantee.

### Tests

```bash
python -m unittest discover -s tests -p 'test_autocalib.py' -v
python tests/test_core.py
```

Synthetic tests exercise three distorted cameras, asynchronous capture phases,
independent walks, camera pose/height recovery, chain overlap, degenerate paths,
timestamp/identity gaps, corrupt artifacts, checkpoint failure, map changes,
export and actual runtime projection. Run a Jetson recording/GUI smoke test and
the independent measured lab walk before adopting a candidate. Automated tests
cannot establish consumer-camera timestamp accuracy or real pose-detection bias.

## Why change

Clicking 4–8 points per camera doesn't scale, and every click carries error
(1–2 px of clicking, lens distortion, ambiguous features, map resolution).
Each camera's clicks are independent, so each camera ends up wrong in its own
direction. That camera-to-camera disagreement is exactly what turns one person
into two dots.

**Idea:** calibrate all cameras **jointly** from a person walking through the
room, then attach the whole rig to the map **once**.

## Steps

1. **Lens intrinsics and distortion, once per camera.** Wave a checkerboard
   (printout or tablet) in front of each camera. Nothing goes on
   the floor. All three Yi cameras now have their own saved lens calibration.
   Tracking then undistorts foot
   points before projecting them. This is required: distorted edges bias
   everything else.

2. **Calibration walk (~2 min).** One person walks the room, making sure to pass
   through the areas where cameras overlap. The NTP-synced cameras record the
   pose keypoints the tracker already produces.
   - Matching anatomical ankles seen by two cameras at the same instant are
     observations of the same 3D point, approximately on a shared ankle plane
     during an upright pause.
     That gives thousands of correspondences, spread over the whole overlap,
     instead of 4–8 clicks.
   - Nose observations supply a soft upright/constant-height constraint.

3. **Joint solve (bundle adjustment).** Estimate every camera's full 3D pose
   (R, t; K refined too) together with the person's floor position at each
   timestamp.
   - Constraints: a flat floor, ankles at a small height above it, and a softly
     upright person with unknown shared nose height.
   - Initialisation: floor-to-floor homographies between camera pairs, computed
     from ankle matches, then decomposed using K.
   - The objective is exactly "all cameras place the same person at the same
     spot". A camera pose has fewer unknowns than a free 8-DoF homography, so it
     can't bend to fit noise, and it yields the horizon and camera height for
     free.
   - Output: rectified-pixel-to-floor homographies
     `H = inverse(K_rect [r1 r2 t])`, plus ankle-plane projection metadata.

4. **Anchor the rig to the map once.** Step 3 fixes how the cameras sit relative
   to each other, but not where the rig sits on the map. Only 4 unknowns are
   left, shared by all cameras: x, y, yaw and scale.
   - Later extension: register one or more cameras to the splat or scan
     (`data/gs_lod2.sog`, `data/smart_lab.las`), using the current pose as the
     starting estimate.
   - Implemented: match **2+ recorded pauses to map points once for the whole system**. An error
     there shifts every camera equally and never splits a person into two dots.

5. **Keep it calibrated (future work).**
   - Continuously measure how far apart cameras place the same person. Above
     about 20–25 cm, re-solve using people who happen to walk by; no new walk is
     needed.
   - Reference-frame motion detection (rotation-only, separate day and IR
     references) catches bumped cameras and corrects them immediately.

## Expected accuracy

- **Camera-to-camera agreement: ~5–10 cm.** This is what fixes the duplicate
  dots. The literature reports 1–3 cm in controlled scenes; the estimate here is
  conservative because camera timing jitter is guessed at 50–200 ms, not measured.
- **Absolute accuracy on the map:** set by the step-4 anchor, roughly 5–20 cm.
- **Adding a camera:** give it some overlap with an existing camera, walk
  through for about a minute, and re-solve.

## Requirements and limits

- Every camera must overlap at least one other camera. Fusion already needs this.
- One person in the room during the calibration walk. Later refinement can
  handle several people by matching them with geometric gating.
- The floor must be flat (the existing assumption).
- Step 1 is mandatory.

## Work breakdown

| Step | Deliverable | Estimate |
|---|---|---|
| 1 | `tools/calib_intrinsics.py`, plus undistortion in `FloorProjector` | 1–2 days (improves accuracy on its own) |
| 2–3 | `autocalib.py record` (synced keypoint log) and `autocalib.py solve` (bundle adjustment) | ~1 week |
| 4 | 2-click anchor; splat anchoring later | included in 2–3 / optional later |
| 5 | Consistency monitor and motion correction | a few days |

Step 1 is present and enabled for all three saved camera configurations. The
historical effort estimates above are planning estimates; lab acceptance of the
implemented steps 2–4 remains to be measured.
