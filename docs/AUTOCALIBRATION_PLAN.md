# Plan: automatic multi-camera calibration

Status: **planned, not started.** Replaces per-camera point clicking
(`calibrate.py`). Background and literature numbers: `docs/REVIEW.md` §6–7.

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
   (printout or tablet) in front of each camera for about 30 s. Nothing goes on
   the floor. The three Yi cameras are the same model, so one calibration may
   cover all of them, refined later in step 3. Tracking then undistorts foot
   points before projecting them. This is required: distorted edges bias
   everything else.

2. **Calibration walk (~2 min).** One person walks the room, making sure to pass
   through the areas where cameras overlap. The NTP-synced cameras record the
   pose keypoints the tracker already produces.
   - Ankles seen by two cameras at the same instant are the same floor point.
     That gives thousands of correspondences, spread over the whole overlap,
     instead of 4–8 clicks.
   - The head sits vertically above the feet at a constant height, which
     constrains each camera's tilt and height.

3. **Joint solve (bundle adjustment).** Estimate every camera's full 3D pose
   (R, t; K refined too) together with the person's floor position at each
   timestamp.
   - Constraints: feet on the floor (z = 0), head vertically above the feet at
     a height that is unknown but constant.
   - Initialisation: floor-to-floor homographies between camera pairs, computed
     from ankle matches, then decomposed using K.
   - The objective is exactly "all cameras place the same person at the same
     spot". A camera pose has fewer unknowns than a free 8-DoF homography, so it
     can't bend to fit noise, and it yields the horizon and camera height for
     free.
   - Output: the floor homography per camera, `H = K [r1 r2 t]`, so the rest of
     the pipeline is unchanged.

4. **Anchor the rig to the map once.** Step 3 fixes how the cameras sit relative
   to each other, but not where the rig sits on the map. Only 4 unknowns are
   left, shared by all cameras: x, y, yaw and scale.
   - Automatic: register one or more cameras to the splat or scan
     (`data/gs_lod2.sog`, `data/smart_lab.las`), using the current pose as the
     starting estimate.
   - Fallback: click **2 points on the map once for the whole system**. An error
     there shifts every camera equally and never splits a person into two dots.

5. **Keep it calibrated.**
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

Start with step 1.
