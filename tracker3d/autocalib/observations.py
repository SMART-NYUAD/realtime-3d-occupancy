"""Strict single-person selection and interpolation at shared capture times."""
from collections import Counter

import cv2
import numpy as np

from .session import lens_from_dict

JOINTS = (15, 16, 0)  # left ankle, right ankle, nose: same anatomical points in every view
MOTION_WINDOW_S = 0.25  # each side of a sample; long enough to average out keypoint jitter
MIN_WINDOW_FRAMES = 3


def _window_speed(frames, ts, t, a, window_s=MOTION_WINDOW_S):
    """Ankle speed around t from medians of the frames just before and after it.

    Consecutive frames are ~50 ms apart, so a few pixels of normal pose-keypoint
    jitter between two frames reads as walking speed and would reject genuine
    pauses. Medians over a window on each side also ignore an isolated
    left/right ankle swap. Only frames of the same segment and track count.
    Returns None when either side has too few valid frames.
    """
    lo, mid, hi = (np.searchsorted(ts, t - window_s), np.searchsorted(ts, t),
                   np.searchsorted(ts, t + window_s, side="right"))
    same = lambda f: f["valid"] and f["segment"] == a["segment"] and f["track"] == a["track"]
    before = [f for f in frames[lo:mid] if same(f)]
    after = [f for f in frames[mid:hi] if same(f)]
    if len(before) < MIN_WINDOW_FRAMES or len(after) < MIN_WINDOW_FRAMES:
        return None
    pb = np.median([f["xy"][:2] for f in before], axis=0)
    pa = np.median([f["xy"][:2] for f in after], axis=0)
    span = np.median([f["t"] for f in after]) - np.median([f["t"] for f in before])
    height = np.median([f["height"] for f in before + after])
    if span <= 0 or height <= 0:
        return None
    return float(np.linalg.norm(pa - pb, axis=1).max() / (span * height))


def prepare(session, kp_conf=0.65, max_gap_s=0.12, sample_hz=5.0,
            max_ankle_speed=0.30):
    """Sample quiet, upright observations; never interpolate over missing detections.

    Speed is ankle motion / person box height / second, measured from medians
    over a short window either side of the sample (see `_window_speed`). Both
    bracketing raw frames must be valid, from the same reconnect generation and track. Ankle
    positions are kept separately (image midpoints are not projectively invariant).
    Nose is optional and receives a softer model in the bundle adjustment.
    """
    if min(max_gap_s, sample_hz, max_ankle_speed) <= 0 or not 0 < kp_conf <= 1:
        raise ValueError("invalid observation sampling settings")
    names = sorted(session["manifest"]["cameras"])
    lenses = {n: lens_from_dict(session["manifest"]["cameras"][n]["lens"]) for n in names}
    frames = {n: [] for n in names}
    counts, seen = Counter(), set()
    previous = {}
    for r in session["records"]:
        if r["kind"] != "frame":
            continue
        n = r["camera"]
        key = (n, r.get("generation"), r.get("frame_id", r["t"]))
        if key in seen:
            counts["duplicate"] += 1
            continue
        seen.add(key)
        # Backwards clocks must split interpolation, even when generation is unchanged.
        segment = previous.get(n, (None, 0, None))[1]
        old_t, _, old_gen = previous.get(n, (None, 0, None))
        if old_t is not None and (r["t"] <= old_t or r.get("generation") != old_gen):
            segment += 1
        previous[n] = (r["t"], segment, r.get("generation"))
        item = {"t": r["t"], "segment": segment, "valid": False}
        frames[n].append(item)
        if r.get("timestamp_source") != "ntp":
            counts["non_ntp"] += 1
            continue
        people = r["people"]
        if len(people) != 1:
            counts["person_count"] += 1
            continue
        p = people[0]
        xy = np.asarray(p.get("kxy", []), float)
        cf = np.asarray(p.get("kconf", []), float)
        box = np.asarray(p.get("bbox", []), float)
        if (xy.shape != (17, 2) or cf.shape != (17,) or box.shape != (4,)
                or not np.isfinite(xy).all() or not np.isfinite(cf).all()
                or not np.isfinite(box).all()):
            counts["invalid_pose"] += 1
            continue
        w, h = lenses[n].image_size
        if (min(cf[15:17]) < kp_conf or min(cf[5:7]) < kp_conf
                or box[3] >= h - 2 or box[1] <= 1 or box[3] - box[1] < 40
                or np.any(xy[15:17] < 2) or np.any(xy[15:17] >= [w - 2, h - 2])):
            counts["occluded_or_clipped"] += 1
            continue
        # Bent/crouched poses: shoulder->hip and hip->ankle must point down.
        if (min(cf[11:13]) < kp_conf
                or xy[11:13, 1].mean() <= xy[5:7, 1].mean()
                or xy[15:17, 1].mean() <= xy[11:13, 1].mean()):
            counts["not_upright"] += 1
            continue
        item.update(valid=True, xy=xy[list(JOINTS)], conf=cf[list(JOINTS)],
                    height=box[3] - box[1], track=p.get("id"))
        counts["usable_frames"] += 1
    # Segment ordering was checked before sorting; interpolation checks its identity.
    for n in names:
        frames[n].sort(key=lambda f: f["t"])
    starts = [v[0]["t"] for v in frames.values() if v]
    ends = [v[-1]["t"] for v in frames.values() if v]
    if len(starts) != len(names):
        raise ValueError("every configured camera needs observations")
    start, end = min(starts), max(ends)
    times = np.arange(start + 0.5 / sample_hz, end, 1 / sample_hz)
    samples = []
    ts = {n: np.array([f["t"] for f in frames[n]]) for n in names}
    for t in times:
        views = {}
        for n in names:
            j = int(np.searchsorted(ts[n], t))
            if j == 0 or j == len(ts[n]):
                continue
            a, b = frames[n][j - 1:j + 1]
            dt = b["t"] - a["t"]
            if (not a["valid"] or not b["valid"] or dt <= 0 or dt > max_gap_s
                    or a["segment"] != b["segment"] or a["track"] != b["track"]):
                continue
            speed = _window_speed(frames[n], ts[n], t, a)
            if speed is None or speed > max_ankle_speed:
                counts["moving"] += 1
                continue
            f = (t - a["t"]) / dt
            bxy = b["xy"]
            # Pose models occasionally exchange left/right ankles for a frame; never
            # interpolate a left ankle towards a right one.
            if (np.linalg.norm(bxy[[1, 0]] - a["xy"][:2], axis=1).sum()
                    < np.linalg.norm(bxy[:2] - a["xy"][:2], axis=1).sum()):
                bxy = bxy[[1, 0, 2]]
            raw = (1 - f) * a["xy"] + f * bxy
            conf = np.minimum(a["conf"], b["conf"])
            conf[conf < kp_conf] = 0
            # Exact point inverse offline; avoid any LUT interpolation in the solve.
            rect = lenses[n]._undistort_exact(raw)
            rays = (np.linalg.inv(lenses[n].K_rect) @ np.c_[rect, np.ones(3)].T).T
            views[n] = {"raw": raw, "rect": rect, "rays": rays[:, :2] / rays[:, 2:],
                        "conf": conf, "bracket_s": dt}
        if len(views) >= 2:
            samples.append({"t": float(t), "views": views})
    if len(samples) < 12:
        raise ValueError("too few shared upright samples; pause at more positions in camera overlaps")
    counts["samples"] = len(samples)
    return samples, lenses, dict(counts)


def split_samples(samples):
    """Withhold contiguous two-second blocks; no random adjacent-frame split."""
    start = samples[0]["t"]
    held = np.array([int((s["t"] - start) // 2) % 5 == 4 for s in samples])
    if held.sum() < 4 or (~held).sum() < 12:
        raise ValueError("walk is too short for independent time-block validation (record >= 20 s)")
    return [s for s, h in zip(samples, held) if not h], [s for s, h in zip(samples, held) if h]


def pair_matches(samples, a, b, field="rays"):
    pairs = [(s["views"][a][field][:2], s["views"][b][field][:2])
             for s in samples if a in s["views"] and b in s["views"]]
    if not pairs:
        return np.empty((0, 2)), np.empty((0, 2))
    return np.concatenate([p[0] for p in pairs]), np.concatenate([p[1] for p in pairs])


def has_spread(pts, min_ratio=0.015):
    if len(pts) < 8:
        return False
    s = np.linalg.svd(pts - pts.mean(axis=0), compute_uv=False)
    area = cv2.contourArea(cv2.convexHull(np.asarray(pts, np.float32)))
    return bool(s[0] > 1e-5 and s[1] / s[0] > min_ratio and area > 0.002)
