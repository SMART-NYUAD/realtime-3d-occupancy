"""Tests for ground-plane blob intersection (src/groundblob.py + Fusion).

Run:  .venv/bin/python test_groundblob.py
Uses the repo's real homographies (yi01, yi04) so the geometry is exercised on
actual calibrations, including yi04 whose homogeneous scale is negative on the
floor (a sign trap the projector must handle).
"""
import os
import sys

import numpy as np

sys.path.insert(0, "src")
from groundblob import (bbox_floor_polygon, convex_intersection,      # noqa: E402
                        polygon_centroid, intersect_blobs)
from fusion import Fusion                                             # noqa: E402

W, H_IMG = 1280, 720
CAMS = ("yi01", "yi04")


def _load():
    Hs = {}
    for n in CAMS:
        p = f"{n}_homography.npy"
        if not os.path.exists(p):
            print(f"SKIP geometry tests: {p} not found")
            return None
        Hs[n] = np.load(p)
    return Hs


# --- pure polygon geometry --------------------------------------------------

def test_convex_intersection():
    a = np.array([[0, 0], [2, 0], [2, 2], [0, 2]], float)
    b = np.array([[1, 1], [3, 1], [3, 3], [1, 3]], float)
    inter = convex_intersection(a, b)
    assert inter is not None
    c = polygon_centroid(inter)
    assert np.allclose(c, [1.5, 1.5], atol=1e-6), c
    # disjoint
    far = np.array([[10, 10], [11, 10], [11, 11], [10, 11]], float)
    assert convex_intersection(a, far) is None
    # winding-independent
    assert convex_intersection(a, b[::-1]) is not None
    print("PASS convex_intersection / centroid")


# --- bbox -> floor trapezoid ------------------------------------------------

def test_polygon_shape(Hs):
    H = Hs["yi04"]   # negative floor sign: catches the horizon-sign bug
    box = (600, 250, 700, 560)
    poly = bbox_floor_polygon(H, box, (W, H_IMG))
    assert poly is not None and len(poly) == 4, "yi04 floor must project (sign-robust)"

    # A clipped box (feet off-frame) must make a longer trapezoid than a tidy one.
    visible = bbox_floor_polygon(H, (600, 250, 700, 540), (W, H_IMG))
    clipped = bbox_floor_polygon(H, (600, 250, 700, 719), (W, H_IMG))

    def depth(p):
        return np.linalg.norm(p[0] - p[3])  # far-left -> near-left edge
    assert depth(clipped) > depth(visible), (depth(clipped), depth(visible))
    print("PASS bbox_floor_polygon (sign-robust; clipped is taller)")


def test_short_box_reaches_nearer(Hs):
    """A partial (torso-only) box -- short relative to its width, bottom edge at
    the waist -- must reach further toward the camera than its near_frac alone,
    while a full standing box is left unchanged. Isolate the short-aspect term by
    turning off the width-based body_aspect reach and avoiding the clip case."""
    H = Hs["yi04"]

    def depth(p):
        return np.linalg.norm(p[0] - p[3])  # far-left -> near-left edge

    common = dict(near_frac=1.0, body_aspect=0.0)
    # Torso-only: h:w = 110:100 = 1.1, below the 1.3 threshold; not frame-clipped.
    short_box = (600, 430, 700, 540)
    off = bbox_floor_polygon(H, short_box, (W, H_IMG), short_aspect=0.0, **common)
    on = bbox_floor_polygon(H, short_box, (W, H_IMG), short_aspect=1.3, **common)
    assert off is not None and on is not None
    assert depth(on) > depth(off), (depth(on), depth(off))

    # A full standing box (h:w ~3.1) is above the threshold -> untouched.
    tall_box = (600, 250, 700, 560)
    t_off = bbox_floor_polygon(H, tall_box, (W, H_IMG), short_aspect=0.0, **common)
    t_on = bbox_floor_polygon(H, tall_box, (W, H_IMG), short_aspect=1.3, **common)
    assert np.allclose(t_off, t_on), "full standing box must be unaffected"
    print("PASS short torso box reaches nearer; full standing box unchanged")


# --- end-to-end fusion on real geometry -------------------------------------

def _frame():
    """Pick a floor point both cameras see at a moderate (recoverable) distance."""
    Hi = {n: np.linalg.inv(Hs[n]) for n in CAMS}

    def w2p(n, F):
        p = Hi[n] @ np.array([F[0], F[1], 1.0]); return p[:2] / p[2]
    for x in np.linspace(2, 13, 90):
        for y in np.linspace(2, 13, 90):
            uv = [w2p(n, (x, y)) for n in CAMS]
            if all(120 < u < 1160 and 430 < v < 560 for u, v in uv):
                return np.array([x, y]), w2p
    return None, w2p


def _det(n, F, w2p, occl=0):
    u, v = w2p(n, F)
    y2 = v - occl
    box = (u - 45, v - 300, u + 45, y2)
    p = Hs[n] @ np.array([u, y2, 1.0])
    return {"cam": n, "id": 1, "world": tuple(p[:2] / p[2]),
            "floor_poly": bbox_floor_polygon(Hs[n], box, (W, H_IMG))}


def _fuse(dets, blob):
    f = Fusion(merge_distance_m=1.2, match_gate_m=1.8, use_blob=blob,
               blob_merge_gate_m=3.0)
    # Tracks must survive n_init frames before they're reported (kills flicker
    # ghosts); the same static detections are stable, so drive n_init frames.
    out = []
    for k in range(f.n_init):
        out = f.update(dets, now=1.0 + 0.03 * k)
    return out


def test_clean_unbiased(F, w2p):
    dets = [_det(n, F, w2p) for n in CAMS]
    out = _fuse(dets, True)
    assert len(out) == 1, "both feet visible -> one person"
    err = np.linalg.norm(np.array(out[0]["world"]) - F)
    assert err < 0.05, f"clean case must stay unbiased, err={err:.3f}"
    print(f"PASS clean case unbiased (err={err:.3f} m)")


def _mean_err(out, F):
    """Distance from the true foot to the *nearest* fused track (a split leaves a
    wrong second dot; a biased single track sits off F)."""
    return min(np.linalg.norm(np.array(g["world"]) - F) for g in out)


def test_occlusion_fix(F, w2p):
    # yi04 can't see the feet. Grow the occlusion until its foot point drifts far
    # enough to actually hurt plain point fusion (split or biased), then check the
    # blob recovers a single, accurate track.
    for occl in (120, 150, 180, 210, 240):
        dets = [_det("yi01", F, w2p), _det("yi04", F, w2p, occl=occl)]
        drift = np.linalg.norm(np.subtract(dets[0]["world"], dets[1]["world"]))
        old, new = _fuse(dets, False), _fuse(dets, True)
        if len(new) == 1 and (len(old) > 1 or _mean_err(old, F) > 0.3):
            assert _mean_err(new, F) < 0.3, f"blob result off F: {_mean_err(new, F):.2f}"
            print(f"PASS occlusion fix: occl {occl}px, drift {drift:.2f} m | "
                  f"old {len(old)} track(s) err {_mean_err(old, F):.2f} -> "
                  f"new 1 track err {_mean_err(new, F):.2f}")
            return
    raise AssertionError("no occlusion level produced a recoverable split to test")


def test_no_over_merge(F, w2p):
    G = F + np.array([1.8, 0.0])   # a second, distinct person
    dets = [_det("yi01", F, w2p), _det("yi04", F, w2p),
            _det("yi01", G, w2p), _det("yi04", G, w2p)]
    dets[2]["id"] = dets[3]["id"] = 2
    out = _fuse(dets, True)
    assert len(out) == 2, f"distinct people must stay separate (got {len(out)})"
    print("PASS distinct people not over-merged")


# --- pose foot estimation + uncertainty-weighted fusion ---------------------

def test_foot_from_pose():
    from geometry import foot_from_pose
    box = (600, 200, 700, 560)          # ~360 px tall standing person
    # Full COCO-17 skeleton, upright, ankles at y=555.
    kxy = np.zeros((17, 2)); kconf = np.ones(17)
    kxy[5], kxy[6] = (610, 260), (690, 260)     # shoulders
    kxy[11], kxy[12] = (615, 380), (685, 380)   # hips
    kxy[13], kxy[14] = (615, 470), (685, 470)   # knees
    kxy[15], kxy[16] = (615, 555), (685, 555)   # ankles

    # Ankles visible -> foot at their midpoint, tight sigma.
    (fx, fy), sig, src = foot_from_pose(kxy, kconf, box)
    assert src == "ankles" and abs(fy - 555) < 1 and abs(fx - 650) < 1, (fx, fy, src)
    ank_sig = sig

    # Hide the ankles -> extrapolate from the knees, and be LESS certain.
    kconf[15] = kconf[16] = 0.0
    (fx, fy), sig, src = foot_from_pose(kxy, kconf, box)
    assert src == "knees", src
    assert fy > 470 and sig > ank_sig, (fy, sig, ank_sig)   # reaches below knees, wider

    # Hide the knees too -> extrapolate down the torso from hips/shoulders.
    kconf[13] = kconf[14] = 0.0
    (fx, fy), sig, src = foot_from_pose(kxy, kconf, box)
    assert src == "hips" and fy > 380 and sig > ank_sig, (fy, sig, src)

    # Nothing usable -> box bottom, most uncertain.
    kconf[:] = 0.0
    (fx, fy), sig, src = foot_from_pose(kxy, kconf, box)
    assert src == "box" and fy == 560, (fy, src)
    print("PASS foot_from_pose: ankles->knees->hips->box, sigma grows with reach")


def test_sigma_weighting_pulls_to_confident_foot():
    """A confident-but-occluded detection (large foot sigma) must NOT drag the dot
    off a camera that sees the ankles (small sigma), even at equal box confidence."""
    dets = [
        {"cam": "a", "id": 1, "world": (5.0, 5.0), "conf": 0.8,
         "foot_sigma_m": 0.05, "floor_poly": None},          # sees ankles
        {"cam": "b", "id": 9, "world": (6.0, 5.0), "conf": 0.8,
         "foot_sigma_m": 0.9,  "floor_poly": None},          # occluded, drifted
    ]
    f = Fusion(merge_distance_m=1.6, match_gate_m=1.8, smoothing=1.0,
               use_blob=False, sigma_weighting=True, conf_drop_ratio=0.0,
               dup_suppress_m=0.0, max_speed_mps=0.0)
    out = []
    for k in range(f.n_init):
        out = f.update(dets, now=1.0 + 0.03 * k)
    assert len(out) == 1, out
    x = out[0]["world"][0]
    assert x < 5.05, f"inverse-variance mean should sit at the tight foot, got x={x:.3f}"
    print(f"PASS sigma weighting: fused x={x:.3f} (pinned to the ankle-visible camera)")


def test_duplicate_suppression():
    """A second detection near an existing track must not spawn a duplicate dot."""
    a = {"cam": "a", "id": 1, "world": (5.0, 5.0), "conf": 0.8,
         "foot_sigma_m": 0.1, "floor_poly": None}
    # a stray same-frame detection 0.6 m away (inside dup_suppress_m), different cam
    b = {"cam": "b", "id": 2, "world": (5.6, 5.0), "conf": 0.8,
         "foot_sigma_m": 0.1, "floor_poly": None}
    f = Fusion(merge_distance_m=0.3, match_gate_m=0.4, use_blob=False,
               dup_suppress_m=1.0)
    out = []
    for k in range(f.n_init + 1):
        out = f.update([a, b], now=1.0 + 0.03 * k)
    assert len(out) == 1, f"dup within suppress radius must not spawn a 2nd dot: {out}"
    print("PASS duplicate suppression: near-by detection did not spawn a 2nd dot")


if __name__ == "__main__":
    test_convex_intersection()
    test_foot_from_pose()
    test_sigma_weighting_pulls_to_confident_foot()
    test_duplicate_suppression()
    Hs = _load()
    if Hs is not None:
        test_polygon_shape(Hs)
        test_short_box_reaches_nearer(Hs)
        F, w2p = _frame()
        if F is None:
            print("SKIP fusion tests: no shared floor point found")
        else:
            test_clean_unbiased(F, w2p)
            test_occlusion_fix(F, w2p)
            test_no_over_merge(F, w2p)
    print("\nAll tests passed.")
