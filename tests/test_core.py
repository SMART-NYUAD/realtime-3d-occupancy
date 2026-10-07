"""Core tests: geometry, ground-blob intersection, fusion, detector glue, privacy.

Run:  .venv/bin/python tests/test_core.py
Uses the repo's real homographies (yi01, yi04) so the geometry is exercised on
actual calibrations, including yi04 whose homogeneous scale is negative on the
floor (a sign trap the projector must handle).
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from tracker3d.groundblob import (bbox_floor_polygon, convex_intersection,      # noqa: E402
                        polygon_centroid)
from tracker3d.fusion import Fusion                                             # noqa: E402

W, H_IMG = 1920, 1080          # the resolution the homographies were calibrated at
CAMS = ("yi01", "yi04")


def _load():
    Hs = {}
    for n in CAMS:
        p = os.path.join(ROOT, "calib", f"{n}_homography.npy")
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

    # The SAME box must reach further toward the camera when it touches the frame
    # bottom (feet off-frame) than when the frame extends below it.
    box = (600, 500, 700, H_IMG - 1)
    visible = bbox_floor_polygon(H, box, (W, H_IMG + 400))
    clipped = bbox_floor_polygon(H, box, (W, H_IMG))

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
    from tracker3d.geometry import foot_from_pose
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
               use_blob=False, conf_drop_ratio=0.0,
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


# --- geometry: calibration persistence, rescale, horizon --------------------

def test_projector_rescale_and_horizon(Hs):
    from tracker3d.geometry import FloorProjector
    H = Hs["yi04"]
    ref = FloorProjector(H, (W, H_IMG)); ref.set_frame_size(W, H_IMG)
    half = FloorProjector(H, (W, H_IMG)); half.set_frame_size(W // 2, H_IMG // 2)
    a = ref.to_floor(900, 800)
    b = half.to_floor(450, 400)          # same scene point in a half-res frame
    assert a is not None and np.allclose(a, b, atol=1e-6), (a, b)
    # yi04's horizon crosses the top of the frame (~v 80-140 px): a pixel above it
    # must be rejected, not projected to a mirrored point behind the camera.
    assert ref.to_floor(100, 5) is None, ref.to_floor(100, 5)
    print("PASS FloorProjector: resolution rescale + horizon rejection")


def test_homography_fit_and_loo():
    from tracker3d.geometry import compute_homography, fit_errors, project
    Htrue = np.array([[0.01, 0.002, -3.0], [0.001, 0.02, -5.0], [0.0, 0.0008, 1.0]])
    rng = np.random.default_rng(0)
    img = rng.uniform([0, 400], [1900, 1070], size=(8, 2))
    wld = np.array([project(Htrue, *p) for p in img])
    H = compute_homography(img, wld)
    assert np.allclose(project(H, 960, 700), project(Htrue, 960, 700), atol=1e-6)
    ins, loo = fit_errors(img, wld)
    assert ins.max() < 1e-6 and len(loo) == 8 and loo.max() < 1e-4
    # 4 points: exact fit, in-sample error is ~0 and there is no LOO estimate.
    ins4, loo4 = fit_errors(img[:4], wld[:4])
    assert ins4.max() < 1e-6 and len(loo4) == 0
    print("PASS homography fit (LSQ <6 pts, RANSAC in meters >=6) + leave-one-out error")


# --- detector glue: ByteTrack index mapping ---------------------------------

def test_bytetrack_global_index():
    """A track kept alive by a LOW-score detection must map back to that
    detection's index in the full list (upstream numbers the low subset from 0,
    which attached another person's keypoints to the track)."""
    from tracker3d.detector import _ByteTracker, _Dets
    from ultralytics.utils import YAML, IterableSimpleNamespace
    args = IterableSimpleNamespace(**YAML.load(os.path.join(ROOT, "bytetrack.yaml")))
    tr = _ByteTracker(args)
    A = [100, 100, 200, 400]
    B = [800, 100, 900, 400]
    for _ in range(3):                                   # establish two tracks
        tr.update(_Dets(np.array([A, B], np.float32), np.array([0.9, 0.9], np.float32)))
    # Next frame: B is confident (idx 0), A drops to a low score (idx 1).
    out = tr.update(_Dets(np.array([B, A], np.float32), np.array([0.9, 0.2], np.float32)))
    by_x = {int(r[0]) // 100: int(r[-1]) for r in out}
    assert by_x.get(1) == 1 and by_x.get(8) == 0, f"index mix-up: {out[:, [0, 4, 7]]}"
    print("PASS ByteTrack reports global detection indices (low-score stage too)")


# --- privacy mask -----------------------------------------------------------

def test_privacy_masks_head_only():
    from tracker3d.privacy import PrivacyMasker
    img = np.tile(np.arange(200, dtype=np.uint8)[None, :, None], (400, 1, 3)).copy()
    orig = img.copy()
    kxy = np.zeros((17, 2)); kcf = np.zeros(17)
    kxy[0], kxy[1], kxy[2] = (100, 60), (92, 52), (108, 52); kcf[:3] = 0.9
    det = {"bbox": (60, 30, 140, 380), "kxy": kxy, "kconf": kcf}
    PrivacyMasker({"method": "solid"}).apply(img, [det])
    assert (img[55, 100] == 0).all(), "face centre must be masked"
    assert (img[300, 100] == orig[300, 100]).all(), "body must be untouched"
    print("PASS privacy mask covers the head from pose keypoints only")


# --- lens correction (synthetic wide-angle camera, no real calibration needed) ---

_LK = np.array([[672.0, 0, 960.0], [0, 672.0, 540.0], [0, 0, 1]])   # ~110 deg over 1920 px
_LD = np.array([-0.28, 0.08, 0.0005, -0.0003, 0.0])                 # barrel distortion


def _lens(dist=_LD):
    from tracker3d.lens import LensModel
    return LensModel(_LK, dist, (W, H_IMG))


def _floor_camera():
    """Camera 2.7 m above the floor, tilted 35 deg down. Returns (floor pts (N,2),
    raw distorted pixels (N,2), rvec, tvec), keeping points inside the image and
    inside the lens model's valid field of view."""
    import cv2
    t = np.radians(35)
    fwd = np.array([0, np.cos(t), -np.sin(t)])
    right = np.array([1.0, 0, 0])
    down = np.cross(fwd, right)
    R = np.vstack([right, down, fwd])
    C = np.array([0, 0, 2.7])
    tvec = -R @ C
    rvec = cv2.Rodrigues(R)[0]
    xs, ys = np.meshgrid(np.arange(-5, 5.01, 0.25), np.arange(0.5, 9.01, 0.25))
    floor = np.column_stack([xs.ravel(), ys.ravel()])
    cam = (R @ np.column_stack([floor, np.zeros(len(floor))]).T).T + tvec
    ok = cam[:, 2] > 0.1
    r = np.hypot(cam[:, 0] / cam[:, 2], cam[:, 1] / cam[:, 2])
    ok &= r < 1.7
    floor, cam3 = floor[ok], np.column_stack([floor[ok], np.zeros(ok.sum())])
    raw = cv2.projectPoints(cam3, rvec, tvec, _LK, _LD)[0].reshape(-1, 2)
    inside = (raw[:, 0] >= 0) & (raw[:, 0] < W) & (raw[:, 1] >= 0) & (raw[:, 1] < H_IMG)
    return floor[inside], raw[inside], rvec, tvec


def test_lens_round_trip():
    lens = _lens()
    us, vs = np.meshgrid(np.linspace(0, W - 1, 25), np.linspace(0, H_IMG - 1, 15))
    raw = np.column_stack([us.ravel(), vs.ravel()])
    back = lens.distort_points(lens.undistort_points(raw))
    err = np.abs(back - raw).max()
    assert err < 0.01, err
    # Below the frame (pose-extrapolated feet): linear extrapolation, monotonic.
    below = np.column_stack([np.full(12, 300.0), np.linspace(H_IMG - 1, H_IMG + 300, 12)])
    ry = lens.undistort_points(below)[:, 1]
    assert np.all(np.diff(ry) > 0), ry
    print(f"PASS lens undistort/distort round trip (max {err:.4f} px) + monotonic beyond frame")


def test_lens_floor_projection():
    from tracker3d.geometry import FloorProjector, compute_homography
    lens = _lens()
    floor, raw, _, _ = _floor_camera()
    # 8 well-spread clicks: the extremes of the visible floor.
    idx = sorted({int(np.argmin(raw[:, 0])), int(np.argmax(raw[:, 0])),
                  int(np.argmin(raw[:, 1])), int(np.argmax(raw[:, 1])),
                  int(np.argmin(raw.sum(1))), int(np.argmax(raw.sum(1))),
                  int(np.argmin(raw[:, 0] - raw[:, 1])), int(np.argmax(raw[:, 0] - raw[:, 1]))})
    H = compute_homography(lens.undistort_points(raw[idx]), floor[idx])
    proj = FloorProjector(H, (W, H_IMG), lens)
    proj.set_frame_size(W, H_IMG)
    err = max(np.hypot(*(np.array(proj.to_floor(*p)) - f)) for p, f in zip(raw, floor))
    assert err < 2e-3, err
    Hraw = compute_homography(raw[idx], floor[idx])           # today's raw-pixel fit
    rproj = FloorProjector(Hraw, (W, H_IMG))
    rproj.set_frame_size(W, H_IMG)
    errs_raw = [np.hypot(*(np.array(rproj.to_floor(*p)) - f)) for p, f in zip(raw, floor)
                if rproj.to_floor(*p) is not None]
    assert max(errs_raw) > 10 * err and max(errs_raw) > 0.05, (max(errs_raw), err)
    print(f"PASS lens-corrected floor projection: max err {err * 1000:.2f} mm "
          f"(raw-pixel homography: {max(errs_raw):.2f} m)")


def test_lens_rescale():
    from tracker3d.geometry import FloorProjector
    lens = _lens()
    floor, raw, _, _ = _floor_camera()
    from tracker3d.geometry import compute_homography
    H = compute_homography(lens.undistort_points(raw[::7]), floor[::7])
    full = FloorProjector(H, (W, H_IMG), lens); full.set_frame_size(W, H_IMG)
    half = FloorProjector(H, (W, H_IMG), lens); half.set_frame_size(W // 2, H_IMG // 2)
    for u, v in raw[::11]:
        a, b = full.to_floor(u, v), half.to_floor(u / 2, v / 2)
        assert np.allclose(a, b, atol=1e-6), (a, b)
    print("PASS lens-corrected projection is resolution independent")


def test_lens_guard():
    import tempfile
    from tracker3d.geometry import FloorProjector, save_calibration
    H = np.array([[0.01, 0.0, -3.0], [0.0, 0.02, -5.0], [0.0, 0.0008, 1.0]])
    lens_a, lens_b = _lens(), _lens(_LD * 1.1)
    with tempfile.TemporaryDirectory() as d:
        with_lens = os.path.join(d, "a_homography.npy")
        save_calibration(with_lens, H, (W, H_IMG), lens=lens_a)
        assert FloorProjector.load(with_lens, lens_a).lens is not None
        for wrong in (lens_b, None):
            try:
                FloorProjector.load(with_lens, wrong)
                raise AssertionError("mismatched lens accepted")
            except ValueError:
                pass
        raw_fit = os.path.join(d, "b_homography.npy")
        save_calibration(raw_fit, H, (W, H_IMG))
        p = FloorProjector.load(raw_fit, lens_a)          # H predates the lens file
        assert p.lens is None and p.lens_status.startswith("IGNORED"), p.lens_status
        p.set_frame_size(W, H_IMG)
        w = H @ np.array([900.0, 800.0, 1.0])
        assert np.allclose(p.to_floor(900, 800), w[:2] / w[2])   # unchanged raw behaviour
    print("PASS lens guard: mismatch refused, pre-lens homography keeps the raw path")


def test_lens_blob_contains_foot():
    from tracker3d.geometry import FloorProjector, compute_homography
    import cv2
    lens = _lens()
    floor, raw, _, _ = _floor_camera()
    H = compute_homography(lens.undistort_points(raw[::5]), floor[::5])
    proj = FloorProjector(H, (W, H_IMG), lens); proj.set_frame_size(W, H_IMG)
    edge = [i for i in range(len(raw)) if raw[i, 0] < 250 and 400 < raw[i, 1] < 900]
    assert edge, "no floor point near the image edge"
    i = edge[0]
    u, v = raw[i]
    bbox = (u - 50, v - 300, u + 50, v)
    kw = dict(far_frac=0.1, near_frac=1.4, clip_inflate=2.5, body_aspect=3.0, short_aspect=1.8)
    poly = bbox_floor_polygon(proj.H0, proj.rectify_bbox(bbox), (W, H_IMG), clipped=False, **kw)
    assert poly is not None
    inside = cv2.pointPolygonTest(poly.astype(np.float32), tuple(map(float, floor[i])), False)
    assert inside >= 0, (poly, floor[i])
    tall = bbox_floor_polygon(proj.H0, proj.rectify_bbox(bbox), (W, H_IMG), clipped=True, **kw)
    assert cv2.contourArea(tall.astype(np.float32)) > cv2.contourArea(poly.astype(np.float32))
    print("PASS lens-corrected blob trapezoid contains the true foot near the image edge")


def test_charuco_board_detect():
    import cv2
    from tracker3d.lens import BOARD_DEFAULTS, detect_board, make_board, make_detector
    board = make_board(**BOARD_DEFAULTS)
    img = board.generateImage((700, 500), marginSize=0, borderBits=1)
    canvas = np.full((800, 1000), 255, np.uint8)
    canvas[150:650, 150:850] = img
    M = np.array([[0.9, 0.12, 40], [-0.05, 0.85, 60], [0.0001, 0.00015, 1.0]])
    warped = cv2.warpPerspective(canvas, M, (1000, 800), borderValue=255)
    det = detect_board(warped, make_detector(board), board)
    n_all = (BOARD_DEFAULTS["cols"] - 1) * (BOARD_DEFAULTS["rows"] - 1)
    assert det is not None and len(det[2]) >= 0.9 * n_all, None if det is None else len(det[2])
    print(f"PASS ChArUco generator <-> detector agree ({len(det[2])}/{n_all} corners)")


def _synthetic_views(seed, n=30):
    """Board views through the synthetic 110 deg lens, ~0.6-1.4 m away (the board
    looks small, as it will from a ladder), with 0.1 px corner noise."""
    import cv2
    from tracker3d.lens import BOARD_DEFAULTS, make_board
    obj = np.asarray(make_board(**BOARD_DEFAULTS).getChessboardCorners(), np.float64).reshape(-1, 3)
    centre = obj.mean(axis=0)
    rng = np.random.default_rng(seed)
    objs, imgs = [], []
    for _ in range(5000):
        if len(objs) == n:
            break
        R = cv2.Rodrigues(rng.uniform(-0.6, 0.6, 3))[0]
        z = rng.uniform(0.6, 1.4)
        nx, ny = rng.uniform(-1.3, 1.3), rng.uniform(-0.75, 0.75)
        tvec = np.array([nx * z, ny * z, z]) - R @ centre
        cam = (R @ obj.T).T + tvec
        if (cam[:, 2] <= 0.1).any() or \
                (np.hypot(cam[:, 0] / cam[:, 2], cam[:, 1] / cam[:, 2]) > 1.7).any():
            continue
        px = cv2.projectPoints(obj, cv2.Rodrigues(R)[0], tvec, _LK, _LD)[0].reshape(-1, 2)
        if (px < 0).any() or (px[:, 0] >= W).any() or (px[:, 1] >= H_IMG).any():
            continue
        objs.append(obj.astype(np.float32))
        imgs.append((px + rng.normal(0, 0.1, px.shape)).astype(np.float32))
    return objs, imgs


def test_solve_intrinsics_synthetic():
    """Seeds 5 and 7 fell into the wide-angle false minimum (fx ~2x, rms > 1 px)
    with OpenCV's default start; the staged multi-start solve must recover them."""
    from tracker3d.lens import solve_with_outliers
    worst = (0.0, 0.0, 0.0)
    for seed in (1, 5, 7):
        objs, imgs = _synthetic_views(seed)
        assert len(objs) >= 20, len(objs)
        res, _ = solve_with_outliers(objs, imgs, (W, H_IMG), "standard")
        fx_err = abs(res["K"][0, 0] - _LK[0, 0]) / _LK[0, 0]
        k1_err = abs(res["dist"][0] - _LD[0])
        assert fx_err < 0.02 and k1_err < 0.02 and res["rms"] < 0.2, (seed, fx_err, k1_err, res["rms"])
        worst = tuple(max(a, b) for a, b in zip(worst, (fx_err, k1_err, res["rms"])))
    print(f"PASS intrinsics solve recovers a 110 deg lens incl. hard cases (worst: fx "
          f"{worst[0] * 100:.2f} %, k1 err {worst[1]:.4f}, rms {worst[2]:.3f} px)")


if __name__ == "__main__":
    test_convex_intersection()
    test_foot_from_pose()
    test_sigma_weighting_pulls_to_confident_foot()
    test_duplicate_suppression()
    test_homography_fit_and_loo()
    test_bytetrack_global_index()
    test_privacy_masks_head_only()
    test_lens_round_trip()
    test_lens_floor_projection()
    test_lens_rescale()
    test_lens_guard()
    test_lens_blob_contains_foot()
    test_charuco_board_detect()
    test_solve_intrinsics_synthetic()
    Hs = _load()
    if Hs is not None:
        test_projector_rescale_and_horizon(Hs)
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
