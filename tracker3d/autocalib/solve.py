"""Calibrated planar initialization and sparse joint camera/person optimization.

Coordinates are right-handed, Z up, with the root camera at (0,0,1). Camera
poses use X_camera = R @ (X_rig - C). Ankles are separate observations on a
shared plane above the floor; nose observations provide a soft upright prior.
"""
from itertools import combinations

import cv2
import numpy as np

from .observations import has_spread, pair_matches, prepare, split_samples
from .session import VERSION, lens_from_dict, seal

SEED_RANSAC_PX = 12.0


def _ankle_pair(view):
    return view["rays"][:2]


def _swap_view(view):
    """Exchange left/right ankles in every per-view field (nose stays last)."""
    for key in ("raw", "rect", "rays", "conf"):
        view[key] = view[key][[1, 0, 2]]


def align_ankle_labels(samples, poses, z):
    """Make left/right ankle labels consistent across cameras within each sample.

    Pose models sometimes exchange the two ankles in one view. The solve only
    needs the same anatomical pairing in every view of a sample, so each view is
    matched to the first view's projected ankles and swapped when that fits
    better. Returns the number of views changed. Mutates `samples`.
    """
    changed = 0
    for s in samples:
        names = [n for n in s["views"] if n in poses]
        projected = {}
        for n in names:
            try:
                projected[n] = intersect_plane(*poses[n], _ankle_pair(s["views"][n]), z)
            except ValueError:
                pass
        if len(projected) < 2:
            continue
        ref_name = next(iter(projected))
        ref = projected[ref_name]
        for n, p in projected.items():
            if n == ref_name:
                continue
            if (np.linalg.norm(p[[1, 0]] - ref, axis=1).sum()
                    < np.linalg.norm(p - ref, axis=1).sum()):
                _swap_view(s["views"][n])
                changed += 1
    return changed


def floor_homography(lens, R, C, z=0.0):
    t = -R @ C + R[:, 2] * z
    A = lens.K_rect @ np.column_stack([R[:, :2], t])
    if abs(np.linalg.det(A)) < 1e-10:
        raise ValueError("camera lies on the projection plane")
    H = np.linalg.inv(A)
    return H / np.linalg.norm(H)


def intersect_plane(R, C, rays, z=0.0):
    d = (R.T @ np.c_[rays, np.ones(len(rays))].T).T
    denom = d[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        distance = (z - C[2]) / denom
        points = C + distance[:, None] * d
    if not np.isfinite(points).all() or np.any(distance <= 0):
        raise ValueError("observed ankle lies on or above the horizon")
    return points[:, :2]


def project_raw(lens, R, C, points, focal_scale=1.0):
    q = (R @ (np.asarray(points) - C).T).T
    # Smooth finite residuals during optimization, including candidates behind camera.
    z = np.maximum(q[:, 2], 1e-4)
    norm = np.clip(q[:, :2] / z[:, None], -20, 20)
    if lens._supported():
        x, y = lens._distort_norm(norm[:, 0], norm[:, 1])
        K = lens.K.copy()
        K[0, 0] *= focal_scale
        K[1, 1] *= focal_scale
        p = K @ np.vstack([x, y, np.ones(len(x))])
        return (p[:2] / p[2]).T, q[:, 2]
    raise ValueError("autocalibration supports standard, rational and fisheye lens models")


def _root_rotation(normal):
    z = -normal / np.linalg.norm(normal)
    x = np.array([1.0, 0, 0])
    x -= z * (x @ z)
    if np.linalg.norm(x) < 0.1:
        x = np.array([0., 1, 0])
        x -= z * (x @ z)
    x /= np.linalg.norm(x)
    return np.column_stack([x, np.cross(z, x), z])


def _decompose(H):
    _, rotations, translations, normals = cv2.decomposeHomographyMat(H, np.eye(3))
    return [(R, t.ravel(), n.ravel()) for R, t, n in zip(rotations, translations, normals)
            if np.linalg.norm(t) > 1e-5 and np.linalg.norm(n) > 0.5]


def _body_score(samples, poses):
    heights, offsets = [], []
    for sample in samples:
        visible = [n for n in poses if n in sample["views"]
                   and sample["views"][n]["conf"][2] > 0]
        if len(visible) < 2:
            continue
        a, b = visible[:2]
        Ra, Ca = poses[a]
        Rb, Cb = poses[b]
        va, vb = sample["views"][a], sample["views"][b]
        Pa, Pb = np.c_[Ra, -Ra @ Ca], np.c_[Rb, -Rb @ Cb]
        X = cv2.triangulatePoints(Pa, Pb, va["rays"][2:].T, vb["rays"][2:].T).ravel()
        if abs(X[3]) < 1e-10:
            continue
        X = X[:3] / X[3]
        if (Ra @ (X - Ca))[2] <= 0 or (Rb @ (X - Cb))[2] <= 0:
            continue
        try:
            feet = intersect_plane(Ra, Ca, va["rays"][:2])
        except ValueError:
            continue
        heights.append(X[2])
        offsets.append(np.linalg.norm(X[:2] - feet.mean(axis=0)))
    if len(heights) < 6 or np.median(heights) <= 0.03:
        return float("inf"), None
    h = float(np.median(heights))
    return float((np.median(offsets) + np.median(np.abs(np.array(heights) - h))) / h), h


def initialize(samples, lenses):
    """Enumerate root branches, propagate plane distance through a connected graph."""
    names = sorted(lenses)
    edges, graph_report = {}, {}
    cv2.setRNGSeed(7)
    for a, b in combinations(names, 2):
        x, y = pair_matches(samples, a, b)
        if len(x) < 24 or not has_spread(x) or not has_spread(y):
            continue
        # Seed only: the bundle adjustment refines it. Pose keypoints carry several
        # pixels of noise in both views, so a tight threshold rejects good overlaps.
        threshold = SEED_RANSAC_PX / min(lenses[b].K_rect[0, 0], lenses[b].K_rect[1, 1])
        H, mask = cv2.findHomography(x, y, cv2.RANSAC, threshold, maxIters=5000,
                                    confidence=0.999)
        if H is None or mask is None or np.linalg.cond(H) > 1e8:
            continue
        keep = mask.ravel().astype(bool)
        if keep.sum() < 20 or keep.mean() < 0.5 or not has_spread(x[keep]):
            continue
        edges[a, b] = (H, int(keep.sum()))
        edges[b, a] = (np.linalg.inv(H), int(keep.sum()))
        graph_report[f"{a}/{b}"] = {"matches": len(x), "inliers": int(keep.sum())}
    if not edges:
        raise ValueError("no reliable overlap: collect more well-spread upright pauses")
    root, second = max(edges, key=lambda e: edges[e][1])
    connected = {root}
    while True:
        bigger = connected | {b for a, b in edges if a in connected}
        if bigger == connected:
            break
        connected = bigger
    if connected != set(names):
        raise ValueError(f"disconnected overlap graph: {sorted(set(names) - connected)}")
    candidates = []
    x, _ = pair_matches(samples, root, second)
    for relR, relt, normal in _decompose(edges[root, second][0]):
        if np.median(np.c_[x, np.ones(len(x))] @ normal) <= 0:
            continue
        R0, C0 = _root_rotation(normal), np.array([0., 0., 1.])
        R1 = relR @ R0
        C1 = -R1.T @ (relR @ (-R0 @ C0) + relt)
        if C1[2] <= 0.05:
            continue
        poses = {root: (R0, C0), second: (R1, C1)}
        while len(poses) < len(names):
            options = sorted((e for e in edges if e[0] in poses and e[1] not in poses),
                             key=lambda e: -edges[e][1])
            added = False
            for a, b in options:
                Ra, Ca = poses[a]
                expected_n = -Ra[:, 2]
                choices = []
                for R, t, n in _decompose(edges[a, b][0]):
                    normal_error = float(1 - n @ expected_n)
                    if normal_error > 0.08:
                        continue
                    Rb = R @ Ra
                    Cb = -Rb.T @ (R @ (-Ra @ Ca) + t * Ca[2])
                    if Cb[2] <= 0.05:
                        continue
                    try:
                        xa, xb = pair_matches(samples, a, b)
                        pa = intersect_plane(Ra, Ca, xa)
                        pb = intersect_plane(Rb, Cb, xb)
                    except ValueError:
                        continue
                    error = np.median(np.linalg.norm(pa - pb, axis=1)) + normal_error
                    choices.append((error, Rb, Cb))
                if choices:
                    _, Rb, Cb = min(choices, key=lambda v: v[0])
                    poses[b] = (Rb, Cb)
                    added = True
                    break
            if not added:
                break
        if len(poses) != len(names):
            continue
        try:
            for sample in samples:
                for n, v in sample["views"].items():
                    intersect_plane(*poses[n], v["rays"][:2])
        except ValueError:
            continue
        score, height = _body_score(samples, poses)
        if np.isfinite(score):
            candidates.append((score, poses, height))
    if not candidates:
        raise ValueError("pose ambiguity unresolved: include visible faces and upright pauses in overlaps")
    candidates.sort(key=lambda c: c[0])
    return root, candidates[:3], graph_report


def _bundle(samples, lenses, root, initial, height, ankle_ratio, refine=False,
            max_nfev=1500):
    from scipy.optimize import least_squares
    from scipy.sparse import lil_matrix

    names = [root] + [n for n in sorted(lenses) if n != root]
    # Initial reconstruction uses ankle plane Z=0. Lift cameras to the floor frame,
    # then normalize root height. The ankle offset is a documented anthropometric prior.
    nose_height = height / (1 - ankle_ratio)
    lift = nose_height * ankle_ratio
    scale = initial[root][1][2] + lift
    poses = {n: (R.copy(), (C + [0, 0, lift]) / scale) for n, (R, C) in initial.items()}
    base = poses[root][0]
    x, lower, upper, cam_slice = [0., 0.], [-0.6, -0.6], [0.6, 0.6], {root: slice(0, 2)}
    for n in names[1:]:
        R, C = poses[n]
        cam_slice[n] = slice(len(x), len(x) + 6)
        x.extend([*cv2.Rodrigues(R)[0].ravel(), C[0], C[1], np.log(C[2])])
        lower.extend([-np.inf] * 5 + [np.log(0.05)])
        upper.extend([np.inf] * 5 + [np.log(50)])
    hi = len(x)
    x.append(np.log(nose_height / scale))
    lower.append(np.log(0.03)); upper.append(np.log(10))
    pi = len(x)
    for s in samples:
        pts = [intersect_plane(*initial[n], v["rays"][:2]) / scale
               for n, v in s["views"].items()]
        x.extend(np.median(pts, axis=0).ravel())
        lower.extend([-np.inf] * 4); upper.extend([np.inf] * 4)
    fi = len(x)
    if refine:
        x.extend([0.] * len(names))
        lower.extend([np.log(0.97)] * len(names)); upper.extend([np.log(1.03)] * len(names))
    obs = {n: [] for n in names}
    for k, s in enumerate(samples):
        for n, v in s["views"].items():
            obs[n].append((k, v))
    obs = {n: (np.array([k for k, v in rows]), np.array([v["raw"] for k, v in rows]),
               np.array([v["conf"] for k, v in rows])) for n, rows in obs.items()}

    def unpack(v):
        p = {root: (base @ cv2.Rodrigues(np.array([v[0], v[1], 0.]))[0],
                    np.array([0., 0., 1.]))}
        for n in names[1:]:
            q = v[cam_slice[n]]
            p[n] = (cv2.Rodrigues(q[:3])[0], np.array([q[3], q[4], np.exp(q[5])]))
        return p, np.exp(v[hi]), v[pi:fi].reshape(-1, 2, 2)

    def residual(v):
        p, h, feet = unpack(v)
        out = []
        for ni, n in enumerate(names):
            idx, raw, conf = obs[n]
            points = np.empty((len(idx), 3, 3))
            points[:, :2, :2] = feet[idx]
            points[:, :2, 2] = ankle_ratio * h
            points[:, 2, :2] = feet[idx].mean(axis=1)
            points[:, 2, 2] = h
            pred, depth = project_raw(lenses[n], *p[n], points.reshape(-1, 3),
                                      np.exp(v[fi + ni]) if refine else 1.)
            error = pred.reshape(-1, 3, 2) - raw
            weights = np.sqrt(conf) / np.array([3., 3., 12.])
            out.append((error * weights[:, :, None]).ravel())
            out.append((np.maximum(0.01 - depth, 0) * 1000).ravel())
        if refine:
            out.append(v[fi:] / 0.01)
        return np.concatenate(out)

    nr = sum(len(rows[0]) * 9 for rows in obs.values()) + (len(names) if refine else 0)
    sparsity = lil_matrix((nr, len(x)), dtype=int)
    row = 0
    for ni, n in enumerate(names):
        idx = obs[n][0]
        for j, k in enumerate(idx):
            cols = list(range(cam_slice[n].start, cam_slice[n].stop)) + [hi]
            cols += list(range(pi + 4 * k, pi + 4 * k + 4))
            if refine:
                cols.append(fi + ni)
            for r in list(range(row + 6 * j, row + 6 * j + 6)) + list(
                    range(row + 6 * len(idx) + 3 * j, row + 6 * len(idx) + 3 * j + 3)):
                sparsity[r, cols] = 1
        row += 9 * len(idx)
    if refine:
        for ni in range(len(names)):
            sparsity[row + ni, fi + ni] = 1
    residual_cost0 = 0.5 * float(np.sum(_soft_l1(residual(np.asarray(x)))))
    result = least_squares(residual, x, bounds=(lower, upper), jac_sparsity=sparsity.tocsr(),
                           loss="soft_l1", f_scale=1., x_scale="jac", max_nfev=max_nfev,
                           ftol=1e-7, xtol=1e-7, gtol=1e-7)
    # Hitting the evaluation limit (status 0) still leaves a usable, improved
    # estimate: held-out agreement and the independent validation walk are the
    # acceptance gate, not the optimizer's own stopping rule. Record it instead.
    if not np.isfinite(result.x).all() or result.status < 0 or (
            result.status == 0 and not result.cost < residual_cost0):
        raise ValueError(f"joint solve did not converge: {result.message}")
    poses, height, feet = unpack(result.x)
    solved_lenses = dict(lenses)
    if refine:
        from tracker3d.lens import LensModel
        for ni, n in enumerate(names):
            K = lenses[n].K.copy()
            K[0, 0] *= np.exp(result.x[fi + ni])
            K[1, 1] *= np.exp(result.x[fi + ni])
            solved_lenses[n] = LensModel(K, lenses[n].dist, lenses[n].image_size, lenses[n].model)
    return poses, height, feet, solved_lenses, {"cost": float(result.cost),
                                               "evaluations": result.nfev,
                                               "converged": bool(result.success),
                                               "message": result.message}


def _soft_l1(r):
    """The robust cost scipy's least_squares uses for loss="soft_l1", f_scale=1."""
    return 2 * (np.sqrt(1 + r * r) - 1)


def statistics(values):
    a = np.asarray(values, float)
    return {"count": len(a), "median": float(np.median(a)),
            "p95": float(np.percentile(a, 95)), "max": float(np.max(a))} if len(a) else None


def evaluate(samples, poses, lenses, ankle_z):
    """Independent per-camera ankle-plane projections, before fusion or smoothing."""
    errors, pairs, pixels = [], {}, {n: [] for n in poses}
    invalid, positions = 0, []
    for s in samples:
        projected = {}
        for n, v in s["views"].items():
            rect = lenses[n]._undistort_exact(v["raw"][:2])
            rays = (np.linalg.inv(lenses[n].K_rect) @ np.c_[rect, np.ones(2)].T).T
            try:
                projected[n] = intersect_plane(*poses[n], rays[:, :2] / rays[:, 2:], ankle_z)
            except ValueError:
                invalid += 1
        if len(projected) < 2:
            continue
        # Compare foot positions, not left/right labels: align each view's pair to
        # the first view's (an exchanged label is not a geometric disagreement).
        first = next(iter(projected))
        for n in projected:
            p = projected[n]
            if n != first and (np.linalg.norm(p[[1, 0]] - projected[first], axis=1).sum()
                               < np.linalg.norm(p - projected[first], axis=1).sum()):
                projected[n] = p[[1, 0]]
        positions.append({"t": s["t"], "xy": np.median(list(projected.values()), axis=0).mean(axis=0).tolist()})
        for a, b in combinations(sorted(projected), 2):
            d = np.linalg.norm(projected[a] - projected[b], axis=1).tolist()
            errors.extend(d)
            pairs.setdefault(f"{a}/{b}", []).extend(d)
        for n in projected:
            other = np.median([v for k, v in projected.items() if k != n], axis=0)
            pred, _ = project_raw(lenses[n], *poses[n], np.c_[other, np.full(2, ankle_z)])
            raw = s["views"][n]["raw"][:2]
            pixels[n].extend(min((np.linalg.norm(pred - raw, axis=1),
                                  np.linalg.norm(pred[[1, 0]] - raw, axis=1)),
                                 key=np.sum).tolist())
    return {"agreement_rig_units": statistics(errors),
            "interpolation_bracket_ms": statistics([1000 * v["bracket_s"] for s in samples
                                                     for v in s["views"].values()]),
            "pairs": {n: statistics(v) for n, v in pairs.items()},
            "reprojection_px": {n: statistics(v) for n, v in pixels.items()},
            "invalid_projections": invalid}, positions


def solve_session(session, ankle_ratio=0.05, refine_focal=False, max_nfev=1500, **sampling):
    if not 0 <= ankle_ratio <= 0.10:
        raise ValueError("ankle-height ratio must be in [0, 0.10]")
    samples, lenses, counts = prepare(session, **sampling)
    train, heldout = split_samples(samples)
    root, candidates, graph = initialize(train, lenses)
    results, failures = [], []
    for _, poses, height in candidates:
        try:
            align_ankle_labels(train, poses, 0.0)   # initial frame: ankle plane is z = 0
            result = _bundle(train, lenses, root, poses, height, ankle_ratio, max_nfev=max_nfev)
            if align_ankle_labels(train, result[0], ankle_ratio * result[1]):
                result = _bundle(train, lenses, root, poses, height, ankle_ratio, max_nfev=max_nfev)
            quality, _ = evaluate(heldout, result[0], result[3], ankle_ratio * result[1])
            if quality["invalid_projections"] or any(v is None or v["count"] < 8
                                                       for v in quality["reprojection_px"].values()):
                raise ValueError("insufficient valid held-out coverage for every camera")
            results.append((result[4]["cost"], result, quality, poses, height))
        except ValueError as e:
            failures.append(str(e))
    if not results:
        raise ValueError("all pose candidates failed: " + "; ".join(failures))
    # Branch selection uses training objective, never withheld data.
    _, result, quality, seed, seed_height = min(results, key=lambda r: r[0])
    refinement = {"requested": refine_focal, "accepted": False}
    if refine_focal:
        try:
            refined = _bundle(train, lenses, root, seed, seed_height, ankle_ratio,
                              refine=True, max_nfev=max_nfev)
            q, _ = evaluate(heldout, refined[0], refined[3], ankle_ratio * refined[1])
            old, new = quality["agreement_rig_units"], q["agreement_rig_units"]
            if (new and not q["invalid_projections"] and new["median"] < old["median"] * 0.99
                    and new["p95"] <= old["p95"] and all(v is not None
                                                                  for v in q["reprojection_px"].values())):
                result, quality = refined, q
                refinement["accepted"] = True
            else:
                refinement["reason"] = "withheld agreement did not improve"
        except ValueError as e:
            refinement["reason"] = str(e)
    poses, height, feet, solved_lenses, optimization = result
    training, _ = evaluate(train, poses, solved_lenses, ankle_ratio * height)
    _, trajectory = evaluate(samples, poses, solved_lenses, ankle_ratio * height)
    cameras = {}
    time_ranges = {}
    for n, (R, C) in poses.items():
        cameras[n] = {"R": R.tolist(), "C": C.tolist(), "lens": solved_lenses[n].to_dict(),
                      "input_lens_fingerprint": lenses[n].fingerprint(),
                      "floor_homography": floor_homography(solved_lenses[n], R, C).tolist()}
        times = [r["t"] for r in session["records"] if r["kind"] == "frame" and r["camera"] == n]
        time_ranges[n] = [min(times), max(times)]
    return seal({"version": VERSION, "kind": "rig", "session_id": session["id"],
                 "observations_id": session["observations_id"], "capture_time_ranges": time_ranges,
                 "root_camera": root, "units": "root_camera_height", "cameras": cameras,
                 "body": {"nose_height": float(height), "ankle_height_ratio": ankle_ratio,
                          "ankle_z": float(ankle_ratio * height)},
                 "sampling": sampling, "counts": counts, "overlap_graph": graph,
                 "training": training, "heldout": quality, "optimization": optimization,
                 "focal_refinement": refinement, "trajectory": trajectory,
                 "markers": [r for r in session["records"] if r["kind"] == "marker"]})


def rig_geometry(rig):
    poses = {n: (np.array(c["R"]), np.array(c["C"])) for n, c in rig["cameras"].items()}
    lenses = {n: lens_from_dict(c["lens"]) for n, c in rig["cameras"].items()}
    return poses, lenses
