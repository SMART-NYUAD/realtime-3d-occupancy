"""One similarity transform attaches every camera to the existing map frame."""
import hashlib
from pathlib import Path

import numpy as np

from .session import VERSION, check_artifact, seal

NOSE_HEIGHT_RANGE_M = (1.2, 2.0)


def map_identity(cfg):
    f = cfg["floorplan"]
    path = Path(f.get("pointcloud_file", "data/smart_lab.las"))
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    # Colour, raster resolution and margin do not change world meters.
    up = str(f.get("up_axis", "auto")).lower()
    if up == "auto":
        import laspy
        from tracker3d.plan import PointCloudPlan
        las = laspy.read(path)
        pts = np.vstack([las.x, las.y, las.z]).T
        up = "xyz"[PointCloudPlan._detect_up_axis(pts)]
    if up not in ("x", "y", "z"):
        raise ValueError("map up_axis must be auto, x, y or z")
    # Scan floor axes are the remaining axes in their original order. A Y-up
    # scan is left-handed in (X,Z,Y) until a single floor-axis flip restores it.
    handedness = -1 if up == "y" else 1
    if bool(f.get("flip_x", False)) != bool(f.get("flip_y", False)):
        handedness *= -1
    return {"scan_sha256": h.hexdigest(), "up_axis": f.get("up_axis", "auto"),
            "resolved_up_axis": up, "handedness": handedness,
            "flip_x": bool(f.get("flip_x", False)), "flip_y": bool(f.get("flip_y", False)),
            "clip_percentile": float(f.get("clip_percentile", 0.2))}


def apply_transform(S, points):
    p = np.asarray(points, float)
    return (np.asarray(S)[:2, :2] @ p.T).T + np.asarray(S)[:2, 2]


def fit_similarity(source, target, min_separation_m=2.0, handedness=1):
    source, target = np.asarray(source, float), np.asarray(target, float)
    if (source.shape != target.shape or source.ndim != 2 or source.shape[1] != 2
            or len(source) < 2 or not np.isfinite(source).all() or not np.isfinite(target).all()):
        raise ValueError("need at least two finite rig/map point pairs")
    distances = np.linalg.norm(target[:, None] - target[None, :], axis=2)
    if distances.max() < min_separation_m:
        raise ValueError(f"anchors must span at least {min_separation_m:g} meters")
    a, b = source.mean(axis=0), target.mean(axis=0)
    X, Y = source - a, target - b
    if np.sum(X * X) < 1e-8:
        raise ValueError("rig anchors coincide")
    U, singular, Vt = np.linalg.svd(Y.T @ X)
    sign = np.linalg.det(U @ Vt)
    if handedness not in (-1, 1):
        raise ValueError("map handedness must be +1 or -1")
    D = np.diag([1., handedness * (1. if sign >= 0 else -1.)])
    R = U @ D @ Vt  # parity is determined by map axes, never fitted freely
    scale = np.sum(singular * np.diag(D)) / np.sum(X * X)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("invalid anchor scale")
    S = np.eye(3)
    S[:2, :2] = scale * R
    S[:2, 2] = b - scale * R @ a
    return S


def resolve_points(points, trajectory, markers):
    if not trajectory:
        raise ValueError("no reconstructed trajectory")
    times = np.array([p["t"] for p in trajectory])
    positions = np.array([p["xy"] for p in trajectory])
    named = {m["name"]: m["t"] for m in markers}
    out = []
    for p in points:
        if "marker" in p:
            if p["marker"] not in named:
                raise ValueError(f"unknown pause marker: {p['marker']}")
            t = named[p["marker"]]
        else:
            t = float(p["t"])
        selected = np.abs(times - t) <= 0.5
        if selected.sum() < 2:
            raise ValueError(f"pause at {t:.3f} needs >=2 valid shared samples within 0.5 s")
        world = np.asarray(p["world"], float)
        if world.shape != (2,) or not np.isfinite(world).all():
            raise ValueError("map point must be finite [x, y] meters")
        role = p.get("role", "anchor")
        if role not in ("anchor", "check"):
            raise ValueError("point role must be anchor or check")
        xy = np.median(positions[selected], axis=0)
        out.append({"t": t, "marker": p.get("marker"), "rig": xy.tolist(),
                    "world": world.tolist(), "role": role,
                    "spread_rig": float(np.max(np.linalg.norm(positions[selected] - xy, axis=1)))})
    if len({p["t"] for p in out}) != len(out):
        raise ValueError("a pause cannot be reused as another anchor or independent checkpoint")
    return out


def anchor_rig(rig, points, cfg):
    check_artifact(rig, "rig")
    resolved = resolve_points(points, rig["trajectory"], rig["markers"])
    fit = [p for p in resolved if p["role"] == "anchor"]
    identity = map_identity(cfg)
    S = fit_similarity([p["rig"] for p in fit], [p["world"] for p in fit],
                       handedness=identity["handedness"])
    scale = float(np.linalg.norm(S[:2, 0]))
    for p in resolved:
        p["error_m"] = float(np.linalg.norm(apply_transform(S, [p["rig"]])[0] - p["world"]))
        if p["spread_rig"] * scale > 0.15:
            raise ValueError("anchor/checkpoint was moving; use a longer upright pause")
    if max(p["error_m"] for p in fit) > 0.25:
        raise ValueError("anchors disagree by >25 cm; check correspondences, orientation and map")
    # Anchors fix the metric scale. The walker's reconstructed nose height is an
    # independent check of it: compare with their real nose height (about 10 cm
    # below standing height). A large mismatch means a wrong anchor or map point.
    nose_m = float(rig["body"]["nose_height"] * scale)
    warnings = []
    if not NOSE_HEIGHT_RANGE_M[0] <= nose_m <= NOSE_HEIGHT_RANGE_M[1]:
        warnings.append(f"reconstructed nose height {nose_m:.2f} m is implausible; "
                        "check anchor correspondences and the map scale")
    return seal({"version": VERSION, "kind": "anchored_rig", "rig": rig,
                 "transform": S.tolist(), "scale_m": scale, "map": identity,
                 "points": resolved, "fit_error_m": [p["error_m"] for p in fit],
                 "nose_height_m": nose_m, "warnings": warnings})


def pick_map_points(rig, cfg):
    """Small OpenCV picker. Each map click identifies a named, recorded pause."""
    import cv2
    from tracker3d.plan import load_plan
    plan = load_plan(cfg)
    markers = rig["markers"]
    if len(markers) < 2:
        raise ValueError("record >=2 pause markers with SPACE, or supply a --points JSON file")
    state = {"index": 0, "role": "anchor", "points": {}}
    window = "Anchor camera rig"

    def clicked(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            marker = markers[state["index"]]["name"]
            state["points"][marker] = {"marker": marker, "world": list(plan.px_to_world(x, y)),
                                        "role": state["role"]}

    cv2.namedWindow(window)
    cv2.setMouseCallback(window, clicked)
    print("Click this pause's floor location. n/p: next/previous pause; c: anchor/check; "
          "u: remove; ENTER: save; q: cancel. Use well-separated anchors and extra checkpoints.")
    try:
        while True:
            canvas = plan.background()
            name = markers[state["index"]]["name"]
            for p in state["points"].values():
                px = plan.world_to_px(*p["world"])
                color = (0, 220, 255) if p["role"] == "anchor" else (255, 180, 0)
                cv2.circle(canvas, px, 5, color, -1)
                cv2.putText(canvas, p["marker"], px, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            cv2.putText(canvas, f"{name}: {state['role']} | n/p pause, c role, ENTER save", (10, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            cv2.imshow(window, canvas)
            key = cv2.waitKey(20) & 0xFF
            if key in (ord("q"), 27):
                raise ValueError("anchoring cancelled; no result saved")
            if key in (10, 13):
                return list(state["points"].values())
            if key in (ord("n"), ord("p")):
                state["index"] = (state["index"] + (1 if key == ord("n") else -1)) % len(markers)
            if key == ord("c"):
                state["role"] = "check" if state["role"] == "anchor" else "anchor"
            if key == ord("u"):
                state["points"].pop(name, None)
    finally:
        cv2.destroyWindow(window)
