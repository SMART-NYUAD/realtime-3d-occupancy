"""Held-out validation and export of a complete candidate tracker configuration."""
from copy import deepcopy
from pathlib import Path
import shutil
import tempfile
import os

import numpy as np
import yaml

from tracker3d.geometry import FloorProjector, save_calibration
from tracker3d.lens import load_lens
from .anchor import apply_transform, map_identity, resolve_points
from .observations import prepare, split_samples
from .session import VERSION, check_artifact, seal, write_json
from .solve import evaluate, floor_homography, rig_geometry, statistics


def validate(anchored, session, cfg, checkpoints=None, allow_unmeasured=False):
    """`allow_unmeasured`: pass on camera agreement alone when no measured
    checkpoints exist (the report then states absolute accuracy was not measured)."""
    check_artifact(anchored, "anchored_rig")
    rig = check_artifact(anchored["rig"], "rig")
    if map_identity(cfg) != anchored["map"]:
        raise ValueError("map frame changed since anchoring; anchor the rig again")
    poses, lenses = rig_geometry(rig)
    recorded = session["manifest"]["cameras"]
    if set(recorded) != set(poses):
        raise ValueError("validation session camera set differs from the solved rig")
    for n in recorded:
        if recorded[n]["lens"]["fingerprint"] != rig["cameras"][n]["input_lens_fingerprint"]:
            raise ValueError(f"{n}: validation session uses a different input lens")
    samples, _, counts = prepare(session, **rig["sampling"])
    independent = (session["id"] != rig["session_id"]
                   and session["observations_id"] != rig["observations_id"])
    for n, (start, end) in rig["capture_time_ranges"].items():
        times = [r["t"] for r in session["records"] if r["kind"] == "frame" and r["camera"] == n]
        if times and max(start, min(times)) <= min(end, max(times)):
            independent = False
    evaluated = samples if independent else split_samples(samples)[1]
    quality, _ = evaluate(evaluated, poses, lenses, rig["body"]["ankle_z"])
    _, trajectory = evaluate(samples, poses, lenses, rig["body"]["ankle_z"])
    scale = anchored["scale_m"]
    agreement = quality["agreement_rig_units"]
    metric = {k: v if k == "count" else v * scale for k, v in agreement.items()} if agreement else None
    markers = [r for r in session["records"] if r["kind"] == "marker"]
    points = []
    if checkpoints:
        points = resolve_points([{**p, "role": "check"} for p in checkpoints], trajectory, markers)
        for p in points:
            p["error_m"] = float(np.linalg.norm(
                apply_transform(anchored["transform"], [p["rig"]])[0] - p["world"]))
    elif not independent:
        points = [p for p in anchored["points"] if p["role"] == "check"]
    absolute = statistics([p["error_m"] for p in points])
    reasons = []
    if not independent:
        reasons.append("record a separate validation walk before exporting")
    if quality["invalid_projections"]:
        reasons.append("some validation observations project above the horizon")
    if any(v is None or v["count"] < 8 for v in quality["reprojection_px"].values()):
        reasons.append("each camera needs >=8 held-out ankle observations")
    connected = {next(iter(poses))}
    edges = [pair.split("/") for pair, st in quality["pairs"].items() if st["count"] >= 8]
    for _ in poses:
        connected |= {n for edge in edges if connected.intersection(edge) for n in edge}
    if connected != set(poses):
        reasons.append("validation overlaps do not connect every camera")
    if not metric or metric["median"] > 0.10 or metric["p95"] > 0.25:
        reasons.append("camera disagreement exceeds median 10 cm / p95 25 cm")
    if not absolute and not (allow_unmeasured and independent):
        reasons.append("supply independent measured floor checkpoints")
    elif absolute and (absolute["median"] > 0.20 or absolute["max"] > 0.40):
        reasons.append("checkpoint error exceeds median 20 cm / maximum 40 cm")
    if any(p["spread_rig"] * scale > 0.15 for p in points):
        reasons.append("checkpoint was moving; pause upright for at least one second")
    # The same observations also exercise the actual runtime ankle projection path.
    runtime_errors = []
    runtime_projectors = {}
    for n, (R, C) in poses.items():
        S = np.asarray(anchored["transform"])
        proj = FloorProjector(S @ floor_homography(lenses[n], R, C), lenses[n].image_size, lenses[n],
                              S @ floor_homography(lenses[n], R, C, rig["body"]["ankle_z"]))
        proj.set_frame_size(*lenses[n].image_size)
        runtime_projectors[n] = proj
    runtime_invalid = 0
    for sample in evaluated:
        per_cam = []
        for n, view in sample["views"].items():
            p = [runtime_projectors[n].to_floor(*raw, plane="ankle") for raw in view["raw"][:2]]
            if any(xy is None for xy in p):
                runtime_invalid += 1
            else:
                per_cam.append(np.mean(p, axis=0))
        for i in range(len(per_cam)):
            for j in range(i):
                runtime_errors.append(float(np.linalg.norm(per_cam[i] - per_cam[j])))
    runtime_quality = statistics(runtime_errors)
    if (runtime_invalid or not runtime_quality or runtime_quality["median"] > 0.10
            or runtime_quality["p95"] > 0.25):
        reasons.append("runtime projection fails the camera agreement thresholds")
    baseline = _baseline(evaluated, cfg)
    return seal({"version": VERSION, "kind": "validation", "anchored_id": anchored["id"],
                 "session_id": session["id"], "independent_recording": independent,
                 "map": anchored["map"], "passed": not reasons, "reasons": reasons,
                 "counts": counts, "agreement_m": metric, "quality": quality,
                 "runtime_center_agreement_m": runtime_quality,
                 "runtime_invalid_projections": runtime_invalid,
                 "checkpoints": points, "absolute_error_m": absolute,
                 "absolute_accuracy_measured": bool(absolute),
                 "anchor": anchored.get("anchor"),
                 "baseline_center_agreement_m": baseline,
                 "thresholds_m": {"agreement_median": 0.10, "agreement_p95": 0.25,
                                  "absolute_median": 0.20, "absolute_max": 0.40}})


def _baseline(samples, cfg):
    projs = {}
    for c in cfg["cameras"]:
        try:
            lens, _ = load_lens(c)
            p = FloorProjector.load(c["homography_file"], lens)
            if lens is None:
                return {"unavailable": "current calibration has no lens"}
            p.set_frame_size(*lens.image_size)
            projs[c["name"]] = p
        except (OSError, ValueError, KeyError):
            return {"unavailable": "current calibration files could not be loaded"}
    distances = []
    for sample in samples:
        xy = []
        for n, v in sample["views"].items():
            if projs[n].ankle_H is not None:
                feet = [projs[n].to_floor(*p, plane="ankle") for p in v["raw"][:2]]
                q = np.mean(feet, axis=0) if all(p is not None for p in feet) else None
            else:
                q = projs[n].to_floor(*v["raw"][:2].mean(axis=0))
            if q is not None:
                xy.append(np.array(q))
        for i in range(len(xy)):
            for j in range(i):
                distances.append(np.linalg.norm(xy[i] - xy[j]))
    return statistics(distances)


def export_candidate(anchored, report, cfg, out):
    check_artifact(anchored, "anchored_rig")
    check_artifact(anchored["rig"], "rig")
    check_artifact(report, "validation")
    if report["anchored_id"] != anchored["id"] or not report["passed"]:
        raise ValueError("export requires a passing validation report for this anchored rig")
    if map_identity(cfg) != anchored["map"] or report["map"] != anchored["map"]:
        raise ValueError("map changed after validation")
    names = {c["name"] for c in cfg["cameras"]}
    if names != set(anchored["rig"]["cameras"]):
        raise ValueError("configuration camera set does not match the rig")
    for c in cfg["cameras"]:
        lens, _ = load_lens(c)
        if lens is None or lens.fingerprint() != anchored["rig"]["cameras"][c["name"]]["input_lens_fingerprint"]:
            raise ValueError(f"{c['name']}: configured input lens changed after recording")
    out = Path(out).resolve()
    if out.exists():
        raise FileExistsError(f"candidate directory already exists: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".autocalib-export-", dir=out.parent))
    try:
        candidate = deepcopy(cfg)
        poses, lenses = rig_geometry(anchored["rig"])
        S = np.array(anchored["transform"])
        for c in candidate["cameras"]:
            n = c["name"]
            lens = lenses[n]
            lens.path = str(out / f"{n}_intrinsics.json")
            c["intrinsics_file"] = lens.path
            c["homography_file"] = str(out / f"{n}_homography.npy")
            write_json(stage / f"{n}_intrinsics.json", lens.to_dict())
            R, C = poses[n]
            H = S @ floor_homography(lens, R, C)
            ankleH = S @ floor_homography(lens, R, C, anchored["rig"]["body"]["ankle_z"])
            path = stage / f"{n}_homography.npy"
            save_calibration(str(path), H, lens.image_size, lens=lens,
                             extra={"source": "autocalib", "rig_id": anchored["rig"]["id"],
                                    "anchored_id": anchored["id"], "map": anchored["map"],
                                    "ankle_homography": ankleH.tolist(),
                                    "ankle_height_m": anchored["rig"]["body"]["ankle_z"] * anchored["scale_m"]})
            p = FloorProjector.load(str(path), lens)
            p.set_frame_size(*lens.image_size)
        # Snapshot all paths used by track.py so the candidate runs from the repo as usual.
        for group, key in (("floorplan", "pointcloud_file"), ("floorplan", "reference_points_file"),
                           ("detector", "model"), ("detector", "tracker")):
            if candidate.get(group, {}).get(key):
                candidate[group][key] = str(Path(candidate[group][key]).resolve())
        for c in candidate["cameras"]:
            source = str(c["source"])
            if not source.startswith(("rtsp://", "usb:")):
                c["source"] = str(Path(source).resolve())
        candidate.setdefault("output", {}).setdefault("mqtt", {})["enabled"] = False
        candidate["autocalibration"] = {"anchored_id": anchored["id"], "map": anchored["map"]}
        with open(stage / "config.yaml", "x") as f:
            yaml.safe_dump(candidate, f, sort_keys=False)
        write_json(stage / "rig.json", anchored)
        write_json(stage / "validation.json", report)
        if out.exists():
            raise FileExistsError(str(out))
        os.rename(stage, out)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return out / "config.yaml"
