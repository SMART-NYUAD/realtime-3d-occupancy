"""Replay a recorded session through the tracking (localization + fusion) offline,
so one recording can be compared with the fixes on and off.

Record any scene (people seated, walking, ...) with the calibration recorder; it
stores each camera's pose keypoints and tracker IDs, never video:
    python autocalib.py record --out calib/walks/office01 --seconds 600
Then replay it:
    python tools/replay_tracking.py calib/walks/office01 --out new.csv
    python tools/replay_tracking.py calib/walks/office01 --legacy --out old.csv
    python tools/tracking_metrics.py new.csv --expected 2
    python tools/tracking_metrics.py old.csv --expected 2 --coast 1.5   # old: unseen people stayed 1.5 s

--legacy reproduces the tracking before the October 2026 fixes (old foot guesses,
distance-only fusion, 1.6 m merge, no seated hold). --set fusion.merge_distance_m=0.8
overrides single settings. The replay uses the calibration in --config, so record
and replay with the same calibration for a fair comparison.
"""
import argparse
import csv
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tracker3d.config import list_cameras, load_config   # noqa: E402
from tracker3d.fusion import Fusion                       # noqa: E402
from tracker3d.geometry import FloorProjector             # noqa: E402
from tracker3d.lens import load_lens                      # noqa: E402
from tracker3d.localize import Localizer                  # noqa: E402
from tracker3d.plan import load_plan                      # noqa: E402

LEGACY = {
    "localize.torso_estimates": False,
    "fusion.blob_only_fallback": False,
    "fusion.merge_distance_m": 1.6, "fusion.blob_merge_gate_m": 3.0, "fusion.dup_suppress_m": 1.0,
    "fusion.identity_matching": False, "fusion.triangulate": False,
    "fusion.seated_persistence": False, "fusion.settled_smoothing": False, "fusion.confirm_s": 0,
    "fusion.dup_merge_m": 0, "fusion.ray_merge_m": 0,
}


def apply_overrides(cfg, overrides):
    for key, value in overrides.items():
        section, name = key.split(".", 1)
        cfg.setdefault(section, {})[name] = value
    return cfg


def parse_set(items):
    out = {}
    for item in items or []:
        key, _, raw = item.partition("=")
        out[key] = json.loads(raw) if raw not in ("true", "false") else raw == "true"
    return out


def projectors(cfg):
    projs = {}
    for cam in list_cameras(cfg):
        lens, _ = load_lens(cam)
        proj = FloorProjector.load(cam["homography_file"], lens)
        if lens is not None:
            proj.set_frame_size(*lens.image_size)
        projs[cam["name"]] = proj
    return projs


def load_frames(session_dir):
    """{camera: [(t, image_size, tracked people)]} sorted by capture time."""
    per_cam = defaultdict(list)
    with open(os.path.join(session_dir, "observations.jsonl")) as f:
        for line in f:
            try:
                r = json.loads(line)
            except ValueError:
                break                              # a crash-truncated last line
            if r.get("kind") != "frame":
                continue
            people = []
            for p in r["people"]:
                if p.get("id") is None:            # the live tracker uses tracked people only
                    continue
                q = {"bbox": tuple(p["bbox"]), "conf": p.get("conf", 1.0), "id": p["id"]}
                if p.get("kxy") is not None:
                    q["kxy"], q["kconf"] = np.asarray(p["kxy"]), np.asarray(p["kconf"])
                people.append(q)
            per_cam[r["camera"]].append((r["t"], tuple(r["image_size"]), people))
    for v in per_cam.values():
        v.sort(key=lambda x: x[0])
    return per_cam


def replay(session_dir, cfg, out_path, step_s=0.05, tol_s=0.075):
    plan = load_plan(cfg)
    projs = projectors(cfg)
    loc = Localizer(cfg, plan)
    fus = Fusion.from_config(cfg.get("fusion", {}))
    coast = float(cfg.get("fusion", {}).get("publish_coast_s", 0.5))
    frames = load_frames(session_dir)
    if not frames:
        raise SystemExit(f"{session_dir}: no camera frames")
    t0 = min(v[0][0] for v in frames.values())
    t1 = max(v[-1][0] for v in frames.values())
    index = {c: 0 for c in frames}
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "kind", "cam", "id", "x", "y", "foot_source", "posture", "settled",
                    "sigma_m", "last_seen_s"])
        steps = 0
        for target in np.arange(t0 + tol_s, t1, step_s):
            dets = []
            hints = fus.posture_hints()
            for cam, rows in frames.items():
                i = index[cam]
                while i + 1 < len(rows) and rows[i + 1][0] <= target:
                    i += 1
                index[cam] = i
                t, size, people = rows[i]
                if t > target or target - t > tol_s:
                    continue
                people = [dict(p) for p in people]
                loc.localize(cam, people, projs.get(cam), size, hints=hints)
                dets += people
            tracks = fus.update(dets, now=float(target))
            for d in dets:
                if d.get("world") is not None:
                    w.writerow([f"{target:.3f}", "det", d["cam"], d["id"], f"{d['world'][0]:.3f}",
                                f"{d['world'][1]:.3f}", d.get("foot_source", ""),
                                d.get("posture_2d", ""), "", f"{d.get('foot_sigma_m') or 0:.3f}", ""])
            for g in tracks:
                w.writerow([f"{target:.3f}", "track", "+".join(g["cams"]), g["gid"],
                            f"{g['world'][0]:.3f}", f"{g['world'][1]:.3f}",
                            "coasting" if g["coasting"] else "", g["posture"], int(g["settled"]),
                            f"{g['sigma']:.3f}", f"{g['last_seen_s']:.2f}"])
            steps += 1
    print(f"replayed {steps} steps ({t1 - t0:.0f} s) from {session_dir} -> {out_path}; "
          f"publish window {coast:g} s")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("session", help="session folder recorded with autocalib.py record")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--out", required=True, help="CSV to write (same format as track.py --log-csv)")
    ap.add_argument("--legacy", action="store_true", help="tracking as before the October 2026 fixes")
    ap.add_argument("--set", nargs="*", help="overrides like fusion.merge_distance_m=0.8")
    ap.add_argument("--step", type=float, default=0.05, help="replay step (s)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.legacy:
        apply_overrides(cfg, LEGACY)
    apply_overrides(cfg, parse_set(args.set))
    replay(args.session, cfg, args.out, args.step)


if __name__ == "__main__":
    main()
