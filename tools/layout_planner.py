"""Find the best positions for a whole set of cameras at once (all moved, or some
kept where they are), from the lab scan and the current calibrations.

    python tools/layout_planner.py                       # best 4 cameras, all free
    python tools/layout_planner.py --keep yi04 --keep yi05   # keep these, place the rest
    python tools/layout_planner.py --cameras 5
    python tools/layout_planner.py --variants --weight 4,8.5,8.3,10.4,4   # count this area 4x
    python tools/layout_planner.py --variants            # best layout for every number
                                                         # of cameras kept in place
    python tools/layout_planner.py --mount-lines data/mount_lines.json
                          # only place cameras on the lines drawn with tools/mount_editor.py
    python tools/layout_planner.py --runs 10 --restarts 60
                          # 10 independent searches (seeds 0-9); keep the best, report all

Floor spots are rated as in tools/coverage_planner.py (good / poor angle / one
camera / unseen, seated and standing by default). Every candidate camera (each
mounting spot along the walls x direction x tilt, plus each current camera exactly
where it is) is scored once; then layouts are searched: a greedy start, swapping
one camera at a time for the candidate that helps most until no swap helps,
repeated from --restarts randomised starts (and, with --runs, the whole search
repeated with --runs different seeds, keeping the best and reporting how they agree). Every camera must share at least
--min-overlap m2 of floor with another and all must link up (the calibration walk
needs it), and cameras stay --min-separation m apart.

--variants tries every choice of which current cameras to keep, for each number
kept, and reports the best of each, so moving cameras can be weighed against the
work it takes (new mounts and a calibration walk for every moved camera).

Writes to --out: layout_keep_<names>.png per layout and layout_plan.json; with
--focus the maps still cover the whole room, with the scored areas outlined. The scan
is a snapshot; check spots with the live camera view before mounting.
"""
import argparse
import itertools
import json
import os
import string
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from coverage_planner import current_cameras, describe, draw_map, far_side   # noqa: E402
from tracker3d import coverage as cov                                       # noqa: E402
from tracker3d.config import load_config                                    # noqa: E402
from tracker3d.plan import load_plan, load_scan_points                      # noqa: E402


def layout_report(plan, search, planner, cams, keep, now, score):
    """Print one layout and return its JSON record and map labels."""
    status, count = search.evaluate(cams)
    after = planner.summaries(status, count)
    kept = ", ".join(keep) if keep else "none"
    print(f"\nLayout keeping {kept}: score {score:.1f}")
    for p in planner.postures:
        print(f"  {p}: {describe(after[p], now[p])}")
    letters = iter(string.ascii_uppercase)
    record, labels = [], []
    for c in cams:
        C = c["C"]
        if c["name"]:
            label = c["name"]
            how = "kept" if c["name"] in keep else "can stay where it is"
            print(f"  {label}: {how} at x={C[0]:.2f}, y={C[1]:.2f}")
        else:
            label = next(letters)
            far = far_side(plan, C, c["yaw"]).round(2) + 0.0
            aim = cov.aim_point(C, c["yaw"], c["tilt"])
            print(f"  Camera {label}: mount at x={C[0]:.2f}, y={C[1]:.2f}, height {C[2]:.2f} m, "
                  f"face towards x={far[0]:.2f}, y={far[1]:.2f}, tilted {c['tilt']:.0f} deg down")
            if aim is not None and plan.in_valid_area(*aim):
                print(f"    (the centre of the picture then lands on floor point "
                      f"x={aim[0]:.2f}, y={aim[1]:.2f})")
        labels.append((c["camera"], label))
        record.append({
            "label": label, "current_camera": c["name"], "kept": bool(c["name"] in keep),
            "position_m": [round(float(v), 2) for v in C], "facing_deg": round(c["yaw"], 1),
            "tilt_down_deg": round(c["tilt"], 1),
            "face_towards_m": [float(v) for v in far_side(plan, C, c["yaw"]).round(2) + 0.0]})
    return {"keep": list(keep), "score": round(score, 2), "after": after,
            "good_gain_m2": {p: round(after[p]["good_m2"] - now[p]["good_m2"], 1) for p in after},
            "cameras": record}, labels, status


def zone_summaries(planner, status, count, zones):
    """Coverage of the whole scored floor and of each rectangle (x1, y1, x2, y2[, w]),
    per posture, in m2."""
    st, ct = planner.split(status), planner.split(count)
    xy = planner.grid.xy
    out = {}
    for rect in [None] + list(zones):
        if rect is None:
            name, inside = "whole room", np.ones(len(xy), bool)
        else:
            x1, y1, x2, y2 = rect[:4]
            name = f"{x1:g},{y1:g},{x2:g},{y2:g}"
            inside = ((xy[:, 0] >= min(x1, x2)) & (xy[:, 0] <= max(x1, x2)) &
                      (xy[:, 1] >= min(y1, y2)) & (xy[:, 1] <= max(y1, y2)))
        out[name] = {p: cov.summary(st[p][inside], ct[p][inside], planner.grid.cell_area)
                     for p in planner.postures}
    return out


def layout_distance(a, b):
    """Largest distance (m, seen from above) between matched cameras of two layouts,
    matching them in the way that keeps the total distance smallest."""
    pa = [np.asarray(c["C"][:2], float) for c in a]
    pb = [np.asarray(c["C"][:2], float) for c in b]
    best = None
    for order in itertools.permutations(range(len(pb))):
        d = [float(np.linalg.norm(pa[i] - pb[j])) for i, j in enumerate(order)]
        if best is None or sum(d) < sum(best):
            best = d
    return max(best) if best else 0.0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--out", default="coverage_layout", help="output folder")
    ap.add_argument("--cameras", type=int, default=4, help="total number of cameras")
    ap.add_argument("--keep", action="append", default=[],
                    help="current camera to keep where it is (repeatable)")
    ap.add_argument("--variants", action="store_true",
                    help="best layout for every number of current cameras kept")
    ap.add_argument("--height", type=float, default=2.5, help="mounting height (m)")
    ap.add_argument("--posture", choices=sorted(cov.POSTURES) + ["both"], default="both")
    ap.add_argument("--cell", type=float, default=0.25, help="floor grid size (m)")
    ap.add_argument("--min-angle", type=float, default=25.0,
                    help="smallest crossing angle of two cameras' lines of sight (deg)")
    ap.add_argument("--max-range", type=float, default=9.0, help="detection range (m)")
    ap.add_argument("--min-tilt", type=float, default=15.0, help="lowest tilt below level (deg)")
    ap.add_argument("--min-overlap", type=float, default=2.0,
                    help="floor each camera must share with another (m2)")
    ap.add_argument("--min-separation", type=float, default=1.0, help="metres between cameras")
    ap.add_argument("--restarts", type=int, default=30, help="randomised search starts")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--runs", type=int, default=1,
                    help="repeat the search with seeds --seed, --seed+1, ...; keep the best")
    ap.add_argument("--focus", action="append",
                    type=lambda t: [float(v) for v in t.split(",")],
                    help="only score floor inside x1,y1,x2,y2 (repeatable)")
    ap.add_argument("--weight", action="append",
                    type=lambda t: [float(v) for v in t.split(",")],
                    help="count floor inside x1,y1,x2,y2 w times when ranking (repeatable)")
    ap.add_argument("--mount-lines", help="only place cameras on these lines, each at its own "
                    "height (the file tools/mount_editor.py saves); replaces --height")
    ap.add_argument("--include-furniture", action="store_true")
    ap.add_argument("--image-format", choices=("png", "jpg"), default="png")
    args = ap.parse_args()

    t0 = time.time()
    cfg = load_config(args.config)
    plan = load_plan(cfg)
    print("[layout] reading the lab scan ...")
    points = load_scan_points(cfg)
    occ = cov.Occupancy(points)
    furniture = None if args.include_furniture else cov.furniture_mask(points, plan)
    if any(len(w) != 5 for w in args.weight or []):
        raise SystemExit("--weight takes x1,y1,x2,y2,w")
    grid = cov.FloorGrid(plan, args.cell, furniture=furniture, focus=args.focus, weights=args.weight)
    cameras, skipped = current_cameras(cfg)
    if skipped:
        print(f"[layout] skipped (no lens or floor calibration): {', '.join(skipped)}")
    if not cameras:
        raise SystemExit("no calibrated cameras to start from")
    names = [c.name for c in cameras]
    unknown = [k for k in args.keep if k not in names]
    if unknown:
        raise SystemExit(f"unknown camera(s) in --keep: {', '.join(unknown)} (have {', '.join(names)})")

    today = cov.Planner(cameras, grid, occ, args.posture, args.min_angle, args.max_range)
    now = today.summaries()
    # Maps always show the whole room; with --focus the scored areas are outlined.
    whole_grid = grid if not args.focus else cov.FloorGrid(plan, args.cell, furniture=furniture)
    empty = cov.Planner([], grid, occ, args.posture, args.min_angle, args.max_range)
    if args.runs < 1:
        raise SystemExit("--runs must be at least 1")
    if args.runs > 1 and args.variants:
        raise SystemExit("--runs cannot be combined with --variants")
    lines, yaws, marked = None, None, None
    candidates = cameras
    if args.mount_lines:
        if args.variants:
            raise SystemExit("--mount-lines cannot be combined with --variants; use --keep")
        lines = cov.load_mount_lines(args.mount_lines)
        marked, dropped = cov.spots_on_lines(plan, occ, lines)
        for d in dropped:
            print(f"[layout] line {d['line'] + 1}: the scan blocks the view in front of "
                  f"x={d['xy'][0]:.2f}, y={d['xy'][1]:.2f}; spot skipped")
        if not marked:
            raise SystemExit(f"no usable mounting spots on the lines in {args.mount_lines}")
        spots = [[s["xy"][0], s["xy"][1], s["height"]] for s in marked]
        yaws = [s["yaw"] for s in marked]
        # Only the marked places count, so a current camera stays only if it is kept.
        candidates = [c for c in cameras if c.name in args.keep]
    else:
        spots = cov.mount_spots(plan, occ, args.height)
    print(f"[layout] scoring candidates at {len(spots)} mounting spots plus "
          f"{len(candidates)} current cameras ...")
    search = cov.LayoutSearch(empty, cameras[0].lens, spots, args.height,
                              mirrored=np.linalg.det(cameras[0].R) < 0, current=candidates,
                              min_separation_m=args.min_separation, min_overlap_m2=args.min_overlap,
                              min_tilt_deg=args.min_tilt, yaws=yaws)
    print(f"  {len(search.cands)} candidate cameras ({time.time() - t0:.0f} s)")
    for p in today.postures:
        print(f"  today, {p}: {describe(now[p])}")

    if args.variants:
        keeps = []
        for k in range(min(len(cameras), args.cameras), -1, -1):
            best = None
            for keep in itertools.combinations(names, k):
                cams, score = search.run(args.cameras, keep, args.restarts, args.seed)
                if len(cams) == args.cameras and (best is None or score > best[1] + 1e-9):
                    best = (cams, score, list(keep))
            if best:
                keeps.append(best)
    else:
        runs = []
        for k in range(args.runs):
            seed = args.seed + k
            cams_k, score_k = search.run(args.cameras, args.keep, args.restarts, seed)
            ok = len(cams_k) == args.cameras and np.isfinite(score_k)
            runs.append((seed, cams_k if ok else None, score_k if ok else None))
            if args.runs > 1:
                print(f"  run {k + 1}/{args.runs} (seed {seed}): "
                      + (f"score {score_k:.1f}" if ok else "no valid layout")
                      + f" ({time.time() - t0:.0f} s)", flush=True)
        valid = [r for r in runs if r[1] is not None]
        if not valid:
            raise SystemExit("no layout meets the overlap and separation rules; "
                             "try a smaller --min-overlap or --min-separation")
        best_seed, cams, score = max(valid, key=lambda r: r[2])
        keeps = [(cams, score, list(args.keep))]

    os.makedirs(args.out, exist_ok=True)
    report = {"postures": today.postures, "cameras_total": args.cameras,
              "mount_height_m": None if lines else args.height, "mount_lines": lines,
              "min_angle_deg": args.min_angle,
              "max_range_m": args.max_range, "min_tilt_deg": args.min_tilt,
              "min_overlap_m2": args.min_overlap, "min_separation_m": args.min_separation,
              "focus": args.focus, "weights": args.weight, "restarts": args.restarts,
              "today": now, "layouts": []}
    both_note = "" if len(today.postures) == 1 else " (each spot: worse of seated / standing)"
    for cams, score, keep in keeps:
        record, labels, status = layout_report(plan, search, today, cams, keep, now, score)
        if marked is not None:
            for row, c in zip(record["cameras"], cams):
                if c["pos"] < len(marked):
                    row["mount_line"] = marked[c["pos"]]["line"] + 1
        record["zones"] = zone_summaries(today, *search.evaluate(cams), args.weight or [])
        report["layouts"].append(record)
        in_layout = {c["name"] for c in cams if c["name"]}
        stay = [cam for cam, label in labels if label in names]
        moved = [(cam, label) for cam, label in labels if label not in names]
        name = "layout_keep_" + ("+".join(keep) if keep else "none")
        title = f"{args.cameras} cameras, keeping {', '.join(keep) or 'none'}{both_note}"
        if args.focus:
            whole = cov.Planner([c["camera"] for c in cams], whole_grid, occ, args.posture,
                                args.min_angle, args.max_range)
            drawn = whole.worst()
        else:
            drawn = today.worst(status)
        img = draw_map(plan, whole_grid, drawn, stay, title, new=moved,
                       faded=[c for c in cameras if c.name not in in_layout],
                       outline=list(args.focus or []) + list(args.weight or []))
        cv2.imwrite(os.path.join(args.out, f"{name}.{args.image_format}"), img,
                    [cv2.IMWRITE_JPEG_QUALITY, 88] if args.image_format == "jpg" else [])
    if not args.variants and args.runs > 1:
        scores = [r[2] for r in runs if r[1] is not None]
        report["runs"] = []
        for seed, cams_k, score_k in runs:
            row = {"seed": seed, "score": None if cams_k is None else round(score_k, 2)}
            if cams_k is not None:
                row["cameras"] = [{"position_m": [round(float(v), 2) for v in c["C"]],
                                   "facing_deg": round(c["yaw"], 1), "tilt_down_deg": round(c["tilt"], 1)}
                                  for c in cams_k]
                row["summaries"] = today.summaries(*search.evaluate(cams_k))
                row["largest_move_from_best_m"] = round(layout_distance(cams_k, keeps[0][0]), 2)
            report["runs"].append(row)
        report["best_seed"] = best_seed
        report["agreement"] = {
            "runs": len(runs), "valid_runs": len(scores),
            "within_half_percent_of_best": int(sum(s >= 0.995 * max(scores) for s in scores)),
            "score_min": round(min(scores), 2), "score_max": round(max(scores), 2)}
        a = report["agreement"]
        print(f"\n[layout] best of {a['runs']} runs: seed {best_seed}; "
              f"{a['within_half_percent_of_best']} runs within 0.5% of the best score "
              f"(scores {a['score_min']} to {a['score_max']})")
    with open(os.path.join(args.out, "layout_plan.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[layout] maps and layout_plan.json in {args.out}/ ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
