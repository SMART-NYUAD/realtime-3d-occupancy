"""Find the best positions for a whole set of cameras at once (all moved, or some
kept where they are), from the lab scan and the current calibrations.

    python tools/layout_planner.py                       # best 4 cameras, all free
    python tools/layout_planner.py --keep yi04 --keep yi05   # keep these, place the rest
    python tools/layout_planner.py --cameras 5
    python tools/layout_planner.py --variants --weight 4,8.5,8.3,10.4,4   # count this area 4x
    python tools/layout_planner.py --variants            # best layout for every number
                                                         # of cameras kept in place
    python tools/layout_planner.py --no-mount 0,3,1.6,6.2 --add-spot 1.35,4.45,2.0
                          # no cameras on that stretch of wall; try a cabinet top 2 m high

Floor spots are rated as in tools/coverage_planner.py (good / poor angle / one
camera / unseen, seated and standing by default). Every candidate camera (each
mounting spot along the walls x direction x tilt, plus each current camera exactly
where it is) is scored once; then layouts are searched: a greedy start, swapping
one camera at a time for the candidate that helps most until no swap helps,
repeated from --restarts randomised starts. Every camera must share at least
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
    ap.add_argument("--focus", action="append",
                    type=lambda t: [float(v) for v in t.split(",")],
                    help="only score floor inside x1,y1,x2,y2 (repeatable)")
    ap.add_argument("--weight", action="append",
                    type=lambda t: [float(v) for v in t.split(",")],
                    help="count floor inside x1,y1,x2,y2 w times when ranking (repeatable)")
    ap.add_argument("--no-mount", action="append", default=[],
                    type=lambda t: [float(v) for v in t.split(",")],
                    help="no mounting spots inside x1,y1,x2,y2, e.g. a pillar with no room "
                         "for a camera (repeatable)")
    ap.add_argument("--add-spot", action="append", default=[],
                    type=lambda t: [float(v) for v in t.split(",")],
                    help="extra mounting spot x,y or x,y,height, e.g. on top of a cabinet "
                         "(repeatable; height defaults to --height)")
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
    if any(len(r) != 4 for r in args.no_mount):
        raise SystemExit("--no-mount takes x1,y1,x2,y2")
    if any(len(s) not in (2, 3) for s in args.add_spot):
        raise SystemExit("--add-spot takes x,y or x,y,height")
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
    spots = [[float(x), float(y), args.height] for x, y in cov.mount_spots(plan, occ, args.height)
             if not any(x1 <= x <= x2 and y1 <= y <= y2 for x1, y1, x2, y2 in args.no_mount)]
    spots += [s if len(s) == 3 else s + [args.height] for s in args.add_spot]
    print(f"[layout] scoring candidates at {len(spots)} mounting spots plus today's "
          f"{len(cameras)} cameras ...")
    search = cov.LayoutSearch(empty, cameras[0].lens, spots, args.height,
                              mirrored=np.linalg.det(cameras[0].R) < 0, current=cameras,
                              min_separation_m=args.min_separation, min_overlap_m2=args.min_overlap,
                              min_tilt_deg=args.min_tilt)
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
        cams, score = search.run(args.cameras, args.keep, args.restarts, args.seed)
        if len(cams) < args.cameras:
            raise SystemExit("no layout meets the overlap and separation rules; "
                             "try a smaller --min-overlap or --min-separation")
        keeps = [(cams, score, list(args.keep))]

    os.makedirs(args.out, exist_ok=True)
    report = {"postures": today.postures, "cameras_total": args.cameras,
              "mount_height_m": args.height, "min_angle_deg": args.min_angle,
              "max_range_m": args.max_range, "min_tilt_deg": args.min_tilt,
              "min_overlap_m2": args.min_overlap, "min_separation_m": args.min_separation,
              "focus": args.focus, "weights": args.weight,
              "no_mount": args.no_mount, "added_spots": args.add_spot,
              "today": now, "layouts": []}
    both_note = "" if len(today.postures) == 1 else " (each spot: worse of seated / standing)"
    for cams, score, keep in keeps:
        record, labels, status = layout_report(plan, search, today, cams, keep, now, score)
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
    with open(os.path.join(args.out, "layout_plan.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[layout] maps and layout_plan.json in {args.out}/ ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
