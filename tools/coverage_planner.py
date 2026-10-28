"""Find the best place to mount one more camera, from the lab scan and the current
camera calibrations.

    python tools/coverage_planner.py                       # seated and standing, 2.5 m mount
    python tools/coverage_planner.py --height 2.4 --top 8
    python tools/coverage_planner.py --posture seated      # desks only (or: standing)
    python tools/coverage_planner.py --focus 0,4,8,10      # only score this rectangle
    python tools/coverage_planner.py --weight 4,8.5,8.3,10.4,4   # count this area 4x
    python tools/coverage_planner.py --spot 7.9,5.2 --spot 4,10.2   # only these spots

Every floor spot (25 cm grid) is rated by how the cameras see a person there,
seated and standing (--posture both, the default, ranks by the average of the two;
the maps show each spot's worse rating):
  good    two cameras see them from directions at least --min-angle apart, so the
          tracker measures their height and posture and keeps neighbours apart
  poor    two or more cameras, but from nearly the same direction
  single  one camera only (position from an assumed height, posture guessed)
  unseen  no camera
"Sees" means head and shoulders are in the picture, within --max-range, and the
lines of sight are not blocked by anything in the scan (walls, shelves, monitors).
Desks and cabinets (found in the scan) are left out, since nobody stands there.

Mounting spots are tried every 0.5 m along walls, pillars and tall cabinets at
--height, aimed into the room at several directions and tilts; the best are refined
and ranked by the floor area that becomes "good". Writes to --out:
  coverage_current.png      today's coverage
  coverage_candidates.png   the ranked mounting spots on today's coverage
  coverage_option_N.png     coverage with option N added
  coverage_plan.json        all numbers
With --focus or --weight the maps still cover the whole room, with those areas
outlined. The m2 figures are plain floor area; weights only change the ranking.
(--image-format jpg for smaller images). The saved lab results are in docs/coverage/.
Map coordinates: meters from the scan's bottom-left corner, x to the right, y up,
as on the tracker's map. The scan is a snapshot: furniture moved since then, or
glass the scanner saw through, is not known. Check a spot with the live camera view
before drilling.
"""
import argparse
import json
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tracker3d import coverage as cov                     # noqa: E402
from tracker3d.config import list_cameras, load_config    # noqa: E402
from tracker3d.geometry import FloorProjector             # noqa: E402
from tracker3d.lens import load_lens                      # noqa: E402
from tracker3d.plan import load_plan, load_scan_points    # noqa: E402

COLOURS = {cov.UNSEEN: (60, 60, 215), cov.SINGLE: (0, 145, 255),
           cov.POOR: (0, 225, 255), cov.GOOD: (70, 185, 60)}         # BGR
LABELS = {cov.UNSEEN: "unseen", cov.SINGLE: "one camera",
          cov.POOR: "2+ cameras, poor angle", cov.GOOD: "good (2 cameras, good angle)"}
SCALE = 2                                  # map pixels per plan pixel in the images
NEW = (220, 60, 220)
OUTLINE = (255, 200, 0)                    # BGR: light blue


def floats(text, n):
    vals = [float(v) for v in text.split(",")]
    if len(vals) not in n:
        raise argparse.ArgumentTypeError(f"expected {' or '.join(map(str, n))} numbers: {text}")
    return vals


def current_cameras(cfg):
    cams, skipped = [], []
    for cam in list_cameras(cfg):
        lens, _ = load_lens(cam)
        if lens is None or not os.path.exists(cam["homography_file"]):
            skipped.append(cam["name"])
            continue
        proj = FloorProjector.load(cam["homography_file"], lens)
        proj.set_frame_size(*lens.image_size)
        M, _, C = proj.pose()
        cams.append(cov.Camera(cam["name"], M, C, lens))
    return cams, skipped


# ---- drawing ------------------------------------------------------------------
def to_px(plan, x, y):
    px, py = plan.world_to_px(x, y)
    return int(px * SCALE), int(py * SCALE)


def draw_map(plan, grid, status, cameras, title, new=None, markers=(), faded=(), outline=()):
    """Coverage map. `cameras` are drawn white; `new` (a camera, or a list of
    (camera, label)) magenta; `faded` cameras as an outlined dot only (e.g. where a
    camera that moves is today); `markers` are numbered (position, facing) pairs;
    `outline` rectangles (x1, y1, x2, y2[, weight]) mark the areas a --focus
    ranking scored or a --weight ranking weighted (labelled "x<weight>")."""
    bg = cv2.resize(plan.background(), None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_NEAREST)
    overlay = bg.copy()
    half = grid.cell / 2
    for (x, y), s in zip(grid.xy, status):
        cv2.rectangle(overlay, to_px(plan, x - half, y + half), to_px(plan, x + half, y - half),
                      COLOURS[int(s)], -1)
    for x, y in grid.furniture_xy:
        cv2.rectangle(overlay, to_px(plan, x - half, y + half), to_px(plan, x + half, y - half),
                      (90, 90, 90), -1)
    img = cv2.addWeighted(overlay, 0.55, bg, 0.45, 0)
    for rect in outline:
        x1, y1, x2, y2 = rect[:4]
        corner = to_px(plan, min(x1, x2), max(y1, y2))
        cv2.rectangle(img, corner, to_px(plan, max(x1, x2), min(y1, y2)), OUTLINE, 3)
        if len(rect) > 4:
            label = f"x{rect[4]:g}"
            cv2.putText(img, label, (corner[0] + 6, corner[1] + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 0, 0), 4)
            cv2.putText(img, label, (corner[0] + 6, corner[1] + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        OUTLINE, 2)
    for m in range(0, int(plan.width_m) + 1):                      # 1 m grid
        cv2.line(img, to_px(plan, m, 0), to_px(plan, m, plan.height_m), (200, 200, 200), 1)
        cv2.putText(img, str(m), (to_px(plan, m, 0)[0] - 4, to_px(plan, m, 0)[1] + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (230, 230, 230), 1)
    for m in range(0, int(plan.height_m) + 1):
        cv2.line(img, to_px(plan, 0, m), to_px(plan, plan.width_m, m), (200, 200, 200), 1)
        cv2.putText(img, str(m), (to_px(plan, 0, m)[0] - 22, to_px(plan, 0, m)[1] + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (230, 230, 230), 1)
    for c in faded:
        p = to_px(plan, *c.C[:2])
        cv2.circle(img, p, 7, (255, 255, 255), 2)
        cv2.putText(img, f"{c.name} today", (p[0] + 9, p[1] + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (0, 0, 0), 3)
        cv2.putText(img, f"{c.name} today", (p[0] + 9, p[1] + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255, 255, 255), 1)
    for c in cameras:
        draw_camera(img, plan, c, (255, 255, 255), c.name)
    if new is not None:
        for cam, label in (new if isinstance(new, list) else [(new, "NEW")]):
            draw_camera(img, plan, cam, NEW, label)
    for k, (C, yaw) in enumerate(markers, 1):
        p = to_px(plan, *C[:2])
        end = to_px(plan, C[0] + 0.8 * np.cos(np.radians(yaw)), C[1] + 0.8 * np.sin(np.radians(yaw)))
        cv2.arrowedLine(img, p, end, NEW, 2, tipLength=0.3)
        cv2.circle(img, p, 11, NEW, -1)
        cv2.putText(img, str(k), (p[0] - 5, p[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    return add_legend(img, title, outlined=bool(outline))


def draw_camera(img, plan, cam, colour, label):
    """Dot plus the floor wedge between the picture's left and right edges."""
    fwd = cam.R[2]
    yaw = np.degrees(np.arctan2(fwd[1], fwd[0]))
    w, h = cam.lens.image_size
    rect = cam.lens.undistort_points([(0, h / 2), (w - 1, h / 2)])
    rays = np.linalg.inv(cam.lens.K_rect) @ np.column_stack([rect, np.ones(2)]).T
    p0 = to_px(plan, *cam.C[:2])
    for ray in rays.T:
        d = cam.R.T @ ray
        d2 = d[:2] / max(np.linalg.norm(d[:2]), 1e-9)
        end = cam.C[:2] + 3.0 * d2
        cv2.line(img, p0, to_px(plan, *end), colour, 1, cv2.LINE_AA)
    tip = cam.C[:2] + 0.7 * np.array([np.cos(np.radians(yaw)), np.sin(np.radians(yaw))])
    cv2.arrowedLine(img, p0, to_px(plan, *tip), colour, 2, tipLength=0.3)
    cv2.circle(img, p0, 7, colour, -1)
    cv2.circle(img, p0, 7, (0, 0, 0), 1)
    cv2.putText(img, label, (p0[0] + 9, p0[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
    cv2.putText(img, label, (p0[0] + 9, p0[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 1)


def add_legend(img, title, outlined=False):
    rows = [(LABELS[s], COLOURS[s]) for s in (cov.GOOD, cov.POOR, cov.SINGLE, cov.UNSEEN)]
    rows.append(("desk / cabinet (not scored)", (90, 90, 90)))
    if outlined:
        rows.append(("outlined: areas the ranking scored only, or weighted (x weight)", OUTLINE))
    panel = np.full((40 + 22 * len(rows), img.shape[1], 3), 245, np.uint8)
    cv2.putText(panel, title, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 1, cv2.LINE_AA)
    for k, (text, colour) in enumerate(rows):
        y = 44 + 22 * k
        cv2.rectangle(panel, (10, y - 12), (28, y + 4), colour, -1)
        cv2.putText(panel, text, (36, y + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 20, 20), 1, cv2.LINE_AA)
    return np.vstack([img, panel])


# ---- main ---------------------------------------------------------------------
def far_side(plan, C, yaw_deg):
    """Where the camera's facing line (seen from above) leaves the map rectangle."""
    d = np.array([np.cos(np.radians(yaw_deg)), np.sin(np.radians(yaw_deg))])
    ts = []
    for k, size in ((0, plan.width_m), (1, plan.height_m)):
        if abs(d[k]) > 1e-9:
            ts += [(edge - C[k]) / d[k] for edge in (0.0, size)]
    t = min(v for v in ts if v > 1e-6)
    return C[:2] + t * d


def describe(s, before=None):
    def part(key, label):
        if before is None:
            return f"{label} {s[key]:.1f} m2"
        return f"{label} {before[key]:.1f} -> {s[key]:.1f} m2"
    return ", ".join([part("good_m2", "good"), part("poor_angle_m2", "poor angle"),
                      part("single_m2", "one camera"), part("unseen_m2", "unseen")])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--out", default="coverage_plan", help="output folder")
    ap.add_argument("--height", type=float, default=2.5, help="mounting height (m)")
    ap.add_argument("--posture", choices=sorted(cov.POSTURES) + ["both"], default="both",
                    help="score floor spots for a seated person, a standing one, or both")
    ap.add_argument("--cell", type=float, default=0.25, help="floor grid size (m)")
    ap.add_argument("--min-angle", type=float, default=25.0,
                    help="smallest crossing angle of two cameras' lines of sight (deg)")
    ap.add_argument("--max-range", type=float, default=9.0, help="detection range (m)")
    ap.add_argument("--top", type=int, default=5, help="number of options to report")
    ap.add_argument("--spot", action="append", type=lambda t: floats(t, (2, 3)),
                    help="only try this mounting point x,y[,z] (repeatable)")
    ap.add_argument("--focus", action="append", type=lambda t: floats(t, (4,)),
                    help="only score floor inside x1,y1,x2,y2 (repeatable)")
    ap.add_argument("--weight", action="append", type=lambda t: floats(t, (5,)),
                    help="count floor inside x1,y1,x2,y2 w times when ranking (repeatable; "
                         "overlaps take the largest weight, 1 elsewhere)")
    ap.add_argument("--include-furniture", action="store_true",
                    help="also score floor under desks and cabinets")
    ap.add_argument("--min-tilt", type=float, default=15.0,
                    help="lowest tilt below level (deg); flatter cameras lose people behind people")
    ap.add_argument("--image-format", choices=("png", "jpg"), default="png",
                    help="jpg makes much smaller files (e.g. for docs/)")
    ap.add_argument("--lens-from", default=None,
                    help="camera whose lens model the new camera uses (default: the first)")
    args = ap.parse_args()

    t0 = time.time()
    cfg = load_config(args.config)
    plan = load_plan(cfg)
    print("[coverage] reading the lab scan ...")
    points = load_scan_points(cfg)
    occ = cov.Occupancy(points)
    furniture = None if args.include_furniture else cov.furniture_mask(points, plan)
    grid = cov.FloorGrid(plan, args.cell, furniture=furniture, focus=args.focus, weights=args.weight)
    cameras, skipped = current_cameras(cfg)
    if skipped:
        print(f"[coverage] skipped (no lens or floor calibration): {', '.join(skipped)}")
    if not cameras:
        raise SystemExit("no calibrated cameras to start from")
    lens_cam = next((c for c in cameras if c.name == args.lens_from), cameras[0])
    mirrored = np.linalg.det(cameras[0].R) < 0

    planner = cov.Planner(cameras, grid, occ, args.posture, args.min_angle, args.max_range)
    # With --focus the ranking scores only those areas, but the maps still show the
    # whole room, with the scored areas outlined.
    whole = planner if not args.focus else cov.Planner(
        cameras, cov.FloorGrid(plan, args.cell, furniture=furniture), occ, args.posture,
        args.min_angle, args.max_range)
    outline = list(args.focus or []) + list(args.weight or [])
    who = " and ".join(f"{p} people" for p in planner.postures)
    now = planner.summaries()
    print(f"[coverage] {grid.cell_area * len(grid.xy):.1f} m2 of floor scored for {who}, "
          f"cameras {', '.join(c.name for c in cameras)}")
    for p in planner.postures:
        print(f"  today, {p}: {describe(now[p])}")
    for c, seen in zip(cameras, planner.seen):
        print(f"    {c.name} at ({c.C[0]:.2f}, {c.C[1]:.2f}, {c.C[2]:.2f}) sees "
              f"{seen.sum() * planner.cell_area:.1f} m2")

    if args.spot:
        spots = np.array([s + [args.height] if len(s) == 2 else s for s in args.spot])
    else:
        spots = cov.mount_spots(plan, occ, args.height, avoid=[c.C for c in cameras])
    print(f"[coverage] trying {len(spots)} mounting spots at {args.height:.2f} m ...")

    def progress(n, total):
        if n % 10 == 0 or n == total:
            print(f"  {n}/{total}", end="\r" if n < total else "\n", flush=True)
    options = planner.search(lens_cam.lens, spots, args.height, top=args.top,
                             mirrored=mirrored, progress=progress, min_tilt_deg=args.min_tilt)

    os.makedirs(args.out, exist_ok=True)

    def save(name, img):
        path = os.path.join(args.out, f"{name}.{args.image_format}")
        cv2.imwrite(path, img, [cv2.IMWRITE_JPEG_QUALITY, 88] if args.image_format == "jpg" else [])

    one = planner.postures[0] if len(planner.postures) == 1 else None
    both_note = "(each spot: worse of seated / standing)"
    save("coverage_current", draw_map(plan, whole.grid, whole.worst(), cameras,
                                      f"Today: {describe(now[one])}" if one else f"Today {both_note}",
                                      outline=outline))
    report = {"postures": planner.postures, "mount_height_m": args.height,
              "min_angle_deg": args.min_angle, "max_range_m": args.max_range,
              "cell_m": args.cell, "focus": args.focus, "weights": args.weight,
              "cameras": {c.name: [round(v, 3) for v in c.C] for c in cameras},
              "today": now, "options": []}
    print(f"\nBest places for a new camera ({who}), best first:")
    for k, o in enumerate(options, 1):
        status, count, vis = planner.with_camera(o["camera"])
        after = planner.summaries(status, count)
        overlap = float(np.sum(vis & planner.seen.any(axis=0)) * planner.cell_area)
        aim = cov.aim_point(o["C"], o["yaw"], o["tilt"])
        C = o["C"]
        print(f"\n  Option {k}: mount at x={C[0]:.2f}, y={C[1]:.2f}, height {C[2]:.2f} m")
        far = far_side(plan, C, o["yaw"]).round(2) + 0.0      # no "-0.00"
        print(f"    face towards x={far[0]:.2f}, y={far[1]:.2f} on the far side of the room, "
              f"tilted {o['tilt']:.0f} deg down")
        if aim is not None and plan.in_valid_area(*aim):
            print(f"    (the centre of the picture then lands on floor point x={aim[0]:.2f}, y={aim[1]:.2f})")
        for p in planner.postures:
            print(f"    {p}: {describe(after[p], now[p])}")
        print(f"    new camera sees {vis.sum() * planner.cell_area:.1f} m2, "
              f"{overlap:.1f} m2 of it shared with today's cameras")
        if overlap < 2.0:
            print("    WARNING: little overlap with the other cameras; calibration needs some")
        report["options"].append({
            "rank": k, "position_m": [round(v, 2) for v in C], "facing_deg": round(o["yaw"], 1),
            "tilt_down_deg": round(o["tilt"], 1),
            "face_towards_m": [round(v, 2) for v in far],
            "aim_floor_point_m": None if aim is None else [round(v, 2) for v in aim],
            "after": after,
            "good_gain_m2": {p: round(after[p]["good_m2"] - now[p]["good_m2"], 1) for p in after},
            "new_camera_sees_m2": round(float(vis.sum() * planner.cell_area), 1),
            "overlap_with_current_m2": round(overlap, 1)})
        title = f"Option {k}: {describe(after[one], now[one])}" if one else f"Option {k} added {both_note}"
        drawn = status if whole is planner else whole.with_camera(o["camera"])[0]
        save(f"coverage_option_{k}", draw_map(plan, whole.grid, whole.worst(drawn), cameras, title,
                                              new=o["camera"], outline=outline))
    save("coverage_candidates", draw_map(plan, whole.grid, whole.worst(), cameras,
                                         "Numbered: best mounting spots (arrow = facing)",
                                         markers=[(o["C"], o["yaw"]) for o in options],
                                         outline=outline))
    with open(os.path.join(args.out, "coverage_plan.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[coverage] images and coverage_plan.json in {args.out}/ ({time.time() - t0:.0f} s)")

if __name__ == "__main__":
    main()
