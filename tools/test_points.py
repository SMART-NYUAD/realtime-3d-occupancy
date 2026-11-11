"""Rate fixed test points under planned layouts, for the validation study.

    python tools/test_points.py --points data/test_points.json \
        --layout docs/coverage/camera_count/2_cameras/layout_plan.json \
        --layout docs/coverage/camera_count/3_cameras/layout_plan.json \
        --layout docs/coverage/camera_count/4_cameras/layout_plan.json \
        --out docs/coverage/validation

For each point, layout and posture it saves the planner's rating, which cameras see
the person's head and shoulders, the widest crossing angle between two of them, the
distance from the nearest of them to the head, and how many of 8 spots 0.3 m around
share the rating (8: well inside an area; low: near an edge, where a small error in
the mark or the mounting can change the rating).

Writes to --out: test_points.json (the points, anchors, route and predictions),
test_points.jpg (the lab scan with the points and the walking route) and
test_points_<n>_cameras.jpg (each layout's coverage map with the points). Rerun it
after measuring the real seat positions, or with the as-built cameras' layout file.
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from coverage_planner import current_cameras, draw_map, fit_text, to_px   # noqa: E402
from tracker3d import coverage as cov                                     # noqa: E402
from tracker3d.config import load_config                                  # noqa: E402
from tracker3d.plan import load_plan, load_scan_points                    # noqa: E402

RATING = {cov.GOOD: "good", cov.POOR: "poor angle", cov.SINGLE: "one camera", cov.UNSEEN: "unseen"}
ROUTE = (200, 40, 200)                        # BGR: magenta
RING = 0.3 * np.array([[np.cos(t), np.sin(t)] for t in np.radians(np.arange(0, 360, 45))])


def layout_cameras(path, lens, mirrored):
    """(name, cameras) of the best layout in a layout_planner.py report."""
    with open(path) as f:
        report = json.load(f)
    cams = [cov.Camera(c["label"], cov.look_rotation(c["facing_deg"], c["tilt_down_deg"], mirrored),
                       c["position_m"], lens) for c in report["layouts"][0]["cameras"]]
    return f"{report['cameras_total']}_cameras", cams


def predict(cams, occ, xy, min_angle, max_range):
    """{posture: [per-point prediction]} for one layout."""
    xy = np.asarray(xy, float)
    spots = np.vstack([xy] + [xy + d for d in RING])          # each point, then its ring
    planner = cov.Planner(cams, cov.PointSet(spots), occ, "both", min_angle, max_range)
    n = len(xy)
    status, seen = planner.split(planner.status), planner.split(planner.seen.T)
    out = {}
    for p, (sh, hd) in cov.POSTURES.items():
        rows = []
        for i, (x, y) in enumerate(xy):
            s = status[p][i]
            ring = status[p][n + i::n]
            who = [c for c, v in zip(cams, seen[p][i]) if v]
            bear = [np.arctan2(c.C[1] - y, c.C[0] - x) for c in who]
            angles = [np.degrees(np.arcsin(min(1.0, abs(np.sin(a - b)))))
                      for k, a in enumerate(bear) for b in bear[k + 1:]]
            head = np.array([x, y, hd])
            rows.append({
                "rating": RATING[int(s)], "cameras": [c.name for c in who],
                "widest_crossing_deg": round(max(angles), 1) if angles else None,
                "nearest_camera_m": round(min(np.linalg.norm(c.C - head) for c in who), 2) if who else None,
                "same_rating_nearby": int(np.sum(ring == s))})
        out[p] = rows
    return out


def draw_points(img, plan, data):
    """Route, then numbered markers: circle for floor points, square for seats,
    an extra ring for anchors."""
    pos = {p["id"]: to_px(plan, p["x"], p["y"]) for p in data["points"]}
    for a, b in zip(data["route"], data["route"][1:]):
        cv2.line(img, pos[a], pos[b], (0, 0, 0), 7, cv2.LINE_AA)
        cv2.line(img, pos[a], pos[b], ROUTE, 4, cv2.LINE_AA)
    for p in data["points"]:
        c, r = pos[p["id"]], 15
        if p["id"] in data["anchors"]:
            cv2.circle(img, c, r + 7, (0, 0, 0), 5, cv2.LINE_AA)
            cv2.circle(img, c, r + 7, (255, 255, 255), 2, cv2.LINE_AA)
        if p["kind"] == "seat":
            cv2.rectangle(img, (c[0] - r, c[1] - r), (c[0] + r, c[1] + r), (255, 255, 255), -1)
            cv2.rectangle(img, (c[0] - r, c[1] - r), (c[0] + r, c[1] + r), (0, 0, 0), 2)
        else:
            cv2.circle(img, c, r, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(img, c, r, (0, 0, 0), 2, cv2.LINE_AA)
        label = str(int(p["id"][1:]))
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.putText(img, label, (c[0] - tw // 2, c[1] + th // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 0, 0), 2, cv2.LINE_AA)


def legend(width, title):
    """Panel explaining the markers, drawn the same way as on the map."""
    ink, step, pad = (20, 20, 20), 40, 14
    rows = ["floor point: stand, then sit on a stool",
            "seat: sit at the desk, then stand beside the chair",
            "anchor: also checked at the start and end of each session",
            "walking route (S2)"]
    panel = np.full((62 + step * len(rows), width, 3), 245, np.uint8)
    cv2.putText(panel, title, (pad, 36), cv2.FONT_HERSHEY_SIMPLEX,
                fit_text(title, 0.85, 2, width - 2 * pad), ink, 2, cv2.LINE_AA)
    for k, text in enumerate(rows):
        y, c = 78 + step * k, (pad + 15, 72 + step * k)
        if k == 0:
            cv2.circle(panel, c, 12, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(panel, c, 12, ink, 2, cv2.LINE_AA)
        elif k == 1:
            cv2.rectangle(panel, (c[0] - 12, c[1] - 12), (c[0] + 12, c[1] + 12), (255, 255, 255), -1)
            cv2.rectangle(panel, (c[0] - 12, c[1] - 12), (c[0] + 12, c[1] + 12), ink, 2)
        elif k == 2:
            cv2.circle(panel, c, 16, ink, 2, cv2.LINE_AA)
            cv2.circle(panel, c, 10, ink, 2, cv2.LINE_AA)
        else:
            cv2.line(panel, (pad, c[1]), (pad + 30, c[1]), ROUTE, 4, cv2.LINE_AA)
        x = pad + 44
        cv2.putText(panel, text, (x, y + 4), cv2.FONT_HERSHEY_SIMPLEX,
                    fit_text(text, 0.75, 2, width - x - pad), ink, 2, cv2.LINE_AA)
    return panel


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--points", default="data/test_points.json")
    ap.add_argument("--layout", action="append", required=True,
                    help="layout_plan.json from tools/layout_planner.py (repeatable)")
    ap.add_argument("--out", default="test_points")
    ap.add_argument("--min-angle", type=float, default=25.0)
    ap.add_argument("--max-range", type=float, default=9.0)
    args = ap.parse_args()

    with open(args.points) as f:
        data = json.load(f)
    ids = [p["id"] for p in data["points"]]
    missing = [i for i in data.get("anchors", []) + data.get("second_person", []) + data.get("route", [])
               if i not in ids]
    if missing or len(set(ids)) != len(ids):
        raise SystemExit(f"{args.points}: unknown or repeated point ids {sorted(set(missing)) or ids}")
    cfg = load_config(args.config)
    plan = load_plan(cfg)
    points = load_scan_points(cfg)
    occ = cov.Occupancy(points)
    furniture = cov.furniture_mask(points, plan)
    cameras, _ = current_cameras(cfg)
    if not cameras:
        raise SystemExit("no calibrated camera to take the lens from")
    lens, mirrored = cameras[0].lens, np.linalg.det(cameras[0].R) < 0
    xy = [(p["x"], p["y"]) for p in data["points"]]
    for p, on in zip(data["points"], furniture(*np.array(xy).T)):
        if on or not plan.in_valid_area(p["x"], p["y"]):
            print(f"[points] warning: {p['id']} is on furniture or outside the room in the scan")

    os.makedirs(args.out, exist_ok=True)
    layouts = [layout_cameras(path, lens, mirrored) for path in args.layout]
    for p in data["points"]:
        p["predicted"] = {}
    for (name, cams), path in zip(layouts, args.layout):
        pred = predict(cams, occ, xy, args.min_angle, args.max_range)
        for k, p in enumerate(data["points"]):
            p["predicted"][name] = {posture: pred[posture][k] for posture in pred}
        print(f"\n{name}  (seated / standing)")
        for p in data["points"]:
            r = p["predicted"][name]
            print(f"  {p['id']} ({p['x']:.2f}, {p['y']:.2f}) {p['kind']:5s}  "
                  f"{r['seated']['rating']:10s} / {r['standing']['rating']}")
        full = cov.Planner(cams, cov.FloorGrid(plan, 0.25, furniture=furniture), occ, "both",
                           args.min_angle, args.max_range)
        img = draw_map(plan, full.grid, full.worst(), [], f"{name.replace('_', ' ')}: test points "
                       "(each square: worse of seated / standing)",
                       new=[(c, c.name) for c in cams], new_colour=(255, 255, 255))
        draw_points(img, plan, data)
        cv2.imwrite(os.path.join(args.out, f"test_points_{name}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 88])
    if data.get("route"):
        pos = {p["id"]: np.array([p["x"], p["y"]]) for p in data["points"]}
        data["route_length_m"] = round(float(sum(np.linalg.norm(pos[a] - pos[b])
                                                 for a, b in zip(data["route"], data["route"][1:]))), 2)
    data["layouts"] = {name: path for (name, _), path in zip(layouts, args.layout)}
    data["settings"] = {"min_angle_deg": args.min_angle, "max_range_m": args.max_range,
                        "postures_m": {p: {"shoulder": sh, "head": hd} for p, (sh, hd) in cov.POSTURES.items()}}

    scale = 2
    bg = cv2.resize(plan.background(), None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    for m in range(0, int(plan.width_m) + 1):
        cv2.line(bg, to_px(plan, m, 0), to_px(plan, m, plan.height_m), (200, 200, 200), 1)
        cv2.putText(bg, str(m), (to_px(plan, m, 0)[0] - 4, to_px(plan, m, 0)[1] + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1)
    for m in range(0, int(plan.height_m) + 1):
        cv2.line(bg, to_px(plan, 0, m), to_px(plan, plan.width_m, m), (200, 200, 200), 1)
        cv2.putText(bg, str(m), (to_px(plan, 0, m)[0] - 22, to_px(plan, 0, m)[1] + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1)
    draw_points(bg, plan, data)
    img = np.vstack([bg, legend(bg.shape[1], f"{len(xy)} test points (1 m grid, lab map metres)")])
    cv2.imwrite(os.path.join(args.out, "test_points.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    with open(os.path.join(args.out, "test_points.json"), "w") as f:
        json.dump(data, f, indent=2)
    print(f"\n[points] test_points.json and maps in {args.out}/")


if __name__ == "__main__":
    main()
