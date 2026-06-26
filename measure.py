"""Measure distances on the BIM floorplan, in meters.

Click points to measure: each consecutive pair shows its distance, and the total
path length is shown too. Clicks SNAP to the nearest DXF vertex (corner) for
precision; a yellow ring means snapped. Use it to verify the model scale or to
plan real-world distances and reference points.

Controls:
  - LEFT-click  : add a measure point (snaps to nearest corner)
  - RIGHT-click : undo last point
  - c : clear      n : start a new separate measurement
  - s : toggle snapping        q : quit

Usage: python measure.py [--config config.yaml]
"""
import argparse
import math
import sys

import cv2
import yaml

sys.path.insert(0, "src")
from floorplan import Floorplan        # noqa: E402

ST = {"pts": [], "cursor": None, "snapped": False, "snap_on": True, "fp": None}
SNAP_M = 0.15


def on_mouse(event, x, y, flags, param):
    fp = ST["fp"]
    wx, wy = fp.px_to_world(x, y)
    if ST["snap_on"]:
        (wx, wy), ST["snapped"] = fp.snap(wx, wy, SNAP_M)
    else:
        ST["snapped"] = False
    ST["cursor"] = (wx, wy)
    if event == cv2.EVENT_LBUTTONDOWN:
        ST["pts"].append((wx, wy))
    elif event == cv2.EVENT_RBUTTONDOWN and ST["pts"]:
        ST["pts"].pop()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    fpc = cfg["floorplan"]
    fp = Floorplan(fpc["file"], px_per_m=fpc["px_per_m"], margin_m=fpc["margin_m"])
    ST["fp"] = fp
    print(f"Floorplan: {fp.width_m:.2f} x {fp.height_m:.2f} m, "
          f"{len(fp.vertices)} BIM vertices for snapping")

    win = "MEASURE (left=add, right=undo, c=clear, n=new, s=snap, q=quit)"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)
    font = cv2.FONT_HERSHEY_SIMPLEX

    while True:
        canvas = fp.background()
        pts = ST["pts"]

        # rubber-band from last point to cursor
        live = list(pts)
        if pts and ST["cursor"] is not None:
            live = pts + [ST["cursor"]]

        total = 0.0
        for i in range(1, len(live)):
            a, b = live[i - 1], live[i]
            pa, pb = fp.world_to_px(*a), fp.world_to_px(*b)
            seg_live = (i == len(live) - 1 and ST["cursor"] is not None
                        and len(live) > len(pts))
            col = (90, 200, 255) if seg_live else (0, 230, 0)
            cv2.line(canvas, pa, pb, col, 2, cv2.LINE_AA)
            d = math.hypot(b[0] - a[0], b[1] - a[1])
            total += d
            mid = ((pa[0] + pb[0]) // 2, (pa[1] + pb[1]) // 2)
            cv2.putText(canvas, f"{d:.2f} m", (mid[0] + 5, mid[1] - 5),
                        font, 0.5, col, 2)

        for p in pts:
            cv2.circle(canvas, fp.world_to_px(*p), 5, (0, 230, 0), -1)

        # cursor + snap indicator + live coordinate
        if ST["cursor"] is not None:
            cpx = fp.world_to_px(*ST["cursor"])
            if ST["snapped"]:
                cv2.circle(canvas, cpx, 9, (0, 255, 255), 2)
            cv2.putText(canvas, f"({ST['cursor'][0]:.2f}, {ST['cursor'][1]:.2f}) m",
                        (cpx[0] + 10, cpx[1] + 18), font, 0.45, (0, 255, 255), 1)

        # header
        hdr = f"points: {len(pts)}"
        if len(pts) >= 2:
            straight = math.hypot(pts[-1][0] - pts[0][0], pts[-1][1] - pts[0][1])
            seg_total = sum(math.hypot(pts[i][0] - pts[i - 1][0],
                                       pts[i][1] - pts[i - 1][1])
                            for i in range(1, len(pts)))
            hdr += f"   path: {seg_total:.2f} m   straight: {straight:.2f} m"
        hdr += f"   snap:{'on' if ST['snap_on'] else 'off'}"
        cv2.putText(canvas, hdr, (10, 22), font, 0.6, (255, 255, 255), 2)

        cv2.imshow(win, canvas)
        k = cv2.waitKey(20) & 0xFF
        if k == ord("q"):
            break
        elif k in (ord("c"), ord("n")):
            ST["pts"] = []
        elif k == ord("s"):
            ST["snap_on"] = not ST["snap_on"]

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
