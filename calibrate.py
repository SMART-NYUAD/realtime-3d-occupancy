"""Shared-reference floorplan calibration.

To make cameras AGREE (so a person merges into one dot), they must calibrate
against the SAME floor locations. This tool keeps a shared set of reference
points (in floorplan meters). When you calibrate a camera:

  - Reference points placed by other cameras show up GRAYED on the floorplan.
  - You click an EXISTING (gray) point to reuse its exact location, then click
    where that point appears in the camera image.  -> cameras stay in agreement.
  - Click empty floor to CREATE a new shared reference point (cyan); it's saved
    for future cameras to snap to.

Collect >=4 correspondences spread as widely as the camera sees (big quad!).
Shared points should sit in the OVERLAP zone between cameras.

Usage:
    python calibrate.py --camera <name> [--config config.yaml] [--frame img.jpg]
Camera window:  SPACE freeze | u undo | ENTER finish | q quit
"""
import argparse
import json
import math
import os
import sys

import cv2
import numpy as np

from tracker3d.capture import open_capture
from tracker3d.config import get_camera, list_cameras, load_config
from tracker3d.geometry import compute_homography, fit_errors, save_calibration
from tracker3d.plan import load_plan

CAM_WIN = "CAMERA"
FP_WIN = "FLOORPLAN (gray=other cameras, cyan=new, green=used)"
SNAP_M = 0.4   # click within this many meters of a ref point snaps to it

state = {
    "cam_scale": 1.0,
    "refs": [],          # [{"id":int,"world":[x,y]}]  existing + new
    "new_ids": set(),    # ids created this session
    "next_id": 0,
    "pending": None,     # {"world":(x,y),"ref_id":int} awaiting a camera click
    "pairs": [],         # [{"cam":(u,v),"world":(x,y),"ref_id":int}]
}


def load_refs(path):
    if os.path.exists(path):
        pts = json.load(open(path)).get("points", [])
        return pts, max((p["id"] for p in pts), default=0)
    return [], 0


def save_refs(path, points):
    json.dump({"points": points}, open(path, "w"), indent=2)


def _fit(img, max_w=1280, max_h=720):
    h, w = img.shape[:2]
    s = min(max_w / w, max_h / h, 1.0)
    state["cam_scale"] = s
    return cv2.resize(img, (int(w * s), int(h * s))) if s < 1.0 else img


def grab_frame(cam, frame_path):
    if frame_path:
        img = cv2.imread(frame_path)
        if img is None:
            raise RuntimeError(f"Could not read {frame_path}")
        return img
    cap = open_capture(cam)
    print("Click the camera window, then press SPACE to freeze a frame ('q' quits).")
    while True:
        ok, f = cap.read()
        if not ok:
            raise RuntimeError("Failed to read from camera.")
        disp = _fit(f)
        cv2.putText(disp, "LIVE - press SPACE to freeze  (q=quit)", (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow(CAM_WIN, disp)
        k = cv2.waitKey(1) & 0xFF
        if k == ord(" "):
            cap.release()
            return f
        if k == ord("q"):
            cap.release(); cv2.destroyAllWindows(); sys.exit(0)


def on_fp(event, x, y, flags, fp):
    if event != cv2.EVENT_LBUTTONDOWN:
        return
    wx, wy = fp.px_to_world(x, y)
    best, bd = None, SNAP_M
    for r in state["refs"]:
        d = math.hypot(r["world"][0] - wx, r["world"][1] - wy)
        if d < bd:
            bd, best = d, r
    if best is None:                              # create new shared reference
        state["next_id"] += 1
        best = {"id": state["next_id"], "world": [round(wx, 3), round(wy, 3)]}
        state["refs"].append(best)
        state["new_ids"].add(best["id"])
        print(f"  + NEW reference R{best['id']} at ({wx:.2f},{wy:.2f}) m")
    else:
        print(f"  using existing reference R{best['id']} "
              f"({best['world'][0]:.2f},{best['world'][1]:.2f}) m")
    state["pending"] = {"world": tuple(best["world"]), "ref_id": best["id"]}
    print("    -> now click that same point in the CAMERA window")


def on_cam(event, x, y, flags, param):
    if event != cv2.EVENT_LBUTTONDOWN:
        return
    if state["pending"] is None:
        print("  pick a reference on the FLOORPLAN first.")
        return
    s = state["cam_scale"]
    pr = state["pending"]
    state["pairs"].append({"cam": (x / s, y / s), "world": pr["world"],
                           "ref_id": pr["ref_id"]})
    print(f"  paired R{pr['ref_id']} <-> camera ({x/s:.0f},{y/s:.0f})  "
          f"[{len(state['pairs'])} pairs]")
    state["pending"] = None


def undo():
    # clear an unpaired pending (removing a just-created new ref), else pop a pair
    if state["pending"] is not None:
        rid = state["pending"]["ref_id"]
        if rid in state["new_ids"] and all(p["ref_id"] != rid for p in state["pairs"]):
            state["refs"][:] = [r for r in state["refs"] if r["id"] != rid]
            state["new_ids"].discard(rid)
        state["pending"] = None
        print("  cleared pending")
    elif state["pairs"]:
        p = state["pairs"].pop()
        rid = p["ref_id"]
        if rid in state["new_ids"] and all(q["ref_id"] != rid for q in state["pairs"]):
            state["refs"][:] = [r for r in state["refs"] if r["id"] != rid]
            state["new_ids"].discard(rid)
        print("  undo last pair")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--camera", default=None, help="camera name (default: first)")
    ap.add_argument("--frame", default=None, help="use a saved image instead of live camera")
    args = ap.parse_args()
    cfg = load_config(args.config)

    cam = get_camera(cfg, args.camera) if args.camera else list_cameras(cfg)[0]
    fpc = cfg["floorplan"]
    fp = load_plan(cfg)
    refs_path = fpc.get("reference_points_file", "data/reference_points.json")

    state["refs"], state["next_id"] = load_refs(refs_path)
    print(f"Calibrating '{cam['name']}'  ({cam['source']})")
    print(f"Map: {fp.width_m:.2f} x {fp.height_m:.2f} m | "
          f"{len(state['refs'])} existing reference points")

    frame = grab_frame(cam, args.frame)
    cv2.namedWindow(CAM_WIN)
    cv2.namedWindow(FP_WIN)
    cv2.setMouseCallback(CAM_WIN, on_cam)
    cv2.setMouseCallback(FP_WIN, on_fp, fp)
    print("\nFLOORPLAN: click a gray point (reuse) or empty floor (new), then click")
    print("it in the CAMERA. >=4 pairs, spread wide. ENTER=finish, u=undo, q=quit.\n")

    while True:
        # camera window
        cam_disp = _fit(frame.copy())
        s = state["cam_scale"]
        for p in state["pairs"]:
            u, v = int(p["cam"][0] * s), int(p["cam"][1] * s)
            cv2.circle(cam_disp, (u, v), 6, (0, 255, 0), -1)
            cv2.putText(cam_disp, f"R{p['ref_id']}", (u + 7, v),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        msg = (f"click R{state['pending']['ref_id']} HERE in camera"
               if state["pending"] else
               f"pairs:{len(state['pairs'])} (need >=4)  pick a floorplan point ->")
        cv2.putText(cam_disp, msg, (12, 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 255), 2)
        cv2.imshow(CAM_WIN, cam_disp)

        # floorplan window
        fp_disp = fp.background()
        used = {p["ref_id"] for p in state["pairs"]}
        for r in state["refs"]:
            px = fp.world_to_px(*r["world"])
            if r["id"] in used:
                col = (0, 255, 0)        # used this session
            elif r["id"] in state["new_ids"]:
                col = (255, 255, 0)      # new this session, not yet paired
            else:
                col = (140, 140, 140)    # from other cameras -> snap to these
            cv2.circle(fp_disp, px, 6, col, -1)
            cv2.putText(fp_disp, f"R{r['id']}", (px[0] + 7, px[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)
        if state["pending"]:
            cv2.circle(fp_disp, fp.world_to_px(*state["pending"]["world"]),
                       11, (0, 255, 255), 2)
        cv2.imshow(FP_WIN, fp_disp)

        k = cv2.waitKey(20) & 0xFF
        if k in (13, 10):
            if len(state["pairs"]) >= 4:
                break
            print("  need at least 4 pairs...")
        elif k == ord("u"):
            undo()
        elif k == ord("q"):
            sys.exit(0)
    cv2.destroyAllWindows()

    cam_pts = [p["cam"] for p in state["pairs"]]
    world_pts = [p["world"] for p in state["pairs"]]

    wp = np.array(world_pts, dtype=np.float32)
    hull_area = cv2.contourArea(cv2.convexHull(wp))
    span_x = float(wp[:, 0].max() - wp[:, 0].min())
    span_y = float(wp[:, 1].max() - wp[:, 1].min())
    print(f"\nSpread: X {span_x:.2f} m, Y {span_y:.2f} m, area {hull_area:.2f} m^2")
    if hull_area < 2.0 or span_x < 1.0 or span_y < 1.0:
        print("*** REFUSED: points clustered / nearly in a line. Spread them wider. "
              "Nothing saved. ***")
        return

    try:
        H = compute_homography(cam_pts, world_pts)
        ins, loo = fit_errors(cam_pts, world_pts)
    except (RuntimeError, ValueError) as e:
        print(f"*** REFUSED: {e} Nothing saved. ***")
        return

    h, w = frame.shape[:2]
    pairs = [{"ref_id": p["ref_id"], "cam": [float(p["cam"][0]), float(p["cam"][1])]}
             for p in state["pairs"]]
    save_calibration(cam["homography_file"], H, (w, h), pairs=pairs)
    save_refs(refs_path, state["refs"])
    print(f"Saved homography for '{cam['name']}' -> {cam['homography_file']} ({w}x{h})")
    print(f"Saved {len(state['refs'])} reference points -> {refs_path}")
    print(f"Fit error on the clicked points: mean {ins.mean():.3f} m (max {ins.max():.3f} m)")
    if len(loo):
        print(f"Leave-one-out error (how far a NEW floor point lands): "
              f"mean {loo.mean():.3f} m (max {loo.max():.3f} m)")
        if loo.mean() > 0.3:
            print("  ^ high: re-click imprecise points or spread them wider.")
    else:
        print("With only 4 points the fit is exact, so its error says nothing — "
              "add a 5th+ point to get a real accuracy estimate.")
    shared = [p['ref_id'] for p in state['pairs'] if p['ref_id'] not in state['new_ids']]
    if shared:
        print(f"Reused shared references: {sorted(set(shared))} "
              "(these keep this camera aligned with the others).")
    else:
        print("NOTE: you created all-new references. For cameras to AGREE, the next "
              "camera should SNAP to these (gray) points in the overlap zone.")


if __name__ == "__main__":
    main()
