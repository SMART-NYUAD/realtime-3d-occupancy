"""Live calibration tuner — drag points while watching your tracked dot.

Stand in view of the cameras. Each camera projects you onto the floorplan in
real time. Drag the calibration points until your dot sits at your true spot AND
both cameras' dots coincide. Then save.

Windows:
  - one per camera: live video + draggable RED point markers (the camera-pixel
    side of each correspondence) + your detection (green box, foot dot).
  - FLOORPLAN: draggable reference points (world side, shared by all cameras) +
    your live position from each camera (color-coded per camera).

Controls (mouse):
  - LEFT-drag a point  : move it (camera point -> that camera only;
                         floorplan point -> all cameras using it). H updates live.
  - RIGHT-click a point : delete it (floorplan: removes the shared reference).
Keys: s = save | r = reload from disk | q = quit

Usage: python recalibrate.py [--config config.yaml]
"""
import argparse
import json
import math
import os
import time

import cv2
import numpy as np

from tracker3d.capture import CameraStream
from tracker3d.config import list_cameras, load_config
from tracker3d.detector import MultiCamDetector
from tracker3d.geometry import (compute_homography, foot_from_pose, foot_point,
                                load_calibration_meta, save_calibration)
from tracker3d.plan import load_plan
from tracker3d.privacy import PrivacyMasker
from tracker3d.render import CAM_COLORS
HIT_PX = 14            # click tolerance for grabbing a point

S = {
    "refs": {},        # ref_id -> [x, y] world meters (shared across cameras)
    "pairs": {},       # cam_name -> [ {"ref_id":int, "cam":[u,v]} ]
    "H": {},           # cam_name -> 3x3 or None
    "err": {},         # cam_name -> mean reprojection error (m) or None
    "scale": {},       # cam_name -> display scale
    "imsize": {},      # cam_name -> (w, h)
    "drag": None,      # ("fp", ref_id) | ("cam", cam_name, index)
    "next_id": 0,
}


def load_refs(path):
    if os.path.exists(path):
        pts = json.load(open(path)).get("points", [])
        return {p["id"]: list(p["world"]) for p in pts}
    return {}


def save_all(cams, refs_path):
    json.dump({"points": [{"id": i, "world": w} for i, w in S["refs"].items()]},
              open(refs_path, "w"), indent=2)
    saved = []
    for cam in cams:
        n = cam["name"]
        if S["H"][n] is not None:
            save_calibration(cam["homography_file"], S["H"][n], S["imsize"][n],
                             pairs=S["pairs"][n])
            saved.append(n)
    print(f"[saved] homographies+points for {saved}, refs -> {refs_path}")


def recompute(name):
    pr = S["pairs"][name]
    if len(pr) < 4:
        S["H"][name], S["err"][name] = None, None
        return
    cam_pts = [p["cam"] for p in pr]
    wld = [S["refs"][p["ref_id"]] for p in pr]
    try:
        H = compute_homography(cam_pts, wld)
        if abs(np.linalg.det(H)) < 1e-9:
            S["H"][name], S["err"][name] = None, None
            return
        errs = []
        for (u, v), (wx, wy) in zip(cam_pts, wld):
            w = H @ np.array([u, v, 1.0])
            errs.append(math.hypot(w[0] / w[2] - wx, w[1] / w[2] - wy)
                        if abs(w[2]) > 1e-9 else 9.9)
        S["H"][name], S["err"][name] = H, float(np.mean(errs))
    except (RuntimeError, ValueError, cv2.error):
        S["H"][name], S["err"][name] = None, None


def bootstrap(cam, frame):
    """Load the clicked points from the calibration sidecar if present, else
    reconstruct them from the saved homography by inverse-projecting the shared
    reference points into the view."""
    n = cam["name"]
    h, w = frame.shape[:2]
    S["imsize"][n] = (w, h)
    pairs = load_calibration_meta(cam["homography_file"]).get("pairs")
    if pairs:
        S["pairs"][n] = [p for p in pairs if p["ref_id"] in S["refs"]]
    elif os.path.exists(cam["homography_file"]) and S["refs"]:
        H = np.load(cam["homography_file"])
        Hinv = np.linalg.inv(H)
        pairs = []
        for rid, wld in S["refs"].items():
            p = Hinv @ np.array([wld[0], wld[1], 1.0])
            if abs(p[2]) < 1e-9:
                continue
            u, v = p[0] / p[2], p[1] / p[2]
            if -60 <= u <= w + 60 and -60 <= v <= h + 60:   # visible-ish in this cam
                pairs.append({"ref_id": rid, "cam": [float(u), float(v)]})
        S["pairs"][n] = pairs
    else:
        S["pairs"][n] = []
    recompute(n)


def make_fp_cb(fp):
    def cb(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            for rid, wld in S["refs"].items():
                px = fp.world_to_px(*wld)
                if math.hypot(px[0] - x, px[1] - y) < HIT_PX:
                    S["drag"] = ("fp", rid)
                    return
        elif event == cv2.EVENT_MOUSEMOVE and S["drag"] and S["drag"][0] == "fp":
            rid = S["drag"][1]
            S["refs"][rid] = list(fp.px_to_world(x, y))
            for n in S["pairs"]:
                if any(p["ref_id"] == rid for p in S["pairs"][n]):
                    recompute(n)
        elif event == cv2.EVENT_LBUTTONUP:
            S["drag"] = None
        elif event == cv2.EVENT_RBUTTONDOWN:
            for rid, wld in list(S["refs"].items()):
                px = fp.world_to_px(*wld)
                if math.hypot(px[0] - x, px[1] - y) < HIT_PX:
                    del S["refs"][rid]
                    for n in S["pairs"]:
                        S["pairs"][n] = [p for p in S["pairs"][n] if p["ref_id"] != rid]
                        recompute(n)
                    print(f"  deleted reference R{rid}")
                    return
    return cb


def make_cam_cb(name):
    def cb(event, x, y, flags, param):
        s = S["scale"][name]
        fx, fy = x / s, y / s
        if event == cv2.EVENT_LBUTTONDOWN:
            for i, p in enumerate(S["pairs"][name]):
                if math.hypot(p["cam"][0] - fx, p["cam"][1] - fy) < HIT_PX / s:
                    S["drag"] = ("cam", name, i)
                    return
        elif (event == cv2.EVENT_MOUSEMOVE and S["drag"]
              and S["drag"][0] == "cam" and S["drag"][1] == name):
            S["pairs"][name][S["drag"][2]]["cam"] = [fx, fy]
            recompute(name)
        elif event == cv2.EVENT_LBUTTONUP:
            S["drag"] = None
        elif event == cv2.EVENT_RBUTTONDOWN:
            for i, p in enumerate(S["pairs"][name]):
                if math.hypot(p["cam"][0] - fx, p["cam"][1] - fy) < HIT_PX / s:
                    S["pairs"][name].pop(i)
                    recompute(name)
                    return
    return cb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)

    cams = list_cameras(cfg)
    fpc = cfg["floorplan"]
    fp = load_plan(cfg)
    refs_path = fpc.get("reference_points_file", "data/reference_points.json")
    S["refs"] = load_refs(refs_path)
    S["next_id"] = max(S["refs"], default=0)

    streams, colors = {}, {}
    for i, cam in enumerate(cams):
        n = cam["name"]
        colors[n] = CAM_COLORS[i % len(CAM_COLORS)]
        print(f"[{n}] connecting {cam['source']} ...")
        streams[n] = CameraStream(cam)
    detector = MultiCamDetector(cfg, [c["name"] for c in cams])
    masker = PrivacyMasker(cfg["output"].get("privacy", {}))
    kp_conf = float(cfg["detector"].get("kp_conf", 0.5))
    time.sleep(3.0)
    for cam in cams:
        _, f = streams[cam["name"]].read()
        while f is None:
            time.sleep(0.1); _, f = streams[cam["name"]].read()
        bootstrap(cam, f)

    fp_cb = make_fp_cb(fp)
    cv2.namedWindow("FLOORPLAN")
    cv2.setMouseCallback("FLOORPLAN", fp_cb)
    for cam in cams:
        n = cam["name"]
        cv2.namedWindow(n)
        cv2.setMouseCallback(n, make_cam_cb(n))
    print("\nDrag points to tune. s=save  r=reload  q=quit.\n")

    while True:
        dots = []   # (cam_name, (wx,wy))
        frames = {}
        for cam in cams:
            _, f = streams[cam["name"]].read()
            if f is not None:
                frames[cam["name"]] = f
        results = detector.detect_and_track(frames)
        for cam in cams:
            n = cam["name"]
            if n not in frames:
                continue
            frame = frames[n]
            people, raw = results[n]
            H = S["H"][n]
            for p in people:
                # Same foot estimate as track.py (pose ankles, else box bottom).
                if p.get("kxy") is not None:
                    (fx, fy), _, _ = foot_from_pose(p["kxy"], p["kconf"], p["bbox"], kp_conf)
                else:
                    fx, fy = foot_point(p["bbox"])
                p["foot"] = (fx, fy)
                if H is not None:
                    w = H @ np.array([fx, fy, 1.0])
                    if abs(w[2]) > 1e-9:
                        w = (w[0] / w[2], w[1] / w[2])
                        if fp.in_bounds(*w):
                            dots.append((n, w))

            # ---- camera window ----
            w0, h0 = S["imsize"][n]
            s = min(1280 / w0, 720 / h0, 1.0)
            S["scale"][n] = s
            disp = cv2.resize(frame, (int(w0 * s), int(h0 * s)))
            masker.apply(disp, raw, s, s)
            for p in people:
                x1, y1, x2, y2 = [int(c * s) for c in p["bbox"]]
                cv2.rectangle(disp, (x1, y1), (x2, y2), colors[n], 2)
                cv2.circle(disp, (int(p["foot"][0] * s), int(p["foot"][1] * s)), 4, (0, 255, 0), -1)
            for pr in S["pairs"][n]:
                u, v = int(pr["cam"][0] * s), int(pr["cam"][1] * s)
                cv2.circle(disp, (u, v), 7, (0, 0, 255), 2)
                cv2.putText(disp, f"R{pr['ref_id']}", (u + 8, v),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            e = S["err"][n]
            etxt = f"err={e:.2f}m" if e is not None else "need >=4 pts"
            cv2.putText(disp, f"{n}: {len(S['pairs'][n])} pts  {etxt}", (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors[n], 2)
            cv2.imshow(n, disp)

        # ---- floorplan ----
        canvas = fp.background()
        for rid, wld in S["refs"].items():
            px = fp.world_to_px(*wld)
            cv2.circle(canvas, px, 6, (200, 200, 200), -1)
            cv2.putText(canvas, f"R{rid}", (px[0] + 7, px[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        for n, w in dots:
            px = fp.world_to_px(*w)
            cv2.circle(canvas, px, 8, colors[n], -1)
            cv2.putText(canvas, n, (px[0] + 9, px[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, colors[n], 1)
        # live cross-camera distance (first two distinct-camera dots)
        if len({n for n, _ in dots}) >= 2:
            (na, wa), (nb, wb) = dots[0], next(d for d in dots if d[0] != dots[0][0])
            dist = math.hypot(wa[0] - wb[0], wa[1] - wb[1])
            cv2.putText(canvas, f"{na}<->{nb}: {dist:.2f} m apart", (10, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 0) if dist < 0.7 else (0, 0, 255), 2)
        cv2.imshow("FLOORPLAN", canvas)

        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            break
        elif k == ord("s"):
            save_all(cams, refs_path)
        elif k == ord("r"):
            S["refs"] = load_refs(refs_path)
            for cam in cams:
                _, f = streams[cam["name"]].read()
                if f is not None:
                    bootstrap(cam, f)
            print("  reloaded from disk")

    for s in streams.values():
        s.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
