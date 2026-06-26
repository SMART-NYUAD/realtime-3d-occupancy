"""Real-time multi-camera people tracking onto a shared floorplan.

Each camera detects + tracks people, projects their feet through that camera's
homography into the SAME floorplan frame (meters), and all people are drawn on
one top-down map. Per-camera annotated video windows are optional.

Usage:
    python track.py [--config config.yaml]
Keys (in any window):  q = quit
"""
import argparse
import os
import sys
import time
from collections import defaultdict, deque

import cv2
import numpy as np
import yaml

sys.path.insert(0, "src")
from capture import list_cameras                               # noqa: E402
from stream import CameraStream                                # noqa: E402
from detector import PersonTracker                             # noqa: E402
from geometry import load_homography, image_to_floor, foot_point  # noqa: E402
from visualizer import draw_camera_view                        # noqa: E402
from plan import load_plan                                     # noqa: E402
from fusion import Fusion                                      # noqa: E402

# one distinct color per camera so you can see which camera sees whom
CAM_COLORS = [(0, 200, 255), (0, 255, 0), (255, 120, 0),
              (255, 0, 255), (0, 165, 255), (200, 200, 0)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--headless", type=float, default=0.0,
                    help="run N seconds with no GUI, saving snapshots to --save-dir")
    ap.add_argument("--save-dir", default="/tmp")
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    cams = list_cameras(cfg)
    floor = load_plan(cfg)

    # NTP-synchronized capture: align all cameras to a common capture instant so a
    # laggy stream doesn't split a moving person into a second track. RTSP only.
    scfg = cfg.get("sync", {})
    sync_on = scfg.get("enabled", False)
    if sync_on and not all(str(c["source"]).startswith("rtsp://") for c in cams):
        print("[sync] disabled: needs all-RTSP sources.")
        sync_on = False

    streams, trackers, homs, colors = {}, {}, {}, {}
    for i, cam in enumerate(cams):
        name = cam["name"]
        colors[name] = CAM_COLORS[i % len(CAM_COLORS)]
        hp = cam["homography_file"]
        if os.path.exists(hp):
            try:
                homs[name] = load_homography(hp)
                print(f"[{name}] homography loaded from {hp}")
            except ValueError as e:
                homs[name] = None
                print(f"[{name}] {e}")
        else:
            homs[name] = None
            print(f"[{name}] no homography ({hp}) — run: python calibrate.py --camera {name}")
        if not sync_on:
            print(f"[{name}] connecting to {cam['source']} ...")
            streams[name] = CameraStream(cam)
        trackers[name] = PersonTracker(cfg)   # separate tracker => per-camera IDs

    group = None
    if sync_on:
        from sync import SyncGroup
        print(f"[sync] NTP-synced capture (decoder={scfg.get('decoder', 'sw')}, "
              f"tol={int(scfg.get('tol_ms', 75))}ms) connecting all cameras ...")
        group = SyncGroup(cams, decoder=scfg.get("decoder", "sw"),
                          tol_s=scfg.get("tol_ms", 75) / 1000.0,
                          buffer_sec=scfg.get("buffer_sec", 1.0),
                          latency_ms=scfg.get("latency_ms", 100))

    show_cams = cfg["output"].get("show_camera_windows", True)
    draw_trails = cfg["output"]["draw_track_trails"]
    trail_len = int(cfg["output"]["trail_length"])
    trails = defaultdict(lambda: deque(maxlen=trail_len))   # key: (cam_name, id)

    fcfg = cfg.get("fusion", {})
    fuse_on = fcfg.get("enabled", False)
    fuser = Fusion(fcfg.get("merge_distance_m", 0.6), fcfg.get("match_gate_m", 1.0),
                   fcfg.get("max_age_s", 1.5), fcfg.get("smoothing", 0.5)) if fuse_on else None
    show_raw = fcfg.get("show_raw_dots", False)

    headless = args.headless > 0.0
    last_views = {}
    fps_t, fps_n, fps = time.time(), 0, 0.0
    start = time.time()
    print("\nTracking all cameras... " + ("(headless)" if headless else "press 'q' to quit."))
    while True:
        all_people = []
        synced = None
        if sync_on:
            _t, synced = group.next_aligned()
            if not synced:
                continue
        for cam in cams:
            name = cam["name"]
            frame = synced.get(name) if sync_on else streams[name].read()
            if frame is None:
                continue
            people = trackers[name].track(frame)
            H = homs[name]
            for p in people:
                p["cam"] = name
                p["world"] = None
                if H is not None:
                    fx, fy = foot_point(p["bbox"])
                    w = image_to_floor(H, fx, fy)
                    # discard detections that map outside the floorplan (distortion/bad foot point)
                    if w is not None and not floor.in_bounds(*w):
                        w = None
                    p["world"] = w
                    if w is not None:
                        trails[(name, p["id"])].append(w)
            all_people.extend(people)

            if show_cams or headless:
                view = draw_camera_view(frame, people)
                cv2.putText(view, f"{name}: {len(people)} people", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors[name], 2)
                view = cv2.resize(view, (640, 360))
                last_views[name] = view
                if show_cams and not headless:
                    cv2.imshow(name, view)

        # one shared top-down map, colored by camera
        canvas = floor.background()
        if draw_trails:
            for (name, _tid), pts in trails.items():
                col = colors[name]
                for j in range(1, len(pts)):
                    if pts[j - 1] and pts[j]:
                        cv2.line(canvas, floor.world_to_px(*pts[j - 1]),
                                 floor.world_to_px(*pts[j]), col, 2, cv2.LINE_AA)
        if fuse_on:
            # faint per-camera dots (optional), then one fused marker per person
            if show_raw:
                for p in all_people:
                    if p["world"] is None:
                        continue
                    px, py = floor.world_to_px(*p["world"])
                    cv2.circle(canvas, (px, py), 4, colors[p["cam"]], 1, cv2.LINE_AA)
            people_now = fuser.update(all_people)
            for g in people_now:
                px, py = floor.world_to_px(*g["world"])
                col = (160, 160, 160) if g["coasting"] else (60, 220, 60)
                cv2.circle(canvas, (px, py), 8, col, -1, cv2.LINE_AA)
                tag = f"P{g['gid']}" + ("" if g["coasting"] else f" [{','.join(g['cams'])}]")
                cv2.putText(canvas, tag, (px + 10, py),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)
            n_people = len(people_now)
        else:
            for p in all_people:
                if p["world"] is None:
                    continue
                col = colors[p["cam"]]
                px, py = floor.world_to_px(*p["world"])
                cv2.circle(canvas, (px, py), 7, col, -1, cv2.LINE_AA)
                cv2.putText(canvas, f"{p['cam']}#{p['id']}", (px + 9, py),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)
            n_people = sum(1 for p in all_people if p["world"] is not None)

        fps_n += 1
        if time.time() - fps_t >= 1.0:
            fps = fps_n / (time.time() - fps_t)
            fps_t, fps_n = time.time(), 0
        label = "people" if fuse_on else "on map"
        cv2.putText(canvas, f"{fps:.1f} FPS | {n_people} {label}", (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        if headless:
            cv2.imwrite(f"{args.save_dir}/floor_map.png", canvas)
            for name, v in last_views.items():
                cv2.imwrite(f"{args.save_dir}/cam_{name}.png", v)
            if fuse_on:
                summary = ", ".join(
                    f"P{g['gid']}({g['world'][0]:.1f},{g['world'][1]:.1f})"
                    f"[{','.join(g['cams'])}]" for g in people_now)
                print(f"  t={time.time()-start:4.1f}s  {fps:.1f}FPS  people={n_people}  {summary}")
            if time.time() - start >= args.headless:
                break
        else:
            cv2.imshow("floor map (all cameras)", canvas)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break

    if group is not None:
        group.release()
    for s in streams.values():
        s.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
