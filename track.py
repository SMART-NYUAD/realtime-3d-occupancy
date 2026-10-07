"""Real-time multi-camera people tracking onto a shared floor map.

Each step: grab one time-aligned frame per camera -> ONE batched YOLO-pose pass
(TensorRT) -> per-camera ByteTrack -> foot point through each camera's
homography -> cross-camera fusion -> top-down map (+ MQTT).

Usage:
    python track.py [--config config.yaml]
    python track.py --headless 60 [--save-dir /tmp]   # no GUI, 60 s, snapshots
    python track.py --log-csv walk.csv                # also log every position
Keys (GUI): q = quit
"""
import argparse
import csv
import os
import signal
import sys
import time

import cv2

from tracker3d.config import load_config, load_env, list_cameras
from tracker3d.detector import MultiCamDetector
from tracker3d.fusion import Fusion
from tracker3d.geometry import FloorProjector
from tracker3d.lens import load_lens
from tracker3d.localize import Localizer
from tracker3d.mqtt_output import MqttPositionPublisher
from tracker3d.plan import load_plan
from tracker3d.privacy import PrivacyMasker
from tracker3d.render import CAM_COLORS, camera_tile, draw_map, mosaic


def load_projectors(cams):
    projs = {}
    for cam in cams:
        name, hp = cam["name"], cam["homography_file"]
        projs[name] = None
        if not os.path.exists(hp):
            print(f"[{name}] no homography ({hp}) — run: python calibrate.py --camera {name}")
            continue
        lens, lens_msg = load_lens(cam)
        try:
            projs[name] = FloorProjector.load(hp, lens)
            print(f"[{name}] homography loaded from {hp}")
            # A configured lens can still be ignored if H predates it (lens_status says so).
            status = lens_msg if lens is None or projs[name].lens is not None \
                else projs[name].lens_status
            print(f"[{name}] lens correction {status}")
        except ValueError as e:
            print(f"[{name}] {e}")
    return projs


class PositionLog:
    """Optional CSV of every per-camera and fused position (for before/after
    accuracy checks, e.g. walking a straight tape line). One row per position:
    t, kind (det|track), cam, id, x, y, foot_source."""

    def __init__(self, path):
        self.f = open(path, "w", newline="")
        self.w = csv.writer(self.f)
        self.w.writerow(["t", "kind", "cam", "id", "x", "y", "foot_source"])
        self._flush_t = time.time()

    def write(self, t, dets, tracks):
        for d in dets:
            if d.get("world") is not None:
                self.w.writerow([f"{t:.3f}", "det", d["cam"], d["id"],
                                 f"{d['world'][0]:.3f}", f"{d['world'][1]:.3f}",
                                 d.get("foot_source", "")])
        for g in tracks:
            self.w.writerow([f"{t:.3f}", "track", "+".join(g.get("cams", [])),
                             g.get("gid", ""), f"{g['world'][0]:.3f}",
                             f"{g['world'][1]:.3f}", "coasting" if g.get("coasting") else ""])
        if time.time() - self._flush_t > 1.0:
            self.f.flush()
            self._flush_t = time.time()

    def close(self):
        self.f.close()


class Sources:
    """Uniform access to the synced (GStreamer/NTP) or plain (cv2) capture path."""

    def __init__(self, cfg, cams):
        scfg = cfg.get("sync", {})
        self.synced = scfg.get("enabled", False)
        if self.synced and not all(str(c["source"]).startswith("rtsp://") for c in cams):
            print("[sync] disabled: needs all-RTSP sources.")
            self.synced = False
        if self.synced:
            from tracker3d.sync import SyncGroup
            print(f"[sync] NTP-synced capture (decoder={scfg.get('decoder', 'hw')}) "
                  "connecting all cameras ...")
            self.group = SyncGroup(cams, scfg)
        else:
            from tracker3d.capture import CameraStream
            stuck = scfg.get("reconnect_stuck_ms", 3000) / 1000.0
            self.streams = {c["name"]: CameraStream(c, reconnect_stuck_s=stuck) for c in cams}

    def next(self):
        """(t, {cam: (key, frame)}). `key` changes only when the frame is new; `t` is
        the capture time (synced) or wall time. Returns quickly when nothing arrives."""
        if self.synced:
            t, got = self.group.next_aligned(timeout=0.25)
            return t, got
        got = {}
        for n, s in self.streams.items():
            seq, f = s.read()
            if f is not None:
                got[n] = (seq, f)
        return time.time(), got

    def health_line(self):
        if not self.synced:
            return None
        spread, info = self.group.health()
        parts = [f"{n} lag={d['lag'] * 1000:4.0f}ms ntp={d['with_ntp']}/{d['frames']}"
                 f"{' rc=%d' % d['reconnects'] if d['reconnects'] else ''}"
                 f"{' STUCK' if d['stuck'] else ''}"
                 for n, d in info.items() if d["lag"] is not None]
        return f"[sync] spread={spread * 1000:4.0f}ms | " + " | ".join(parts)

    def release(self):
        if self.synced:
            self.group.release()
        else:
            for s in self.streams.values():
                s.release()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--headless", type=float, default=0.0,
                    help="run N seconds with no GUI, saving snapshots to --save-dir")
    ap.add_argument("--save-dir", default="/tmp")
    ap.add_argument("--log-csv", default=None,
                    help="write every per-camera and fused position to this CSV")
    args = ap.parse_args()
    load_env()
    cfg = load_config(args.config)
    ocfg, fcfg = cfg["output"], cfg.get("fusion", {})

    cams = list_cameras(cfg)
    names = [c["name"] for c in cams]
    colors = {n: CAM_COLORS[i % len(CAM_COLORS)] for i, n in enumerate(names)}
    plan = load_plan(cfg)
    projs = load_projectors(cams)
    detector = MultiCamDetector(cfg, names)
    print(f"[detector] {detector.model_path} (batch of {len(names)}, "
          f"{'fp16' if detector.half else 'fp32'}) warming up ...")
    detector.warmup(n=len(names))
    localizer = Localizer(cfg, plan)
    fuser = Fusion.from_config(fcfg) if fcfg.get("enabled", True) else None
    masker = PrivacyMasker(ocfg.get("privacy", {}))
    mqtt_pub = MqttPositionPublisher(ocfg.get("mqtt", {}))
    mqtt_pub.start()
    sources = Sources(cfg, cams)
    poslog = PositionLog(args.log_csv) if args.log_csv else None

    headless = args.headless > 0.0
    gui = ocfg.get("show_window", True) and not headless
    show_cams = ocfg.get("show_camera_windows", True)
    preview_w = int(ocfg.get("preview_width", 640))
    show_raw, show_blobs = fcfg.get("show_raw_dots", True), fcfg.get("show_blobs", False)

    stop = {"flag": False}
    signal.signal(signal.SIGTERM, lambda *_: stop.update(flag=True))

    last_key = {}                 # cam -> key of the frame last run through the detector
    cache = {}                    # cam -> (people, raw, frame) for that frame
    people_now, blobs = [], []
    n_steps, t_det, t_draw = 0, 0.0, 0.0
    stat_t = snap_t = start = time.time()
    print("\nTracking all cameras... " + ("(headless)" if headless else
                                         "press 'q' to quit." if gui else "Ctrl+C to stop."))
    try:
        while not stop["flag"]:
            t_cap, got = sources.next()
            if got:
                # Only frames we haven't processed go through the detector; a
                # camera whose aligned frame is re-served reuses its last result
                # (running ByteTrack twice on one frame corrupts its motion model).
                new = {n: f for n, (k, f) in got.items() if last_key.get(n) != k}
                t0 = time.perf_counter()
                for n, (people, raw) in detector.detect_and_track(new).items():
                    frame = new[n]
                    h, w = frame.shape[:2]
                    proj = projs[n]
                    if proj is not None:
                        proj.set_frame_size(w, h)
                    localizer.localize(n, people, proj, (w, h))
                    cache[n] = (people, raw, frame)
                    last_key[n] = got[n][0]
                t_det += time.perf_counter() - t0

                all_dets = [p for n in got if n in cache for p in cache[n][0]]
                if fuser is not None:
                    people_now = fuser.update(all_dets, now=t_cap)
                    blobs = fuser.last_blobs
                else:
                    people_now = [p for p in all_dets if p["world"] is not None]
                mqtt_pub.publish_positions(people_now)
                if poslog is not None and new:
                    poslog.write(t_cap, all_dets, people_now)
                n_steps += 1 if new else 0
                if not new and not sources.synced:
                    time.sleep(0.002)          # plain capture: wait for a fresh frame

                # Headless only renders when a snapshot is due (1/s), not every step.
                snap_due = headless and time.time() - snap_t >= 1.0
                if gui or snap_due:
                    t0 = time.perf_counter()
                    canvas = draw_map(plan, people_now, all_dets, colors, fuser is not None,
                                      show_raw, show_blobs, blobs)
                    tiles = None
                    if show_cams or headless:
                        tiles = [camera_tile(cache[n][2], cache[n][0], cache[n][1], masker,
                                             preview_w, f"{n}: {len(cache[n][0])} people",
                                             colors[n])
                                 for n in names if n in cache]
                    if gui:
                        cv2.imshow("floor map", canvas)
                        if tiles:
                            cv2.imshow("cameras", mosaic(tiles))
                    else:
                        snap_t = time.time()
                        cv2.imwrite(f"{args.save_dir}/floor_map.png", canvas)
                        if tiles:
                            cv2.imwrite(f"{args.save_dir}/cameras.png", mosaic(tiles))
                    t_draw += time.perf_counter() - t0

            # UI + stats + exit run every iteration, even when no frame arrived,
            # so the window stays responsive and --headless always terminates.
            if gui and (cv2.waitKey(1) & 0xFF) == ord("q"):
                break
            now = time.time()
            if now - stat_t >= 1.0:
                dt = now - stat_t
                fps = n_steps / dt
                det_ms = 1000 * t_det / max(n_steps, 1)
                draw_ms = 1000 * t_draw / max(n_steps, 1)
                summary = ", ".join(f"P{g['gid']}({g['world'][0]:.1f},{g['world'][1]:.1f})"
                                    for g in people_now if "gid" in g)
                print(f"{fps:5.1f} FPS | detect {det_ms:4.1f} ms | draw {draw_ms:4.1f} ms | "
                      f"{len(people_now)} people {summary}", flush=True)
                hl = sources.health_line()
                if hl:
                    print(hl, flush=True)
                stat_t, n_steps, t_det, t_draw = now, 0, 0.0, 0.0
            if headless and now - start >= args.headless:
                break
    except KeyboardInterrupt:
        pass

    print("shutting down ...", flush=True)
    sources.release()
    if poslog is not None:
        poslog.close()
    mqtt_pub.stop()
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    sys.stdout.flush()
    # GStreamer/NVDEC/TensorRT native threads don't always join at interpreter
    # exit on Jetson ("terminate called without an active exception" + hang).
    # Everything is torn down above, so exit hard.
    os._exit(0)


if __name__ == "__main__":
    main()
