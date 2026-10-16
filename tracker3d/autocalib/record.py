"""Live NTP walk recorder. Imported only by the record command."""
from collections import Counter
from itertools import combinations
import time

import cv2
import numpy as np

from tracker3d.config import list_cameras
from tracker3d.lens import load_lens
from .session import SessionWriter


def record_walk(cfg, out, seconds=120, headless=False):
    if seconds <= 0:
        raise ValueError("recording duration must be positive")
    cams = list_cameras(cfg)
    if not cfg.get("sync", {}).get("enabled") or not all(
            str(c["source"]).startswith("rtsp://") for c in cams):
        raise ValueError("calibration recording requires sync.enabled and all RTSP sources")
    snapshots = {}
    for c in cams:
        lens, status = load_lens(c)
        if lens is None:
            raise ValueError(f"{c['name']}: calibrate the lens before recording a walk")
        snapshots[c["name"]] = {"lens": lens.to_dict(),
                                 "calibration_quality": {k: lens.meta.get(k)
                                                          for k in ("rms_px", "coverage", "stability")}}
        print(f"[{c['name']}] lens {status}")
        stability = lens.meta.get("stability") or {}
        if stability.get("fx_diff_pct", 0) > 1:
            print(f"[{c['name']}] lens stability warning: check edges before final validation")
    if len(snapshots) != len(cams):
        raise ValueError("camera names must be unique")
    from tracker3d.detector import MultiCamDetector
    from tracker3d.sync import SyncGroup
    from tracker3d.privacy import PrivacyMasker
    from tracker3d.render import camera_tile, mosaic

    detector = MultiCamDetector(cfg, list(snapshots))
    detector.warmup(n=len(cams))
    # Always mask recorder previews; only keypoints are written to the session.
    masker = PrivacyMasker({**cfg.get("output", {}).get("privacy", {}), "enabled": True})
    writer = SessionWriter(out, snapshots, cfg["detector"])
    group, complete = None, False
    counts, pairs, coverage = Counter(), Counter(), {n: set() for n in snapshots}
    seen, seen_pairs, cache = set(), set(), {}
    marker, last_target = 0, None
    print("One person only. Walk all overlaps in several directions, pausing upright for 1–2 s "
          "at >=12 well-spread positions with feet visible. SPACE marks a named map/checkpoint "
          "pause; q finishes. Headless sessions can identify pauses by timestamps in a points JSON.")
    start, last_stats = time.monotonic(), time.monotonic()
    try:
        group = SyncGroup(cams, cfg["sync"])
        while time.monotonic() - start < seconds:
            target, got = group.next_aligned(timeout=0.25, with_metadata=True)
            new, keys = {}, {}
            for n, (t, frame, meta) in got.items():
                key = (n, meta["generation"], meta["frame_id"])
                if key not in seen:
                    new[n], keys[n] = frame, key
                    seen.add(key)
            detected = detector.detect_and_track(new)
            for n, (tracked, raw) in detected.items():
                t, frame, meta = got[n]
                h, w = frame.shape[:2]
                if [w, h] != snapshots[n]["lens"]["image_size"]:
                    raise ValueError(f"{n}: stream size differs from its lens calibration")
                ids = {tuple(p["bbox"]): p["id"] for p in tracked}
                people = []
                for p in raw:
                    person = {"bbox": list(p["bbox"]), "conf": p["conf"],
                              "id": ids.get(tuple(p["bbox"]))}
                    if p.get("kxy") is not None:
                        person.update(kxy=np.asarray(p["kxy"]).tolist(),
                                      kconf=np.asarray(p["kconf"]).tolist())
                    people.append(person)
                writer.append({"kind": "frame", "camera": n, "t": t,
                               "image_size": [w, h], **meta, "people": people})
                cache[n] = (frame, raw, t, meta, keys[n])
                counts[n] += 1
            qualified = []
            for n in got:
                if n not in cache:
                    continue
                frame, raw, t, meta, key = cache[n]
                if (meta["timestamp_source"] != "ntp" or len(raw) != 1
                        or raw[0].get("kconf") is None or min(raw[0]["kconf"][15:17]) < 0.65):
                    continue
                qualified.append(n)
                h, w = frame.shape[:2]
                for x, y in raw[0]["kxy"][15:17]:
                    coverage[n].add((min(7, max(0, int(x / w * 8))),
                                     min(5, max(0, int(y / h * 6)))))
            for a, b in combinations(sorted(qualified), 2):
                pair_key = (cache[a][4], cache[b][4])
                if pair_key not in seen_pairs:
                    pairs[f"{a}/{b}"] += 1
                    seen_pairs.add(pair_key)
            if target is not None:
                last_target = target
            if not headless and cache:
                tiles = [camera_tile(fr, [], raw, masker, 480,
                                     f"{n}: {meta['timestamp_source']} | {len(coverage[n])}/48 cells",
                                     (0, 220, 255)) for n, (fr, raw, t, meta, key) in cache.items()]
                cv2.imshow("Calibration walk: SPACE marker, q finish", mosaic(tiles))
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord(" ") and last_target is not None:
                    marker += 1
                    writer.append({"kind": "marker", "name": f"M{marker}", "t": last_target})
                    print(f"Marked M{marker} at capture time {last_target:.3f}; stay still for another second")
            if time.monotonic() - last_stats >= 5:
                print(f"{time.monotonic() - start:.0f}s | frames {dict(counts)} | "
                      f"candidate ankle pairs {dict(pairs)} | "
                      f"coverage { {n: len(v) for n, v in coverage.items()} }")
                last_stats = time.monotonic()
            if not counts and time.monotonic() - start > 20:
                raise ValueError("no camera frames after 20 seconds; check streams and NTP")
        complete = True
    except KeyboardInterrupt:
        complete = True
    finally:
        writer.close(complete=complete)
        if group is not None:
            group.release()
        if not headless:
            cv2.destroyAllWindows()
    print(f"Saved {writer.count} records in {out}. Solve checks timing, upright poses and overlap coverage.")
