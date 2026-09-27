"""Live check of NTP-synced capture: per-camera lag and cross-camera skew.

    python tools/check_sync.py [--config config.yaml] [--seconds 8]

Uses the `sync:` settings from config.yaml (decoder, protocol, ...). A healthy
setup shows ntp=N/N (every frame NTP-stamped) and a spread of a few hundred ms
at most; `match` is how close each camera's nearest frame is to the common
target instant.
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tracker3d.config import list_cameras, load_config   # noqa: E402
from tracker3d.sync import SyncGroup                      # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--seconds", type=int, default=8)
    args = ap.parse_args()
    cfg = load_config(args.config)
    scfg = cfg.get("sync", {})
    group = SyncGroup(list_cameras(cfg), scfg)
    print(f"opened {len(group.streams)} cameras (decoder={scfg.get('decoder', 'hw')}, "
          f"protocol={scfg.get('protocol', 'udp')}); warming up ...")
    time.sleep(3.0)
    try:
        for _ in range(args.seconds):
            time.sleep(1.0)
            lasts = {n: s.latest_capture_t() for n, s in group.streams.items()}
            have = {n: t for n, t in lasts.items() if t is not None}
            if len(have) < len(lasts):
                print("waiting for:", [n for n in lasts if lasts[n] is None])
                continue
            target = min(have.values())
            line = []
            for n, s in group.streams.items():
                got = s.frame_at(target, tol=0.20)
                st = s.stats()
                match = f"{(got[0] - target) * 1000:+4.0f}ms" if got else " miss"
                line.append(f"{n}: lag={(time.time() - have[n]) * 1000:4.0f}ms match={match} "
                            f"ntp={st['with_ntp']}/{st['frames']}")
            spread = (max(have.values()) - min(have.values())) * 1000
            print(f"spread={spread:4.0f}ms | " + " | ".join(line))
    finally:
        group.release()
        sys.stdout.flush()
        os._exit(0)          # GStreamer threads may not join cleanly on Jetson


if __name__ == "__main__":
    main()
