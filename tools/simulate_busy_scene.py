"""Compare the old and new tracking on a simulated busy scene.

Uses the simulated lab from tests/test_tracking.py (three calibrated cameras with a
Yi-like lens, 2.55 m high). Two people sit at desks 1 m apart for 60 s, each camera
misses each of them in 25% of frames, and a third person walks past for 12 s. The
same frames go through the tracking with the fixes on (new) and off (old):

    python tools/simulate_busy_scene.py
    python tools/simulate_busy_scene.py --seeds 1 2 3 --miss 0.5

Reported per run (first 5 s skipped as warm-up):
  wrong count   frames where the number of published people is not the true number
  IDs           distinct track IDs handed out (3 people = 3 is ideal)
  wobble cm     std dev of each seated person's dot
Simulated detections are idealised, so these show each fix works as designed, not
the exact lab performance; use tools/tracking_metrics.py on real logs for that.
"""
import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))
import test_tracking as sim                    # noqa: E402
from tracker3d.fusion import publishable       # noqa: E402

FRAMES, WARMUP = 600, 50                       # 60 s at 10 frames/s, 5 s warm-up
WALK = (200, 320)                              # frames the walker is in view
SEAT_RADIUS = 0.35                             # walker passes 0.5 m from seat B


def scene(seed, miss):
    rng = np.random.default_rng(seed)
    a, b = sim.SEAT, sim.SEAT + np.array([0.0, 1.0])
    frames = []
    for k in range(FRAMES):
        frame = {c: [] for c in sim.POSES}
        for c in sim.POSES:
            if rng.random() > miss:
                frame[c].append(sim.detection(c, a, "sitting", pid=1, desk=True, rng=rng))
            if rng.random() > miss:
                frame[c].append(sim.detection(c, b, "sitting", pid=2, desk=True, rng=rng))
            if WALK[0] <= k < WALK[1]:
                xy = np.array([1.5 + 0.03 * (k - WALK[0]), 5.7])
                frame[c].append(sim.detection(c, xy, "standing", pid=3, rng=rng))
        frames.append(frame)
    return frames, (a, b)


def evaluate(frames, seats, cfg, fusion_cfg):
    _, history = sim.run(frames, cfg, fusion_cfg)
    # the walker needs the confirmation time before they count as present
    expected = [3 if WALK[0] + 5 <= k < WALK[1] else 2 for k in range(FRAMES)]
    wrong = sum(len(publishable(history[k][1])) != expected[k] for k in range(WARMUP, FRAMES))
    ids = len({g["gid"] for _, tracks, _ in history for g in tracks})
    wobble = []
    for seat in seats:
        pts = np.array([g["world"] for k in range(WARMUP, FRAMES) for g in history[k][1]
                        if np.linalg.norm(np.subtract(g["world"], seat)) < SEAT_RADIUS])
        wobble.append(round(float(pts.std(axis=0).max() * 100), 1) if len(pts) else None)
    return wrong, ids, wobble


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seeds", type=int, nargs="+", default=[11, 12, 13, 14, 15],
                    help="random seeds, one run each")
    ap.add_argument("--miss", type=float, default=0.25,
                    help="chance a camera misses a seated person in a frame")
    args = ap.parse_args()
    n = FRAMES - WARMUP
    for seed in args.seeds:
        frames, seats = scene(seed, args.miss)
        new = evaluate(frames, seats, sim.NEW_CFG, None)
        old = evaluate(frames, seats, sim.OLD_CFG, sim.OLD_FUSION)
        print(f"seed {seed}: new wrong count {new[0]:3d}/{n}, IDs {new[1]}, wobble cm {new[2]} | "
              f"old wrong count {old[0]:3d}/{n}, IDs {old[1]}, wobble cm {old[2]}")


if __name__ == "__main__":
    main()
