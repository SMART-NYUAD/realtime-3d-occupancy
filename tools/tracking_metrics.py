"""Summarise a tracker position log (track.py --log-csv) into tracking-quality numbers.

    python tools/tracking_metrics.py walk.csv
    python tools/tracking_metrics.py seated.csv --expected 2 --json seated_metrics.json
    python tools/tracking_metrics.py a.csv b.csv          # side by side (e.g. fixes off / on)

Counts only people that would be published to the digital twin (seen within
--coast seconds, or settled). Reported:
  frames, duration           how much was logged
  people per frame           mean, and % of frames equal to --expected (if given)
  duplicate-dot frames       % of frames with two published people closer than --dup-m
  distinct IDs, births/min   how often new IDs appear (lower = steadier identities)
  short-lived IDs            IDs shown for less than --short-s (flicker / ghosts)
  seated jitter (cm)         per settled person: std dev of their position while settled
  seated continuity          per settled person: share of time from first settled to
                             last seen that they were on the map
  posture share              how often people were sitting / standing / unknown
Older logs without posture/settled columns still work (those metrics are skipped).
"""
import argparse
import csv
import json
from collections import defaultdict

import numpy as np


def load(path):
    frames = defaultdict(list)
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if row["kind"] != "track":
                continue
            frames[float(row["t"])].append(row)
    return dict(sorted(frames.items()))


def published(row, coast_s):
    settled = row.get("settled") in ("1", "True", "true")
    last_seen = float(row["last_seen_s"]) if row.get("last_seen_s") not in (None, "") else (
        1e9 if row.get("foot_source") == "coasting" else 0.0)
    return settled or last_seen <= coast_s


def metrics(path, expected=None, coast_s=0.5, dup_m=1.0, short_s=2.0):
    frames = load(path)
    if not frames:
        raise SystemExit(f"{path}: no track rows (was it written with track.py --log-csv?)")
    times = np.array(list(frames))
    duration = float(times[-1] - times[0]) if len(times) > 1 else 0.0
    counts, dups = [], 0
    seen = defaultdict(list)                  # gid -> times published
    settled_pos = defaultdict(list)
    first_settled = {}
    postures = defaultdict(int)
    for t, rows in frames.items():
        pub = [r for r in rows if published(r, coast_s)]
        counts.append(len(pub))
        xy = np.array([[float(r["x"]), float(r["y"])] for r in pub]) if pub else np.empty((0, 2))
        if len(xy) > 1:
            d = np.linalg.norm(xy[:, None] - xy[None], axis=2)
            if (d[np.triu_indices(len(xy), 1)] < dup_m).any():
                dups += 1
        for r in pub:
            gid = r["id"]
            seen[gid].append(t)
            postures[r.get("posture") or "unknown"] += 1
            if r.get("settled") in ("1", "True", "true"):
                settled_pos[gid].append((float(r["x"]), float(r["y"])))
                first_settled.setdefault(gid, t)
    counts = np.array(counts)
    lifetimes = {g: ts[-1] - ts[0] for g, ts in seen.items()}
    step = float(np.median(np.diff(times))) if len(times) > 1 else 0.1
    continuity = {}
    for gid, t0 in first_settled.items():
        ts = np.array(seen[gid])
        span = ts[-1] - t0
        if span > 5 * step:
            continuity[gid] = round(float(np.sum(ts >= t0) * step / span), 3)
    out = {
        "file": path,
        "frames": int(len(times)),
        "duration_s": round(duration, 1),
        "people_per_frame_mean": round(float(counts.mean()), 3),
        "duplicate_dot_frames_pct": round(100.0 * dups / len(times), 2),
        "distinct_ids": len(seen),
        "id_births_per_min": round(len(seen) / max(duration / 60.0, 1e-9), 2),
        "short_lived_ids": int(sum(v < short_s for v in lifetimes.values())),
        "seated_jitter_cm": {g: round(float(np.std(p, axis=0).max() * 100), 2)
                             for g, p in settled_pos.items() if len(p) > 10},
        "seated_continuity": continuity,
        "posture_share_pct": {k: round(100.0 * v / max(sum(postures.values()), 1), 1)
                              for k, v in sorted(postures.items())},
    }
    if expected is not None:
        out["frames_with_expected_count_pct"] = round(100.0 * np.mean(counts == expected), 2)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("logs", nargs="+", help="CSV files written by track.py --log-csv")
    ap.add_argument("--expected", type=int, default=None, help="true number of people present")
    ap.add_argument("--coast", type=float, default=0.5, help="publish window for unseen people (s)")
    ap.add_argument("--dup-m", type=float, default=1.0, help="two published people closer than this = duplicate")
    ap.add_argument("--short-s", type=float, default=2.0, help="IDs shown less than this count as short-lived")
    ap.add_argument("--json", default=None, help="also write the numbers to this JSON file")
    args = ap.parse_args()
    results = [metrics(p, args.expected, args.coast, args.dup_m, args.short_s) for p in args.logs]
    keys = [k for k in results[0] if k != "file"]
    width = max(len(k) for k in keys) + 2
    print(" " * width + " | ".join(r["file"][-28:].rjust(28) for r in results))
    for k in keys:
        print(k.ljust(width) + " | ".join(str(r.get(k, "")).rjust(28)[:60] for r in results))
    if args.json:
        with open(args.json, "w") as f:
            json.dump(results if len(results) > 1 else results[0], f, indent=2)
        print(f"saved {args.json}")


if __name__ == "__main__":
    main()
