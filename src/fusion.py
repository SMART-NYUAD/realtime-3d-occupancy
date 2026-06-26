"""Cross-camera fusion: merge per-camera detections that land near each other
on the floorplan into a single global person with a stable ID.

Per frame:
  1. cluster current world positions (from all cameras) within merge_distance_m
  2. match clusters to existing global tracks (nearest, within match_gate_m)
  3. smooth position (EMA), spawn new tracks, age out stale ones

Result: one marker per real person, even when several cameras see them, and the
marker survives a few frames of occlusion (coasts at last position).
"""
import time
import numpy as np


class _Track:
    __slots__ = ("gid", "pos", "last_t", "cams", "members")

    def __init__(self, gid, pos, now, cams, members):
        self.gid = gid
        self.pos = pos            # np.array([x, y]) meters
        self.last_t = now
        self.cams = cams          # set of camera names contributing now
        self.members = members    # list of (cam, per_camera_id)


class Fusion:
    def __init__(self, merge_distance_m=0.6, match_gate_m=1.0,
                 max_age_s=1.5, smoothing=0.5):
        self.merge = float(merge_distance_m)
        self.gate = float(match_gate_m)
        self.max_age = float(max_age_s)
        self.s = float(smoothing)
        self.tracks = {}
        self.next_id = 1

    def _cluster(self, dets):
        """Greedy-merge detections whose world points are within merge_distance."""
        obs = []
        for d in dets:
            w = np.array(d["world"], dtype=float)
            for o in obs:
                if np.linalg.norm(o["pos"] - w) <= self.merge:
                    o["members"].append((d["cam"], d["id"]))
                    o["cams"].add(d["cam"])
                    k = len(o["members"])
                    o["pos"] = (o["pos"] * (k - 1) + w) / k   # running mean
                    break
            else:
                obs.append({"pos": w, "cams": {d["cam"]},
                            "members": [(d["cam"], d["id"])]})
        return obs

    def update(self, detections, now=None):
        now = time.time() if now is None else now
        dets = [d for d in detections if d.get("world") is not None]
        obs = self._cluster(dets)

        # greedy nearest-neighbour match of observations to existing tracks
        pairs = []
        for oi, o in enumerate(obs):
            for gid, tr in self.tracks.items():
                dist = np.linalg.norm(tr.pos - o["pos"])
                if dist <= self.gate:
                    pairs.append((dist, oi, gid))
        pairs.sort(key=lambda p: p[0])
        obs_done, gid_done, obs_to_gid = set(), set(), {}
        for dist, oi, gid in pairs:
            if oi in obs_done or gid in gid_done:
                continue
            obs_done.add(oi); gid_done.add(gid); obs_to_gid[oi] = gid

        for oi, o in enumerate(obs):
            if oi in obs_to_gid:
                tr = self.tracks[obs_to_gid[oi]]
                tr.pos = self.s * o["pos"] + (1 - self.s) * tr.pos
                tr.last_t = now
                tr.cams = o["cams"]
                tr.members = o["members"]
            else:
                gid = self.next_id
                self.next_id += 1
                self.tracks[gid] = _Track(gid, o["pos"], now, o["cams"], o["members"])

        # age out stale tracks
        for gid in [g for g, t in self.tracks.items() if now - t.last_t > self.max_age]:
            del self.tracks[gid]

        out = []
        for tr in self.tracks.values():
            out.append({
                "gid": tr.gid,
                "world": (float(tr.pos[0]), float(tr.pos[1])),
                "cams": sorted(tr.cams),
                "coasting": tr.last_t < now,   # not seen this frame (occluded)
            })
        return out
