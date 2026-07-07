"""Cross-camera fusion: merge per-camera detections that land near each other
on the floorplan into a single global person with a stable ID.

Per frame:
  1. cluster current detections (from all cameras): same person if their foot
     points are within merge_distance_m -- or, with `use_blob`, if their floor
     trapezoids overlap (see src/groundblob.py), which rejoins a camera whose
     foot point drifted because it couldn't see the feet
  2. with `use_blob`, refine each cluster's position to where the cameras'
     trapezoids intersect (else keep the foot-point mean)
  3. match clusters to existing global tracks (nearest, within match_gate_m)
  4. smooth position (EMA), spawn new tracks, age out stale ones

Result: one marker per real person, even when several cameras see them, and the
marker survives a few frames of occlusion (coasts at last position).
"""
import time
import numpy as np

from groundblob import intersect_blobs, convex_intersection


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
                 max_age_s=1.5, smoothing=0.5, use_blob=False,
                 blob_merge_gate_m=2.5):
        self.merge = float(merge_distance_m)
        self.gate = float(match_gate_m)
        self.max_age = float(max_age_s)
        self.s = float(smoothing)
        self.use_blob = bool(use_blob)
        self.blob_gate = float(blob_merge_gate_m)
        self.tracks = {}
        self.next_id = 1
        self.last_blobs = []   # intersection polygons from the last update (viz)

    def _merges(self, o, w, poly):
        """Does detection (point w, trapezoid poly) belong to observation o?

        A detection joins when its foot point is within merge_distance, OR -- with
        blobs on -- when its trapezoid overlaps one already in the cluster and the
        foot points are within the (looser) blob gate. The second rule is what
        catches a camera that can't see the feet: its foot point drifts too far to
        cluster by distance, but its tall trapezoid still overlaps the others."""
        d = np.linalg.norm(o["pos"] - w)
        if d <= self.merge:
            return True
        if not self.use_blob or poly is None or d > self.blob_gate:
            return False
        return any(convex_intersection(poly, mp) is not None
                   for mp in o["polys"] if mp is not None)

    def _cluster(self, dets):
        """Greedy-merge detections that share a person (see `_merges`)."""
        obs = []
        for d in dets:
            w = np.array(d["world"], dtype=float)
            poly = d.get("floor_poly")
            for o in obs:
                # A single camera that split two people into two boxes is the
                # strongest evidence they ARE two people: never merge detections
                # from the same camera (the blob-overlap rule is cross-camera only).
                if d["cam"] in o["cams"]:
                    continue
                if self._merges(o, w, poly):
                    o["members"].append((d["cam"], d["id"]))
                    o["cams"].add(d["cam"])
                    o["polys"].append(poly)
                    k = len(o["members"])
                    o["pos"] = (o["pos"] * (k - 1) + w) / k   # running mean
                    break
            else:
                obs.append({"pos": w, "cams": {d["cam"]},
                            "members": [(d["cam"], d["id"])],
                            "polys": [poly]})
        return obs

    def _refine_blob(self, obs):
        """Replace each multi-camera observation's running-mean position with the
        centroid where the cameras' floor trapezoids intersect. The point-mean is
        kept whenever the trapezoids don't overlap (e.g. calibration error)."""
        self.last_blobs = []
        for o in obs:
            centroid, poly = intersect_blobs(o["polys"])
            if centroid is not None:
                o["pos"] = centroid
                self.last_blobs.append(poly)

    def update(self, detections, now=None):
        now = time.time() if now is None else now
        dets = [d for d in detections if d.get("world") is not None]
        obs = self._cluster(dets)
        if self.use_blob:
            self._refine_blob(obs)

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
