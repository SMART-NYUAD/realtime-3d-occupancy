"""Cross-camera fusion: merge per-camera detections that land near each other
on the floorplan into a single global person with a stable ID.

Per frame:
  1. cluster current detections (from all cameras): same person if their foot
     points are within merge_distance_m -- or, with `use_blob`, if their floor
     trapezoids overlap (see src/groundblob.py), which rejoins a camera whose
     foot point drifted because it couldn't see the feet
  2. resolve each cluster to one position by INVERSE-VARIANCE weighting of the
     members' foot estimates -- an occluded (high-sigma) foot barely moves the
     result, even if the detector was confident. With `use_blob`, refine to the
     trapezoid intersection only when the feet are genuinely uncertain.
  3. match clusters to existing global tracks (nearest, within match_gate_m),
     capping how far a track may move per frame (a bad frame can't drag it)
  4. spawn/age tracks: a new track must persist `n_init` frames before it shows
     (kills flicker ghosts) and is not born within `dup_suppress_m` of an
     existing track (kills duplicate dots)

Result: one marker per real person, even when several cameras see them, and the
marker survives a few frames of occlusion (coasts at last position).
"""
import time
import numpy as np

from .groundblob import intersect_blobs, convex_intersection

_SIG_MIN = 1e-3   # floor on a foot-estimate sigma (m) so 1/sigma^2 stays finite


class _Track:
    __slots__ = ("gid", "pos", "last_t", "cams", "members", "sigma",
                 "hits", "confirmed")

    def __init__(self, gid, pos, now, cams, members, sigma):
        self.gid = gid
        self.pos = pos            # np.array([x, y]) meters
        self.last_t = now
        self.cams = cams          # set of camera names contributing now
        self.members = members    # list of (cam, per_camera_id)
        self.sigma = sigma        # fused position uncertainty (m)
        self.hits = 1             # frames matched (for confirmation)
        self.confirmed = False    # shown only once it has survived n_init frames


class Fusion:
    def __init__(self, merge_distance_m=0.6, match_gate_m=1.0,
                 max_age_s=1.5, smoothing=0.5, use_blob=False,
                 blob_merge_gate_m=2.5, conf_drop_ratio=0.5,
                 blob_min_sigma_m=0.15, n_init=3, max_age_tentative_s=0.4,
                 dup_suppress_m=1.0, max_speed_mps=2.5, default_sigma_m=0.4):
        self.merge = float(merge_distance_m)
        self.gate = float(match_gate_m)
        self.max_age = float(max_age_s)
        self.s = float(smoothing)
        self.use_blob = bool(use_blob)
        self.blob_gate = float(blob_merge_gate_m)
        # Members are weighted by inverse foot-sigma variance (occluded feet ->
        # large sigma -> tiny weight); conf_drop_ratio additionally drops members
        # whose box confidence is far below the cluster's best (likely false hits).
        self.conf_drop_ratio = float(conf_drop_ratio)
        self.blob_min_sigma = float(blob_min_sigma_m)
        # Track lifecycle gates.
        self.n_init = int(n_init)
        self.max_age_tent = float(max_age_tentative_s)
        self.dup_suppress = float(dup_suppress_m)
        self.max_speed = float(max_speed_mps)
        self.default_sigma = float(default_sigma_m)
        self.tracks = {}
        self.next_id = 1
        self.last_blobs = []   # intersection polygons from the last update (viz)

    @classmethod
    def from_config(cls, f):
        """Build from the `fusion:` block of config.yaml."""
        return cls(f.get("merge_distance_m", 1.6), f.get("match_gate_m", 1.8),
                   f.get("max_age_s", 1.5), f.get("smoothing", 0.35),
                   use_blob=f.get("use_blob", True),
                   blob_merge_gate_m=f.get("blob_merge_gate_m", 3.0),
                   conf_drop_ratio=f.get("conf_drop_ratio", 0.5),
                   blob_min_sigma_m=f.get("blob_min_sigma_m", 0.15),
                   n_init=f.get("n_init", 3),
                   max_age_tentative_s=f.get("max_age_tentative_s", 0.4),
                   dup_suppress_m=f.get("dup_suppress_m", 1.0),
                   max_speed_mps=f.get("max_speed_mps", 2.5))

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
            c = float(d.get("conf", 1.0))
            sig = float(d.get("foot_sigma_m") or self.default_sigma)
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
                    o["pts"].append(w)
                    o["confs"].append(c)
                    o["sigmas"].append(sig)
                    k = len(o["members"])
                    # Unweighted running mean, used only as the gate reference for
                    # the next candidate. The reported position is recomputed
                    # (weighted) in `_resolve` once the cluster closes.
                    o["pos"] = (o["pos"] * (k - 1) + w) / k
                    break
            else:
                obs.append({"pos": w, "cams": {d["cam"]},
                            "members": [(d["cam"], d["id"])],
                            "polys": [poly], "pts": [w], "confs": [c],
                            "sigmas": [sig]})
        for o in obs:
            self._resolve(o)
        return obs

    def _resolve(self, o):
        """Collapse a cluster's per-camera detections into one position + a fused
        uncertainty, weighting each camera by how well it can localize the feet.

        Inverse-variance weighting on the per-detection foot sigma, so a
        confident-but-occluded view (large sigma) barely moves the dot.
        `conf_drop_ratio` is a secondary relative drop (and nulls dropped
        trapezoids so they can't veto the blob intersection). A lone detection is
        passed through unchanged.
        """
        pts = np.asarray(o["pts"], dtype=float)
        sig = np.clip(np.asarray(o["sigmas"], dtype=float), _SIG_MIN, None)
        confs = np.asarray(o["confs"], dtype=float)

        w = 1.0 / sig**2

        # Secondary: relative confidence dominance drop.
        if self.conf_drop_ratio > 0.0 and len(confs) > 1:
            keep = confs >= self.conf_drop_ratio * confs.max()
            if not keep.any():
                keep[confs.argmax()] = True
            w = w * keep
            o["polys"] = [p if k else None for p, k in zip(o["polys"], keep)]

        if not np.any(w > 0):
            w = np.ones(len(pts))
        o["pos"] = np.average(pts, axis=0, weights=w)
        # Inverse-variance fusion of the kept members' variances.
        kept = w > 0
        o["sigma"] = float(1.0 / np.sqrt(np.sum(1.0 / sig[kept]**2)))

    def _refine_blob(self, obs):
        """Replace an observation's weighted-mean position with the trapezoid
        intersection centroid -- but ONLY when the feet are genuinely uncertain
        (fused sigma above blob_min_sigma). When a camera sees the ankles clearly
        the weighted point is better than the trapezoid, so leave it alone."""
        self.last_blobs = []
        for o in obs:
            if self.blob_min_sigma > 0.0 and o.get("sigma", 1.0) < self.blob_min_sigma:
                continue
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
                dt = max(now - tr.last_t, 1e-3)
                ema = self.s * o["pos"] + (1 - self.s) * tr.pos
                # Motion cap: a person can't teleport, so limit how far the marker
                # moves in one frame. A single drifted (occluded) observation then
                # only nudges the dot and it recovers next frame, instead of being
                # yanked across the room.
                move = ema - tr.pos
                d = float(np.linalg.norm(move))
                max_step = self.max_speed * dt
                if self.max_speed > 0.0 and d > max_step:
                    ema = tr.pos + move * (max_step / d)
                tr.pos = ema
                tr.last_t = now
                tr.cams = o["cams"]
                tr.members = o["members"]
                tr.sigma = o["sigma"]
                tr.hits += 1
                if tr.hits >= self.n_init:
                    tr.confirmed = True
            else:
                # Spawn gate: don't birth a track right next to an existing one --
                # that's almost always the same person seen a second way (the
                # source of duplicate dots), not a new person.
                if self.dup_suppress > 0.0 and any(
                        np.linalg.norm(tr.pos - o["pos"]) <= self.dup_suppress
                        for tr in self.tracks.values()):
                    continue
                gid = self.next_id
                self.next_id += 1
                self.tracks[gid] = _Track(gid, o["pos"], now, o["cams"],
                                          o["members"], o["sigma"])

        # age out stale tracks (unconfirmed ones die fast so flicker never sticks)
        for gid in [g for g, t in self.tracks.items()
                    if now - t.last_t > (self.max_age if t.confirmed
                                         else self.max_age_tent)]:
            del self.tracks[gid]

        out = []
        for tr in self.tracks.values():
            if not tr.confirmed:
                continue
            out.append({
                "gid": tr.gid,
                "world": (float(tr.pos[0]), float(tr.pos[1])),
                "cams": sorted(tr.cams),
                "coasting": tr.last_t < now,   # not seen this frame (occluded)
            })
        return out
