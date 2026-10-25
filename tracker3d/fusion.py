"""Cross-camera fusion: merge per-camera detections that land near each other
on the floorplan into a single global person with a stable ID.

Per frame:
  1. cluster current detections (from all cameras): same person if their floor
     estimates are within merge_distance_m -- or, with `use_blob`, if their floor
     trapezoids overlap (only fallback estimates carry one)
  2. resolve each cluster to one position by INVERSE-VARIANCE weighting of the
     members' estimates; with `triangulate`, the shoulders seen by two or more
     cameras are located directly where the lines of sight cross, which also
     measures shoulder height (posture). Blob intersection refines only clusters
     made of fallback guesses
  3. match clusters to tracks: first by the per-camera tracker IDs the track
     owned last frame (`identity_matching`), then nearest within match_gate_m
     (a settled seated person only accepts nearby observations)
  4. spawn/age tracks: a new track must be seen for `confirm_s` before it shows
     and is not born within `dup_suppress_m` of an existing one. Seated people who
     sit still become "settled": their position is a median with a dead-band (it
     does not wander) and they are kept for `settled_max_age_s` when undetected
  5. a torso / head estimate whose assumed height was wrong still joins when its
     line of sight passes over a confident ankle estimate (`ray_merge_m`)
  6. merge duplicates (`dup_merge_m`): when one cluster holds camera IDs owned by
     two tracks, or two tracks stay close and move together for `dup_merge_s`
     while no camera sees both, they are one person: the younger track is folded
     into the older one, camera ID ownership included

New behaviour defaults OFF when constructing Fusion directly (legacy behaviour);
Fusion.from_config turns it on per config.yaml.
"""
import time
from collections import deque
from itertools import combinations

import numpy as np

from .geometry import closest_point_between_rays
from .groundblob import intersect_blobs, convex_intersection

_SIG_MIN = 1e-3   # floor on a foot-estimate sigma (m) so 1/sigma^2 stays finite
FALLBACK_SOURCES = ("knees", "hips", "box")


class _Track:
    __slots__ = ("gid", "pos", "last_t", "born_t", "cams", "members", "sigma", "hits",
                 "confirmed", "posture", "shoulder_h", "heights", "votes", "history",
                 "settled", "far_since", "legs_hidden")

    def __init__(self, gid, pos, now, cams, members, sigma):
        self.gid = gid
        self.pos = pos            # np.array([x, y]) meters
        self.last_t = now
        self.born_t = now
        self.cams = cams          # set of camera names contributing now
        self.members = set(members)   # {(cam, per_camera_id)} from the last match
        self.sigma = sigma        # fused position uncertainty (m)
        self.hits = 1
        self.confirmed = False    # shown only once it has survived confirmation
        self.posture = "unknown"  # sitting | standing | unknown
        self.shoulder_h = None    # recent measured shoulder height (m) or None
        self.heights = deque(maxlen=300)   # (t, triangulated shoulder z)
        self.votes = deque(maxlen=300)     # (t, "sitting" | "standing")
        self.history = deque(maxlen=600)   # (t, observed xy)
        self.settled = False
        self.far_since = None     # dead-band: since when observations stay far away
        self.legs_hidden = False


class Fusion:
    def __init__(self, merge_distance_m=0.6, match_gate_m=1.0,
                 max_age_s=1.5, smoothing=0.5, use_blob=False,
                 blob_merge_gate_m=2.5, conf_drop_ratio=0.5,
                 blob_min_sigma_m=0.15, n_init=3, max_age_tentative_s=0.4,
                 dup_suppress_m=1.0, max_speed_mps=2.5, default_sigma_m=0.4,
                 identity_matching=False, identity_gate_m=3.0, triangulate=False,
                 tri_max_gap_m=0.25, tri_min_angle_deg=25.0, sitting_shoulder_max_m=1.20,
                 blob_only_fallback=False,
                 seated_persistence=False, settled_max_age_s=120.0, settled_gate_m=0.6,
                 standup_gate_m=0.35,
                 settled_smoothing=False, settle_speed_mps=0.2, settle_window_s=3.0,
                 deadband_m=0.3, deadband_s=1.0, confirm_s=0.0,
                 dup_merge_m=0.0, dup_merge_s=1.0, dup_merge_speed_mps=0.4,
                 ray_merge_m=0.0, shoulder_band_m=(0.9, 1.75)):
        self.merge = float(merge_distance_m)
        self.gate = float(match_gate_m)
        self.max_age = float(max_age_s)
        self.s = float(smoothing)
        self.use_blob = bool(use_blob)
        self.blob_gate = float(blob_merge_gate_m)
        self.conf_drop_ratio = float(conf_drop_ratio)
        self.blob_min_sigma = float(blob_min_sigma_m)
        self.n_init = int(n_init)
        self.max_age_tent = float(max_age_tentative_s)
        self.dup_suppress = float(dup_suppress_m)
        self.max_speed = float(max_speed_mps)
        self.default_sigma = float(default_sigma_m)
        self.identity = bool(identity_matching)
        self.identity_gate = float(identity_gate_m)
        self.triangulate = bool(triangulate)
        self.tri_gap = float(tri_max_gap_m)
        self.tri_min_sin = float(np.sin(np.radians(tri_min_angle_deg)))
        self.sit_max = float(sitting_shoulder_max_m)
        self.blob_only_fallback = bool(blob_only_fallback)
        self.persistence = bool(seated_persistence)
        self.settled_max_age = float(settled_max_age_s)
        self.settled_gate = float(settled_gate_m)
        self.standup_gate = float(standup_gate_m)
        self.settled_smoothing = bool(settled_smoothing)
        self.settle_speed = float(settle_speed_mps)
        self.settle_window = float(settle_window_s)
        self.deadband = float(deadband_m)
        self.deadband_s = float(deadband_s)
        self.confirm_s = float(confirm_s)
        self.dup_merge = float(dup_merge_m)
        self.dup_merge_s = float(dup_merge_s)
        self.dup_merge_speed = float(dup_merge_speed_mps)
        self._close_pairs = {}    # (older gid, younger gid) -> [first close t, last close t]
        self.ray_merge = float(ray_merge_m)
        self.shoulder_band = tuple(shoulder_band_m)
        self.tracks = {}
        self.next_id = 1
        self.last_blobs = []   # intersection polygons from the last update (viz)
        # (cam, per-camera id) -> recent (t, gid) votes: which global person each
        # camera's own track has belonged to over the last few seconds.
        self.owner_votes = {}
        self._now = 0.0

    @classmethod
    def from_config(cls, f):
        """Build from the `fusion:` block of config.yaml (new behaviour on by default)."""
        return cls(f.get("merge_distance_m", 1.0), f.get("match_gate_m", 1.8),
                   f.get("max_age_s", 1.5), f.get("smoothing", 0.35),
                   use_blob=f.get("use_blob", True),
                   blob_merge_gate_m=f.get("blob_merge_gate_m", 1.5),
                   conf_drop_ratio=f.get("conf_drop_ratio", 0.5),
                   blob_min_sigma_m=f.get("blob_min_sigma_m", 0.15),
                   n_init=f.get("n_init", 3),
                   max_age_tentative_s=f.get("max_age_tentative_s", 0.4),
                   dup_suppress_m=f.get("dup_suppress_m", 0.7),
                   max_speed_mps=f.get("max_speed_mps", 2.5),
                   identity_matching=f.get("identity_matching", True),
                   identity_gate_m=f.get("identity_gate_m", 3.0),
                   triangulate=f.get("triangulate", True),
                   tri_max_gap_m=f.get("tri_max_gap_m", 0.25),
                   tri_min_angle_deg=f.get("tri_min_angle_deg", 25.0),
                   sitting_shoulder_max_m=f.get("sitting_shoulder_max_m", 1.20),
                   blob_only_fallback=f.get("blob_only_fallback", True),
                   seated_persistence=f.get("seated_persistence", True),
                   settled_max_age_s=f.get("settled_max_age_s", 120.0),
                   settled_gate_m=f.get("settled_gate_m", 0.6),
                   standup_gate_m=f.get("standup_gate_m", 0.35),
                   settled_smoothing=f.get("settled_smoothing", True),
                   settle_speed_mps=f.get("settle_speed_mps", 0.2),
                   settle_window_s=f.get("settle_window_s", 3.0),
                   deadband_m=f.get("deadband_m", 0.3),
                   deadband_s=f.get("deadband_s", 1.0),
                   confirm_s=f.get("confirm_s", 0.3),
                   dup_merge_m=f.get("dup_merge_m", 0.5),
                   dup_merge_s=f.get("dup_merge_s", 1.0),
                   dup_merge_speed_mps=f.get("dup_merge_speed_mps", 0.4),
                   ray_merge_m=f.get("ray_merge_m", 0.35))

    # ---- clustering -----------------------------------------------------------
    def _ray_relation(self, r1, r2):
        """'cross' (the two cameras look at the same shoulders), 'miss' (different
        people) or 'unsure', plus the crossing point.

        Unsure when the lines of sight meet at a shallow angle seen from above (two
        cameras facing each other, or side by side, looking along the same line):
        then any two people on that line give crossing lines, so a crossing proves
        nothing and the crossing point is poorly defined."""
        (_, d1), (_, d2) = r1, r2
        h1, h2 = d1[:2], d2[:2]
        n1, n2 = np.linalg.norm(h1), np.linalg.norm(h2)
        if n1 < 1e-6 or n2 < 1e-6 or abs(h1[0] * h2[1] - h1[1] * h2[0]) / (n1 * n2) < self.tri_min_sin:
            return "unsure", None
        res = closest_point_between_rays(r1, r2)
        if res is None:
            return "unsure", None
        p, gap, in_front = res
        if in_front and gap <= self.tri_gap and 0.75 <= p[2] <= 1.9:
            return "cross", (p, gap)
        return "miss", None

    def _ray_over(self, guess, xy):
        """Horizontal distance from floor point xy to the stretch of a guess line of
        sight between plausible heights (shoulder_band, raised by guess["above"]
        for a head line of sight)."""
        (C, d), above = guess["ray"], guess.get("above", 0.0)
        if d[2] > -1e-6:
            return np.inf
        ts = sorted(max((z + above - C[2]) / d[2], 0.0) for z in self.shoulder_band)
        a, b = C[:2] + ts[0] * d[:2], C[:2] + ts[1] * d[:2]
        ab = b - a
        u = 0.0 if not ab.any() else float(np.clip(np.dot(xy - a, ab) / np.dot(ab, ab), 0.0, 1.0))
        return float(np.linalg.norm(xy - (a + u * ab)))

    def _guess_fits(self, o, member):
        """A torso / head estimate placed at an assumed height can be far off when
        the posture guess is wrong (a standing person whose legs are hidden is
        taken as seated). Its line of sight is still right: it fits a confident
        ankle estimate it passes over at a plausible height. Returns that distance
        (m) when within ray_merge_m, else None."""
        guess = lambda m: m["kind"] in ("torso", "head") and m.get("guess") is not None
        sure = lambda m: m["kind"] in ("ankle", "ankles") and m["sigma"] <= 0.3
        if guess(member):
            anchors = [m for m in o["members"] if sure(m)]
            if anchors:
                wts = np.array([1.0 / max(m["sigma"], _SIG_MIN) ** 2 for m in anchors])
                xy = np.average([m["w"] for m in anchors], axis=0, weights=wts)
                dist = self._ray_over(member["guess"], xy)
                return dist if dist <= self.ray_merge else None
        elif sure(member) and all(guess(m) for m in o["members"]):
            dist = max(self._ray_over(m["guess"], member["w"]) for m in o["members"])
            return dist if dist <= self.ray_merge else None
        return None

    def _merge_score(self, o, member):
        """How well a detection fits cluster o (lower is better), or None if it
        does not belong there.

        Camera IDs owned by different tracks keep detections apart, unless the
        lines of sight cross (below). With `triangulate`, shoulder lines of sight
        from different cameras decide where they can: any line that clearly misses a member's line means a
        different person (even when the floor estimates are close: neighbours,
        someone walking past a desk); a clear crossing means the same person,
        wherever the single-camera guesses landed (e.g. a standing person whose
        hidden legs made each camera guess "seated"), provided the floor estimates
        agree within their uncertainties. Shallow-angle pairs decide nothing. When no
        pair is decisive, a detection joins when its floor estimate is within
        merge_distance, or -- with blobs on -- when its trapezoid overlaps one in the
        cluster within the (looser) blob gate."""
        w, poly, ray = member["w"], member["poly"], member["ray"]
        # Each camera's own tracker has kept these apart for seconds -- unless the
        # shoulders' lines of sight cross below AND the floor estimates are within
        # dup_merge_m: one person split in two. (Lines of sight to two different
        # people often cross by coincidence; their estimates are then further apart.)
        conflicting = [m for m in o["members"] if member["owner"] is not None
                       and m["owner"] is not None and m["owner"] != member["owner"]]
        owner_conflict = bool(conflicting)
        split_ok = self.dup_merge > 0.0 and all(
            np.linalg.norm(m["w"] - w) <= self.dup_merge for m in conflicting)
        d = float(np.linalg.norm(o["pos"] - w))
        compatible = all(np.linalg.norm(m["w"] - w) <= 3.0 * np.hypot(m["sigma"], member["sigma"]) + 0.1
                         for m in o["members"])
        if self.triangulate and ray is not None and d <= self.identity_gate:
            rel = [self._ray_relation(m["ray"], ray) for m in o["members"] if m["ray"] is not None]
            if any(r == "miss" for r, _ in rel):
                return None
            gaps = [x[1] for r, x in rel if r == "cross"]
            if gaps and (not owner_conflict or split_ok):
                return float(np.mean(gaps)) if compatible else None
        if owner_conflict:
            return None
        if self.ray_merge > 0.0:
            fit = self._guess_fits(o, member)
            if fit is not None:
                return fit
        if d <= self.merge and compatible:
            return d
        if not self.use_blob or poly is None or d > self.blob_gate:
            return None
        if any(convex_intersection(poly, mp) is not None for mp in o["polys"] if mp is not None):
            return d
        return None

    def _cluster(self, dets):
        """Assign each detection to the cluster it fits best (see `_merge_score`)."""
        obs = []
        for d in dets:
            w = np.array(d["world"], dtype=float)
            member = {"cam": d["cam"], "id": d.get("id"), "w": w, "poly": d.get("floor_poly"),
                      "conf": float(d.get("conf", 1.0)),
                      "sigma": float(d.get("foot_sigma_m") or self.default_sigma),
                      "kind": d.get("foot_source", "box"), "ray": d.get("shoulder_ray"),
                      "guess": d.get("guess_ray"),
                      "posture": d.get("posture_2d", "unknown"),
                      "legs_hidden": bool(d.get("legs_hidden", False))}
            member["owner"] = self._owner((d["cam"], d.get("id"))) if self.identity else None
            best, best_score = None, None
            for o in obs:
                # A single camera that split two people into two boxes is the
                # strongest evidence they ARE two people: never merge detections
                # from the same camera.
                if d["cam"] in o["cams"]:
                    continue
                score = self._merge_score(o, member)
                if score is not None and (best_score is None or score < best_score):
                    best, best_score = o, score
            if best is None:
                obs.append({"pos": w, "cams": {d["cam"]}, "members": [member],
                            "polys": [member["poly"]]})
                continue
            best["members"].append(member)
            best["cams"].add(d["cam"])
            best["polys"].append(member["poly"])
            k = len(best["members"])
            # Unweighted running mean: only the gate reference for the next
            # candidate; the reported position is recomputed in `_resolve`.
            best["pos"] = (best["pos"] * (k - 1) + w) / k
        for o in obs:
            self._resolve(o)
        return obs

    def _triangulate(self, members):
        """Shoulder position and height from lines of sight of two or more cameras."""
        rays = [m["ray"] for m in members if m["ray"] is not None]
        points = []
        for r1, r2 in combinations(rays, 2):
            relation, hit = self._ray_relation(r1, r2)
            if relation == "cross":          # well-conditioned crossings only
                points.append(hit[0])
        if not points:
            return None
        points = np.array(points)
        return points[:, :2].mean(axis=0), float(np.median(points[:, 2]))

    def _resolve(self, o):
        """Collapse a cluster's per-camera detections into one position + a fused
        uncertainty, weighting each camera by how well it can localize the person
        (inverse variance), plus the triangulated shoulders when available."""
        ms = o["members"]
        pts = np.array([m["w"] for m in ms], dtype=float)
        sig = np.clip(np.array([m["sigma"] for m in ms], dtype=float), _SIG_MIN, None)
        confs = np.array([m["conf"] for m in ms], dtype=float)
        w = 1.0 / sig**2
        # Secondary: drop members whose box confidence is far below the cluster's best.
        if self.conf_drop_ratio > 0.0 and len(confs) > 1:
            keep = confs >= self.conf_drop_ratio * confs.max()
            if not keep.any():
                keep[confs.argmax()] = True
            w = w * keep
            o["polys"] = [p if k else None for p, k in zip(o["polys"], keep)]
        if not np.any(w > 0):
            w = np.ones(len(pts))
        o["shoulder_z"] = None
        if self.triangulate and len(ms) >= 2:
            tri = self._triangulate(ms)
            if tri is not None:
                o["shoulder_z"] = tri[1]
                # The measured shoulders replace single-camera shoulder/head guesses
                # (which assumed a height); visible-ankle estimates still count.
                guess = np.array([m["kind"] in ("torso", "head") for m in ms])
                w = np.where(guess, 0.0, w)
                pts = np.vstack([pts, tri[0]])
                sig = np.append(sig, 0.08)
                w = np.append(w, 1.0 / 0.08**2)
        o["pos"] = np.average(pts, axis=0, weights=w)
        kept = w > 0
        o["sigma"] = float(1.0 / np.sqrt(np.sum(1.0 / sig[kept]**2)))
        o["kinds"] = [m["kind"] for m in ms]
        o["postures"] = [m["posture"] for m in ms]
        o["legs_hidden"] = all(m["legs_hidden"] for m in ms)
        o["member_ids"] = [(m["cam"], m["id"]) for m in ms]

    def _refine_blob(self, obs):
        """Replace a cluster's weighted position with the trapezoid intersection --
        only when the estimates are genuinely uncertain and (with
        `blob_only_fallback`) made only of fallback guesses, never over ankle,
        shoulder or triangulated estimates."""
        self.last_blobs = []
        for o in obs:
            if self.blob_min_sigma > 0.0 and o.get("sigma", 1.0) < self.blob_min_sigma:
                continue
            if self.blob_only_fallback and (o["shoulder_z"] is not None or any(
                    k not in FALLBACK_SOURCES for k in o["kinds"])):
                continue
            centroid, poly = intersect_blobs(o["polys"])
            if centroid is not None:
                o["pos"] = centroid
                self.last_blobs.append(poly)

    # ---- per-camera identity --------------------------------------------------------
    def _owner(self, member, window=3.0, share=0.6, min_votes=3):
        """The global track a camera's own track has mostly belonged to recently,
        or None. A majority over seconds, so one wrong grouping cannot stick."""
        votes = [g for t, g in self.owner_votes.get(member, ())
                 if self._now - t <= window and g in self.tracks]
        if len(votes) < min_votes:
            return None
        values, counts = np.unique(votes, return_counts=True)
        k = int(np.argmax(counts))
        return int(values[k]) if counts[k] >= share * len(votes) else None

    def _vote_owners(self, obs, obs_to_gid, now):
        for oi, gid in obs_to_gid.items():
            for m in obs[oi]["member_ids"]:
                self.owner_votes.setdefault(m, deque(maxlen=60)).append((now, gid))
        for m in [m for m, v in self.owner_votes.items() if now - v[-1][0] > 10.0]:
            del self.owner_votes[m]

    # ---- track state ----------------------------------------------------------------
    def _recent(self, seq, now, window):
        return [v for t, v in seq if now - t <= window]

    def _is_still(self, tr, now):
        """Moved less than settle_speed over the last settle_window seconds."""
        pts = [(t, xy) for t, xy in tr.history if now - t <= self.settle_window]
        if len(pts) < 4 or pts[-1][0] - pts[0][0] < 0.6 * self.settle_window:
            return False
        t0, t1 = pts[0][0], pts[-1][0]
        first = np.median([xy for t, xy in pts if t - t0 <= 1.0], axis=0)
        last = np.median([xy for t, xy in pts if t1 - t <= 1.0], axis=0)
        return np.linalg.norm(last - first) / max(t1 - t0 - 1.0, 0.5) < self.settle_speed

    def _update_posture(self, tr, now):
        # The last 2 s decide first, so standing up (or sitting down) registers
        # within a few frames instead of being outvoted by older measurements.
        latest = self._recent(tr.heights, now, 2.0)
        heights = latest if len(latest) >= 2 else self._recent(tr.heights, now, 10.0)
        if len(heights) >= 2:
            tr.shoulder_h = float(np.median(heights))
            tr.posture = "sitting" if tr.shoulder_h < self.sit_max else "standing"
            return
        tr.shoulder_h = None
        votes = self._recent(tr.votes, now, 5.0)
        sit, stand = votes.count("sitting"), votes.count("standing")
        if sit >= 2 and sit > stand:
            tr.posture = "sitting"
        elif stand >= 2 and stand > sit:
            tr.posture = "standing"

    def _update_track(self, tr, o, now):
        dt = max(now - tr.last_t, 1e-3)
        tr.history.append((now, o["pos"].copy()))
        if o["shoulder_z"] is not None:
            tr.heights.append((now, o["shoulder_z"]))
        for v in o["postures"]:
            if v != "unknown":
                tr.votes.append((now, v))
        tr.legs_hidden = o["legs_hidden"]
        tr.hits += 1
        if self.confirm_s > 0:
            tr.confirmed = tr.confirmed or (now - tr.born_t >= self.confirm_s and tr.hits >= 2)
        elif tr.hits >= self.n_init:
            tr.confirmed = True
        still = self._is_still(tr, now)
        if not still and len(tr.history) >= 2:
            # Walking means standing: a moving person's posture vote.
            t0, p0 = tr.history[max(0, len(tr.history) - 10)]
            if now - t0 > 0.2 and np.linalg.norm(o["pos"] - p0) / (now - t0) > 0.5:
                tr.votes.append((now, "standing"))
        self._update_posture(tr, now)
        tr.settled = (self.persistence and tr.confirmed and still
                      and (tr.posture == "sitting"
                           or (tr.posture == "unknown" and tr.legs_hidden)))
        if tr.settled and self.settled_smoothing:
            recent = [xy for t, xy in tr.history if now - t <= self.settle_window]
            med = np.median(recent, axis=0)
            if np.linalg.norm(med - tr.pos) <= self.deadband:
                tr.pos, tr.far_since = med, None
            elif tr.far_since is None:
                tr.far_since = now
            elif now - tr.far_since >= self.deadband_s:      # really moved (e.g. rolled the chair)
                tr.pos, tr.far_since = med, None
        else:
            tr.far_since = None
            ema = self.s * o["pos"] + (1 - self.s) * tr.pos
            # Motion cap: a person can't teleport, so limit how far the marker
            # moves in one frame.
            move = ema - tr.pos
            d = float(np.linalg.norm(move))
            max_step = self.max_speed * dt
            if self.max_speed > 0.0 and d > max_step:
                ema = tr.pos + move * (max_step / d)
            tr.pos = ema
        tr.last_t = now
        tr.cams = o["cams"]
        tr.members = set(o["member_ids"])
        tr.sigma = o["sigma"]

    # ---- duplicate merging -----------------------------------------------------------
    def _absorb(self, keep_gid, drop_gid, now):
        """Fold track drop_gid into keep_gid: one person, the older ID survives."""
        keep, drop = self.tracks[keep_gid], self.tracks.pop(drop_gid)
        if keep.last_t == now and drop.last_t == now:
            wk, wd = 1.0 / max(keep.sigma, _SIG_MIN) ** 2, 1.0 / max(drop.sigma, _SIG_MIN) ** 2
            keep.pos = (wk * keep.pos + wd * drop.pos) / (wk + wd)
            keep.cams = set(keep.cams) | set(drop.cams)
        elif drop.last_t > keep.last_t:
            keep.pos, keep.cams = drop.pos, drop.cams
        keep.members |= drop.members
        keep.last_t = max(keep.last_t, drop.last_t)
        keep.born_t = min(keep.born_t, drop.born_t)
        keep.hits += drop.hits
        keep.confirmed = keep.confirmed or drop.confirmed
        for name in ("heights", "votes"):
            merged = sorted(list(getattr(keep, name)) + list(getattr(drop, name)), key=lambda v: v[0])
            getattr(keep, name).clear()
            getattr(keep, name).extend(merged)
        self._update_posture(keep, now)
        for votes in self.owner_votes.values():     # camera IDs now belong to the survivor
            for k, (t, g) in enumerate(votes):
                if g == drop_gid:
                    votes[k] = (t, keep_gid)
        self._close_pairs = {p: v for p, v in self._close_pairs.items() if drop_gid not in p}

    def _older(self, a, b):
        ta, tb = self.tracks[a], self.tracks[b]
        return (a, b) if (ta.born_t, a) <= (tb.born_t, b) else (b, a)

    def _velocity(self, tr, now, window=1.0):
        pts = [(t, xy) for t, xy in tr.history if now - t <= window]
        if len(pts) < 2 or pts[-1][0] - pts[0][0] < 0.3:
            return None
        return (pts[-1][1] - pts[0][1]) / (pts[-1][0] - pts[0][0])

    def _merge_split_owners(self, obs, obs_to_gid, now):
        """A cluster whose camera IDs are owned by two tracks (allowed in when the
        lines of sight crossed) shows the other track is the same person."""
        assigned = set(obs_to_gid.values())
        for oi, gid in list(obs_to_gid.items()):
            for m in obs[oi]["member_ids"]:
                other = self._owner(m)
                if (other is not None and other != gid and other in self.tracks
                        and gid in self.tracks and other not in assigned):
                    keep, drop = self._older(gid, other)
                    self._absorb(keep, drop, now)
                    for k, g in obs_to_gid.items():
                        if g == drop:
                            obs_to_gid[k] = keep
                    assigned = set(obs_to_gid.values())

    def _merge_close_tracks(self, now):
        """Two tracks seen this frame that stay within dup_merge_m, move together
        and share no camera for dup_merge_s are one person seen twice (e.g. where
        two cameras look along the same line and their lines of sight cannot tell)."""
        seen = [g for g, tr in self.tracks.items() if tr.last_t == now]
        for a, b in combinations(seen, 2):
            if a not in self.tracks or b not in self.tracks:
                continue
            ta, tb = self.tracks[a], self.tracks[b]
            pair = self._older(a, b)
            close = (np.linalg.norm(ta.pos - tb.pos) <= self.dup_merge
                     and not (set(ta.cams) & set(tb.cams)))
            if close:
                va, vb = self._velocity(ta, now), self._velocity(tb, now)
                close = va is not None and vb is not None
                if close:
                    speeds = sorted((np.linalg.norm(va), np.linalg.norm(vb)))
                    # Never a still person and a moving one (someone walking past a desk).
                    close = (np.linalg.norm(va - vb) <= self.dup_merge_speed
                             and not (speeds[0] < 0.1 and speeds[1] > 0.2))
            if close:
                first, _ = self._close_pairs.get(pair, (now, now))
                self._close_pairs[pair] = [first, now]
                if now - first >= self.dup_merge_s:
                    self._absorb(*pair, now)
            elif pair in self._close_pairs and now - self._close_pairs[pair][1] > 0.3:
                del self._close_pairs[pair]

    # ---- per-frame update ---------------------------------------------------------
    def update(self, detections, now=None):
        now = time.time() if now is None else now
        self._now = now
        dets = [d for d in detections if d.get("world") is not None]
        obs = self._cluster(dets)
        if self.use_blob:
            self._refine_blob(obs)

        obs_to_gid, free = {}, set(self.tracks)
        # Stage 1: identity. The per-camera tracker keeps a person's ID through a
        # sudden jump in floor position; follow it instead of spawning a new track.
        if self.identity:
            pairs = []
            for oi, o in enumerate(obs):
                owners = [self._owner(m) for m in o["member_ids"]]
                for gid, tr in self.tracks.items():
                    shared = sum(1 for m, own in zip(o["member_ids"], owners)
                                 if own == gid or (own is None and m in tr.members))
                    dist = float(np.linalg.norm(tr.pos - o["pos"]))
                    if shared and dist <= self.identity_gate:
                        pairs.append((-shared, dist, oi, gid))
            for _, _, oi, gid in sorted(pairs):
                if oi not in obs_to_gid and gid in free:
                    obs_to_gid[oi] = gid
                    free.discard(gid)
        # Stage 2: nearest within the gate. A settled (seated) person only accepts
        # a nearby observation, so someone walking past can't take over their track.
        pairs = []
        for oi, o in enumerate(obs):
            if oi in obs_to_gid:
                continue
            owners = {self._owner(m) for m in o["member_ids"]} - {None}
            standing = o["postures"].count("standing") > o["postures"].count("sitting")
            for gid in free:
                tr = self.tracks[gid]
                if self.identity and owners and gid not in owners:
                    continue        # a camera's own tracker says this is someone else
                gate = self.gate
                if self.persistence and tr.settled:
                    # A seated person accepts nearby observations only; a standing
                    # one only right at the chair (getting up), not someone passing.
                    gate = self.standup_gate if standing else self.settled_gate
                dist = float(np.linalg.norm(tr.pos - o["pos"]))
                if dist <= gate:
                    pairs.append((dist, oi, gid))
        for _, oi, gid in sorted(pairs):
            if oi not in obs_to_gid and gid in free:
                obs_to_gid[oi] = gid
                free.discard(gid)

        if self.dup_merge > 0.0 and self.identity:
            self._merge_split_owners(obs, obs_to_gid, now)
        for oi, o in enumerate(obs):
            if oi in obs_to_gid:
                self._update_track(self.tracks[obs_to_gid[oi]], o, now)
            else:
                # Spawn gate: don't birth a track right next to an existing one --
                # almost always the same person seen a second way.
                if self.dup_suppress > 0.0 and any(
                        np.linalg.norm(tr.pos - o["pos"]) <= self.dup_suppress
                        for tr in self.tracks.values()):
                    continue
                gid = self.next_id
                self.next_id += 1
                tr = _Track(gid, o["pos"], now, o["cams"], o["member_ids"], o["sigma"])
                tr.history.append((now, o["pos"].copy()))
                if o["shoulder_z"] is not None:
                    tr.heights.append((now, o["shoulder_z"]))
                self.tracks[gid] = tr
                obs_to_gid[oi] = gid
        self._vote_owners(obs, obs_to_gid, now)
        if self.dup_merge > 0.0:
            self._merge_close_tracks(now)

        # Age out stale tracks: tentative ones die fast (flicker never sticks),
        # settled seated people are held much longer when undetected.
        for gid in [g for g, t in self.tracks.items() if now - t.last_t > (
                (self.settled_max_age if t.settled else self.max_age) if t.confirmed
                else self.max_age_tent)]:
            del self.tracks[gid]
        self._close_pairs = {p: v for p, v in self._close_pairs.items()
                             if p[0] in self.tracks and p[1] in self.tracks}

        out = []
        for tr in self.tracks.values():
            if not tr.confirmed:
                continue
            out.append({
                "gid": tr.gid,
                "world": (float(tr.pos[0]), float(tr.pos[1])),
                "cams": sorted(tr.cams),
                "coasting": tr.last_t < now,   # not seen this frame (occluded)
                "last_seen_s": float(now - tr.last_t),
                "posture": tr.posture,
                "settled": bool(tr.settled),
                "shoulder_h": tr.shoulder_h,
                "sigma": float(tr.sigma),
            })
        return out

    def posture_hints(self):
        """{(cam, per_camera_id): {"posture", "shoulder_h"}} from the last update, so
        the next frame's single-camera estimates use the person's known posture and
        measured shoulder height."""
        hints = {}
        for tr in self.tracks.values():
            if tr.posture == "unknown" and tr.shoulder_h is None:
                continue
            for member in tr.members:
                hints[member] = {"posture": tr.posture, "shoulder_h": tr.shoulder_h}
        return hints


def publishable(tracks, publish_coast_s=0.5):
    """Tracks to send to the digital twin: seen recently, or settled (a seated
    person briefly hidden is still there; a walker who vanished is not)."""
    return [t for t in tracks
            if t.get("settled") or t.get("last_seen_s", 0.0) <= publish_coast_s]
