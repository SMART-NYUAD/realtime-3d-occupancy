"""Per-camera localization: detection -> floor position (+ sigma, + the shoulder
line of sight used for triangulation, + the ground-blob trapezoid for fallbacks).

Order of preference for each detection:
  1. ankles visible            -> the ankle plane (feet), unless the person is known
                                  to be sitting (their feet are ahead of the chair)
  2. shoulders visible + pose  -> where the line of sight to the shoulders crosses
                                  the person's shoulder height (seated or standing)
  3. head keypoints + pose     -> the same at head height
  4. otherwise                 -> the old foot guess (knees / hips / box bottom)
The camera pose comes from the lens-corrected floor calibration
(FloorProjector.pose); without a lens only paths 1 and 4 exist, as before.
"""
import numpy as np

from .geometry import foot_from_pose, foot_point, posture_2d, ray_at_height
from .groundblob import bbox_floor_polygon

_HEAD = (0, 1, 2, 3, 4)
_SHOULDERS = (5, 6)
_KNEES = (13, 14)
_ANKLES = (15, 16)
FALLBACK_SOURCES = ("knees", "hips", "box")


class Localizer:
    def __init__(self, cfg, plan):
        f = cfg.get("fusion", {})
        loc = cfg.get("localize", {})
        self.plan = plan
        self.kp_conf = float(cfg["detector"].get("kp_conf", 0.5))
        self.use_blob = bool(f.get("use_blob", True))
        self.blob_kw = dict(far_frac=f.get("blob_far_frac", 0.1),
                            near_frac=f.get("blob_near_frac", 1.4),
                            clip_inflate=f.get("blob_clip_inflate", 2.5),
                            body_aspect=f.get("blob_body_aspect", 3.0),
                            short_aspect=f.get("blob_short_aspect", 1.8))
        self.torso = bool(loc.get("torso_estimates", True))
        self.seated_use_torso = bool(loc.get("seated_use_torso", True))
        self.blob_only_fallback = bool(f.get("blob_only_fallback", True))
        self.h_stand = float(loc.get("standing_shoulder_m", 1.40))
        self.h_seat = float(loc.get("seated_shoulder_m", 1.05))
        self.h_unknown = float(loc.get("unknown_shoulder_m", 1.22))
        self.h_head = float(loc.get("head_above_shoulder_m", 0.22))

    # ---- helpers ----------------------------------------------------------------
    def _visible(self, p, idxs):
        return [i for i in idxs if p["kconf"][i] >= self.kp_conf]

    def _shoulder_height(self, posture, hint, legs_hidden, clipped):
        """Plane height (m) for the shoulders and its 1-sigma uncertainty. The
        person's measured height is used only while the posture still agrees
        (after standing up, their seated height no longer applies)."""
        if hint and hint.get("shoulder_h") and hint.get("posture") == posture:
            return float(hint["shoulder_h"]), 0.07          # measured by triangulation
        if posture == "sitting":
            return self.h_seat, 0.12
        if posture == "standing":
            return self.h_stand, 0.12
        if legs_hidden and not clipped:
            return self.h_seat, 0.18                        # legs hidden by a desk: likely seated
        return self.h_unknown, 0.25

    @staticmethod
    def _plane_estimate(ray, h, sh):
        xy = ray_at_height(ray, h)
        if xy is None:
            return None, None
        lo, hi = ray_at_height(ray, h - sh), ray_at_height(ray, h + sh)
        if lo is None or hi is None:
            return xy, 0.5
        return xy, float(np.clip(np.hypot(hi[0] - lo[0], hi[1] - lo[1]) / 2.0, 0.05, 2.0))

    # ---- main -----------------------------------------------------------------
    def localize(self, cam, people, proj, frame_wh, hints=None):
        """Annotate each tracked detection in place with: cam, foot_px, foot_source,
        world (None if unmapped / outside the usable area), foot_sigma_m, floor_poly,
        posture_2d, legs_hidden, shoulder_ray (map-space line of sight or None)."""
        hints = hints or {}
        for p in people:
            p["cam"] = cam
            p["world"] = p["floor_poly"] = p["foot_sigma_m"] = p["shoulder_ray"] = None
            p["posture_2d"], p["legs_hidden"] = "unknown", False
            has_kp = p.get("kxy") is not None and p.get("kconf") is not None
            if has_kp:
                (fx, fy), sig_px, src = foot_from_pose(p["kxy"], p["kconf"], p["bbox"],
                                                       kp_thresh=self.kp_conf)
            else:
                (fx, fy), src = foot_point(p["bbox"]), "box"
                sig_px = 0.5 * (p["bbox"][3] - p["bbox"][1])
            p["foot_px"], p["foot_source"] = (fx, fy), src
            if proj is None:
                continue

            pose = proj.pose() if (self.torso and has_kp) else None
            hint = hints.get((cam, p.get("id")))
            clipped = p["bbox"][3] >= frame_wh[1] - 2.0
            ankles = self._visible(p, _ANKLES) if has_kp else []
            if has_kp:
                p["posture_2d"] = posture_2d(p["kxy"], p["kconf"], self.kp_conf)
                p["legs_hidden"] = not ankles and not self._visible(p, _KNEES)
            # What this view clearly shows wins; the track's remembered posture only
            # fills in when the view is ambiguous (legs hidden, odd angle).
            posture = (p["posture_2d"] if p["posture_2d"] != "unknown"
                       else (hint or {}).get("posture", "unknown"))
            if pose is not None and len(self._visible(p, _SHOULDERS)) == 2:
                p["shoulder_ray"] = proj.ray(*np.mean([p["kxy"][i] for i in _SHOULDERS], axis=0))

            w, sigma = None, None
            seated_torso = (self.seated_use_torso and posture == "sitting"
                            and p["shoulder_ray"] is not None)
            if ankles and not seated_torso and src in ("ankle", "ankles"):
                w, sigma = self._ankles(p, proj, ankles, fx, fy, sig_px, src)
            if w is None and p["shoulder_ray"] is not None:
                h, sh = self._shoulder_height(posture, hint, p["legs_hidden"], clipped)
                w, sigma = self._plane_estimate(p["shoulder_ray"], h, sh)
                p["foot_source"] = "torso"
            if w is None and pose is not None:
                head = self._visible(p, _HEAD)
                if head:
                    h, sh = self._shoulder_height(posture, hint, p["legs_hidden"], clipped)
                    ray = proj.ray(*np.mean([p["kxy"][i] for i in head], axis=0))
                    w, sigma = self._plane_estimate(ray, h + self.h_head, sh)
                    p["foot_source"] = "head"
            if w is None:                   # no usable torso/head: feet if seen, else old guess
                if ankles and src in ("ankle", "ankles"):
                    w, sigma = self._ankles(p, proj, ankles, fx, fy, sig_px, src)
                    p["foot_source"] = src
                else:
                    w, sigma = self._fallback(proj, fx, fy, sig_px)
                    p["foot_source"] = src
            # Misplaced points can land in impossible places (outside the room);
            # drop those before they reach fusion.
            if w is None or not self.plan.in_valid_area(*w):
                continue
            p["world"], p["foot_sigma_m"] = w, sigma
            if self.use_blob and (not self.blob_only_fallback
                                  or p["foot_source"] in FALLBACK_SOURCES):
                # Built in H0's (rectified) pixel space; "clipped" is judged on the
                # raw box, since the frame edge is only straight in raw pixels.
                p["floor_poly"] = bbox_floor_polygon(
                    proj.H0, proj.rectify_bbox(p["bbox"]), frame_wh,
                    clipped=clipped, **self.blob_kw)
        return people

    def _ankles(self, p, proj, ankles, fx, fy, sig_px, src):
        """Feet from the visible ankles. With a joint calibration each ankle is
        projected on the ankle plane separately, then averaged in world space
        (averaging pixels first is not invariant under perspective)."""
        def locate(dy=0):
            if proj.ankle_H is None:
                return proj.to_floor(fx, fy + dy)
            xy = [proj.to_floor(p["kxy"][i][0], p["kxy"][i][1] + dy, plane="ankle") for i in ankles]
            return tuple(np.mean(xy, axis=0)) if all(q is not None for q in xy) else None
        return self._with_sigma(locate, sig_px)

    def _fallback(self, proj, fx, fy, sig_px):
        return self._with_sigma(lambda dy=0: proj.to_floor(fx, fy + dy), sig_px)

    @staticmethod
    def _with_sigma(locate, sig_px=None):
        w = locate()
        if w is None:
            return None, None
        # Pixel sigma -> meters through the homography: distance-aware for free.
        w2 = locate(sig_px) if sig_px is not None else None
        return w, (float(np.clip(np.hypot(w2[0] - w[0], w2[1] - w[1]), 0.03, 2.0))
                   if w2 is not None else 0.5)
