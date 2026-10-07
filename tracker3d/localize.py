"""Per-camera localization: detection -> foot pixel -> floor position (+ sigma,
+ the ground-blob trapezoid used by fusion)."""
import numpy as np

from .geometry import foot_from_pose, foot_point
from .groundblob import bbox_floor_polygon


class Localizer:
    def __init__(self, cfg, plan):
        f = cfg.get("fusion", {})
        self.plan = plan
        self.kp_conf = float(cfg["detector"].get("kp_conf", 0.5))
        self.use_blob = bool(f.get("use_blob", True))
        self.blob_kw = dict(far_frac=f.get("blob_far_frac", 0.1),
                            near_frac=f.get("blob_near_frac", 1.4),
                            clip_inflate=f.get("blob_clip_inflate", 2.5),
                            body_aspect=f.get("blob_body_aspect", 3.0),
                            short_aspect=f.get("blob_short_aspect", 1.8))

    def localize(self, cam, people, proj, frame_wh):
        """Annotate each tracked detection in place with cam, foot_px, foot_source,
        world (None if unmapped / outside the usable area), foot_sigma_m, floor_poly."""
        for p in people:
            p["cam"] = cam
            p["world"] = p["floor_poly"] = p["foot_sigma_m"] = None
            if p.get("kxy") is not None and p.get("kconf") is not None:
                (fx, fy), sig_px, src = foot_from_pose(p["kxy"], p["kconf"], p["bbox"],
                                                       kp_thresh=self.kp_conf)
            else:
                (fx, fy), src = foot_point(p["bbox"]), "box"
                sig_px = 0.5 * (p["bbox"][3] - p["bbox"][1])
            p["foot_px"], p["foot_source"] = (fx, fy), src
            if proj is None:
                continue
            w = proj.to_floor(fx, fy)
            # Occluded/clipped foot points can land in impossible places (outside
            # the room); drop those before they reach fusion.
            if w is None or not self.plan.in_valid_area(*w):
                continue
            p["world"] = w
            # Pixel sigma -> meters through the homography: distance-aware for free
            # (the same pixel error is more meters far from the camera).
            w2 = proj.to_floor(fx, fy + sig_px)
            p["foot_sigma_m"] = (float(np.clip(np.hypot(w2[0] - w[0], w2[1] - w[1]), 0.03, 2.0))
                                 if w2 is not None else 0.5)
            if self.use_blob:
                # Built in H0's (rectified) pixel space; "clipped" is judged on the
                # raw box, since the frame edge is only straight in raw pixels.
                p["floor_poly"] = bbox_floor_polygon(
                    proj.H0, proj.rectify_bbox(p["bbox"]), frame_wh,
                    clipped=p["bbox"][3] >= frame_wh[1] - 2.0, **self.blob_kw)
        return people
