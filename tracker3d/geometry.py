"""Image <-> floor (world) mapping via a ground-plane homography.

A person standing on a flat floor has their feet ON the ground plane, and a
homography H maps any image pixel on that plane to floor meters:

    [Xw*s, Yw*s, s]^T = H * [u, v, 1]^T   ->   (Xw, Yw)

Each camera's H is stored as `calib/<cam>_homography.npy` plus a JSON sidecar
recording the image size it was calibrated at (and the clicked points), so a
stream delivered at another resolution is rescaled instead of silently wrong.

With a lens file (see lens.py) H maps *rectified* (undistorted) pixels to the
floor, and the sidecar records the lens fingerprint it was fitted with.
"""
import json
import os

import cv2
import numpy as np

_EPS = 1e-9


def compute_homography(image_points, world_points, ransac_thresh_m=0.25):
    """Fit image px -> floor m from N>=4 correspondences.

    With few points RANSAC can't reject anything (and its default 3.0 threshold
    would be in *meters* here, i.e. accept everything), so use a plain
    least-squares fit below 6 points and a meter-scaled RANSAC above."""
    img = np.asarray(image_points, dtype=np.float64)
    wld = np.asarray(world_points, dtype=np.float64)
    if len(img) < 4:
        raise ValueError("Need at least 4 point correspondences for a homography.")
    if len(img) < 6:
        H, _ = cv2.findHomography(img, wld, 0)
    else:
        H, _ = cv2.findHomography(img, wld, cv2.RANSAC, ransac_thresh_m)
    if H is None or abs(np.linalg.det(H)) < _EPS:
        raise RuntimeError("Homography fit failed/degenerate — spread the points wider.")
    return H


def fit_errors(image_points, world_points):
    """(in-sample errors, leave-one-out errors) in meters. The in-sample error of
    a 4-point fit is always ~0, so the leave-one-out error (needs >=5 points) is
    the honest estimate of how far off a *new* floor point will land."""
    img = np.asarray(image_points, np.float64)
    wld = np.asarray(world_points, np.float64)
    H = compute_homography(img, wld)
    ins = [np.hypot(*(project(H, u, v) - w)) for (u, v), w in zip(img, wld)]
    loo = []
    if len(img) >= 5:
        for i in range(len(img)):
            m = np.arange(len(img)) != i
            try:
                Hi = compute_homography(img[m], wld[m])
            except (RuntimeError, ValueError):
                continue
            loo.append(float(np.hypot(*(project(Hi, *img[i]) - wld[i]))))
    return np.array(ins), np.array(loo)


def project(H, u, v):
    w = H @ np.array([u, v, 1.0])
    return w[:2] / w[2]


# ---- persistence -------------------------------------------------------------

def _sidecar(path):
    return os.path.splitext(path)[0] + ".json"


def save_calibration(path, H, image_size, pairs=None, lens=None):
    """Write H (.npy) + sidecar JSON {image_size: [w, h], pairs: [...], lens: {...}}.

    `lens` is the LensModel H was fitted with (H then expects rectified pixels);
    its fingerprint lets FloorProjector refuse a mismatched lens later."""
    np.save(path, H)
    meta = {"image_size": [int(image_size[0]), int(image_size[1])]}
    if lens is not None:
        meta["lens"] = {"file": lens.path, "fingerprint": lens.fingerprint()}
    if pairs is not None:
        meta["pairs"] = pairs
    with open(_sidecar(path), "w") as f:
        json.dump(meta, f, indent=2)


def load_calibration_meta(path):
    sc = _sidecar(path)
    if os.path.exists(sc):
        with open(sc) as f:
            return json.load(f)
    return {}


class FloorProjector:
    """One camera's image -> floor mapping, adapted to the actual frame size and
    safe against the horizon (pixels above it would otherwise project to a
    mirrored point on the far side of the camera).

    Every raw pixel takes one path: scale to the calibration resolution ->
    undistort (if a lens is attached) -> H0. Without a lens this equals the old
    H0 @ diag(scale)."""

    def __init__(self, H, calib_size=None, lens=None):
        self.H0 = np.asarray(H, np.float64)
        if abs(np.linalg.det(self.H0)) < _EPS:
            raise ValueError("homography is degenerate (singular) — recalibrate.")
        self.calib_size = tuple(calib_size) if calib_size else None
        if lens is not None and self.calib_size:
            lens = lens.scaled(*self.calib_size)
        self.lens = lens
        self.lens_status = "on" if lens is not None else "off"
        self.frame_size = None
        self.H = self.H0          # raw-pixel homography; only meaningful without a lens
        self._scale = (1.0, 1.0)  # frame px -> calibration px
        self._sign = 0.0

    @classmethod
    def load(cls, path, lens=None):
        """Load H + sidecar, checking it against the camera's lens model:
          - fitted with lens X but X is missing/different -> ValueError (H expects
            pixels from a lens model we don't have; projecting would be wrong);
          - fitted on raw pixels but a lens is configured -> keep the raw path
            (as good as before) and say so in `lens_status` until re-clicked."""
        meta = load_calibration_meta(path)
        need = (meta.get("lens") or {}).get("fingerprint")
        have = lens.fingerprint() if lens is not None else None
        if need and need != have:
            raise ValueError(
                f"{path} was fitted with lens {need} but the configured lens is "
                f"{have or 'none'} — fix intrinsics_file or recalibrate (calibrate.py).")
        ignored = lens is not None and not need
        proj = cls(np.load(path), meta.get("image_size"), None if ignored else lens)
        if ignored:
            proj.lens_status = ("IGNORED: homography predates the lens file — "
                                "re-click with calibrate.py to use it")
        return proj

    def set_frame_size(self, w, h):
        """Rescale if frames arrive at a different resolution than calibrated."""
        if self.frame_size == (w, h):
            return
        self.frame_size = (w, h)
        if self.calib_size and tuple(self.calib_size) != (w, h):
            self._scale = (self.calib_size[0] / w, self.calib_size[1] / h)
        else:
            self._scale = (1.0, 1.0)
        self.H = self.H0 @ np.diag([self._scale[0], self._scale[1], 1.0])
        # The bottom-centre pixel is floor in every sane camera placement; its
        # homogeneous sign marks the floor side of the horizon.
        r = self.to_rect(w / 2.0, h - 1.0)
        wz = (self.H0 @ np.array([r[0], r[1], 1.0]))[2]
        self._sign = float(np.sign(wz)) if abs(wz) > _EPS else 0.0

    def to_rect_many(self, pts):
        """Raw frame pixels (N,2) -> the pixel space H0 expects (N,2)."""
        p = np.asarray(pts, np.float64).reshape(-1, 2) * np.asarray(self._scale)
        return self.lens.undistort_points(p) if self.lens is not None else p

    def to_rect(self, u, v):
        return self.to_rect_many([(u, v)])[0]

    def rect_to_floor(self, x, y):
        """(x, y) meters for a pixel already in H0 space, or None past the horizon."""
        w = self.H0 @ np.array([x, y, 1.0])
        if abs(w[2]) < _EPS or (self._sign and np.sign(w[2]) != self._sign):
            return None
        return float(w[0] / w[2]), float(w[1] / w[2])

    def to_floor(self, u, v):
        """(x, y) meters for a raw frame pixel, or None if on/above the horizon."""
        r = self.to_rect(u, v)
        return self.rect_to_floor(r[0], r[1])

    def rectify_bbox(self, bbox):
        """A raw (x1, y1, x2, y2) box expressed in H0's pixel space."""
        sx, sy = self._scale
        x1, y1, x2, y2 = (float(b) for b in bbox)
        b = (x1 * sx, y1 * sy, x2 * sx, y2 * sy)
        return self.lens.rectify_bbox(b) if self.lens is not None else b


def foot_point(bbox):
    """Bottom-center of an (x1, y1, x2, y2) bbox = where the feet touch the floor."""
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2.0, y2


# COCO-17 pose keypoint indices (Ultralytics pose models emit these).
_L_SHO, _R_SHO = 5, 6
_L_HIP, _R_HIP = 11, 12
_L_KNEE, _R_KNEE = 13, 14
_L_ANK, _R_ANK = 15, 16


def foot_from_pose(kxy, kconf, bbox, kp_thresh=0.5):
    """Estimate the floor contact point (feet) in image pixels from pose keypoints,
    together with a pixel-space uncertainty and a label for how it was derived.

    Unlike the box bottom, this survives lower-body occlusion: when a desk hides
    the legs the ankles are gone but the hips/shoulders are still detected, so we
    extrapolate down the body axis and *flag the estimate as uncertain* instead of
    silently placing the feet at the waist. The returned sigma scales with how far
    we had to extrapolate, so fusion can down-weight inferred feet.

    Args:
        kxy:       (17, 2) keypoint pixel coords (COCO order).
        kconf:     (17,)   per-keypoint visibility confidence.
        bbox:      (x1, y1, x2, y2) detection box (fallback + scale reference).
        kp_thresh: min visibility for a keypoint to be trusted.

    Returns (foot_xy, sigma_px, source):
        foot_xy   : (x, y) float pixel estimate of the feet.
        sigma_px  : 1-sigma uncertainty of foot_xy along the body axis, in pixels.
        source    : "ankles" | "ankle" | "knees" | "hips" | "box".
    """
    kxy = np.asarray(kxy, dtype=np.float64)
    kconf = np.asarray(kconf, dtype=np.float64)
    x1, y1, x2, y2 = (float(b) for b in bbox)
    box_h = max(y2 - y1, 1.0)

    def vis(i):
        return kconf[i] >= kp_thresh

    def mean_pts(idxs):
        return np.mean([kxy[i] for i in idxs], axis=0)

    # 1. Ankles visible -> the feet are observed directly (tightest).
    ank = [i for i in (_L_ANK, _R_ANK) if vis(i)]
    if len(ank) == 2:
        return tuple(mean_pts(ank)), max(0.04 * box_h, 3.0), "ankles"
    if len(ank) == 1:
        return tuple(kxy[ank[0]]), max(0.07 * box_h, 4.0), "ankle"

    knees = [i for i in (_L_KNEE, _R_KNEE) if vis(i)]
    hips = [i for i in (_L_HIP, _R_HIP) if vis(i)]

    # 2. Knees + hips: the shank ~ the thigh, so the ankle sits about as far below
    #    the knee as the knee is below the hip. Extrapolate one segment down.
    if knees and hips:
        kc, hc = mean_pts(knees), mean_pts(hips)
        seg = kc - hc
        if seg[1] > 0:                              # knees below hips: upright
            foot = kc + seg
            return tuple(foot), max(0.35 * np.linalg.norm(seg), 6.0), "knees"

    shoulders = [i for i in (_L_SHO, _R_SHO) if vis(i)]

    # 3. Hips + shoulders: no legs at all. The hip-to-ankle span is ~1.7x the
    #    shoulder-to-hip span on a standing body; extrapolate down that axis.
    if hips and shoulders:
        hc, sc = mean_pts(hips), mean_pts(shoulders)
        axis = hc - sc
        if axis[1] > 0:                             # hips below shoulders: upright
            foot = hc + 1.7 * axis
            return tuple(foot), max(0.4 * np.linalg.norm(1.7 * axis), 8.0), "hips"

    # 4. Nothing usable -> fall back to the box bottom (very uncertain).
    return ((x1 + x2) / 2.0, y2), 0.5 * box_h, "box"
