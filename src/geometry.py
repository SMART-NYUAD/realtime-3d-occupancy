"""Image <-> floor (world) coordinate mapping via a ground-plane homography.

A person standing on a flat floor has their feet (bottom-center of the bounding
box) ON the ground plane. A homography H maps any image point that lies on the
ground plane to its real-world floor coordinate in meters:

    [Xw*s, Yw*s, s]^T = H * [u, v, 1]^T   ->   (Xw, Yw) = (Xw*s/s, Yw*s/s)
"""
import numpy as np
import cv2


def compute_homography(image_points, world_points):
    """image_points, world_points: (N>=4, 2) arrays. Returns 3x3 H (img->world)."""
    img = np.asarray(image_points, dtype=np.float64)
    wld = np.asarray(world_points, dtype=np.float64)
    if len(img) < 4:
        raise ValueError("Need at least 4 point correspondences for a homography.")
    H, _ = cv2.findHomography(img, wld, method=cv2.RANSAC)
    if H is None:
        raise RuntimeError("findHomography failed — check your point correspondences.")
    return H


def save_homography(H, path):
    np.save(path, H)


def load_homography(path):
    H = np.load(path)
    if abs(np.linalg.det(H)) < 1e-9:
        raise ValueError(
            f"Homography in '{path}' is degenerate (singular). Re-run calibration "
            "with 4 well-spread, non-collinear floor points."
        )
    return H


def image_to_floor(H, u, v):
    """Map a single image pixel (u, v) on the ground plane to (Xw, Yw) meters."""
    p = np.array([u, v, 1.0], dtype=np.float64)
    w = H @ p
    if abs(w[2]) < 1e-9:
        return None
    return float(w[0] / w[2]), float(w[1] / w[2])


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
