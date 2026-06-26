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
