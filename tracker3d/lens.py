"""Lens model (intrinsics + distortion) and ChArUco calibration helpers.

The Yi cameras are wide-angle with barrel distortion: straight floor lines bow
toward the image edges, which a floor homography (a straight-line-preserving
warp) cannot represent. So every pixel is first moved to where an ideal pinhole
camera would have seen it, and the homography maps *those* pixels to the floor.

Coordinate spaces:
  raw pixels        what the camera delivers (the detector runs on these)
  rectified pixels  the undistorted image with camera matrix K_rect, chosen with
                    alpha=1 so the whole raw image stays visible. A homography
                    fitted with a lens file maps rectified pixels -> floor meters,
                    and the calibration tools display this same image, so a click
                    is directly a rectified coordinate.

Only the handful of projected points per detection are undistorted, never whole
frames, so this costs ~nothing in the tracking loop.

The lens file is calib/<cam>_intrinsics.json, written by tools/calib_intrinsics.py.
Its fingerprint is stored in every homography sidecar fitted with it, so a
homography is never silently used with a different lens model.
"""
import hashlib
import json
import os

import cv2
import numpy as np

# Default printed board: A3, 7x5 squares of 55 mm with 41 mm 4x4 markers. Large
# squares because the wide-angle cameras see the board small even from a ladder;
# an odd row count avoids OpenCV's legacy-pattern ambiguity for even rows.
BOARD_DEFAULTS = {"cols": 7, "rows": 5, "square_mm": 55.0, "marker_mm": 41.0,
                  "dictionary": "DICT_4X4_50"}
MIN_CORNERS = 8          # fewer ChArUco corners than this per view is not used
_LUT_STEP = 4            # raw -> rectified lookup table spacing (px)

# Only for coefficient sets not modelled here (thin prism / tilted sensor).
# OpenCV's default (5 iterations) is far off in the corners of a 110 deg lens.
_UNDISTORT_CRITERIA = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 300, 1e-12)


# --- ChArUco board ----------------------------------------------------------

def make_board(cols=7, rows=5, square_mm=55.0, marker_mm=41.0, dictionary="DICT_4X4_50"):
    """The board used both to print and to detect, so the two always match.
    Only the marker:square ratio matters for detection; the absolute size only
    scales the (unused) board-to-camera distance, never the lens parameters."""
    d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary))
    return cv2.aruco.CharucoBoard((int(cols), int(rows)), square_mm / 1000.0,
                                  marker_mm / 1000.0, d)


def board_from_params(p):
    return make_board(p["cols"], p["rows"], p["square_mm"], p["marker_mm"], p["dictionary"])


def make_detector(board):
    return cv2.aruco.CharucoDetector(board)


def detect_board(img, detector, board, min_corners=MIN_CORNERS):
    """Find ChArUco corners. Returns (obj_pts (N,3), img_pts (N,2), ids (N,)) as
    float32/int arrays, or None if too few corners (or they lie on one line,
    which carries no information about the lens)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    corners, ids, _, _ = detector.detectBoard(gray)
    if ids is None or len(ids) < min_corners:
        return None
    if board.checkCharucoCornersCollinear(ids):
        return None
    obj, imgp = board.matchImagePoints(corners, ids)
    return (obj.reshape(-1, 3).astype(np.float32), imgp.reshape(-1, 2).astype(np.float32),
            ids.ravel().astype(int))


# --- calibration solve --------------------------------------------------------

def _fisheye_flag(name):
    return getattr(cv2.fisheye, name, getattr(cv2, name, 0))


def _per_view_rms(proj, img):
    return np.array([float(np.sqrt(np.mean(np.sum((p.reshape(-1, 2) - i.reshape(-1, 2)) ** 2,
                                                   axis=1))))
                     for p, i in zip(proj, img)])


_CALIB_CRITERIA = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 100, 1e-10)


def _pinhole_path(obj, img, size, stages, K=None, D=None):
    """Run calibrateCamera through a sequence of flag sets, each starting from the
    previous result."""
    r = None
    for fl in stages:
        if K is not None:
            fl |= cv2.CALIB_USE_INTRINSIC_GUESS
            if D is None:
                D = np.zeros(8 if fl & cv2.CALIB_RATIONAL_MODEL else 5)
            elif fl & cv2.CALIB_RATIONAL_MODEL and len(D) < 8:
                D = np.concatenate([D, np.zeros(8 - len(D))])
        r = cv2.calibrateCamera(obj, img, size, K, D, flags=fl, criteria=_CALIB_CRITERIA)
        K, D = r[1], r[2].ravel()
    return r


def solve_intrinsics(obj_list, img_list, image_size, model="standard"):
    """Solve K + distortion from per-view board correspondences.

    model: "standard" (k1 k2 p1 p2 k3), "rational" (adds k4-k6, for stronger
    distortion) or "fisheye" (equidistant k1-k4).
    Returns {"K", "dist", "rms", "per_view"} (pixels).

    Wide-angle lenses with a board that looks small in the image have a
    false minimum ("zoomed-in lens with huge distortion") that OpenCV's default
    start often falls into. So the pinhole models are solved in stages (no
    distortion -> k1 -> k1,k2 -> all), from several starting focal lengths, each
    finished with a full refinement, and the lowest-error result wins."""
    size = (int(image_size[0]), int(image_size[1]))
    if model == "fisheye":
        obj = [np.asarray(o, np.float64).reshape(-1, 1, 3) for o in obj_list]
        img = [np.asarray(i, np.float64).reshape(-1, 1, 2) for i in img_list]
        flags = _fisheye_flag("CALIB_RECOMPUTE_EXTRINSIC") | _fisheye_flag("CALIB_FIX_SKEW")
        crit = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 200, 1e-9)
        rms, K, D, rvecs, tvecs = cv2.fisheye.calibrate(obj, img, size, np.eye(3), np.zeros(4),
                                                        flags=flags, criteria=crit)
        proj = [cv2.fisheye.projectPoints(o, r, t, K, D)[0] for o, r, t in zip(obj, rvecs, tvecs)]
        return {"K": np.asarray(K, np.float64), "dist": np.asarray(D, np.float64).ravel(),
                "rms": float(rms), "per_view": _per_view_rms(proj, img), "model": model}

    obj = [np.asarray(o, np.float32).reshape(-1, 3) for o in obj_list]
    img = [np.asarray(i, np.float32).reshape(-1, 2) for i in img_list]
    final = cv2.CALIB_RATIONAL_MODEL if model == "rational" else 0
    no_k = cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3 | cv2.CALIB_ZERO_TANGENT_DIST
    k1 = cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3 | cv2.CALIB_ZERO_TANGENT_DIST
    k12 = cv2.CALIB_FIX_K3 | cv2.CALIB_ZERO_TANGENT_DIST
    w, h = size
    starts = [None] + [w / 2 / np.tan(np.radians(fov / 2)) for fov in (90, 110, 130)]
    best = None
    for f0 in starts:
        try:
            if f0 is None:
                r = _pinhole_path(obj, img, size, [no_k, k1, k12, final, final])
            else:
                K0 = np.array([[f0, 0, w / 2], [0, f0, h / 2], [0, 0, 1.0]])
                r = _pinhole_path(obj, img, size, [k1, k12, final, final], K=K0)
        except cv2.error:
            continue
        if best is None or r[0] < best[0]:
            best = r
    if best is None:
        raise cv2.error("calibration failed from every starting point")
    rms, K, D, rvecs, tvecs = best
    proj = [cv2.projectPoints(o, r, t, K, D)[0] for o, r, t in zip(obj, rvecs, tvecs)]
    return {"K": np.asarray(K, np.float64), "dist": np.asarray(D, np.float64).ravel(),
            "rms": float(rms), "per_view": _per_view_rms(proj, img), "model": model}


def solve_with_outliers(obj_list, img_list, image_size, model="standard", min_views=10):
    """solve_intrinsics, then repeatedly drop views far worse than the rest (blur,
    a bent board, a misdetection) and re-solve. One bad view can drag the whole
    solve into a false minimum, so this runs until nothing is dropped (max 3x).
    Returns (result, keep mask over the input views)."""
    keep = np.ones(len(obj_list), bool)
    res = solve_intrinsics(obj_list, img_list, image_size, model)
    for _ in range(3):
        med = float(np.median(res["per_view"]))
        bad = res["per_view"] > max(2.5 * med, 0.5)
        if not bad.any() or keep.sum() - bad.sum() < min_views:
            break
        keep[np.where(keep)[0][bad]] = False
        res = solve_intrinsics([o for o, k in zip(obj_list, keep) if k],
                               [i for i, k in zip(img_list, keep) if k], image_size, model)
    return res, keep


# --- lens model -----------------------------------------------------------------

def _scale_K(K, sx, sy):
    S = np.diag([sx, sy, 1.0])
    return S @ np.asarray(K, np.float64)


class LensModel:
    """One camera's intrinsics + distortion at a given image size."""

    def __init__(self, K, dist, image_size, model="standard", K_rect=None, path=None,
                 meta=None):
        self.K = np.asarray(K, np.float64)
        self.dist = np.asarray(dist, np.float64).ravel()
        self.image_size = (int(image_size[0]), int(image_size[1]))
        self.model = model
        self.path = path
        self.meta = meta or {}
        if K_rect is None:
            K_rect = self._optimal_rect_K()
        self.K_rect = np.asarray(K_rect, np.float64)
        self._K_inv = np.linalg.inv(self.K)
        self._K_rect_inv = np.linalg.inv(self.K_rect)
        self._maps = {}
        self._table = None

    def _optimal_rect_K(self):
        w, h = self.image_size
        if self.model == "fisheye":
            return cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
                self.K, self.dist, (w, h), np.eye(3), balance=1.0)
        K_new, _ = cv2.getOptimalNewCameraMatrix(self.K, self.dist, (w, h), 1.0, (w, h))
        return K_new

    # ---- persistence ----
    @classmethod
    def load(cls, path):
        with open(path) as f:
            d = json.load(f)
        return cls(d["camera_matrix"], d["dist_coeffs"], d["image_size"],
                   d.get("model", "standard"), d.get("rectified_camera_matrix"), path=path,
                   meta=d)

    def to_dict(self):
        return {"image_size": list(self.image_size), "model": self.model,
                "camera_matrix": self.K.tolist(), "dist_coeffs": self.dist.tolist(),
                "rectified_camera_matrix": self.K_rect.tolist(),
                "fingerprint": self.fingerprint()}

    def fingerprint(self):
        """Short hash of everything that defines rectified pixel space."""
        parts = [self.model, str(self.image_size)]
        for a in (self.K, self.dist, self.K_rect):
            parts.append(",".join(f"{v:.9g}" for v in np.ravel(a)))
        return hashlib.sha1("|".join(parts).encode()).hexdigest()[:12]

    def scaled(self, w, h):
        """The same lens for frames delivered at another resolution."""
        if (int(w), int(h)) == self.image_size:
            return self
        sx, sy = w / self.image_size[0], h / self.image_size[1]
        return LensModel(_scale_K(self.K, sx, sy), self.dist, (w, h), self.model,
                         _scale_K(self.K_rect, sx, sy), path=self.path, meta=self.meta)

    # ---- point mapping ----
    def _supported(self):
        """Distortion models evaluated here: fisheye (k1-k4) and pinhole with
        4/5/8 coefficients. Anything else (thin prism, tilted) goes through OpenCV."""
        return self.model == "fisheye" or len(self.dist) in (4, 5, 8)

    def _distort_norm(self, x, y):
        """OpenCV's distortion model on normalized coordinates (ideal -> distorted)."""
        if self.model == "fisheye":
            k1, k2, k3, k4 = self.dist[:4]
            r = np.hypot(x, y)
            t = np.arctan(r)
            t2 = t * t
            td = t * (1 + t2 * (k1 + t2 * (k2 + t2 * (k3 + t2 * k4))))
            sc = np.where(r > 1e-12, td / np.maximum(r, 1e-12), 1.0)
            return x * sc, y * sc
        k1, k2, p1, p2, k3, k4, k5, k6 = np.concatenate([self.dist, np.zeros(8 - len(self.dist))])
        r2 = x * x + y * y
        radial = (1 + r2 * (k1 + r2 * (k2 + r2 * k3))) / (1 + r2 * (k4 + r2 * (k5 + r2 * k6)))
        xd = x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
        yd = y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
        return xd, yd

    def _undistort_norm(self, xd, yd):
        """Invert the distortion with Newton's method (numerical Jacobian): ~1e-12
        in a handful of iterations across the whole field of view. (OpenCV's
        default fixed-point iteration is off by hundreds of pixels in the corners
        of a 110 deg lens.)"""
        x, y = xd.copy(), yd.copy()
        h = 1e-7
        for _ in range(50):
            fx, fy = self._distort_norm(x, y)
            ex, ey = fx - xd, fy - yd
            if max(np.abs(ex).max(), np.abs(ey).max()) < 1e-13:
                break
            ax, ay = self._distort_norm(x + h, y)
            bx, by = self._distort_norm(x, y + h)
            j00, j01, j10, j11 = (ax - fx) / h, (bx - fx) / h, (ay - fy) / h, (by - fy) / h
            det = j00 * j11 - j01 * j10
            det = np.where(np.abs(det) < 1e-12, 1e-12, det)
            x = x - (j11 * ex - j01 * ey) / det
            y = y - (-j10 * ex + j00 * ey) / det
        return x, y

    def _undistort_exact(self, pts):
        """Raw px -> rectified px, solved per point (exact, ~0.1-0.3 ms per call)."""
        p = np.asarray(pts, np.float64).reshape(-1, 2)
        if not self._supported():
            src = p.reshape(-1, 1, 2)
            if hasattr(cv2, "undistortPointsIter"):                     # OpenCV 4.x
                out = cv2.undistortPointsIter(src, self.K, self.dist, np.eye(3), self.K_rect,
                                              _UNDISTORT_CRITERIA)
            else:                                                       # OpenCV 5.x
                out = cv2.undistortPoints(src, self.K, self.dist, R=np.eye(3), P=self.K_rect,
                                          criteria=_UNDISTORT_CRITERIA)
            return out.reshape(-1, 2)
        n = (self._K_inv @ np.column_stack([p, np.ones(len(p))]).T)
        x, y = self._undistort_norm(n[0] / n[2], n[1] / n[2])
        r = self.K_rect @ np.vstack([x, y, np.ones_like(x)])
        return (r[:2] / r[2]).T

    def _lut(self):
        """Dense raw -> rectified table every _LUT_STEP px (built once, ~0.2 s), so
        the per-detection lookups in the tracking loop are a cheap bilinear
        interpolation instead of an iterative solve."""
        if self._table is None:
            w, h = self.image_size
            gx = np.append(np.arange(0, w - 1, _LUT_STEP, dtype=np.float64), w - 1)
            gy = np.append(np.arange(0, h - 1, _LUT_STEP, dtype=np.float64), h - 1)
            uu, vv = np.meshgrid(gx, gy)
            r = self._undistort_exact(np.column_stack([uu.ravel(), vv.ravel()]))
            self._table = (gx, gy, r[:, 0].reshape(uu.shape), r[:, 1].reshape(uu.shape))
        return self._table

    def _undistort_inside(self, pts):
        gx, gy, tx, ty = self._lut()
        u, v = pts[:, 0], pts[:, 1]
        i = np.clip(np.searchsorted(gx, u, side="right") - 1, 0, len(gx) - 2)
        j = np.clip(np.searchsorted(gy, v, side="right") - 1, 0, len(gy) - 2)
        fu = (u - gx[i]) / (gx[i + 1] - gx[i])
        fv = (v - gy[j]) / (gy[j + 1] - gy[j])

        def bil(t):
            return ((1 - fu) * (1 - fv) * t[j, i] + fu * (1 - fv) * t[j, i + 1] +
                    (1 - fu) * fv * t[j + 1, i] + fu * fv * t[j + 1, i + 1])
        return np.column_stack([bil(tx), bil(ty)])

    def undistort_points(self, pts):
        """Raw pixels (N,2) -> rectified pixels (N,2).

        The distortion polynomial is only fitted inside the image, and can fold
        back on itself beyond it. Points outside the frame (feet extrapolated
        from pose, trapezoid rows below the frame) are therefore extrapolated
        linearly from the nearest in-frame point using the local Jacobian."""
        pts = np.asarray(pts, np.float64).reshape(-1, 2)
        out = np.empty_like(pts)
        if not len(pts):
            return out
        w, h = self.image_size
        inside = ((pts[:, 0] >= 0) & (pts[:, 0] <= w - 1) &
                  (pts[:, 1] >= 0) & (pts[:, 1] <= h - 1))
        if inside.all():
            return self._undistort_inside(pts)
        if inside.any():
            out[inside] = self._undistort_inside(pts[inside])
        q = pts[~inside]
        c = np.column_stack([np.clip(q[:, 0], 0, w - 1), np.clip(q[:, 1], 0, h - 1)])
        e = float(_LUT_STEP)                        # inward finite-difference step (px)
        sx = np.where(c[:, 0] > w / 2, -e, e)
        sy = np.where(c[:, 1] > h / 2, -e, e)
        zero = np.zeros_like(sx)
        r0 = self._undistort_inside(c)
        jx = (self._undistort_inside(c + np.column_stack([sx, zero])) - r0) / sx[:, None]
        jy = (self._undistort_inside(c + np.column_stack([zero, sy])) - r0) / sy[:, None]
        d = q - c
        out[~inside] = r0 + jx * d[:, :1] + jy * d[:, 1:]
        return out

    def distort_points(self, rect_pts):
        """Rectified pixels (N,2) -> raw pixels (N,2) (inverse of undistort_points)."""
        r = np.asarray(rect_pts, np.float64).reshape(-1, 2)
        if not len(r):
            return r.copy()
        n = self._K_rect_inv @ np.column_stack([r, np.ones(len(r))]).T
        x, y = n[0] / n[2], n[1] / n[2]
        if not self._supported():
            obj = np.column_stack([x, y, np.ones_like(x)]).reshape(-1, 1, 3)
            out, _ = cv2.projectPoints(obj, np.zeros(3), np.zeros(3), self.K, self.dist)
            return out.reshape(-1, 2)
        xd, yd = self._distort_norm(x, y)
        out = self.K @ np.vstack([xd, yd, np.ones_like(xd)])
        return (out[:2] / out[2]).T

    def rectify_bbox(self, bbox):
        """Axis-aligned rectified box enclosing a raw (x1, y1, x2, y2) box (its edges
        curve after undistortion, so sample corners and edge midpoints)."""
        x1, y1, x2, y2 = (float(v) for v in bbox)
        xm, ym = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
        r = self.undistort_points([(x1, y1), (xm, y1), (x2, y1), (x2, ym),
                                   (x2, y2), (xm, y2), (x1, y2), (x1, ym)])
        return (float(r[:, 0].min()), float(r[:, 1].min()),
                float(r[:, 0].max()), float(r[:, 1].max()))

    # ---- images (display only) ----
    def undistort_image(self, img, out_size=None):
        """Undistort a raw frame for display. The output is rectified pixel space
        scaled to `out_size` (default: the input size), so a display pixel times
        (image_size / out_size) is a rectified coordinate."""
        h, w = img.shape[:2]
        out_size = (int(out_size[0]), int(out_size[1])) if out_size else (w, h)
        key = (w, h, out_size)
        if key not in self._maps:
            W, H = self.image_size
            K_in = _scale_K(self.K, w / W, h / H)
            P = _scale_K(self.K_rect, out_size[0] / W, out_size[1] / H)
            if self.model == "fisheye":
                m = cv2.fisheye.initUndistortRectifyMap(K_in, self.dist, np.eye(3), P,
                                                        out_size, cv2.CV_16SC2)
            else:
                m = cv2.initUndistortRectifyMap(K_in, self.dist, None, P, out_size,
                                                cv2.CV_16SC2)
            self._maps[key] = m
        m1, m2 = self._maps[key]
        return cv2.remap(img, m1, m2, cv2.INTER_LINEAR)


def load_lens(cam):
    """(LensModel or None, status text) for one merged camera dict. A configured
    but missing file is not an error: the camera simply runs without correction."""
    path = cam.get("intrinsics_file")
    if not path:
        return None, "off (no intrinsics_file)"
    if not os.path.exists(path):
        return None, f"off — {path} not found (run tools/calib_intrinsics.py)"
    lens = LensModel.load(path)
    rms = lens.meta.get("rms_px")
    rms_txt = f", rms {rms:.2f} px" if isinstance(rms, (int, float)) else ""
    return lens, f"on ({lens.model}{rms_txt}, {lens.fingerprint()})"
