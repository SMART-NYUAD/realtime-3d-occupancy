"""The shared world frame: a top-down metric map rendered from the lab's .las scan.

World frame:
  - units: meters
  - origin (0, 0): min corner of the (outlier-clipped) floor footprint
  - +X right, +Y up on the map; height is only used for ceiling removal/shading

The scan (`data/smart_lab.las`) is to scale but in its own local frame (no CRS,
Y-up). It is projected straight down and rasterized; the render is cached next
to the .las (`<file>.topdown.npz`) and reused while the file and render
parameters are unchanged.

Every camera's homography maps into THIS frame, so positions agree across
cameras. Changing render parameters that move the frame (flip_x/flip_y,
clip_percentile, up_axis) invalidates every calibration.
"""
import json
import os

import cv2
import numpy as np

_AXIS = {"x": 0, "y": 1, "z": 2}
_EMPTY = 25            # background grey for empty map pixels


class PointCloudPlan:
    def __init__(self, las_path, px_per_m=50, margin_m=0.5, up_axis="auto",
                 color_mode="rgb", ceiling_trim_m=0.4, clip_percentile=0.2,
                 flip_x=False, flip_y=False, rgb_autocontrast=True, cache=True):
        self.px_per_m = float(px_per_m)
        self.margin_m = float(margin_m)
        params = dict(px_per_m=self.px_per_m, margin_m=self.margin_m,
                      up_axis=str(up_axis).lower(), color_mode=color_mode,
                      ceiling_trim_m=ceiling_trim_m, clip_percentile=clip_percentile,
                      flip_x=bool(flip_x), flip_y=bool(flip_y), point_px=1,
                      rgb_autocontrast=bool(rgb_autocontrast))
        cached = self._load_cache(las_path, params) if cache else None
        if cached is not None:
            self._bg, self.width_m, self.height_m, self.up_axis = cached
            print(f"[map] {os.path.basename(las_path)}: cached top-down map "
                  f"{self.width_m:.2f} x {self.height_m:.2f} m")
        else:
            self._build(las_path, params)
            if cache:
                self._save_cache(las_path, params)
        self.canvas_h, self.canvas_w = self._bg.shape[:2]
        self.valid_mask = None
        self.valid_area = None
        self.valid_area_tol_m = 0.0

    # ---- coordinate transforms ----------------------------------------------
    def world_to_px(self, wx, wy):
        return (int((wx + self.margin_m) * self.px_per_m),
                int(self.canvas_h - (wy + self.margin_m) * self.px_per_m))

    def px_to_world(self, px, py):
        return (px / self.px_per_m - self.margin_m,
                (self.canvas_h - py) / self.px_per_m - self.margin_m)

    def in_bounds(self, wx, wy, tol=1.0):
        return -tol <= wx <= self.width_m + tol and -tol <= wy <= self.height_m + tol

    def in_valid_area(self, wx, wy):
        """True if a world point is inside the usable floor area."""
        if not self.in_bounds(wx, wy):
            return False
        if self.valid_mask is not None:
            px, py = self.world_to_px(wx, wy)
            return (0 <= px < self.canvas_w and 0 <= py < self.canvas_h
                    and bool(self.valid_mask[py, px]))
        if self.valid_area is not None:
            d = cv2.pointPolygonTest(self.valid_area, (float(wx), float(wy)), True)
            return d >= -self.valid_area_tol_m
        return True

    def background(self):
        return self._bg.copy()

    # ---- build ---------------------------------------------------------------
    def _build(self, las_path, p):
        import laspy
        print(f"[map] reading {os.path.basename(las_path)} ...")
        las = laspy.read(las_path)
        pts = np.vstack([las.x, las.y, las.z]).T.astype(np.float64)

        up = self._detect_up_axis(pts) if p["up_axis"] == "auto" else _AXIS[p["up_axis"]]
        self.up_axis = "xyz"[up]
        fa, fb = [a for a in (0, 1, 2) if a != up]
        fx, fy, h = pts[:, fa].copy(), pts[:, fb].copy(), pts[:, up]
        # Flips mirror the world axis itself (not just the image) so world<->px stays valid.
        if p["flip_x"]:
            fx = -fx
        if p["flip_y"]:
            fy = -fy

        has_rgb = {"red", "green", "blue"}.issubset(set(las.point_format.dimension_names))
        rgb = None
        if has_rgb:
            rgb = np.vstack([las.red, las.green, las.blue]).T.astype(np.float64)
            if rgb.max() > 300:                      # 16-bit colour -> 8-bit
                rgb /= 256.0
            rgb = rgb.clip(0, 255).astype(np.uint8)

        # Robust footprint: percentile-clip strays so a few far points don't blow up the canvas.
        c = float(p["clip_percentile"])
        xlo, xhi = np.percentile(fx, [c, 100 - c])
        ylo, yhi = np.percentile(fy, [c, 100 - c])
        keep = (fx >= xlo) & (fx <= xhi) & (fy >= ylo) & (fy <= yhi)
        fx, fy, h = fx[keep], fy[keep], h[keep]
        if has_rgb:
            rgb = rgb[keep]

        self.width_m, self.height_m = float(xhi - xlo), float(yhi - ylo)
        W = max(1, int((self.width_m + 2 * self.margin_m) * self.px_per_m))
        H = max(1, int((self.height_m + 2 * self.margin_m) * self.px_per_m))
        u = ((fx - xlo + self.margin_m) * self.px_per_m).astype(np.int32)
        v = (H - ((fy - ylo + self.margin_m) * self.px_per_m)).astype(np.int32)
        inb = (u >= 0) & (u < W) & (v >= 0) & (v < H)
        u, v, h = u[inb], v[inb], h[inb]
        if has_rgb:
            rgb = rgb[inb]
        flat = v * W + u

        if p["color_mode"] == "rgb" and has_rgb:
            sel = np.ones(len(h), bool)
            trim = float(p["ceiling_trim_m"])
            if trim > 0 and len(h):
                sel = h <= (h.max() - trim)          # drop the ceiling so the floor shows
            fl, col = flat[sel], rgb[sel].astype(np.float64)
            npix = H * W
            cnt = np.bincount(fl, minlength=npix).astype(np.float64)
            written = cnt > 0
            bg = np.full((npix, 3), float(_EMPTY))
            for c_out, c_in in ((0, 2), (1, 1), (2, 0)):   # BGR <- RGB, mean colour per pixel
                s = np.bincount(fl, weights=col[:, c_in], minlength=npix)
                bg[written, c_out] = s[written] / cnt[written]
            if p["rgb_autocontrast"]:
                bg = self._autocontrast(bg, written)
            bg = bg.clip(0, 255).astype(np.uint8).reshape(H, W, 3)
        else:                                        # height colormap
            order = np.argsort(h)                    # topmost point wins each pixel
            himg = np.full(H * W, -np.inf)
            himg[flat[order]] = h[order]
            himg = himg.reshape(H, W)
            valid = np.isfinite(himg)
            hn = np.zeros((H, W), np.uint8)
            if valid.any():
                hmin, hmax = float(h.min()), float(h.max())
                hn[valid] = ((himg[valid] - hmin) / max(hmax - hmin, 1e-9) * 255).astype(np.uint8)
            bg = cv2.applyColorMap(hn, cv2.COLORMAP_JET)
            bg[~valid] = (_EMPTY,) * 3
        self._bg = bg
        print(f"[map] {os.path.basename(las_path)}: built top-down map "
              f"{self.width_m:.2f} x {self.height_m:.2f} m, {W}x{H}px (up={self.up_axis})")

    @staticmethod
    def _autocontrast(bg, written, lo=1.0, hi=99.0):
        """Stretch luminance percentiles to 0..255 with one affine for all channels."""
        if not written.any():
            return bg
        a, b = np.percentile(bg[written].mean(axis=1), [lo, hi])
        if b - a < 1e-6:
            return bg
        out = bg.copy()
        out[written] = (bg[written] - a) * (255.0 / (b - a))
        return out

    @staticmethod
    def _detect_up_axis(pts, slab_m=0.2):
        """The vertical axis is the one with the most points in one horizontal slab (the floor)."""
        best_frac, best = -1.0, 1
        for a in range(3):
            v = pts[:, a]
            lo, hi = float(v.min()), float(v.max())
            if hi - lo < 1e-6:
                return a
            frac = np.histogram(v, bins=np.arange(lo, hi + slab_m, slab_m))[0].max() / len(v)
            if frac > best_frac:
                best_frac, best = frac, a
        return best

    # ---- render cache --------------------------------------------------------
    @staticmethod
    def _key(las_path, params):
        st = os.stat(las_path)
        return json.dumps({**params, "mtime": st.st_mtime, "size": st.st_size}, sort_keys=True)

    def _load_cache(self, las_path, params):
        path = las_path + ".topdown.npz"
        if not (os.path.exists(path) and os.path.exists(las_path)):
            return None
        try:
            d = np.load(path, allow_pickle=False)
            if str(d["key"]) != self._key(las_path, params):
                return None
            return d["bg"], float(d["width_m"]), float(d["height_m"]), str(d["up_axis"])
        except Exception:
            return None

    def _save_cache(self, las_path, params):
        try:
            np.savez_compressed(las_path + ".topdown.npz", bg=self._bg,
                                width_m=self.width_m, height_m=self.height_m,
                                up_axis=self.up_axis, key=self._key(las_path, params))
        except Exception as e:
            print(f"[map] cache write skipped: {e}")


def load_plan(cfg):
    """Build the shared top-down map from config `floorplan`."""
    fpc = cfg["floorplan"]
    plan = PointCloudPlan(
        fpc.get("pointcloud_file", "data/smart_lab.las"),
        px_per_m=fpc.get("px_per_m", 50), margin_m=fpc.get("margin_m", 0.5),
        up_axis=fpc.get("up_axis", "auto"), color_mode=fpc.get("color_mode", "rgb"),
        ceiling_trim_m=fpc.get("ceiling_trim_m", 0.4),
        clip_percentile=fpc.get("clip_percentile", 0.2),
        flip_x=fpc.get("flip_x", False), flip_y=fpc.get("flip_y", False),
        rgb_autocontrast=fpc.get("rgb_autocontrast", True))
    _configure_valid_area(plan, fpc)
    return plan


def _configure_valid_area(plan, fpc):
    """Attach the usable-floor gate: 'auto' (from the rendered scan), an explicit
    polygon [[x, y], ...], or [] / null to disable."""
    raw = fpc.get("valid_area", "auto")
    plan.valid_area_tol_m = float(fpc.get("valid_area_tol_m", 0.0))
    if raw in (None, False) or raw == []:
        return
    if str(raw).lower() == "auto":
        bg = plan.background()
        non_empty = np.max(np.abs(bg.astype(np.int16) - _EMPTY), axis=2) > 8
        tol_px = max(1, int(round(plan.valid_area_tol_m * plan.px_per_m)))
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * tol_px + 1, 2 * tol_px + 1))
        mask = cv2.morphologyEx(non_empty.astype(np.uint8), cv2.MORPH_CLOSE, k)
        plan.valid_mask = cv2.dilate(mask, k).astype(bool)
        return
    poly = np.asarray(raw, dtype=np.float32)
    if poly.ndim != 2 or poly.shape[1] != 2 or len(poly) < 3:
        raise ValueError("floorplan.valid_area must be 'auto', [] or a list of >= 3 [x, y] points")
    plan.valid_area = poly
