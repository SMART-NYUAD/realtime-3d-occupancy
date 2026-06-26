"""Top-down ("bottom-up") metric map rendered from a .las point cloud.

The lab scan (`smart_lab.las`) is to scale (meters) but in its OWN local frame:
it has no CRS, and it is Y-up (the floor is the X-Z plane, Y is height) — unlike
the DXF, which is Z-up. This class projects the cloud straight down onto its
floor plane and rasterizes it into a top-down image, then exposes the SAME world
frame + transforms as Floorplan (plan.PlanBase) so calibrate/track/measure work
unchanged.

World frame here:
  - origin (0,0): min corner of the (outlier-clipped) floor footprint
  - +X / +Y : the two horizontal axes, in meters
  - the vertical axis (height) is only used for shading / ceiling removal

Because this is a different frame from the DXF (different origin, orientation
and up-axis), homographies/reference points calibrated against the BIM do NOT
carry over — recalibrate each camera after switching `floorplan.source`.

The rasterized background is cached next to the .las (`<file>.topdown.npz`) and
reused while the file and render parameters are unchanged, so startup is fast.
"""
import json
import os

import cv2
import numpy as np

from plan import PlanBase

_AXIS = {"x": 0, "y": 1, "z": 2}


class PointCloudPlan(PlanBase):
    def __init__(self, las_path, px_per_m=50, margin_m=0.5, up_axis="auto",
                 color_mode="rgb", ceiling_trim_m=0.4, clip_percentile=0.2,
                 flip_x=False, flip_y=False, point_px=1, rgb_autocontrast=True,
                 cache=True):
        self.px_per_m = float(px_per_m)
        self.margin_m = float(margin_m)
        self.vertices = np.empty((0, 2), np.float32)   # no corner snapping for a cloud

        params = dict(px_per_m=self.px_per_m, margin_m=self.margin_m,
                      up_axis=str(up_axis).lower(), color_mode=color_mode,
                      ceiling_trim_m=ceiling_trim_m, clip_percentile=clip_percentile,
                      flip_x=bool(flip_x), flip_y=bool(flip_y), point_px=int(point_px),
                      rgb_autocontrast=bool(rgb_autocontrast))

        cached = self._load_cache(las_path, params) if cache else None
        if cached is not None:
            self._bg, self.width_m, self.height_m, self.up_axis = cached
            print(f"[pointcloud] {os.path.basename(las_path)}: cached top-down map "
                  f"{self.width_m:.2f} x {self.height_m:.2f} m (up={self.up_axis})")
        else:
            self._build(las_path, params)
            if cache:
                self._save_cache(las_path, params)
        self.canvas_h, self.canvas_w = self._bg.shape[:2]

    # ---- build ---------------------------------------------------------------
    def _build(self, las_path, p):
        import laspy
        print(f"[pointcloud] reading {os.path.basename(las_path)} ...")
        las = laspy.read(las_path)
        pts = np.vstack([las.x, las.y, las.z]).T.astype(np.float64)

        up = self._detect_up_axis(pts) if p["up_axis"] == "auto" else _AXIS[p["up_axis"]]
        self.up_axis = "xyz"[up]
        fa, fb = [a for a in (0, 1, 2) if a != up]      # the two floor (horizontal) axes
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
            if rgb.max() > 300:          # 16-bit colour -> 8-bit
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

        self.width_m = float(xhi - xlo)
        self.height_m = float(yhi - ylo)
        W = max(1, int((self.width_m + 2 * self.margin_m) * self.px_per_m))
        H = max(1, int((self.height_m + 2 * self.margin_m) * self.px_per_m))
        # Pixel coords consistent with PlanBase.world_to_px (world origin = floor min corner).
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
                sel = h <= (h.max() - trim)     # drop the ceiling so furniture/floor show
            fl = flat[sel]
            col = rgb[sel].astype(np.float64)   # (M,3) in R,G,B order
            npix = H * W
            cnt = np.bincount(fl, minlength=npix).astype(np.float64)
            written = cnt > 0
            bg = np.full((npix, 3), 25.0)       # BGR for OpenCV
            for c_out, c_in in ((0, 2), (1, 1), (2, 0)):   # B<-blue, G<-green, R<-red
                s = np.bincount(fl, weights=col[:, c_in], minlength=npix)
                bg[written, c_out] = s[written] / cnt[written]   # mean colour per pixel
            if p["rgb_autocontrast"]:
                bg = self._autocontrast(bg, written)             # lift the flat indoor greys
            bg = bg.clip(0, 255).astype(np.uint8).reshape(H, W, 3)
        else:                                   # height colormap (RGB-independent)
            order = np.argsort(h)               # topmost point wins each pixel
            himg = np.full(H * W, -np.inf)
            himg[flat[order]] = h[order]
            himg = himg.reshape(H, W)
            valid = np.isfinite(himg)
            hn = np.zeros((H, W), np.uint8)
            if valid.any():
                hmin, hmax = float(h.min()), float(h.max())
                hn[valid] = ((himg[valid] - hmin) / max(hmax - hmin, 1e-9) * 255).astype(np.uint8)
            bg = cv2.applyColorMap(hn, cv2.COLORMAP_JET)
            bg[~valid] = (25, 25, 25)

        pp = int(p["point_px"])
        if pp > 1:                              # thicken sparse points for a denser look
            bg = cv2.dilate(bg, np.ones((pp, pp), np.uint8))
        self._bg = bg
        print(f"[pointcloud] {os.path.basename(las_path)}: built top-down map "
              f"{self.width_m:.2f} x {self.height_m:.2f} m, {W}x{H}px "
              f"(up={self.up_axis}, mode={p['color_mode']}, {int(keep.sum())} pts)")

    @staticmethod
    def _autocontrast(bg, written, lo=1.0, hi=99.0):
        """Stretch the [lo, hi] luminance percentiles to 0..255, applying the SAME
        affine to all channels so colour balance (and 'realness') is preserved."""
        if not written.any():
            return bg
        lum = bg[written].mean(axis=1)
        a, b = np.percentile(lum, [lo, hi])
        if b - a < 1e-6:
            return bg
        out = bg.copy()
        out[written] = (bg[written] - a) * (255.0 / (b - a))
        return out

    @staticmethod
    def _detect_up_axis(pts, slab_m=0.2):
        """The vertical axis is the one with the strongest floor: the most points
        packed into a single horizontal slab."""
        best_frac, best = -1.0, 1
        for a in range(3):
            v = pts[:, a]
            lo, hi = float(v.min()), float(v.max())
            if hi - lo < 1e-6:
                return a
            bins = np.arange(lo, hi + slab_m, slab_m)
            frac = np.histogram(v, bins=bins)[0].max() / len(v)
            if frac > best_frac:
                best_frac, best = frac, a
        return best

    # ---- background cache ----------------------------------------------------
    def _cache_path(self, las_path):
        return las_path + ".topdown.npz"

    def _key(self, las_path, params):
        st = os.stat(las_path)
        return json.dumps({**params, "mtime": st.st_mtime, "size": st.st_size},
                          sort_keys=True)

    def _load_cache(self, las_path, params):
        path = self._cache_path(las_path)
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
            np.savez_compressed(self._cache_path(las_path), bg=self._bg,
                                width_m=self.width_m, height_m=self.height_m,
                                up_axis=self.up_axis, key=self._key(las_path, params))
        except Exception as e:
            print(f"[pointcloud] cache write skipped: {e}")
