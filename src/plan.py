"""Shared top-down metric map: world<->pixel transforms, rendering, snapping.

A "plan" is any top-down map of the space in a common world frame:
  - units: meters
  - origin (0,0): bottom-left corner of the map's bounding box
  - +X to the right, +Y "up" (away). The vertical axis is not modeled here.

Two concrete plans share this frame and API so every tool (calibrate, track,
measure, recalibrate) works against either:
  - Floorplan       (floorplan.py)        -> rasterized from the DXF/BIM
  - PointCloudPlan  (pointcloud_plan.py)  -> top-down view of the .las scan

`load_plan(cfg)` picks one from config (`floorplan.source`). Subclasses must set
these attributes in __init__: px_per_m, margin_m, width_m, height_m, canvas_w,
canvas_h, _bg (HxWx3 uint8 background image), vertices ((N,2) float32 snap pts).
"""
import cv2
import numpy as np


class PlanBase:
    # ---- coordinate transforms ----------------------------------------------
    def world_to_px(self, wx, wy):
        """world meters (origin at bbox min corner) -> canvas pixel."""
        px = int((wx + self.margin_m) * self.px_per_m)
        py = int(self.canvas_h - (wy + self.margin_m) * self.px_per_m)
        return px, py

    def px_to_world(self, px, py):
        """canvas pixel -> world meters (origin at bbox min corner)."""
        wx = px / self.px_per_m - self.margin_m
        wy = (self.canvas_h - py) / self.px_per_m - self.margin_m
        return wx, wy

    def in_bounds(self, wx, wy, tol=1.0):
        """True if a world point is inside the map (+/- tol meters)."""
        return (-tol <= wx <= self.width_m + tol and
                -tol <= wy <= self.height_m + tol)

    def in_valid_area(self, wx, wy):
        """True if a world point is inside the configured usable floor area."""
        if not self.in_bounds(wx, wy):
            return False
        poly = getattr(self, "valid_area", None)
        mask = getattr(self, "valid_mask", None)
        if mask is not None:
            px, py = self.world_to_px(wx, wy)
            if px < 0 or px >= self.canvas_w or py < 0 or py >= self.canvas_h:
                return False
            return bool(mask[py, px])
        if poly is None:
            return True
        tol = float(getattr(self, "valid_area_tol_m", 0.0))
        dist = cv2.pointPolygonTest(poly, (float(wx), float(wy)), True)
        return dist >= -tol

    def snap(self, wx, wy, tol_m=0.15):
        """Nearest snap vertex to (wx, wy) within tol_m, else the point itself.

        Returns ((x, y), snapped_bool). Plans with no vertices never snap.
        """
        if len(self.vertices) == 0:
            return (wx, wy), False
        d = np.hypot(self.vertices[:, 0] - wx, self.vertices[:, 1] - wy)
        i = int(np.argmin(d))
        if d[i] <= tol_m:
            return (float(self.vertices[i, 0]), float(self.vertices[i, 1])), True
        return (wx, wy), False

    # ---- rendering -----------------------------------------------------------
    def background(self):
        return self._bg.copy()

    def render(self, people, trails=None, color_fn=None):
        canvas = self._bg.copy()
        if trails:
            for tid, pts in trails.items():
                col = color_fn(tid) if color_fn else (0, 200, 255)
                for i in range(1, len(pts)):
                    if pts[i - 1] is None or pts[i] is None:
                        continue
                    cv2.line(canvas, self.world_to_px(*pts[i - 1]),
                             self.world_to_px(*pts[i]), col, 2, cv2.LINE_AA)
        for p in people:
            if p.get("world") is None:
                continue
            col = color_fn(p["id"]) if color_fn else (0, 200, 255)
            px, py = self.world_to_px(*p["world"])
            cv2.circle(canvas, (px, py), 7, col, -1, cv2.LINE_AA)
            cv2.putText(canvas, str(p["id"]), (px + 9, py),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)
        return canvas


def load_plan(cfg):
    """Build the top-down map selected by config `floorplan.source`.

    source: "pointcloud" (default) -> PointCloudPlan from `pointcloud_file`
            "dxf"                   -> Floorplan from `file` (legacy BIM)
    """
    fpc = cfg["floorplan"]
    source = str(fpc.get("source", "pointcloud")).lower()
    px_per_m = fpc.get("px_per_m", 50)
    margin_m = fpc.get("margin_m", 0.5)

    if source in ("pointcloud", "pc", "las", "scan"):
        from pointcloud_plan import PointCloudPlan
        plan = PointCloudPlan(
            fpc.get("pointcloud_file", "smart_lab.las"),
            px_per_m=px_per_m, margin_m=margin_m,
            up_axis=fpc.get("up_axis", "auto"),
            color_mode=fpc.get("color_mode", "rgb"),
            ceiling_trim_m=fpc.get("ceiling_trim_m", 0.4),
            clip_percentile=fpc.get("clip_percentile", 0.2),
            flip_x=fpc.get("flip_x", False),
            flip_y=fpc.get("flip_y", False),
            rgb_autocontrast=fpc.get("rgb_autocontrast", True),
        )
        _configure_valid_area(plan, fpc)
        return plan
    if source == "dxf":
        from floorplan import Floorplan
        plan = Floorplan(fpc["file"], px_per_m=px_per_m, margin_m=margin_m)
        _configure_valid_area(plan, fpc)
        return plan
    raise ValueError(f"Unknown floorplan.source '{source}' (use 'pointcloud' or 'dxf').")


def _configure_valid_area(plan, fpc):
    """Attach an optional room polygon used to reject impossible projections."""
    raw = fpc.get("valid_area", "auto")
    plan.valid_area = None
    plan.valid_mask = None
    plan.valid_area_tol_m = float(fpc.get("valid_area_tol_m", 0.0))
    if raw in (None, False) or raw == []:
        return
    if str(raw).lower() == "auto":
        plan.valid_mask = _auto_valid_mask(plan)
        return
    poly = np.asarray(raw, dtype=np.float32)
    if poly.ndim != 2 or poly.shape[1] != 2 or len(poly) < 3:
        raise ValueError("floorplan.valid_area must be 'auto' or a list of at least three [x, y] points")
    plan.valid_area = poly


def _auto_valid_mask(plan):
    """Build a usable-area mask from rendered map pixels.

    The renderers use a dark background for empty space. Anything drawn by the
    scan/DXF is treated as part of the usable footprint, then expanded/closed by
    the configured tolerance so sparse point clouds do not create tiny holes.
    """
    bg = plan.background()
    non_empty = np.max(np.abs(bg.astype(np.int16) - 25), axis=2) > 8
    tol_px = max(1, int(round(float(plan.valid_area_tol_m) * plan.px_per_m)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * tol_px + 1, 2 * tol_px + 1))
    mask = cv2.morphologyEx(non_empty.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    mask = cv2.dilate(mask, kernel)
    return mask.astype(bool)
