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
        return PointCloudPlan(
            fpc.get("pointcloud_file", "smart_lab.las"),
            px_per_m=px_per_m, margin_m=margin_m,
            up_axis=fpc.get("up_axis", "auto"),
            color_mode=fpc.get("color_mode", "height"),
            ceiling_trim_m=fpc.get("ceiling_trim_m", 0.4),
            clip_percentile=fpc.get("clip_percentile", 0.2),
            flip_x=fpc.get("flip_x", False),
            flip_y=fpc.get("flip_y", False),
        )
    if source == "dxf":
        from floorplan import Floorplan
        return Floorplan(fpc["file"], px_per_m=px_per_m, margin_m=margin_m)
    raise ValueError(f"Unknown floorplan.source '{source}' (use 'pointcloud' or 'dxf').")
