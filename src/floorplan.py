"""Load a DXF floorplan, render it as a top-down map, and map world<->pixels.

World frame (shared by every camera):
  - units: meters
  - origin (0,0): bottom-left corner of the floorplan bounding box
  - +X to the right, +Y "up" (away). Z is not modeled (flat floor).

The same world_to_px transform draws the floorplan AND places tracked people,
so a person's (X,Y) in meters always lands in the right spot on the map.
"""
import math
import cv2
import numpy as np
import ezdxf


def _arc_points(cx, cy, r, a0_deg, a1_deg, step_deg=6.0):
    if a1_deg < a0_deg:
        a1_deg += 360.0
    n = max(2, int((a1_deg - a0_deg) / step_deg) + 1)
    pts = []
    for i in range(n):
        a = math.radians(a0_deg + (a1_deg - a0_deg) * i / (n - 1))
        pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    return pts


class Floorplan:
    def __init__(self, dxf_path, px_per_m=50, margin_m=0.5, line_color=(180, 180, 180)):
        self.px_per_m = float(px_per_m)
        self.margin_m = float(margin_m)

        doc = ezdxf.readfile(dxf_path)
        msp = doc.modelspace()
        # DXF is in millimeters (INSUNITS=4). Collect geometry in meters.
        segments = []   # list of polylines, each a list of (x_m, y_m)
        xs, ys = [], []
        for e in msp:
            t = e.dxftype()
            if t == "LINE":
                p = [(e.dxf.start.x / 1000.0, e.dxf.start.y / 1000.0),
                     (e.dxf.end.x / 1000.0, e.dxf.end.y / 1000.0)]
            elif t == "CIRCLE":
                c = e.dxf.center
                p = _arc_points(c.x / 1000.0, c.y / 1000.0, e.dxf.radius / 1000.0, 0, 360)
            elif t == "ARC":
                c = e.dxf.center
                p = _arc_points(c.x / 1000.0, c.y / 1000.0, e.dxf.radius / 1000.0,
                                e.dxf.start_angle, e.dxf.end_angle)
            else:
                continue
            segments.append(p)
            for x, y in p:
                xs.append(x); ys.append(y)

        min_x, min_y = min(xs), min(ys)
        self.width_m = max(xs) - min_x
        self.height_m = max(ys) - min_y
        # shift geometry so the bbox min corner becomes world origin (0, 0)
        segments = [[(x - min_x, y - min_y) for x, y in poly] for poly in segments]

        self.canvas_w = int((self.width_m + 2 * self.margin_m) * self.px_per_m)
        self.canvas_h = int((self.height_m + 2 * self.margin_m) * self.px_per_m)

        # pre-render the static background once; collect vertices for snapping
        bg = np.full((self.canvas_h, self.canvas_w, 3), 25, dtype=np.uint8)
        verts = []
        for poly in segments:
            pix = np.array([self.world_to_px(x, y) for x, y in poly], dtype=np.int32)
            cv2.polylines(bg, [pix], False, line_color, 1, cv2.LINE_AA)
            verts.extend(poly)
        self._bg = bg
        self.vertices = np.array(verts, dtype=np.float32)   # (N, 2) world meters

    def snap(self, wx, wy, tol_m=0.15):
        """Nearest DXF vertex to (wx, wy) within tol_m, else the point itself."""
        if len(self.vertices) == 0:
            return (wx, wy), False
        d = np.hypot(self.vertices[:, 0] - wx, self.vertices[:, 1] - wy)
        i = int(np.argmin(d))
        if d[i] <= tol_m:
            return (float(self.vertices[i, 0]), float(self.vertices[i, 1])), True
        return (wx, wy), False

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
        """True if a world point is inside the floorplan (+/- tol meters)."""
        return (-tol <= wx <= self.width_m + tol and
                -tol <= wy <= self.height_m + tol)

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
