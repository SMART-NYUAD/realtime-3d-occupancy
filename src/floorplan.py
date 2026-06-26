"""Load a DXF floorplan, render it as a top-down map, and map world<->pixels.

World frame (shared by every camera) and the world<->pixel transforms live in
plan.PlanBase; this class just rasterizes the DXF into that frame. See plan.py.
"""
import math
import cv2
import numpy as np
import ezdxf

from plan import PlanBase


def _arc_points(cx, cy, r, a0_deg, a1_deg, step_deg=6.0):
    if a1_deg < a0_deg:
        a1_deg += 360.0
    n = max(2, int((a1_deg - a0_deg) / step_deg) + 1)
    pts = []
    for i in range(n):
        a = math.radians(a0_deg + (a1_deg - a0_deg) * i / (n - 1))
        pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    return pts


class Floorplan(PlanBase):
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

    # snap (to DXF corners), world<->px transforms, in_bounds, background and
    # render are all inherited from plan.PlanBase.
