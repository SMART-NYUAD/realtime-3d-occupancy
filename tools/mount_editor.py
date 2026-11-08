"""Mark where cameras can be mounted: draw lines on the lab map in your browser, each
with its own height. Selecting a line shows the scan along it from floor to ceiling,
and clicking that side view sets the line's height. Every change is saved, and the
map shows the exact spots the planner will try. Then feed the file to the planner:

    python tools/mount_editor.py
    python tools/layout_planner.py --mount-lines data/mount_lines.json

The page is only served to this computer.
"""
import argparse
import base64
import json
import math
import os
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tracker3d import coverage as cov                     # noqa: E402
from tracker3d.config import load_config                  # noqa: E402
from tracker3d.plan import load_plan, load_scan_points    # noqa: E402

PAGE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mount_editor.html")
MAX_BODY = 1 << 20
SIDE_PX_PER_M = 100          # side view resolution: 1 pixel per cm
SIDE_MARGIN_M = 0.3          # scan shown beyond each end of the line
SIDE_MAX_LENGTH_M = 30.0
SIDE_DEPTHS_M = (0.2, 0.4, 0.8)


def room_side(plan, start, end):
    """+1 if the room is to the left of the line going from start to end, -1 if to the
    right, 0 if unclear (judged from the usable floor a little way off each side)."""
    a, b = np.asarray(start, float), np.asarray(end, float)
    length = float(np.linalg.norm(b - a))
    if length < 1e-9:
        return 0
    u = (b - a) / length
    n = np.array([-u[1], u[0]])
    votes = 0
    for t in (0.25, 0.5, 0.75):
        p = a + t * (b - a)
        for k in (0.5, 1.0, 1.5):
            votes += int(plan.in_valid_area(*(p + k * n))) - int(plan.in_valid_area(*(p - k * n)))
    return int(np.sign(votes))


def side_view(points, rgb, start, end, depth=0.4, px_per_m=SIDE_PX_PER_M, margin=SIDE_MARGIN_M,
              viewer_side=0):
    """The scan seen face-on along a line: every point within `depth` m of the vertical
    plane through the line, drawn by distance along the line and height (floor at the
    bottom). Where points overlap, the one nearest the plane wins, so the wall the line
    is drawn on shows in front of things behind it. Start is on the left, unless
    viewer_side (see room_side) puts the viewer on the line's left, when the image is
    mirrored so it looks as it does from there. Returns (BGR image, {"s0", "s1", "z0",
    "z1", "px_per_m", "mirrored"}): s is metres along the line, z is height."""
    a, b = np.asarray(start, float), np.asarray(end, float)
    length = float(np.linalg.norm(b - a))
    u = (b - a) / length if length > 1e-9 else np.array([1.0, 0.0])
    n = np.array([-u[1], u[0]])
    z0, z1 = -0.1, float(np.percentile(points[:, 2], 99.9)) + 0.1
    s0, s1 = -margin, length + margin
    w = max(1, int(math.ceil((s1 - s0) * px_per_m)))
    h = max(1, int(math.ceil((z1 - z0) * px_per_m)))
    rel = points[:, :2] - a
    s, d, z = rel @ u, rel @ n, points[:, 2]
    keep = (np.abs(d) <= depth) & (s >= s0) & (s < s1) & (z >= z0) & (z < z1)
    img = np.full((h, w, 3), 40, np.uint8)
    if keep.any():
        col = ((s[keep] - s0) * px_per_m).astype(np.int64).clip(0, w - 1)
        row = ((z1 - z[keep]) * px_per_m).astype(np.int64).clip(0, h - 1)
        near = np.abs(d[keep])
        cell = row * w + col
        order = np.lexsort((near, cell))
        first = order[np.unique(cell[order], return_index=True)[1]]
        if rgb is not None:
            colour = rgb[keep][first][:, ::-1].astype(np.float64)
            lo, hi = np.percentile(colour, 1), np.percentile(colour, 99)
            colour = (colour - lo) / max(hi - lo, 1.0) * 255.0
        else:
            colour = np.repeat(((z[keep][first] - z0) / (z1 - z0) * 200 + 55)[:, None], 3, axis=1)
        colour *= (1.0 - 0.5 * near[first] / depth)[:, None]           # farther = darker
        img.reshape(-1, 3)[cell[first]] = colour.clip(0, 255).astype(np.uint8)
        filled = np.zeros((h, w), np.uint8)
        filled.reshape(-1)[cell[first]] = 1
        for _ in range(2):                    # close small gaps between scan points
            grown = cv2.dilate(img, np.ones((3, 3), np.uint8))
            gap = (filled == 0) & (cv2.dilate(filled, np.ones((3, 3), np.uint8)) > 0)
            img[gap] = grown[gap]
            filled[gap] = 1
    mirrored = viewer_side > 0
    if mirrored:
        img = np.ascontiguousarray(img[:, ::-1])
    return img, {"s0": s0, "s1": s1, "z0": z0, "z1": z1, "px_per_m": px_per_m, "w": w, "h": h,
                 "mirrored": mirrored}


def preview(plan, occ, lines):
    """What the planner will make of these lines, as JSON-ready values."""
    spots, dropped = cov.spots_on_lines(plan, occ, lines)
    return {"spots": [{"line": s["line"], "xy": [round(float(v), 3) for v in s["xy"]]} for s in spots],
            "dropped": dropped}


def save_lines(path, lines):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"lines": lines}, f, indent=2)
        f.write("\n")
    os.replace(tmp, path)


def make_handler(plan, occ, out_path, points=None, rgb=None):
    """points (N, 3) and rgb are the scan for the side view; without points the side
    view is not offered."""
    lock = threading.Lock()
    ok, png = cv2.imencode(".png", plan.background())
    if not ok:
        raise RuntimeError("could not encode the map")
    map_png = png.tobytes()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _send(self, code, body, kind="application/json"):
            if not isinstance(body, bytes):
                body = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            url = urlparse(self.path)
            if url.path == "/":
                with open(PAGE, "rb") as f:
                    self._send(200, f.read(), "text/html; charset=utf-8")
            elif url.path == "/map.png":
                self._send(200, map_png, "image/png")
            elif url.path == "/state":
                lines = cov.load_mount_lines(out_path) if os.path.exists(out_path) else []
                h, w = plan.background().shape[:2]
                self._send(200, {"map": {"w": w, "h": h, "px_per_m": plan.px_per_m,
                                         "margin_m": plan.margin_m, "width_m": plan.width_m,
                                         "height_m": plan.height_m},
                                 "side_view": points is not None, "depths_m": SIDE_DEPTHS_M,
                                 "file": out_path, "lines": lines, **preview(plan, occ, lines)})
            elif url.path == "/side" and points is not None:
                try:
                    q = parse_qs(url.query)
                    x1, y1, x2, y2, depth = (float(q[k][0]) for k in ("x1", "y1", "x2", "y2", "depth"))
                    if not all(map(math.isfinite, (x1, y1, x2, y2))):
                        raise ValueError("coordinates must be numbers")
                    if math.hypot(x2 - x1, y2 - y1) > SIDE_MAX_LENGTH_M:
                        raise ValueError(f"lines longer than {SIDE_MAX_LENGTH_M:g} m are not shown")
                    if depth not in SIDE_DEPTHS_M:
                        raise ValueError(f"depth must be one of {SIDE_DEPTHS_M}")
                except (KeyError, ValueError) as e:
                    return self._send(400, {"error": str(e)})
                img, meta = side_view(points, rgb, (x1, y1), (x2, y2), depth,
                                      viewer_side=room_side(plan, (x1, y1), (x2, y2)))
                ok, png = cv2.imencode(".png", img)
                self._send(200, {**meta, "png": base64.b64encode(png.tobytes()).decode()})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):
            # A JSON content type stops other websites posting here from a browser.
            if self.path != "/lines" or self.headers.get("Content-Type") != "application/json":
                return self._send(404, {"error": "not found"})
            size = int(self.headers.get("Content-Length") or 0)
            if size > MAX_BODY:
                return self._send(413, {"error": "too large"})
            try:
                lines = cov.check_mount_lines(json.loads(self.rfile.read(size)).get("lines"))
            except (ValueError, AttributeError) as e:
                return self._send(400, {"error": str(e)})
            with lock:
                save_lines(out_path, lines)
                result = preview(plan, occ, lines)
            self._send(200, result)

    return Handler


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--out", default="data/mount_lines.json", help="file the lines are saved to")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true", help="don't open the page automatically")
    args = ap.parse_args()

    cfg = load_config(args.config)
    plan = load_plan(cfg)
    print("[mounts] reading the lab scan ...")
    points, rgb = load_scan_points(cfg, colours=True)
    occ = cov.Occupancy(points)
    server = ThreadingHTTPServer(("127.0.0.1", args.port),
                                 make_handler(plan, occ, args.out, points, rgb))
    url = f"http://127.0.0.1:{args.port}/"
    print(f"[mounts] open {url}  (saving to {args.out}; press Ctrl+C here when finished)")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print(f"\n[mounts] saved. Next: python tools/layout_planner.py --mount-lines {args.out}")


if __name__ == "__main__":
    main()
