"""Preview rendering: annotated camera tiles and the top-down floor map.

Everything is drawn on a DOWNSCALED copy of the frame (the preview size), never
on the full-resolution frame the detector reads — cheaper, and the capture
buffers stay pristine.
"""
import cv2
import numpy as np

# One colour per camera (BGR).
CAM_COLORS = [(0, 200, 255), (0, 255, 0), (255, 120, 0),
              (255, 0, 255), (0, 165, 255), (200, 200, 0)]

# foot-source -> marker colour: which detections see the ankles vs extrapolate.
_FOOT_COLORS = {"ankles": (0, 255, 0), "ankle": (0, 255, 128),
                "knees": (0, 200, 255), "hips": (0, 128, 255), "box": (0, 0, 255)}


# Calibration markers (BGR): bright, distinct from each other and from the
# grey/brown scan background.
MARK_SHARED = (255, 0, 255)    # magenta: reference point from another camera / shared
MARK_USED = (0, 255, 0)        # green: paired in this session
MARK_NEW = (0, 255, 255)       # yellow: new this session, not paired yet
MARK_PENDING = (255, 255, 0)   # cyan: ring around the point awaiting its camera click


def draw_marker(img, pt, color, label=None, size=6):
    """Small crosshair whose centre is the exact point, with a thin dark outline
    so it stays visible on any background. Precise to click/drag against."""
    x, y = int(round(pt[0])), int(round(pt[1]))
    for (x0, y0, x1, y1) in ((x - size, y, x + size, y), (x, y - size, x, y + size)):
        cv2.line(img, (x0, y0), (x1, y1), (0, 0, 0), 3, cv2.LINE_AA)
        cv2.line(img, (x0, y0), (x1, y1), color, 1, cv2.LINE_AA)
    if label:
        org = (x + size + 3, y - 3)
        cv2.putText(img, label, org, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, label, org, cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)


def _id_color(tid):
    h = (tid * 2654435761) & 0xFFFFFF
    return 60 + (h & 0xFF) % 196, 60 + (h >> 8 & 0xFF) % 196, 60 + (h >> 16 & 0xFF) % 196


def camera_tile(frame, people, raw, masker, width, label, color):
    """Downscale `frame` to `width`, mask heads, draw boxes/foot points. Returns a new image."""
    fh, fw = frame.shape[:2]
    s = width / fw
    tile = cv2.resize(frame, (width, int(round(fh * s))), interpolation=cv2.INTER_LINEAR)
    masker.apply(tile, raw, s, s)
    for p in people:
        x1, y1, x2, y2 = (int(v * s) for v in p["bbox"])
        col = _id_color(p["id"])
        cv2.rectangle(tile, (x1, y1), (x2, y2), col, 1)
        src = p.get("foot_source", "box")
        fx, fy = p.get("foot_px", ((p["bbox"][0] + p["bbox"][2]) / 2, p["bbox"][3]))
        cv2.circle(tile, (int(fx * s), int(fy * s)), 4, _FOOT_COLORS.get(src, (0, 0, 255)), -1)
        txt = f"{p['id']}"
        if p.get("world") is not None:
            txt += f" ({p['world'][0]:.1f},{p['world'][1]:.1f})"
        cv2.putText(tile, txt, (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1)
    cv2.putText(tile, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    return tile


def mosaic(tiles, cols=None):
    """Grid the camera tiles into one image (one window instead of N)."""
    if not tiles:
        return None
    cols = cols or (1 if len(tiles) == 1 else 2)
    h = max(t.shape[0] for t in tiles)
    w = max(t.shape[1] for t in tiles)
    rows = (len(tiles) + cols - 1) // cols
    out = np.zeros((rows * h, cols * w, 3), np.uint8)
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols)
        out[r * h:r * h + t.shape[0], c * w:c * w + t.shape[1]] = t
    return out


def _poly_px(plan, poly):
    return np.array([plan.world_to_px(x, y) for x, y in poly], np.int32)


def draw_map(plan, people_now, all_dets, colors, fused, show_raw, show_blobs, blobs):
    canvas = plan.background()
    if show_blobs:
        for d in all_dets:
            if d.get("floor_poly") is not None:
                cv2.polylines(canvas, [_poly_px(plan, d["floor_poly"])], True,
                              colors[d["cam"]], 1, cv2.LINE_AA)
        for poly in blobs:
            cv2.polylines(canvas, [_poly_px(plan, poly)], True, (255, 255, 255), 2, cv2.LINE_AA)
    if show_raw or not fused:
        r = 4 if fused else 7
        for d in all_dets:
            if d.get("world") is None:
                continue
            cv2.circle(canvas, plan.world_to_px(*d["world"]), r, colors[d["cam"]],
                       1 if fused else -1, cv2.LINE_AA)
    if fused:
        for g in people_now:
            px, py = plan.world_to_px(*g["world"])
            col = (160, 160, 160) if g["coasting"] else (60, 220, 60)
            cv2.circle(canvas, (px, py), 8, col, -1, cv2.LINE_AA)
            tag = f"P{g['gid']}" + ("" if g["coasting"] else f" [{','.join(g['cams'])}]")
            cv2.putText(canvas, tag, (px + 10, py), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)
    return canvas
