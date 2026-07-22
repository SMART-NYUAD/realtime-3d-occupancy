"""Draw the annotated camera view and a top-down floor map of tracked people."""
import cv2
import numpy as np

# deterministic-ish color per track id
def _color(tid):
    np.random.seed(tid * 7919 % 2**31)
    c = np.random.randint(60, 255, size=3)
    return int(c[0]), int(c[1]), int(c[2])


# foot-source -> marker color (BGR), so you can see at a glance which detections
# are trusting the ankles vs extrapolating through occlusion.
_FOOT_COLORS = {"ankles": (0, 255, 0), "ankle": (0, 255, 128),
                "knees": (0, 200, 255), "hips": (0, 128, 255), "box": (0, 0, 255)}


def draw_camera_view(frame, people, draw_trails=False, trails=None):
    for p in people:
        x1, y1, x2, y2 = map(int, p["bbox"])
        col = _color(p["id"])
        cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)
        # foot point: the pose-derived estimate if present, else box bottom.
        src = p.get("foot_source", "box")
        fpx = p.get("foot_px", ((x1 + x2) / 2.0, y2))
        fx, fy = int(fpx[0]), int(fpx[1])
        fcol = _FOOT_COLORS.get(src, (0, 0, 255))
        cv2.circle(frame, (fx, fy), 5, fcol, -1)
        label = f"ID {p['id']} [{src}]"
        if p.get("world") is not None:
            wx, wy = p["world"]
            label += f"  ({wx:.1f},{wy:.1f})m"
        cv2.putText(frame, label, (x1, max(0, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)
    return frame
