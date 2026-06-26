"""Draw the annotated camera view and a top-down floor map of tracked people."""
import cv2
import numpy as np

# deterministic-ish color per track id
def _color(tid):
    np.random.seed(tid * 7919 % 2**31)
    c = np.random.randint(60, 255, size=3)
    return int(c[0]), int(c[1]), int(c[2])


def draw_camera_view(frame, people, draw_trails=False, trails=None):
    for p in people:
        x1, y1, x2, y2 = map(int, p["bbox"])
        col = _color(p["id"])
        cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)
        fx, fy = int((x1 + x2) / 2), int(y2)
        cv2.circle(frame, (fx, fy), 4, col, -1)        # foot point
        label = f"ID {p['id']}"
        if p.get("world") is not None:
            wx, wy = p["world"]
            label += f"  ({wx:.1f},{wy:.1f})m"
        cv2.putText(frame, label, (x1, max(0, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)
    return frame
