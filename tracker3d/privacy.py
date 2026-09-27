"""Privacy masking for preview/snapshot frames, driven by the pose keypoints the
person detector already produces — no second face model.

The previous approach ran a separate face detector (deface/CenterFace) on the
CPU: ~600 ms per 1080p frame on the Thor, which starved the tracking loop down to
~4 FPS. The pose model already localizes every person's nose/eyes/ears, so the
head region comes for free, is never stale (same frame, same detections), and is
masked on the small preview image instead of the full-resolution frame.

Coverage: every detection above `detector.conf` is masked, including low-score
ones ByteTrack hasn't confirmed. A person the detector misses entirely is not
masked — keep `detector.conf` low (it is 0.15 by default).

Only the displayed/saved preview is masked; the frames fed to the detector are
never modified, so tracking accuracy is unaffected.
"""
import cv2
import numpy as np

_HEAD = (0, 1, 2, 3, 4)          # COCO nose, eyes, ears
_L_SHO, _R_SHO = 5, 6


def head_ellipse(det, kp_thresh=0.3):
    """(cx, cy, rx, ry) of a detection's head in source-frame pixels.

    From the head keypoints when visible, else the top of the box (a person seen
    from behind or with a hidden face still gets the top of the box masked)."""
    x1, y1, x2, y2 = det["bbox"]
    bw, bh = max(x2 - x1, 1.0), max(y2 - y1, 1.0)
    kxy, kcf = det.get("kxy"), det.get("kconf")
    if kxy is not None and kcf is not None:
        pts = np.array([kxy[i] for i in _HEAD if kcf[i] >= kp_thresh])
        if len(pts):
            # Head size: widest of the head-keypoint spread, a fraction of the
            # shoulder width, or of the box height (robust when only one kp shows).
            spread = float(np.ptp(pts[:, 0])) if len(pts) > 1 else 0.0
            sho = 0.0
            if kcf[_L_SHO] >= kp_thresh and kcf[_R_SHO] >= kp_thresh:
                sho = float(abs(kxy[_L_SHO][0] - kxy[_R_SHO][0]))
            r = max(0.75 * spread, 0.35 * sho, 0.07 * bh, 0.25 * bw, 6.0)
            cx, cy = pts.mean(axis=0)
            return float(cx), float(cy - 0.1 * r), r, 1.4 * r
    # No usable head keypoints: cover the top of the box.
    r = max(0.35 * bw, 0.08 * bh, 6.0)
    return (x1 + x2) / 2.0, y1 + 0.9 * r, r, 1.2 * r


class PrivacyMasker:
    def __init__(self, cfg):
        self.enabled = bool(cfg.get("enabled", True))
        self.method = str(cfg.get("method", "pixelate"))     # pixelate | blur | solid
        self.scale = float(cfg.get("scale", 1.3))             # grow the head ellipse
        self.kp_thresh = float(cfg.get("kp_thresh", 0.3))

    def apply(self, img, dets, sx=1.0, sy=1.0):
        """Mask every detection's head IN PLACE on `img` (a preview copy).
        sx, sy map source-frame pixels to `img` pixels."""
        if not self.enabled or not dets:
            return img
        H, W = img.shape[:2]
        for d in dets:
            cx, cy, rx, ry = head_ellipse(d, self.kp_thresh)
            cx, cy = cx * sx, cy * sy
            rx, ry = max(3, int(rx * sx * self.scale)), max(3, int(ry * sy * self.scale))
            x0, y0 = max(0, int(cx - rx)), max(0, int(cy - ry))
            x1, y1 = min(W, int(cx + rx) + 1), min(H, int(cy + ry) + 1)
            if x1 - x0 < 2 or y1 - y0 < 2:
                continue
            roi = img[y0:y1, x0:x1]
            if self.method == "solid":
                cover = np.zeros_like(roi)
            elif self.method == "blur":
                k = (max(roi.shape[:2]) // 2) | 1
                cover = cv2.GaussianBlur(roi, (k, k), 0)
            else:                                            # pixelate
                h, w = roi.shape[:2]
                small = cv2.resize(roi, (max(1, w // 8), max(1, h // 8)),
                                   interpolation=cv2.INTER_AREA)
                cover = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
            mask = np.zeros(roi.shape[:2], np.uint8)
            cv2.ellipse(mask, (int(cx) - x0, int(cy) - y0), (rx, ry), 0, 0, 360, 255, -1)
            roi[mask > 0] = cover[mask > 0]
        return img
