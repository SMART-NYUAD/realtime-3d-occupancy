"""Face blurring for privacy-preserving preview windows, via the `deface` library.

Detects faces with `deface`'s CenterFace model and blurs them on a *copy* of the
frame. Only the preview/snapshot path is blurred; the frames fed to the person
detector, homography and fusion stay untouched, so tracking is unaffected.

Notes on getting faces to actually blur in live footage:
  * Detection runs at FULL frame resolution by default (`det_size: 0`). Surveillance
    faces are small; downscaling first makes them vanish from the detector.
  * The default threshold is low (0.2, deface's own default) for the same reason.
  * Our frames are BGR (OpenCV/GStreamer); CenterFace expects RGB, so we convert
    before detection. The blur itself is applied to the BGR frame (box coords are
    colour-independent), so preview colours stay correct.

If `deface` isn't installed or the model can't load, FaceBlurrer degrades to a
no-op (logs once) so it never takes tracking down.
"""
import cv2

try:
    from deface.deface import CenterFace, anonymize_frame
    _DEFACE_OK = True
except Exception as _e:          # library missing / import error
    _DEFACE_OK = False
    _IMPORT_ERR = _e

# config method -> deface `replacewith` mode
_METHODS = {"blur": "blur", "pixelate": "mosaic", "mosaic": "mosaic", "solid": "solid"}


class FaceBlurrer:
    """Blur faces on a copy of a frame for privacy-preserving previews.

    Configured from the ``output.blur_faces`` block.
    """

    def __init__(self, cfg):
        self.enabled = bool(cfg.get("enabled", False))
        self.threshold = float(cfg.get("threshold", 0.2))
        self.mask_scale = float(cfg.get("mask_scale", 1.3))   # grow box before blurring
        self.replacewith = _METHODS.get(str(cfg.get("method", "blur")), "blur")
        self.ellipse = bool(cfg.get("ellipse", True))
        self.mosaicsize = int(cfg.get("mosaicsize", 20))
        # 0 = detect at full frame resolution (best for small/far faces). A positive
        # value caps the detector's long side for speed, at the cost of missing
        # smaller faces.
        self.det_size = int(cfg.get("det_size", 0))
        self._cf = None
        self._warned = False

        if not self.enabled:
            return
        if not _DEFACE_OK:
            print(f"[deface] `deface` not importable ({_IMPORT_ERR}) — face blur off. "
                  f"Install: python3 -m pip install deface", flush=True)
            self.enabled = False
            return
        try:
            # onnx_path=None -> use deface's bundled centerface.onnx. A local override
            # can be given via `model:` in config.
            model = cfg.get("model") or None
            self._cf = CenterFace(onnx_path=model, in_shape=None, backend="auto")
            print(f"[deface] face blur on (method={self.replacewith}, "
                  f"thresh={self.threshold}, det_size={self.det_size or 'full'})", flush=True)
        except Exception as e:
            print(f"[deface] failed to init CenterFace: {e} — face blur off.", flush=True)
            self.enabled = False

    def blur(self, frame):
        """Return a copy of `frame` with detected faces blurred. On any failure
        (or when disabled) returns the frame unchanged."""
        if not self.enabled or self._cf is None:
            return frame
        try:
            dets = self._detect(frame)
        except Exception as e:
            if not self._warned:
                print(f"[deface] detection error: {e} — passing frames through.", flush=True)
                self._warned = True
            return frame
        if len(dets) == 0:
            return frame
        out = frame.copy()
        anonymize_frame(dets, out, mask_scale=self.mask_scale,
                        replacewith=self.replacewith, ellipse=self.ellipse,
                        draw_scores=False, replaceimg=None, mosaicsize=self.mosaicsize)
        return out

    def _detect(self, frame):
        # CenterFace was trained on RGB; our frames are BGR.
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        if self.det_size and max(h, w) > self.det_size:
            s = self.det_size / max(h, w)
            small = cv2.resize(rgb, (int(round(w * s)), int(round(h * s))))
            dets, _ = self._cf(small, threshold=self.threshold)
            if len(dets) > 0:
                dets[:, :4] /= s           # scale boxes back to full-res coords
            return dets
        dets, _ = self._cf(rgb, threshold=self.threshold)
        return dets
