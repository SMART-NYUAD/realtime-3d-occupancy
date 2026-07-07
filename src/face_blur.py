"""Face blurring for privacy-preserving preview windows, via the `deface` library.

Detects faces with `deface`'s CenterFace model and blurs them on a *copy* of the
frame. Only the preview/snapshot path is blurred; the frames fed to the person
detector, homography and fusion stay untouched, so tracking is unaffected.

Performance: CenterFace runs on the CPU here (no onnxruntime-gpu wheel for this
JetPack, and pip OpenCV's DNN is CPU-only), which is ~100+ ms per full-res frame
— far too slow to run inline for several cameras. So detection runs in a
BACKGROUND thread per camera and the render loop just applies the most recent
boxes to the current frame. Faces move slowly, so slightly-stale boxes look fine
in a preview, and the render FPS no longer waits on detection. Set
`async_detect: false` to detect inline (simpler, but throttles FPS to detection
speed).

Getting faces to actually blur in live footage:
  * Detection runs at FULL frame resolution by default (`det_size: 0`);
    surveillance faces are small and vanish if you downscale first.
  * Low threshold (0.2, deface's default) for the same reason.
  * Frames are BGR (OpenCV/GStreamer); CenterFace wants RGB, so we convert before
    detection. Boxes are colour-independent, so the blur goes on the BGR frame.

If `deface` isn't installed or the model can't load, FaceBlurrer degrades to a
no-op (logs once) so it never takes tracking down.
"""
import threading

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

    Configured from the ``output.blur_faces`` block. A single instance is shared
    across cameras; pass a distinct ``key`` per camera to ``blur()`` so each keeps
    its own detection state.
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
        # Detect in a background thread so the render loop never blocks on the
        # (slow, CPU-bound) face model. Applies the most recent boxes each frame.
        self.async_detect = bool(cfg.get("async_detect", True))
        self._cf = None
        self._warned = False
        self._states = {}                 # key -> per-camera detection state
        self._states_lock = threading.Lock()

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
            self._cf_lock = threading.Lock()   # CenterFace/cv2.dnn net isn't reentrant
            mode = "async" if self.async_detect else "inline"
            print(f"[deface] face blur on ({mode}, method={self.replacewith}, "
                  f"thresh={self.threshold}, det_size={self.det_size or 'full'})", flush=True)
        except Exception as e:
            print(f"[deface] failed to init CenterFace: {e} — face blur off.", flush=True)
            self.enabled = False

    def blur(self, frame, key=0):
        """Return a copy of `frame` with detected faces blurred. On any failure
        (or when disabled) returns the frame unchanged. `key` identifies the
        camera so each keeps its own detection state."""
        if not self.enabled or self._cf is None:
            return frame
        try:
            dets = self._get_dets(frame, key)
        except Exception as e:
            if not self._warned:
                print(f"[deface] detection error: {e} — passing frames through.", flush=True)
                self._warned = True
            return frame
        if dets is None or len(dets) == 0:
            return frame
        out = frame.copy()
        anonymize_frame(dets, out, mask_scale=self.mask_scale,
                        replacewith=self.replacewith, ellipse=self.ellipse,
                        draw_scores=False, replaceimg=None, mosaicsize=self.mosaicsize)
        return out

    # ---- detection scheduling ------------------------------------------------
    def _get_dets(self, frame, key):
        if not self.async_detect:
            return self._detect(frame)
        with self._states_lock:
            st = self._states.get(key)
            if st is None:
                st = {"dets": None, "busy": False}
                self._states[key] = st
        # First frame for this camera: detect synchronously so faces are covered
        # immediately instead of flashing unblurred until the first async result.
        if st["dets"] is None and not st["busy"]:
            st["busy"] = True
            try:
                st["dets"] = self._detect(frame)
            finally:
                st["busy"] = False
            return st["dets"]
        # Otherwise kick off a background refresh (if none in flight) and return
        # the most recent boxes we have.
        if not st["busy"]:
            st["busy"] = True
            snap = frame.copy()          # detach from the buffer the loop reuses
            threading.Thread(target=self._detect_worker, args=(st, snap),
                             daemon=True).start()
        return st["dets"]

    def _detect_worker(self, st, frame):
        try:
            st["dets"] = self._detect(frame)
        except Exception as e:
            if not self._warned:
                print(f"[deface] detection error: {e} — passing frames through.", flush=True)
                self._warned = True
        finally:
            st["busy"] = False

    def _detect(self, frame):
        # CenterFace was trained on RGB; our frames are BGR.
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        if self.det_size and max(h, w) > self.det_size:
            s = self.det_size / max(h, w)
            rgb = cv2.resize(rgb, (int(round(w * s)), int(round(h * s))))
        else:
            s = 1.0
        with self._cf_lock:
            dets, _ = self._cf(rgb, threshold=self.threshold)
        if s != 1.0 and len(dets) > 0:
            dets[:, :4] /= s             # scale boxes back to full-res coords
        return dets
