"""Batched person detection (YOLO pose, TensorRT on Jetson) + per-camera ByteTrack.

All cameras' frames go through ONE batched inference call per step instead of one
call per camera, and each camera keeps its own ByteTrack instance (Ultralytics'
`model.track()` on a list shares a single tracker across the batch, which would
mix IDs between cameras, so tracking is driven here directly).

Model selection (`detector.model` in config.yaml):
  - `*.engine` : TensorRT engine (build it with `python tools/export_engine.py`).
                 If it's missing, the matching `.pt` is used with a warning.
  - `*.pt`     : PyTorch weights (auto-downloaded), FP16 on CUDA.
"""
import os

import numpy as np
from ultralytics import YOLO
from ultralytics.trackers.byte_tracker import BYTETracker
from ultralytics.utils import YAML, IterableSimpleNamespace

PERSON_CLASS = 0


def resolve_model_path(path):
    if path.endswith(".engine") and not os.path.exists(path):
        pt = path[:-len(".engine")] + ".pt"
        print(f"[detector] {path} not found — using {pt} (slower). "
              f"Build the engine once with: python tools/export_engine.py", flush=True)
        return pt
    return path


class _Dets:
    """Minimal Boxes-like view ByteTrack can consume (numpy, boolean-indexable).
    `gidx` carries each detection's index in the full, unsplit list."""

    def __init__(self, xyxy, conf, gidx=None):
        self.xyxy = xyxy
        self.conf = conf
        self.cls = np.zeros(len(conf), np.float32)
        self.gidx = np.arange(len(conf)) if gidx is None else gidx

    @property
    def xywh(self):
        x1, y1, x2, y2 = self.xyxy.T
        return np.stack([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], axis=1)

    def __len__(self):
        return len(self.conf)

    def __getitem__(self, idx):
        return _Dets(self.xyxy[idx], self.conf[idx], self.gidx[idx])


class _ByteTracker(BYTETracker):
    """ByteTrack that reports each track's index into the FULL detection list.

    Upstream splits detections into high/low-score subsets and numbers each
    subset from 0, so a track matched in the low-score (second) stage reports an
    index into the low subset — and `result[idx]` then attaches another person's
    keypoints/confidence to it. Re-stamp the global index at track creation."""

    def init_track(self, results, img=None):
        tracks = super().init_track(results, img)
        for t, g in zip(tracks, results.gidx):
            t.idx = int(g)
        return tracks


def tracker_input_mask(xyxy, conf, kxy=None, kcf=None, zones=(), nested_overlap=0.75,
                       kp_conf=0.5):
    """Boolean mask of detections to pass to the tracker.

    Drops (1) a box whose centre lies in an ignore zone (a camera-pixel polygon,
    e.g. over a monitor or poster that keeps being detected as a person), and
    (2) a duplicate of a stronger box: mostly inside it (a torso box within a full
    body box survives NMS at IoU 0.5) or with both shoulders on the same spot.
    A duplicate would otherwise become a second, permanent dot for one person."""
    import cv2
    n = len(conf)
    keep = np.ones(n, bool)
    for i in range(n):
        cx, cy = (xyxy[i, 0] + xyxy[i, 2]) / 2, (xyxy[i, 1] + xyxy[i, 3]) / 2
        if any(cv2.pointPolygonTest(z, (float(cx), float(cy)), False) >= 0 for z in zones):
            keep[i] = False
    if nested_overlap <= 0:
        return keep
    order = np.argsort(-np.asarray(conf))
    kept = []
    for i in order:
        if not keep[i]:
            continue
        x1, y1, x2, y2 = xyxy[i]
        area = max((x2 - x1) * (y2 - y1), 1e-6)
        dup = False
        for j in kept:
            ix = max(0.0, min(x2, xyxy[j, 2]) - max(x1, xyxy[j, 0]))
            iy = max(0.0, min(y2, xyxy[j, 3]) - max(y1, xyxy[j, 1]))
            if ix * iy / area >= nested_overlap:
                dup = True
            elif kxy is not None and kcf is not None and min(kcf[i][5:7]) >= kp_conf \
                    and min(kcf[j][5:7]) >= kp_conf:
                tol = 0.15 * max(y2 - y1, 1.0)
                if np.all(np.linalg.norm(kxy[i][5:7] - kxy[j][5:7], axis=1) < tol):
                    dup = True
            if dup:
                break
        if dup:
            keep[i] = False
        else:
            kept.append(i)
    return keep


class MultiCamDetector:
    def __init__(self, cfg, cam_names):
        d = cfg["detector"]
        self.model_path = resolve_model_path(d["model"])
        self.model = YOLO(self.model_path, task="pose")
        self.conf = float(d.get("conf", 0.15))
        self.iou = float(d.get("iou", 0.5))
        self.imgsz = int(d.get("imgsz", 640))
        self.device = d.get("device", "cuda:0")
        self.half = bool(d.get("half", True)) and str(self.device).startswith("cuda")
        self.max_batch = int(d.get("max_batch", 4))
        targs = IterableSimpleNamespace(**YAML.load(d.get("tracker", "bytetrack.yaml")))
        self.trackers = {n: _ByteTracker(targs) for n in cam_names}
        self.suppress_nested = bool(d.get("suppress_nested", True))
        self.nested_overlap = float(d.get("nested_overlap", 0.75))
        self.kp_conf = float(d.get("kp_conf", 0.5))
        self.zones = {c["name"]: [np.asarray(z, np.float32) for z in c.get("ignore_zones", [])]
                      for c in cfg.get("cameras", [])}

    def warmup(self, shape=(1080, 1920, 3), n=3):
        dummy = [np.zeros(shape, np.uint8)] * max(1, min(n, self.max_batch))
        for _ in range(2):
            self._infer(dummy)

    def _infer(self, frames):
        out = []
        for i in range(0, len(frames), self.max_batch):
            out.extend(self.model.predict(
                frames[i:i + self.max_batch], classes=[PERSON_CLASS], conf=self.conf,
                iou=self.iou, imgsz=self.imgsz, device=self.device, half=self.half,
                verbose=False))
        return out

    def detect_and_track(self, frames):
        """frames: {cam: BGR image}. Returns {cam: (people, raw)}.

        people: tracked detections [{id, bbox, conf, kxy?, kconf?}] (ByteTrack-confirmed)
        raw:    every detection above `conf` [{bbox, kxy?, kconf?}] — used for privacy
                masking so an untracked (low-score) person is still blurred.
        """
        names = list(frames)
        if not names:
            return {}
        results = self._infer([frames[n] for n in names])
        out = {}
        for n, r in zip(names, results):
            boxes = r.boxes
            xyxy = boxes.xyxy.cpu().numpy().astype(np.float32)
            conf = boxes.conf.cpu().numpy().astype(np.float32)
            kp = getattr(r, "keypoints", None)
            kxy = kp.xy.cpu().numpy() if kp is not None and kp.xy is not None else None
            kcf = kp.conf.cpu().numpy() if kp is not None and kp.conf is not None else None

            raw = []
            for i in range(len(conf)):
                d = {"bbox": tuple(float(v) for v in xyxy[i]), "conf": float(conf[i])}
                if kxy is not None and kcf is not None:
                    d["kxy"], d["kconf"] = kxy[i], kcf[i]
                raw.append(d)

            # The tracker sees only kept boxes; `raw` (privacy masking) keeps all.
            keep = tracker_input_mask(xyxy, conf, kxy, kcf, self.zones.get(n, ()),
                                      self.nested_overlap if self.suppress_nested else 0.0,
                                      self.kp_conf)
            tracks = self.trackers[n].update(_Dets(xyxy[keep], conf[keep], np.flatnonzero(keep)))
            people = []
            for row in tracks:
                idx = int(row[-1])
                p = dict(raw[idx])
                p["id"] = int(row[4])
                people.append(p)
            out[n] = (people, raw)
        return out
