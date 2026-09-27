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

            tracks = self.trackers[n].update(_Dets(xyxy, conf))
            people = []
            for row in tracks:
                idx = int(row[-1])
                p = dict(raw[idx])
                p["id"] = int(row[4])
                people.append(p)
            out[n] = (people, raw)
        return out
