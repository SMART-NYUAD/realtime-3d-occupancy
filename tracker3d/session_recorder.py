"""Record a live tracking run as keypoint-only sessions, for replaying later.

`track.py --record-session DIR` writes what every camera detected in every frame:
boxes, body keypoints and each camera's own tracker IDs. Never video. The run is
split into parts of `part_minutes`, each a complete session in the format
`autocalib.py record` writes (manifest.json + observations.jsonl), so a full work
day can be replayed part by part without loading it all into memory:

    DIR/recording.json        start and end, cameras, code version, settings
    DIR/calib_snapshot/       the calibration files in use when recording started
    DIR/part_0001/ ...        sessions, one per part

tools/replay_tracking.py replays DIR (all parts in order) with any tracker
settings and any subset of the cameras. Recording stops (tracking carries on)
when free disk space falls below `min_free_gb`.

With `video` ({"width": 960, "fps": 10}), each part also gets one video per
camera and the capture time of every video frame (tracker3d/video_recorder.py).
"""
import json
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from tracker3d.autocalib.session import SessionWriter, write_json
from tracker3d.lens import load_lens


def _git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                              timeout=5, cwd=Path(__file__).resolve().parents[1]).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def _jsonable(value):
    return json.loads(json.dumps(value, default=str))


class SessionRecorder:
    def __init__(self, root, cams, cfg, part_minutes=30.0, min_free_gb=2.0, video=None):
        self.root = Path(root)
        if self.root.exists():
            raise ValueError(f"{self.root} already exists; record each run to a new folder")
        self.snapshots, missing = {}, []
        for cam in cams:
            lens, _ = load_lens(cam)
            if lens is None:
                missing.append(cam["name"])
            else:
                self.snapshots[cam["name"]] = {"lens": lens.to_dict()}
        if missing:
            raise ValueError("--record-session needs every camera's lens calibration; missing: "
                             + ", ".join(missing))
        self.root.mkdir(parents=True)
        snap = self.root / "calib_snapshot"
        snap.mkdir()
        for cam in cams:
            for key in ("homography_file", "intrinsics_file"):
                src = Path(cam.get(key) or "")
                for path in (src, src.with_suffix(".json")):
                    if path.is_file() and not (snap / path.name).exists():
                        shutil.copy2(path, snap / path.name)
        self.detector_cfg = cfg.get("detector", {})
        self.part_s = max(float(part_minutes), 1e-3) * 60.0
        self.min_free = float(min_free_gb) * 1e9
        self.info = {"kind": "live_recording", "started_utc": datetime.now(timezone.utc).isoformat(),
                     "ended_utc": None, "cameras": [c["name"] for c in cams],
                     "part_minutes": float(part_minutes), "git_commit": _git_commit(),
                     "config": _jsonable(cfg), "parts": [], "frames": 0, "stopped_early": None}
        self.writer, self.frames, self.stopped = None, 0, False
        self.video = None
        if video:
            from tracker3d.video_recorder import VideoRecorder
            self.video = VideoRecorder(list(self.snapshots), video.get("width", 960),
                                       video.get("fps", 10.0))
            self.info["video"] = {"codec": self.video.fourcc, "width": self.video.width,
                                  "fps": self.video.fps}
        self._last_disk_check = float("-inf")     # check on the first frame, then every minute
        self._open_part()

    def _save_info(self):
        write_json(self.root / "recording.json", self.info, overwrite=True)

    def _open_part(self):
        name = f"part_{len(self.info['parts']) + 1:04d}"
        self.writer = SessionWriter(self.root / name, self.snapshots, self.detector_cfg)
        self.part_started = time.monotonic()
        self.info["parts"].append(name)
        self._save_info()

    def _stop(self, reason):
        self.writer.close(complete=True)
        if self.video is not None:
            self.info["video"] = self.video.close()
            self.video = None
        self.stopped = True
        self.info["stopped_early"] = reason
        self._save_info()
        print(f"[record] stopped recording ({reason}); tracking continues", flush=True)

    def frame(self, cam, t, image_size, meta, tracked, raw, image=None):
        """Write one camera frame: every raw box with its keypoints, and the camera
        tracker's ID for the boxes it tracks (None for the others). With video on,
        `image` (the camera frame) goes to that camera's video."""
        if self.stopped:
            return
        now = time.monotonic()
        if now - self._last_disk_check >= 60.0:
            self._last_disk_check = now
            if shutil.disk_usage(self.root).free < self.min_free:
                self._stop(f"less than {self.min_free / 1e9:g} GB of disk space left")
                return
        if now - self.part_started >= self.part_s:
            self.writer.close(complete=True)
            self._open_part()
        ids = {tuple(p["bbox"]): p.get("id") for p in tracked}
        people = []
        for p in raw:
            person = {"bbox": [float(v) for v in p["bbox"]], "conf": float(p["conf"]),
                      "id": ids.get(tuple(p["bbox"]))}
            if p.get("kxy") is not None:
                person.update(kxy=np.asarray(p["kxy"], float).tolist(),
                              kconf=np.asarray(p["kconf"], float).tolist())
            people.append(person)
        self.writer.append({"kind": "frame", "camera": cam, "t": float(t),
                            "image_size": [int(image_size[0]), int(image_size[1])],
                            **meta, "people": people})
        self.frames += 1
        if self.video is not None and image is not None:
            self.video.push(cam, self.writer.path, t, meta, image)

    def close(self):
        if not self.stopped and self.writer is not None:
            self.writer.close(complete=True)
        if self.video is not None:
            self.info["video"] = self.video.close()
            self.video = None
        self.info.update(ended_utc=datetime.now(timezone.utc).isoformat(), frames=self.frames)
        self._save_info()
        print(f"[record] {self.frames} camera frames in {len(self.info['parts'])} part(s) "
              f"saved to {self.root}", flush=True)
