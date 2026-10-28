"""Video for a live recording (track.py --record-session DIR --record-video).

Each camera gets a background thread that encodes its frames, so tracking never
waits for the disk or the encoder: if a camera's queue fills up, frames are left
out of the video and counted instead. Videos follow the keypoint recording's
parts, one file per camera per part, with every frame's capture time (the same
clock as the keypoints) in a CSV beside it:

    DIR/part_0001/video/yi01.mp4
    DIR/part_0001/video/yi01_frames.csv     frame, t, frame_id, generation, timestamp_source

Frames are scaled down to `width` pixels across (0 = full size) and limited to
`fps` per camera to keep the size manageable. The codec is the first that works
of H.264 (avc1), MPEG-4 (mp4v) and Motion JPEG.
"""
import csv
import os
import queue
import tempfile
import threading

import cv2
import numpy as np

CODECS = (("avc1", ".mp4"), ("mp4v", ".mp4"), ("MJPG", ".avi"))


def pick_codec():
    """(fourcc, extension) of the first codec this OpenCV build can write."""
    with tempfile.TemporaryDirectory() as tmp:
        for fourcc, ext in CODECS:
            path = os.path.join(tmp, "probe" + ext)
            w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*fourcc), 10, (64, 48))
            ok = w.isOpened()
            if ok:
                w.write(np.zeros((48, 64, 3), np.uint8))
            w.release()
            if ok and os.path.getsize(path) > 0:
                return fourcc, ext
    raise RuntimeError("no working video codec in this OpenCV build (tried avc1, mp4v, MJPG)")


class _CameraVideo(threading.Thread):
    def __init__(self, cam, fourcc, ext, fps, queue_size):
        super().__init__(name=f"video-{cam}", daemon=True)
        self.cam, self.fourcc, self.ext, self.fps = cam, fourcc, ext, fps
        self.q = queue.Queue(maxsize=queue_size)
        self.part_dir = self.writer = self.csv_file = self.csv = None
        self.index = self.written = self.dropped = 0
        self.error = None

    def push(self, item):
        try:
            self.q.put_nowait(item)
        except queue.Full:
            self.dropped += 1

    def _close_part(self):
        if self.writer is not None:
            self.writer.release()
            self.csv_file.close()
        self.writer = self.csv_file = self.csv = None

    def _open_part(self, part_dir, frame):
        self._close_part()
        video_dir = os.path.join(part_dir, "video")
        os.makedirs(video_dir, exist_ok=True)
        h, w = frame.shape[:2]
        path = os.path.join(video_dir, self.cam + self.ext)
        self.writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*self.fourcc), self.fps, (w, h))
        if not self.writer.isOpened():
            raise RuntimeError(f"could not open {path} for writing")
        self.csv_file = open(os.path.join(video_dir, f"{self.cam}_frames.csv"), "w", newline="")
        self.csv = csv.writer(self.csv_file)
        self.csv.writerow(["frame", "t", "frame_id", "generation", "timestamp_source"])
        self.part_dir, self.index, self.size = part_dir, 0, (w, h)

    def run(self):
        while True:
            item = self.q.get()
            if item is None:
                break
            if self.error is not None:
                continue
            part_dir, t, meta, frame = item
            try:
                if part_dir != self.part_dir:
                    self._open_part(part_dir, frame)
                if (frame.shape[1], frame.shape[0]) != self.size:   # stream size changed mid-part
                    frame = cv2.resize(frame, self.size, interpolation=cv2.INTER_AREA)
                self.writer.write(frame)
                self.csv.writerow([self.index, f"{t:.6f}", meta.get("frame_id", ""),
                                   meta.get("generation", ""), meta.get("timestamp_source", "")])
                self.index += 1
                self.written += 1
            except Exception as e:      # keep tracking; report at the end
                self.error = str(e)
                print(f"[video] {self.cam}: {e}; video for this camera stopped", flush=True)
        self._close_part()


class VideoRecorder:
    def __init__(self, cameras, width=960, fps=10.0, queue_size=60):
        self.width, self.fps = int(width), float(fps)
        self.fourcc, self.ext = pick_codec()
        self.workers = {c: _CameraVideo(c, self.fourcc, self.ext, self.fps, queue_size) for c in cameras}
        self.next_due = {c: None for c in cameras}
        self.skipped = {c: 0 for c in cameras}
        for w in self.workers.values():
            w.start()
        print(f"[video] recording {len(cameras)} camera(s) at up to {self.fps:g} frames/s, "
              f"{self.width or 'full'} px wide, codec {self.fourcc}", flush=True)

    def push(self, cam, part_dir, t, meta, frame):
        """Queue one frame (never blocks). Frames are kept on a 1/fps schedule, so the
        video averages `fps` frames a second whatever the camera delivers; kept
        frames are scaled to `width` and copied, since the capture buffer reuses
        its frames."""
        if cam not in self.workers:
            return
        if self.fps > 0:
            due, period = self.next_due[cam], 1.0 / self.fps
            if due is not None and t < due:
                self.skipped[cam] += 1
                return
            # On schedule: next slot. After a gap (camera stalled): restart from now.
            self.next_due[cam] = t + period if due is None or t - due > period else due + period
        h, w = frame.shape[:2]
        if self.width and w > self.width:
            frame = cv2.resize(frame, (self.width, int(round(h * self.width / w / 2)) * 2),
                               interpolation=cv2.INTER_AREA)
        else:
            frame = frame.copy()
        self.workers[cam].push((str(part_dir), float(t), dict(meta), frame))

    def close(self, timeout=30.0):
        for w in self.workers.values():
            w.q.put(None)
        for w in self.workers.values():
            w.join(timeout)
        return {"codec": self.fourcc, "width": self.width, "fps": self.fps,
                "cameras": {c: {"frames": w.written, "dropped_queue_full": w.dropped,
                                "error": w.error} for c, w in self.workers.items()}}
