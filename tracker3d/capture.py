"""Camera capture for the non-synced path: USB (V4L2), video files, or RTSP via
FFMPEG. Multi-camera RTSP runs through the NTP-synced GStreamer path instead
(gst_stream.py / sync.py).

Note: the pip OpenCV build has no GStreamer backend, so Jetson CSI / NVDEC
pipelines can't be opened through cv2 here; the synced path does its own
GStreamer (with NVDEC) through PyGObject.
"""
import sys
import threading
import time

import cv2


def open_capture(cam):
    """Open a cv2.VideoCapture for one merged camera dict (see config.list_cameras)."""
    source = str(cam["source"])
    if source.startswith("rtsp://"):
        cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)
        cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    elif source.startswith("usb:"):
        # V4L2 is Linux-only (the Thor); elsewhere (e.g. a Mac webcam) let OpenCV pick.
        backend = cv2.CAP_V4L2 if sys.platform.startswith("linux") else cv2.CAP_ANY
        cap = cv2.VideoCapture(int(source.split(":", 1)[1]), backend)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(cam.get("width", 1280)))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(cam.get("height", 720)))
        cap.set(cv2.CAP_PROP_FPS, int(cam.get("fps", 30)))
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    else:
        cap = cv2.VideoCapture(source)           # video file
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera source '{source}'. "
                           "Check the device and the source string in config.yaml.")
    return cap


class CameraStream:
    """Threaded reader that keeps only the newest frame, so a slow or stalled
    camera never blocks the others. Reconnects a stream that stops delivering."""

    def __init__(self, cam, reconnect_stuck_s=3.0):
        self.name = cam["name"]
        self._cam = cam
        self.cap = open_capture(cam)
        self._frame = None
        self._seq = 0                         # increments per new frame
        self._last_recv = time.time()
        self._reconnects = 0
        self._reconnect_stuck_s = float(reconnect_stuck_s)
        self._lock = threading.Lock()
        self._running = True
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()

    def _loop(self):
        while self._running:
            ok, f = self.cap.read()
            now = time.time()
            if not ok:
                if self._reconnect_stuck_s > 0 and now - self._last_recv > self._reconnect_stuck_s:
                    self._reconnect()
                else:
                    time.sleep(0.02)
                continue
            with self._lock:
                self._frame = f
                self._seq += 1
            self._last_recv = now

    def _reconnect(self):
        print(f"[{self.name}] stuck — restarting connection (#{self._reconnects + 1})", flush=True)
        try:
            self.cap.release()
        except Exception:
            pass
        try:
            self.cap = open_capture(self._cam)
        except Exception as e:
            print(f"[{self.name}] reconnect failed: {e}", flush=True)
            time.sleep(1.0)
            return
        self._reconnects += 1
        self._last_recv = time.time()

    def read(self):
        """(seq, frame) of the newest frame, or (0, None) before the first one.
        The frame is owned by the caller only until the next read; don't draw on it."""
        with self._lock:
            return self._seq, self._frame

    def release(self):
        self._running = False
        self._t.join(timeout=1.0)
        self.cap.release()
