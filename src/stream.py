"""Threaded RTSP/camera reader: each camera reads in its own thread and keeps
only the latest frame, so a slow or stalled stream never blocks the others and
processing always sees fresh frames (no buffer lag)."""
import threading
import time
import cv2

from capture import open_capture


class CameraStream:
    def __init__(self, cam):
        self.name = cam["name"]
        self.cap = open_capture(cam)
        self._frame = None
        self._lock = threading.Lock()
        self._running = True
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()

    def _loop(self):
        while self._running:
            ok, f = self.cap.read()
            if not ok:
                time.sleep(0.02)
                continue
            with self._lock:
                self._frame = f

    def read(self):
        """Latest frame (a copy), or None if nothing received yet."""
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    def release(self):
        self._running = False
        self._t.join(timeout=1.0)
        self.cap.release()
