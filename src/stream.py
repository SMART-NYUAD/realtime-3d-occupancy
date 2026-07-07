"""Threaded RTSP/camera reader: each camera reads in its own thread and keeps
only the latest frame, so a slow or stalled stream never blocks the others and
processing always sees fresh frames (no buffer lag)."""
import threading
import time
import cv2

from capture import open_capture


class CameraStream:
    def __init__(self, cam, reconnect_stuck_s=5.0):
        self.name = cam["name"]
        self._cam = cam
        self.cap = open_capture(cam)
        self._frame = None
        self._last_recv = time.time()        # wall clock a frame last arrived
        self._reconnects = 0
        # No fresh frame for this long -> reopen the connection (fixes a stuck /
        # laggy stream ~95% of the time). 0 disables auto-reconnect.
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
                # Stream stalled or ended: if it stays down, reconnect instead of
                # spinning forever on a dead capture.
                if self._reconnect_stuck_s > 0 and now - self._last_recv > self._reconnect_stuck_s:
                    self._reconnect()
                else:
                    time.sleep(0.02)
                continue
            with self._lock:
                self._frame = f
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
            time.sleep(1.0)              # back off before the next attempt
            return
        self._reconnects += 1
        self._last_recv = time.time()   # grace period so it isn't judged stuck immediately

    def read(self):
        """Latest frame (a copy), or None if nothing received yet."""
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    def release(self):
        self._running = False
        self._t.join(timeout=1.0)
        self.cap.release()
