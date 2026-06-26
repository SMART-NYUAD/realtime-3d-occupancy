"""Coordinate several NTP-synced camera streams into time-aligned frame sets.

Each step we choose a common target capture time = the newest instant ALL
cameras can cover (the most-delayed camera's latest frame), then return the
frame nearest that instant from each camera. Detection + fusion then see one
synchronized snapshot of the room, so a laggy stream no longer places a moving
person at a stale spot and splits them into a second track.

Frames are delivered at the camera's native resolution (no rescaling) so they
match the pixels the homographies were calibrated against. Needs system gi +
GStreamer (see gst_stream.py); RTSP/H.264 sources only.
"""
import time

from gst_stream import SyncCameraStream


class SyncGroup:
    def __init__(self, cams, decoder="sw", tol_s=0.075, buffer_sec=1.0, latency_ms=100):
        self.streams = {
            c["name"]: SyncCameraStream(c, decoder=decoder, buffer_sec=buffer_sec,
                                        latency_ms=latency_ms)   # native resolution
            for c in cams
        }
        self.tol = float(tol_s)
        self._last_target = 0.0

    def next_aligned(self, wait=True, timeout=2.0):
        """Return (target_t, {name: frame}) for the newest common capture time not
        yet served, blocking up to `timeout`s for one. Cameras lacking a frame
        within tol of the target are omitted. (None, {}) on timeout."""
        t0 = time.time()
        while True:
            lasts = {n: s.latest_capture_t() for n, s in self.streams.items()}
            if all(t is not None for t in lasts.values()):
                target = min(lasts.values())          # paced by the most-delayed camera
                if target > self._last_target:
                    self._last_target = target
                    frames = {}
                    for n, s in self.streams.items():
                        got = s.frame_at(target, self.tol)
                        if got is not None:
                            frames[n] = got[1]
                    return target, frames
            if not wait or time.time() - t0 > timeout:
                return None, {}
            time.sleep(0.005)

    def stats(self):
        return {n: s.stats() for n, s in self.streams.items()}

    def release(self):
        for s in self.streams.values():
            s.release()
