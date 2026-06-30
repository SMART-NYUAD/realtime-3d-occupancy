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
    def __init__(self, cams, decoder="sw", tol_s=0.075, buffer_sec=1.0, latency_ms=100,
                 lag_budget_s=0.5, protocol="tcp", drop_on_latency=False,
                 retransmission=True):
        self.streams = {
            c["name"]: SyncCameraStream(c, decoder=decoder, buffer_sec=buffer_sec,
                                        latency_ms=latency_ms, protocol=protocol,
                                        drop_on_latency=drop_on_latency,
                                        retransmission=retransmission)  # native resolution
            for c in cams
        }
        self.tol = float(tol_s)
        self.lag_budget = float(lag_budget_s)
        self._last_target = 0.0

    def next_aligned(self, wait=True, timeout=2.0):
        """Return (target_t, {name: frame}) for the newest capture time the *live*
        cameras share, not yet served, blocking up to `timeout`s for one. A camera
        lagging more than `lag_budget` behind the most-live one is treated as stuck
        and excluded from pacing, so it can't drag every camera back to its time
        (which freezes them once the gap exceeds the per-camera buffer). Cameras
        lacking a frame within tol of the target are omitted. (None, {}) on timeout."""
        t0 = time.time()
        while True:
            lasts = {}
            for n, s in self.streams.items():
                t = s.latest_capture_t()
                if t is not None:
                    lasts[n] = t
            if lasts:
                newest = max(lasts.values())
                # Pace off the newest instant the non-stuck cameras can all cover.
                live = [t for t in lasts.values() if newest - t <= self.lag_budget]
                target = min(live)
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

    def health(self):
        """Snapshot for live monitoring: (spread_s, {name: {lag, with_ntp, frames,
        missing, stuck}}). `lag` is how far behind wall-clock each camera's newest
        frame is; `stuck` flags cameras beyond `lag_budget` of the most-live one."""
        now = time.time()
        info, lasts = {}, []
        for n, s in self.streams.items():
            st = s.stats()
            lt = st["last_t"]
            if lt is not None:
                lasts.append(lt)
            info[n] = {"lag": (now - lt) if lt is not None else None,
                       "with_ntp": st["with_ntp"], "frames": st["frames"],
                       "missing": st["missing_ntp"], "stuck": False}
        if lasts:
            newest = max(lasts)
            for n, d in info.items():
                lt = now - d["lag"] if d["lag"] is not None else None
                d["stuck"] = lt is not None and newest - lt > self.lag_budget
        spread = (max(lasts) - min(lasts)) if len(lasts) >= 2 else 0.0
        return spread, info

    def stats(self):
        return {n: s.stats() for n, s in self.streams.items()}

    def release(self):
        for s in self.streams.values():
            s.release()
