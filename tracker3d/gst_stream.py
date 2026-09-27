"""NTP-synchronized RTSP capture via GStreamer.

Each camera is a chrony/NTP client of this host, so the RTCP sender-report NTP
timestamp on every frame is a clock SHARED across all cameras. We use that as a
true capture time, which lets the fusion loop line cameras up to the same instant
instead of fusing whatever each one happened to have buffered (the source of
"two dots for one person" when a stream lags — see README "Synchronisation").

Why a pad probe instead of reading the frame's metadata directly: the NTP
reference-timestamp meta is reliably present on the parsed H.264 buffer but some
(hardware) decoders drop it. So we read NTP -> capture-time keyed by buffer PTS
on the parser's src pad (pre-decode), then re-join it to the decoded BGR frame by
PTS in the appsink. Decoder choice then can't break the timestamps.

Each camera keeps a short time-ordered buffer of (capture_t, frame); `frame_at()`
returns the frame nearest a requested capture time. Requires system PyGObject
(`gi`) on sys.path — see setup.sh / the `.pth` it installs into the venv.
"""
import threading
import time
from collections import deque

import numpy as np

import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst  # noqa: E402

Gst.init(None)

NTP_UNIX_OFFSET = 2208988800           # seconds between the 1900 (NTP) and 1970 (Unix) epochs

# decode stage (parsed H.264 -> BGR-able raw). hw = Jetson NVDEC path.
DECODERS = {
    "hw":    "nvv4l2decoder ! nvvidconv ! video/x-raw,format=BGRx",
    "nvdec": "nvh264dec ! videoconvert",
    "sw":    "openh264dec",
}


def _ntp_ns_to_unix(ns):
    s = ns / 1e9
    return s - NTP_UNIX_OFFSET if s > NTP_UNIX_OFFSET else s


class SyncCameraStream:
    def __init__(self, cam, decoder="sw", buffer_sec=1.0, width=None, height=None,
                 latency_ms=100, protocol="tcp", drop_on_latency=False,
                 retransmission=True, reconnect_stuck_s=3.0, reconnect_lag_s=0.0):
        self.name = cam["name"]
        self.source = str(cam["source"])
        self._buf = deque()            # (capture_t, frame), oldest -> newest
        self._ntp_by_pts = {}          # buffer PTS (ns) -> capture_t (unix s)
        self._lock = threading.Lock()
        self._buffer_sec = float(buffer_sec)
        self._last_t = None
        self._last_recv = time.time()  # wall clock a frame last arrived (stuck watchdog)
        self._frames = 0
        self._with_ntp = 0
        self._missing_ntp = 0          # decoded frames whose NTP capture time was lost
        self._ntp_seen = False         # has this stream ever delivered an NTP timestamp?
        self._reconnects = 0           # how many times the connection was restarted
        self._lag_since = None         # wall time lag first crossed the laggy threshold
        self._running = True

        # Pipeline build params (kept so the watchdog can rebuild on restart).
        self._decoder = decoder
        self._width, self._height = width, height
        self._latency_ms = int(latency_ms)
        self._protocol = protocol
        self._drop_on_latency = drop_on_latency
        self._retransmission = retransmission
        self._build_lock = threading.Lock()
        self._reconnect_stuck_s = float(reconnect_stuck_s)
        self._reconnect_lag_s = float(reconnect_lag_s)

        self._start_pipeline()

        # Watchdog: a camera that goes stuck (frames stop) or too laggy (frames
        # arrive but fall far behind real time) is fixed ~95% of the time by just
        # restarting its connection, so do exactly that. See config.yaml `sync:`.
        if self._reconnect_stuck_s > 0 or self._reconnect_lag_s > 0:
            threading.Thread(target=self._watchdog, daemon=True).start()

    def _start_pipeline(self):
        """(Re)build and start the GStreamer pipeline from the stored params."""
        dec = DECODERS.get(self._decoder, DECODERS["sw"])
        scale = "videoscale ! " if (self._width and self._height) else ""
        dims = (f",width={int(self._width)},height={int(self._height)}"
                if (self._width and self._height) else "")
        # Transport tuning. tcp is reliable but under loss it *delays* (the jitter
        # buffer ratchets up and never drains, so a camera drifts seconds behind the
        # rest); udp drops instead, keeping latency live. drop-on-latency / no
        # retransmission keep it bounded. See the `sync:` block in config.yaml.
        extra = ""
        if self._drop_on_latency:
            extra += " drop-on-latency=true"
        if not self._retransmission:
            extra += " do-retransmission=false"
        pipeline = (
            f"rtspsrc location={self.source} protocols={self._protocol} latency={self._latency_ms} "
            f"ntp-sync=true ntp-time-source=ntp add-reference-timestamp-meta=true{extra} name=src "
            f"! rtph264depay ! h264parse name=parse ! {dec} ! videoconvert ! {scale}"
            f"video/x-raw,format=BGR{dims} ! appsink name=sink emit-signals=true "
            f"max-buffers=2 drop=true sync=false"
        )
        self.pipe = Gst.parse_launch(pipeline)
        self.pipe.get_by_name("parse").get_static_pad("src").add_probe(
            Gst.PadProbeType.BUFFER, self._on_parsed)
        self._sink = self.pipe.get_by_name("sink")
        self._sink.connect("new-sample", self._on_sample)   # fires on the streaming thread
        self.pipe.set_state(Gst.State.PLAYING)

    def restart(self, why="manual"):
        """Tear the pipeline down and bring it back up. Clears buffered/in-flight
        state so the fresh connection starts clean (no stale frames or PTS map)."""
        with self._build_lock:
            if not self._running:
                return
            self._reconnects += 1
            print(f"[{self.name}] {why} — restarting connection (#{self._reconnects})", flush=True)
            try:
                self.pipe.set_state(Gst.State.NULL)
            except Exception as e:
                print(f"[{self.name}] teardown error: {e}", flush=True)
            with self._lock:
                self._buf.clear()
                self._ntp_by_pts.clear()
                self._ntp_seen = False
                self._last_t = None
                self._last_recv = time.time()   # reset grace so the watchdog won't re-fire instantly
            self._lag_since = None
            try:
                self._start_pipeline()
            except Exception as e:
                print(f"[{self.name}] restart failed: {e}", flush=True)

    def _watchdog(self):
        """Restart the connection when the stream goes stuck or stays too laggy."""
        # Give a fresh connection a moment to deliver its first frame before judging.
        grace = max(self._latency_ms / 1000.0 + self._buffer_sec, 1.5)
        while self._running:
            time.sleep(0.5)
            now = time.time()
            with self._lock:
                last_recv, last_t = self._last_recv, self._last_t
            if now - last_recv < grace:
                continue
            # Stuck: no decoded frame has arrived in a while -> pipeline is wedged.
            if self._reconnect_stuck_s > 0 and (now - last_recv) > self._reconnect_stuck_s:
                self.restart(why=f"stuck ({now - last_recv:.1f}s no frames)")
                continue
            # Laggy: frames still arrive, but their capture time trails real time by
            # more than the budget and stays there. Require it sustained so a single
            # network hiccup doesn't trigger a needless reconnect.
            if self._reconnect_lag_s > 0 and last_t is not None:
                lag = now - last_t
                if lag > self._reconnect_lag_s:
                    if self._lag_since is None:
                        self._lag_since = now
                    elif now - self._lag_since > self._reconnect_lag_s:
                        self.restart(why=f"laggy ({lag:.1f}s behind)")
                else:
                    self._lag_since = None

    # ---- pipeline callbacks --------------------------------------------------
    def _on_parsed(self, pad, info):
        """Pre-decode: stash NTP capture time keyed by PTS."""
        buf = info.get_buffer()
        meta = buf.get_reference_timestamp_meta()
        if meta is not None and buf.pts != Gst.CLOCK_TIME_NONE:
            self._ntp_by_pts[buf.pts] = _ntp_ns_to_unix(meta.timestamp)
            if len(self._ntp_by_pts) > 240:        # bound: drop oldest unmatched
                for k in list(self._ntp_by_pts)[:120]:
                    self._ntp_by_pts.pop(k, None)
        return Gst.PadProbeReturn.OK

    def _on_sample(self, sink):
        sample = sink.emit("pull-sample")
        if sample is None:
            return Gst.FlowReturn.OK
        buf = sample.get_buffer()
        s = sample.get_caps().get_structure(0)
        w, h = s.get_value("width"), s.get_value("height")
        cap_t = self._ntp_by_pts.pop(buf.pts, None)
        if cap_t is not None:
            self._with_ntp += 1
            if not self._ntp_seen:
                # First RTCP sender report: frames buffered so far carry ARRIVAL
                # time (a later clock than NTP capture time). Drop them so the
                # buffer never mixes the two clocks (wrong order, bad alignment).
                with self._lock:
                    self._buf.clear()
                self._ntp_seen = True
        else:
            self._missing_ntp += 1
            # An NTP stream hitting a dropout: stamping with arrival time (now) would
            # put this frame on a *different* clock than the buffered NTP frames
            # (which are one pipeline-latency old), inverting their order and making
            # the buffer_sec trim evict good frames. Drop it instead -- the camera
            # just has no fresh frame until NTP resumes, which is the honest state.
            if self._ntp_seen:
                return Gst.FlowReturn.OK
            cap_t = time.time()                    # never had NTP -> arrival time is all we have
        ok, mi = buf.map(Gst.MapFlags.READ)
        if not ok:
            return Gst.FlowReturn.OK
        try:
            row = mi.size // h                     # bytes per row (handles stride padding)
            frame = (np.frombuffer(mi.data, np.uint8)
                     .reshape(h, row // 3, 3)[:, :w, :].copy())
        finally:
            buf.unmap(mi)
        with self._lock:
            self._frames += 1
            self._buf.append((cap_t, frame))
            cutoff = cap_t - self._buffer_sec
            while self._buf and self._buf[0][0] < cutoff:
                self._buf.popleft()
            self._last_t = cap_t
            self._last_recv = time.time()       # wall clock: feeds the stuck watchdog
        return Gst.FlowReturn.OK

    # ---- accessors -----------------------------------------------------------
    def latest_capture_t(self):
        with self._lock:
            return self._last_t

    def latest(self):
        """Newest (capture_t, frame) or None."""
        with self._lock:
            return self._buf[-1] if self._buf else None

    def frame_at(self, t, tol):
        """(capture_t, frame) whose capture time is nearest t within tol seconds, else None."""
        with self._lock:
            best, bd = None, tol
            for ct, fr in self._buf:
                d = abs(ct - t)
                if d <= bd:
                    bd, best = d, (ct, fr)
            return best

    def stats(self):
        with self._lock:
            return {"frames": self._frames, "with_ntp": self._with_ntp,
                    "missing_ntp": self._missing_ntp, "reconnects": self._reconnects,
                    "buffered": len(self._buf), "last_t": self._last_t}

    def release(self):
        # Stop the watchdog first so it can't resurrect the pipeline mid-teardown,
        # and take _build_lock so we don't race a restart already in flight.
        self._running = False
        with self._build_lock:
            try:
                self.pipe.set_state(Gst.State.NULL)
                # NULL is asynchronous (rtspsrc returns ASYNC); block until the
                # streaming threads have actually stopped, else they get destroyed
                # while still joinable at interpreter exit -> "terminate called
                # without an active exception" and a hang. Bounded so a wedged
                # stream can't block shutdown forever.
                self.pipe.get_state(3 * Gst.SECOND)
            except Exception as e:
                print(f"[{self.name}] release error: {e}", flush=True)
