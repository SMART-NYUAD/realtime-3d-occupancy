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
                 retransmission=True):
        self.name = cam["name"]
        self.source = str(cam["source"])
        self._buf = deque()            # (capture_t, frame), oldest -> newest
        self._ntp_by_pts = {}          # buffer PTS (ns) -> capture_t (unix s)
        self._lock = threading.Lock()
        self._buffer_sec = float(buffer_sec)
        self._last_t = None
        self._frames = 0
        self._with_ntp = 0
        self._missing_ntp = 0          # decoded frames whose NTP capture time was lost
        self._ntp_seen = False         # has this stream ever delivered an NTP timestamp?
        self._running = True

        dec = DECODERS.get(decoder, DECODERS["sw"])
        scale = "videoscale ! " if (width and height) else ""
        dims = f",width={int(width)},height={int(height)}" if (width and height) else ""
        # Transport tuning. tcp is reliable but under loss it *delays* (the jitter
        # buffer ratchets up and never drains, so a camera drifts seconds behind the
        # rest); udp drops instead, keeping latency live. drop-on-latency / no
        # retransmission keep it bounded. See the `sync:` block in config.yaml.
        extra = ""
        if drop_on_latency:
            extra += " drop-on-latency=true"
        if not retransmission:
            extra += " do-retransmission=false"
        pipeline = (
            f"rtspsrc location={self.source} protocols={protocol} latency={int(latency_ms)} "
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
                    "missing_ntp": self._missing_ntp,
                    "buffered": len(self._buf), "last_t": self._last_t}

    def release(self):
        self._running = False
        self.pipe.set_state(Gst.State.NULL)


# ---- live self-test: python src/gst_stream.py [config.yaml] [decoder] [protocol] --
if __name__ == "__main__":
    import sys
    import yaml
    sys.path.insert(0, "src")
    from capture import list_cameras

    cfg = yaml.safe_load(open(sys.argv[1] if len(sys.argv) > 1 else "config.yaml"))
    decoder = sys.argv[2] if len(sys.argv) > 2 else "sw"
    protocol = sys.argv[3] if len(sys.argv) > 3 else "tcp"
    cams = list_cameras(cfg)
    streams = [SyncCameraStream(c, decoder=decoder, protocol=protocol,
                                drop_on_latency=(protocol != "tcp"),
                                retransmission=(protocol == "tcp"),
                                width=c.get("width"), height=c.get("height")) for c in cams]
    print(f"opened {len(streams)} cameras (decoder={decoder}, protocol={protocol}); warming up...")
    time.sleep(3.0)
    try:
        for _ in range(8):
            time.sleep(1.0)
            lasts = {s.name: s.latest_capture_t() for s in streams}
            have = {n: t for n, t in lasts.items() if t is not None}
            if len(have) < len(streams):
                print("waiting for:", [n for n in lasts if lasts[n] is None]); continue
            # common target = newest time all cameras can cover = min of their latest
            target = min(have.values())
            line = []
            for s in streams:
                got = s.frame_at(target, tol=0.20)
                age = (got[0] - target) * 1000 if got else None
                st = s.stats()
                line.append(f"{s.name}: lag={(time.time()-have[s.name])*1000:4.0f}ms "
                            f"match={'%+4.0fms' % age if age is not None else ' miss'} "
                            f"ntp={st['with_ntp']}/{st['frames']}"
                            f"{' miss=%d' % st['missing_ntp'] if st['missing_ntp'] else ''}")
            spread = (max(have.values()) - min(have.values())) * 1000
            print(f"latest-spread={spread:4.0f}ms | " + " | ".join(line))
    finally:
        for s in streams:
            s.release()
