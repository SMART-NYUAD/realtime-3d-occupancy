"""Live-run recording (track.py --record-session) and replaying recordings with a
subset of the cameras (tools/replay_tracking.py --cameras).

Run:  python -m unittest tests/test_recording.py -v
"""
import csv
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
from test_tracking import H, LENS, POSES, SEAT, W, detection, projectors   # noqa: E402
from tracker3d import session_recorder                                     # noqa: E402
from tracker3d.autocalib.session import load_session, write_json           # noqa: E402
from tracker3d.config import load_config                                   # noqa: E402
from tracker3d.geometry import save_calibration                            # noqa: E402


def tool(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


class RecordingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        tmp = self.tmp_path = Path(self.tmp.name)
        with open(ROOT / "config.yaml") as f:
            cfg = yaml.safe_load(f)
        cfg.pop("autocalibration", None)
        cfg["floorplan"]["pointcloud_file"] = str(ROOT / cfg["floorplan"]["pointcloud_file"])
        cfg["floorplan"]["valid_area"] = []
        cfg["cameras"] = []
        for n, p in projectors().items():
            lens_path = tmp / f"{n}_intrinsics.json"
            write_json(lens_path, LENS.to_dict())
            save_calibration(str(tmp / f"{n}_homography.npy"), p.H0, (W, H), lens=LENS)
            cfg["cameras"].append({"name": n, "source": "rtsp://x", "intrinsics_file": str(lens_path),
                                   "homography_file": str(tmp / f"{n}_homography.npy")})
        with open(tmp / "cfg.yaml", "w") as f:
            yaml.safe_dump(cfg, f)
        self.cfg = load_config(str(tmp / "cfg.yaml"))
        self.cams = self.cfg["cameras"]

    def tearDown(self):
        self.tmp.cleanup()

    def record(self, root, seconds=30.0, part_minutes=0.25, fps=10, disk_free=1e12):
        """Simulate a live run: every camera sees one seated person at SEAT."""
        clock, rng = Clock(), np.random.default_rng(3)
        usage = mock.Mock(free=disk_free)
        with mock.patch.object(session_recorder.time, "monotonic", clock), \
                mock.patch.object(session_recorder.shutil, "disk_usage", return_value=usage):
            rec = session_recorder.SessionRecorder(root, self.cams, self.cfg, part_minutes)
            for k in range(int(seconds * fps)):
                clock.t = k / fps
                for ci, cam in enumerate(POSES):
                    d = detection(cam, SEAT, "sitting", desk=True, rng=rng)
                    raw = [{"bbox": d["bbox"], "conf": 0.85, "kxy": d["kxy"], "kconf": d["kconf"]},
                           {"bbox": (5.0, 5.0, 40.0, 90.0), "conf": 0.3, "kxy": None, "kconf": None}]
                    tracked = [{"bbox": d["bbox"], "id": 7}]
                    rec.frame(cam, 1000.0 + k / fps + 0.01 * ci, (W, H),
                              {"timestamp_source": "ntp", "generation": 0, "frame_id": k}, tracked, raw)
            rec.close()
        return rec

    def test_live_recording_is_split_into_replayable_parts(self):
        root = self.tmp_path / "day1"
        rec = self.record(root, seconds=30.0, part_minutes=0.25)     # 15 s parts
        info = json.loads((root / "recording.json").read_text())
        self.assertEqual(info["parts"], ["part_0001", "part_0002"])
        self.assertEqual(info["frames"], 30 * 10 * len(POSES))
        self.assertEqual(info["cameras"], list(POSES))
        self.assertIsNotNone(info["ended_utc"])
        self.assertTrue((root / "calib_snapshot" / "yi01_homography.npy").exists())
        self.assertTrue((root / "calib_snapshot" / "yi01_intrinsics.json").exists())
        # Each part is a complete session in the autocalib.py record format.
        session = load_session(root / "part_0001")
        frames = [r for r in session["records"] if r["kind"] == "frame"]
        self.assertTrue(frames)
        people = frames[0]["people"]
        self.assertEqual([p["id"] for p in people], [7, None])      # tracked box and untracked box
        self.assertEqual(len(people[0]["kxy"]), 17)
        self.assertNotIn("kxy", people[1])
        self.assertEqual(rec.frames, info["frames"])

    def test_refuses_existing_folder_and_missing_lens(self):
        root = self.tmp_path / "exists"
        root.mkdir()
        with self.assertRaises(ValueError):
            session_recorder.SessionRecorder(root, self.cams, self.cfg)
        cams = [dict(self.cams[0], intrinsics_file=str(self.tmp_path / "missing.json"))] + self.cams[1:]
        with self.assertRaises(ValueError) as e:
            session_recorder.SessionRecorder(self.tmp_path / "new", cams, self.cfg)
        self.assertIn("yi01", str(e.exception))
        self.assertFalse((self.tmp_path / "new").exists())

    def test_recording_stops_when_disk_is_nearly_full(self):
        root = self.tmp_path / "full"
        rec = self.record(root, seconds=5.0, disk_free=1e9)          # below the 2 GB floor
        info = json.loads((root / "recording.json").read_text())
        self.assertIn("disk space", info["stopped_early"])
        self.assertEqual(rec.frames, 0)

    def test_replay_whole_day_and_camera_subsets(self):
        replay = tool("replay_tracking")
        root = self.tmp_path / "day1"
        self.record(root, seconds=30.0, part_minutes=0.25)
        out_all, out_sub = self.tmp_path / "all.csv", self.tmp_path / "sub.csv"
        replay.replay(str(root), self.cfg, str(out_all))
        replay.replay(str(root), self.cfg, str(out_sub), cameras=["yi01", "yi05"])
        rows = {}
        for name, path in (("all", out_all), ("sub", out_sub)):
            with open(path) as f:
                rows[name] = list(csv.DictReader(f))
        det_cams = {name: {r["cam"] for r in rs if r["kind"] == "det"} for name, rs in rows.items()}
        self.assertEqual(det_cams["all"], set(POSES))
        self.assertEqual(det_cams["sub"], {"yi01", "yi05"})
        for rs in rows.values():                    # one seated person, one ID across both parts
            ids = {r["id"] for r in rs if r["kind"] == "track"}
            self.assertEqual(len(ids), 1, ids)
            times = [float(r["t"]) for r in rs if r["kind"] == "track"]
            self.assertGreater(max(times) - min(times), 25.0)
        self.assertEqual(replay.recorded_cameras(str(root)), set(POSES))
        with self.assertRaises(SystemExit):
            replay.replay(str(root), self.cfg, str(self.tmp_path / "bad.csv"), cameras=["yi09"])

    def test_track_sources_report_frame_metadata(self):
        sys.argv = ["track.py"]
        spec = importlib.util.spec_from_file_location("track", ROOT / "track.py")
        track = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(track)
        src = track.Sources.__new__(track.Sources)
        src.synced = True
        src.group = mock.Mock()
        src.group.next_aligned.return_value = (
            5.0, {"yi01": (4.98, "frame", {"timestamp_source": "ntp", "generation": 1, "frame_id": 42})})
        t, got = src.next()
        self.assertEqual((t, got), (5.0, {"yi01": (4.98, "frame")}))
        self.assertEqual(src.meta["yi01"], {"t": 4.98, "timestamp_source": "ntp", "generation": 1,
                                            "frame_id": 42})
        src.group.next_aligned.assert_called_with(timeout=0.25, with_metadata=True)
        plain = track.Sources.__new__(track.Sources)
        plain.synced = False
        plain.streams = {"yi04": mock.Mock(read=mock.Mock(return_value=(9, "frame")))}
        t, got = plain.next()
        self.assertEqual(got, {"yi04": (9, "frame")})
        self.assertEqual(plain.meta["yi04"]["frame_id"], 9)
        self.assertEqual(plain.meta["yi04"]["timestamp_source"], "wall")


if __name__ == "__main__":
    unittest.main()
