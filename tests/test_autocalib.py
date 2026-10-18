"""CPU-only synthetic walk -> solve -> anchor -> independent validation -> export.

Run: .venv/bin/python -m unittest discover -s tests -p 'test_autocalib.py' -v
No cameras, detector weights, CUDA or GStreamer are needed.
"""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import threading
import types
import unittest
from unittest.mock import patch
from collections import deque

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tracker3d.autocalib.anchor import (anchor_rig, apply_transform, fit_similarity,
                                       map_identity, resolve_points)
from tracker3d.autocalib.observations import prepare, split_samples
from tracker3d.autocalib.session import (SessionWriter, check_artifact, lens_from_dict,
                                        load_session, read_json, seal, write_json)
from tracker3d.autocalib.solve import (evaluate, floor_homography, initialize,
                                      project_raw, rig_geometry, solve_session)
from tracker3d.autocalib.validation import export_candidate, validate
from tracker3d.geometry import FloorProjector
from tracker3d.lens import LensModel
from tracker3d.localize import Localizer


def look_at(C, target):
    forward = np.array(target) - C
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0, 0, 1.])
    right /= np.linalg.norm(right)
    return np.vstack([right, np.cross(forward, right), forward])


WAYPOINTS = np.array([[1, 1], [3.6, 1], [1.2, 3.8], [3.6, 3.6], [2.4, 2.5],
                      [0.8, 2.6], [3.4, 2.1], [2.3, 3.5], [2.2, 0.8], [1.5, 2.2],
                      [3.5, 3.4], [1.4, 3.5], [3.2, 1.2], [1.1, 1.4], [2.8, 2.7]])


def position(t, variant=0):
    waypoints = WAYPOINTS if variant == 0 else WAYPOINTS[::-1]
    index = min(int(t // 2.5), len(waypoints) - 1)
    f = np.clip((t % 2.5 - 2.) / 0.5, 0, 1)
    return (1 - f) * waypoints[index] + f * waypoints[min(index + 1, len(waypoints) - 1)]


def synthetic(path, variant=0, noise=0.20, swap=0.0):
    rng = np.random.default_rng(15 + variant)
    K = np.array([[1050., 0, 960], [0, 1045., 540], [0, 0, 1.]])
    lens = LensModel(K, [-0.18, 0.03, 0.0002, -0.0001, 0.], (1920, 1080))
    centres = {"a": np.array([0., -3., 3.]), "b": np.array([6., 0., 3.5]),
               "c": np.array([-3., 5., 3.2])}
    poses = {n: (look_at(C, [2, 2, 0.8]), C) for n, C in centres.items()}
    cameras = {n: {"lens": lens.to_dict()} for n in poses}
    writer = SessionWriter(path, cameras)
    epoch = 1000 + 100 * variant
    for k, t in enumerate(np.arange(0, len(WAYPOINTS) * 2.5, 0.05)):
        for ni, (n, (R, C)) in enumerate(poses.items()):
            capture = t + ni * 0.009  # different frame phases on the same NTP clock
            xy = position(capture, variant)
            xyz = np.tile([*xy, 0.8], (17, 1))
            xyz[0] = [*xy, 1.6]
            xyz[5], xyz[6] = [xy[0] - .18, xy[1], 1.35], [xy[0] + .18, xy[1], 1.35]
            xyz[11], xyz[12] = [xy[0] - .12, xy[1], .8], [xy[0] + .12, xy[1], .8]
            xyz[15], xyz[16] = [xy[0] - .10, xy[1], .08], [xy[0] + .10, xy[1], .08]
            px, _ = project_raw(lens, R, C, xyz)
            px += rng.normal(0, noise, px.shape)
            if swap and rng.random() < swap:       # pose model exchanged left/right ankles
                px[[15, 16]] = px[[16, 15]]
            box = [float(px[:, 0].min() - 20), float(px[:, 1].min() - 20),
                   float(px[:, 0].max() + 20), float(px[:, 1].max() + 12)]
            writer.append({"kind": "frame", "camera": n, "t": epoch + capture,
                           "generation": 0, "frame_id": k, "timestamp_source": "ntp",
                           "image_size": [1920, 1080],
                           "people": [{"id": 1, "bbox": box, "conf": .95,
                                       "kxy": px.tolist(), "kconf": [0.95] * 17}]})
    markers = []
    for i in range(len(WAYPOINTS)):
        t = epoch + 2.5 * i + 1.
        writer.append({"kind": "marker", "name": f"M{i + 1}", "t": t})
        markers.append({"marker": f"M{i + 1}", "world": position(t - epoch, variant).tolist()})
    writer.close()
    return load_session(path), poses, lens, markers


class CalibrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.base = Path(cls.temp.name)
        cls.session, cls.poses, cls.lens, cls.markers = synthetic(cls.base / "walk")
        cls.validation_session, _, _, cls.checks = synthetic(cls.base / "check", variant=1)
        cls.samples, cls.lenses, cls.counts = prepare(cls.session)
        cls.rig = solve_session(cls.session, max_nfev=300)
        scan = cls.base / "scan.las"
        scan.write_bytes(b"synthetic map identity; rendering not needed in headless tests")
        cls.cfg = {"cameras": [], "floorplan": {"pointcloud_file": str(scan), "flip_x": True, "up_axis": "y"},
                   "detector": {"model": "yolo11m-pose.engine", "tracker": "bytetrack.yaml"},
                   "output": {"mqtt": {"enabled": True}}}
        for n in cls.poses:
            lens_path = cls.base / f"{n}_intrinsics.json"
            write_json(lens_path, cls.lens.to_dict())
            cls.cfg["cameras"].append({"name": n, "source": f"rtsp://example.invalid/{n}",
                                       "intrinsics_file": str(lens_path),
                                       "homography_file": str(cls.base / f"{n}_old.npy")})
        points = [{**p, "role": "anchor" if i < 3 else "check"}
                  for i, p in enumerate(cls.markers[:4])]
        cls.anchored = anchor_rig(cls.rig, points, cls.cfg)
        cls.report = validate(cls.anchored, cls.validation_session, cls.cfg, cls.checks[:5])

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_joint_geometry_and_independent_accuracy(self):
        self.assertTrue(self.report["passed"], self.report["reasons"])
        self.assertLess(self.report["agreement_m"]["p95"], .025)
        self.assertLess(self.report["absolute_error_m"]["max"], .025)
        self.assertEqual(set(self.rig["cameras"]), set(self.poses))
        for c in self.rig["cameras"].values():
            R = np.array(c["R"])
            np.testing.assert_allclose(R.T @ R, np.eye(3), atol=1e-8)
            self.assertAlmostEqual(np.linalg.det(R), 1.)
            self.assertGreater(c["C"][2], 0)

    def test_pose_heights_and_relative_rotation(self):
        solved, _ = rig_geometry(self.rig)
        scale = self.anchored["scale_m"]
        for n, (R, C) in solved.items():
            self.assertLess(abs(C[2] * scale - self.poses[n][1][2]), .03)
        for a, b in (("a", "b"), ("b", "c")):
            true = self.poses[b][0] @ self.poses[a][0].T
            got = solved[b][0] @ solved[a][0].T
            angle = np.linalg.norm(cv2.Rodrigues(got @ true.T)[0])
            self.assertLess(np.degrees(angle), .5)

    def test_session_roundtrip_and_exclusive_output(self):
        loaded = load_session(self.base / "walk")
        self.assertEqual(loaded["id"], self.session["id"])
        with self.assertRaises(FileExistsError):
            SessionWriter(self.base / "walk", self.session["manifest"]["cameras"])
        path = self.base / "exclusive.json"
        write_json(path, {"a": 1})
        with self.assertRaises(FileExistsError):
            write_json(path, {"a": 2})
        self.assertEqual(read_json(path), {"a": 1})

    def test_corrupt_lens_rejected(self):
        data = deepcopy(self.lens.to_dict())
        data["camera_matrix"][0][0] *= 1.1
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            lens_from_dict(data)

    def test_interrupted_session_recovery(self):
        path = self.base / "interrupted"
        writer = SessionWriter(path, self.session["manifest"]["cameras"])
        writer.append(self.session["records"][0])
        writer.close(complete=False)
        with open(path / "observations.jsonl", "a") as f:
            f.write('{"kind":')
        with self.assertRaisesRegex(ValueError, "incomplete"):
            load_session(path)
        recovered = load_session(path, allow_incomplete=True)
        self.assertEqual(len(recovered["records"]), 1)

    def test_timestamps_duplicates_and_ambiguity(self):
        altered = deepcopy(self.session)
        original = altered["records"][0]
        altered["records"].insert(1, deepcopy(original))
        for r in altered["records"][2:30]:
            if r["kind"] == "frame":
                r["timestamp_source"] = "arrival"
        for r in altered["records"][30:60]:
            if r["kind"] == "frame":
                r["people"] *= 2
        samples, _, counts = prepare(altered)
        self.assertEqual(counts["duplicate"], 1)
        self.assertGreater(counts["non_ntp"], 0)
        self.assertGreater(counts["person_count"], 0)
        self.assertLess(len(samples), len(self.samples))

    def test_no_interpolation_across_drops_or_reconnect(self):
        altered = deepcopy(self.session)
        for r in altered["records"]:
            if r["kind"] == "frame" and 1005 < r["t"] < 1006:
                r["generation"] = r["frame_id"]  # every frame belongs to another connection
        samples, _, _ = prepare(altered)
        self.assertFalse(any(1005.1 < s["t"] < 1005.9 for s in samples))

    def test_no_interpolation_over_missing_person(self):
        altered = deepcopy(self.session)
        for r in altered["records"]:
            if r["kind"] == "frame" and 1005 < r["t"] < 1006:
                r["people"] = []
        samples, _, _ = prepare(altered)
        self.assertFalse(any(1005.1 < s["t"] < 1005.9 for s in samples))

    def test_disconnected_camera_rejected(self):
        samples = deepcopy(self.samples)
        for s in samples:
            s["views"].pop("c", None)
        with self.assertRaisesRegex(ValueError, "disconnected"):
            initialize(samples, self.lenses)

    def test_collinear_walk_rejected(self):
        samples = deepcopy(self.samples)
        for s in samples:
            for v in s["views"].values():
                v["rays"][:2, 1] = .3
        with self.assertRaisesRegex(ValueError, "overlap"):
            initialize(samples, self.lenses)

    def test_chain_overlap_initialization(self):
        samples = deepcopy(self.samples)
        for i, s in enumerate(samples):
            s["views"].pop("a" if i % 2 else "c", None)  # some samples already lack a view
        root, seeds, graph = initialize(samples, self.lenses)
        self.assertEqual(set(seeds[0][1]), {"a", "b", "c"})
        self.assertNotIn("a/c", graph)

    def test_similarity_and_anchor_guards(self):
        theta, scale = .4, 2.7
        S = np.array([[scale * np.cos(theta), -scale * np.sin(theta), 5],
                      [scale * np.sin(theta), scale * np.cos(theta), 3], [0, 0, 1]])
        xy = np.array([[0, 0], [3, 0], [0, 3], [1, 2.]])
        np.testing.assert_allclose(fit_similarity(xy, apply_transform(S, xy)), S, atol=1e-10)
        with self.assertRaisesRegex(ValueError, "span"):
            fit_similarity([[0, 0], [1, 0]], [[0, 0], [.1, 0]])
        with self.assertRaisesRegex(ValueError, "coincide"):
            fit_similarity([[0, 0], [0, 0]], [[0, 0], [3, 0]])
        with self.assertRaisesRegex(ValueError, "reused"):
            resolve_points([self.markers[0], self.markers[0]], self.rig["trajectory"], self.rig["markers"])

    def test_mirrored_map_uses_configured_handedness(self):
        cfg = deepcopy(self.cfg)
        cfg["floorplan"]["flip_x"] = False
        points = deepcopy(self.markers[:4])
        for p in points:
            p["world"][0] = 8 - p["world"][0]
        anchor = anchor_rig(self.rig, points, cfg)
        self.assertEqual(anchor["map"]["handedness"], -1)
        self.assertLess(np.linalg.det(np.array(anchor["transform"])[:2, :2]), 0)
        self.assertLess(max(anchor["fit_error_m"]), .025)

    def test_map_and_artifact_mismatch(self):
        cfg = deepcopy(self.cfg)
        cfg["floorplan"]["flip_x"] = False
        self.assertNotEqual(map_identity(cfg), self.anchored["map"])
        with self.assertRaisesRegex(ValueError, "map frame changed"):
            validate(self.anchored, self.validation_session, cfg, self.checks)
        corrupt = deepcopy(self.rig)
        corrupt["body"]["nose_height"] *= 2
        with self.assertRaisesRegex(ValueError, "modified"):
            check_artifact(corrupt, "rig")

    def test_validation_requires_independence_and_checkpoints(self):
        own = validate(self.anchored, self.session, self.cfg)
        self.assertFalse(own["passed"])
        self.assertIn("separate", " ".join(own["reasons"]))
        no_check = validate(self.anchored, self.validation_session, self.cfg)
        self.assertFalse(no_check["passed"])
        self.assertIn("checkpoints", " ".join(no_check["reasons"]))
        duplicate = deepcopy(self.session)
        duplicate["id"] = "changed-manifest"
        report = validate(self.anchored, duplicate, self.cfg)
        self.assertFalse(report["independent_recording"])

    def test_focal_refinement_is_bounded_and_guarded(self):
        refined = solve_session(self.session, refine_focal=True, max_nfev=300)
        self.assertTrue(refined["focal_refinement"]["requested"])
        for n, camera in refined["cameras"].items():
            solved = lens_from_dict(camera["lens"])
            multiplier = solved.K[0, 0] / self.lens.K[0, 0]
            self.assertGreaterEqual(multiplier, .97)
            self.assertLessEqual(multiplier, 1.03)
        if refined["focal_refinement"]["accepted"]:
            self.assertLess(refined["heldout"]["agreement_rig_units"]["median"],
                            self.rig["heldout"]["agreement_rig_units"]["median"])

    def test_realistic_keypoint_noise_and_ankle_swaps(self):
        """Real pose keypoints jitter by pixels and sometimes exchange left/right
        ankles. Pauses must still be found and the rig must still validate."""
        noisy, poses, _, marks = synthetic(self.base / "noisy", noise=4.0, swap=0.05)
        check, _, _, checks = synthetic(self.base / "noisy_check", variant=1, noise=4.0, swap=0.05)
        rig = solve_session(noisy)
        points = [{**p, "role": "anchor"} for p in marks[:4]]
        anchored = anchor_rig(rig, points, self.cfg)
        report = validate(anchored, check, self.cfg, checks[:8])
        self.assertTrue(report["passed"], report["reasons"])
        self.assertLess(report["absolute_error_m"]["max"], 0.05)
        for n, c in rig["cameras"].items():
            self.assertAlmostEqual(c["C"][2] * anchored["scale_m"], poses[n][1][2], delta=0.08)

    def test_solo_walk_pauses_replace_markers(self):
        """One person alone: no SPACE presses. Long pauses found afterwards stand
        in for markers in both anchoring and validation."""
        from tracker3d.autocalib.pauses import points_template, session_pauses
        pauses, _ = session_pauses(self.rig, self.session, min_seconds=1.0)
        truth = [m["world"] for m in self.markers]
        self.assertEqual(len(pauses), len(truth))           # one per waypoint, in order
        anchors = points_template(pauses[:4], "anchor")
        for p, world in zip(anchors, truth):
            p["world"] = world
        anchored = anchor_rig(self.rig, anchors, self.cfg)
        self.assertLess(max(anchored["fit_error_m"]), 0.02)
        found, _ = session_pauses(anchored, self.validation_session, min_seconds=1.0)
        checks = points_template(found[:6], "check")
        for p, f, c in zip(checks, found, self.checks):
            self.assertLess(np.linalg.norm(np.subtract(f["map_xy"], c["world"])), 0.05)
            p["world"] = c["world"]
        report = validate(anchored, self.validation_session, self.cfg, checks)
        self.assertTrue(report["passed"], report["reasons"])
        with self.assertRaises(ValueError):                  # an unfilled template is refused
            anchor_rig(self.rig, points_template(pauses[:3], "anchor"), self.cfg)

    def test_anchor_from_current_calibration(self):
        """No tape measuring: the existing per-camera calibrations place the rig.
        Each 'current' calibration is deliberately off by 5 cm in a different
        direction; averaging over many points must still land close to truth."""
        from copy import deepcopy
        from tracker3d.autocalib.anchor import anchor_from_current
        from tracker3d.geometry import save_calibration
        cfg = deepcopy(self.cfg)
        shifts = {"a": (0.05, 0.0), "b": (-0.05, 0.0), "c": (0.0, 0.05)}
        for c in cfg["cameras"]:
            R, C = self.poses[c["name"]]
            T = np.eye(3); T[:2, 2] = shifts[c["name"]]
            H = T @ floor_homography(self.lens, R, C)
            path = self.base / f"{c['name']}_current.npy"
            lens = LensModel.load(c["intrinsics_file"])
            save_calibration(str(path), H, self.lens.image_size, lens=lens)
            c["homography_file"] = str(path)
        anchored = anchor_from_current(self.rig, self.session, cfg)
        self.assertEqual(anchored["anchor"]["method"], "current_calibration")
        self.assertGreater(anchored["anchor"]["points_used"], 100)
        measured = validate(anchored, self.validation_session, cfg, self.checks[:5])
        self.assertTrue(measured["passed"], measured["reasons"])
        self.assertLess(measured["absolute_error_m"]["max"], 0.06)
        unmeasured = validate(anchored, self.validation_session, cfg, None, allow_unmeasured=True)
        self.assertTrue(unmeasured["passed"], unmeasured["reasons"])
        self.assertFalse(unmeasured["absolute_accuracy_measured"])
        strict = validate(anchored, self.validation_session, cfg, None)
        self.assertIn("supply independent measured floor checkpoints", strict["reasons"])

    def test_robust_solve_with_pose_outliers(self):
        samples = deepcopy(self.samples)
        rng = np.random.default_rng(6)
        for i in rng.choice(len(samples), len(samples) // 12, replace=False):
            view = samples[i]["views"]["b"]
            view["raw"][:2] += [35, -25]
            view["rect"] = self.lens._undistort_exact(view["raw"])
            rays = (np.linalg.inv(self.lens.K_rect) @ np.c_[view["rect"], np.ones(3)].T).T
            view["rays"] = rays[:, :2] / rays[:, 2:]
        # Inject after temporal filtering to exercise RANSAC and robust BA directly.
        with patch('tracker3d.autocalib.solve.prepare', return_value=(samples, self.lenses, self.counts)):
            rig = solve_session(self.session, max_nfev=300)
        anchor = anchor_rig(rig, self.markers[:4], self.cfg)
        report = validate(anchor, self.validation_session, self.cfg, self.checks[:5])
        self.assertTrue(report["passed"], report["reasons"])

    def test_stale_lens_and_report_refuse_export(self):
        cfg = deepcopy(self.cfg)
        wrong = self.base / "wrong_lens.json"
        lens = LensModel(self.lens.K * np.array([[1.01, 1, 1], [1, 1.01, 1], [1, 1, 1]]),
                         self.lens.dist, self.lens.image_size)
        write_json(wrong, lens.to_dict())
        cfg["cameras"][0]["intrinsics_file"] = str(wrong)
        with self.assertRaisesRegex(ValueError, "input lens changed"):
            export_candidate(self.anchored, self.report, cfg, self.base / "badlens")
        report = seal({**self.report, "anchored_id": "another-rig"})
        with self.assertRaisesRegex(ValueError, "passing validation"):
            export_candidate(self.anchored, report, self.cfg, self.base / "badreport")

    def test_wrong_checkpoints_fail(self):
        wrong = deepcopy(self.checks[:3])
        for p in wrong:
            p["world"][0] += 1
        report = validate(self.anchored, self.validation_session, self.cfg, wrong)
        self.assertFalse(report["passed"])
        with self.assertRaisesRegex(ValueError, "passing validation"):
            export_candidate(self.anchored, report, self.cfg, self.base / "rejected")
        self.assertFalse((self.base / "rejected").exists())

    def test_export_runtime_floor_and_ankle_projection(self):
        path = export_candidate(self.anchored, self.report, self.cfg, self.base / "candidate")
        import yaml
        cfg = yaml.safe_load(path.read_text())
        self.assertFalse(cfg["output"]["mqtt"]["enabled"])
        self.assertEqual(read_json(path.parent / "rig.json")["id"], self.anchored["id"])
        for cam in cfg["cameras"]:
            n = cam["name"]
            lens = LensModel.load(cam["intrinsics_file"])
            proj = FloorProjector.load(cam["homography_file"], lens)
            proj.set_frame_size(1920, 1080)
            self.assertIsNotNone(proj.ankle_H)
            xyz = np.array([[2.1, 2.3, 0], [2.1, 2.3, .08]])
            raw, _ = project_raw(self.lens, *self.poses[n], xyz)
            np.testing.assert_allclose(proj.to_floor(*raw[0]), xyz[0, :2], atol=.025)
            np.testing.assert_allclose(proj.to_floor(*raw[1], plane="ankle"), xyz[1, :2], atol=.025)
            proj.set_frame_size(960, 540)
            np.testing.assert_allclose(proj.to_floor(*(raw[1] / 2), plane="ankle"), xyz[1, :2], atol=.025)
        with self.assertRaises(FileExistsError):
            export_candidate(self.anchored, self.report, self.cfg, path.parent)

    def test_localizer_projects_ankles_individually(self):
        class Plan:
            @staticmethod
            def in_valid_area(x, y):
                return True
        n = "a"
        R, C = self.poses[n]
        proj = FloorProjector(floor_homography(self.lens, R, C), (1920, 1080), self.lens,
                              floor_homography(self.lens, R, C, .08))
        proj.set_frame_size(1920, 1080)
        points = np.array([[1.5, 2, .08], [2.5, 3, .08]])
        raw, _ = project_raw(self.lens, R, C, points)
        kxy, kcf = np.zeros((17, 2)), np.zeros(17)
        kxy[15:17], kcf[15:17] = raw, 1.
        person = {"bbox": (500, 200, 1100, 900), "kxy": kxy, "kconf": kcf}
        localizer = Localizer({"detector": {"kp_conf": .5}, "fusion": {"use_blob": False}}, Plan())
        out = localizer.localize(n, [person], proj, (1920, 1080))[0]
        np.testing.assert_allclose(out["world"], points[:, :2].mean(axis=0), atol=.001)

    def test_help_and_offline_imports(self):
        import subprocess
        r = subprocess.run([sys.executable, "autocalib.py", "--help"], cwd=ROOT,
                           capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("record,solve,pauses,anchor,validate,export", r.stdout)
        r = subprocess.run([sys.executable, "-c", "import sys; import autocalib; "
                            "import tracker3d.autocalib.solve; "
                            "assert 'ultralytics' not in sys.modules; assert 'gi' not in sys.modules"],
                           cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)

    def test_cli_pipeline(self):
        import subprocess
        import yaml
        cfg_path = self.base / "cli-config.yaml"
        cfg_path.write_text(yaml.safe_dump(self.cfg))
        points, checkpoints = self.base / "cli-points.json", self.base / "cli-checkpoints.json"
        write_json(points, self.markers[:4])
        write_json(checkpoints, self.checks[:5])
        rig, anchored, report = (self.base / name for name in ("cli-rig.json", "cli-anchor.json", "cli-report.json"))
        commands = [
            ["solve", "--session", str(self.base / "walk"), "--out", str(rig)],
            ["anchor", "--rig", str(rig), "--points", str(points), "--out", str(anchored)],
            ["validate", "--rig", str(anchored), "--session", str(self.base / "check"),
             "--checkpoints", str(checkpoints), "--out", str(report)],
            ["export", "--rig", str(anchored), "--report", str(report), "--out", str(self.base / "cli-candidate")]]
        for command in commands:
            r = subprocess.run([sys.executable, "autocalib.py", *command, "--config", str(cfg_path)],
                               cwd=ROOT, capture_output=True, text=True, timeout=120)
            self.assertEqual(r.returncode, 0, f"{command[0]}: {r.stderr}\n{r.stdout}")
        self.assertTrue((self.base / "cli-candidate/config.yaml").exists())

    def test_watchdog_reconnects_a_laggy_stream_that_still_delivers_frames(self):
        """A camera whose frames keep arriving (every ~50 ms) but trail real time
        must still be reconnected; the start-up grace must not hide the lag."""
        module = self._gst_module()
        stream = module.SyncCameraStream.__new__(module.SyncCameraStream)
        stream._lock, stream._lag_since = threading.Lock(), None
        stream._latency_ms, stream._buffer_sec = 100, 1.0
        stream._reconnect_stuck_s, stream._reconnect_lag_s = 3.0, 0.7
        now = 10_000.0
        stream._connected_at = now - 600          # running for 10 minutes
        stream._last_recv = now - 0.05            # a frame arrived 50 ms ago
        stream._last_t = now - 0.9                # ...captured 0.9 s ago
        self.assertIsNone(stream._watchdog_check(now))           # starts the lag timer
        self.assertIn("laggy", stream._watchdog_check(now + 0.8))  # sustained -> reconnect
        stream._connected_at = now + 0.5          # just reconnected: grace period
        stream._lag_since = None
        self.assertIsNone(stream._watchdog_check(now + 1.0))
        stream._connected_at, stream._last_recv = now - 600, now - 5   # frames stopped
        self.assertIn("stuck", stream._watchdog_check(now))
        stream._last_recv, stream._last_t = now - 0.05, now - 0.2      # healthy
        self.assertIsNone(stream._watchdog_check(now))

    def _gst_module(self):
        gst = types.SimpleNamespace(init=lambda _: None,
                                    FlowReturn=types.SimpleNamespace(OK=0),
                                    PadProbeReturn=types.SimpleNamespace(OK=0),
                                    MapFlags=types.SimpleNamespace(READ=1))
        gi = types.ModuleType("gi")
        gi.require_version = lambda *args: None
        repository = types.ModuleType("gi.repository")
        repository.Gst = gst
        with patch.dict(sys.modules, {"gi": gi, "gi.repository": repository}):
            spec = importlib.util.spec_from_file_location("capture_under_test", ROOT / "tracker3d/gst_stream.py")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        return module

    def test_capture_provenance_and_legacy_frame_api(self):
        gst = types.SimpleNamespace(init=lambda _: None,
                                    FlowReturn=types.SimpleNamespace(OK=0),
                                    PadProbeReturn=types.SimpleNamespace(OK=0),
                                    MapFlags=types.SimpleNamespace(READ=1))
        gi = types.ModuleType("gi")
        gi.require_version = lambda *args: None
        repository = types.ModuleType("gi.repository")
        repository.Gst = gst
        with patch.dict(sys.modules, {"gi": gi, "gi.repository": repository}):
            spec = importlib.util.spec_from_file_location("capture_under_test", ROOT / "tracker3d/gst_stream.py")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        stream = module.SyncCameraStream.__new__(module.SyncCameraStream)
        stream._lock, stream._buf = threading.Lock(), deque()
        stream._reconnects, stream._frames = 0, 0
        stream._ntp_seen, stream._ntp_by_pts = False, {}
        stream._missing_ntp, stream._with_ntp, stream._buffer_sec = 0, 0, 1.
        buf = types.SimpleNamespace(pts=1, map=lambda flags: (True, types.SimpleNamespace(size=12, data=bytes(12))),
                                    unmap=lambda info: None)
        struct = types.SimpleNamespace(get_value=lambda key: 2)
        sample = types.SimpleNamespace(get_buffer=lambda: buf,
                                       get_caps=lambda: types.SimpleNamespace(get_structure=lambda index: struct))
        sink = types.SimpleNamespace(emit=lambda signal: sample)
        stream._on_sample(sink, 0)
        self.assertEqual(stream._buf[-1][2]["timestamp_source"], "arrival")
        stream._ntp_by_pts[1] = 1000.
        stream._on_sample(sink, 0)
        self.assertEqual(len(stream._buf), 1)  # arrival-clock buffer was cleared
        t, frame, meta = stream.frame_at_with_metadata(1000., .01)
        self.assertEqual(meta["timestamp_source"], "ntp")
        self.assertEqual(meta["generation"], 0)
        self.assertEqual(len(stream.frame_at(1000., .01)), 2)
        self.assertEqual(len(stream.latest()), 2)
        meta["generation"] = 99
        self.assertEqual(stream.frame_at_with_metadata(1000., .01)[2]["generation"], 0)
        stream._on_sample(sink, 0)  # missing NTP after lock-on is dropped
        self.assertEqual(len(stream._buf), 1)
        stream._reconnects = 1
        self.assertEqual(stream._on_sample(None, 0), 0)  # stale callback must not touch sink
        self.assertEqual(stream._on_parsed(None, None, 0), 0)


if __name__ == "__main__":
    unittest.main()
