"""Tracking tests on a simulated lab: three calibrated cameras (Yi-like lens,
2.55 m high) watching 3D people, seated or standing, with desks hiding legs.

Each fix is checked against the old behaviour on the same scene:
  - pose from calibration, shoulder-plane localization, triangulation/posture
  - steady seated positions (ankle flicker), seated hold and its release
  - duplicate boxes, identity through jumps, publishing, neighbours kept apart

Run:  python -m unittest tests/test_tracking.py -v
"""
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
from tracker3d.autocalib.solve import floor_homography, project_raw   # noqa: E402
from tracker3d.detector import tracker_input_mask                      # noqa: E402
from tracker3d.fusion import Fusion, publishable                       # noqa: E402
from tracker3d.geometry import FloorProjector                          # noqa: E402
from tracker3d.lens import LensModel                                    # noqa: E402
from tracker3d.localize import Localizer                               # noqa: E402

W, H = 1920, 1080
LENS = LensModel(np.array([[1290., 0, 950], [0, 1290., 600], [0, 0, 1.]]),
                 [-0.32, 0.10, -0.004, -0.001, -0.016], (W, H))
CAMERAS = {"yi01": np.array([1.4, 9.8, 2.55]), "yi04": np.array([0.5, 2.5, 2.55]),
           "yi05": np.array([5.4, 0.5, 2.55])}
TARGET = np.array([3.2, 5.0, 0.8])


def look_at(C, target):
    forward = np.array(target) - C
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0, 0, 1.])
    right /= np.linalg.norm(right)
    return np.vstack([right, np.cross(forward, right), forward])


POSES = {n: (look_at(C, TARGET), C) for n, C in CAMERAS.items()}


def projectors():
    out = {}
    for n, (R, C) in POSES.items():
        p = FloorProjector(floor_homography(LENS, R, C), (W, H), LENS)
        p.set_frame_size(W, H)
        out[n] = p
    return out


class _Plan:
    def in_valid_area(self, x, y):
        return -1 <= x <= 10 and -1 <= y <= 12


NEW_CFG = {"detector": {"kp_conf": 0.5}, "fusion": {"use_blob": True}, "localize": {}}
OLD_CFG = {"detector": {"kp_conf": 0.5},
           "fusion": {"use_blob": True, "blob_only_fallback": False},
           "localize": {"torso_estimates": False}}
OLD_FUSION = {"merge_distance_m": 1.6, "blob_merge_gate_m": 3.0, "dup_suppress_m": 1.0,
              "identity_matching": False, "triangulate": False, "blob_only_fallback": False,
              "seated_persistence": False, "settled_smoothing": False, "confirm_s": 0}


def skeleton(xy, posture, facing=(1.0, 0.0)):
    """COCO-17 keypoints in 3D (meters) for a person at floor position xy."""
    d = np.array(facing, float) / np.linalg.norm(facing)
    side = np.array([-d[1], d[0]])
    xy = np.asarray(xy, float)
    P = np.zeros((17, 3))

    def put(i, base, fwd, lat, z):
        P[i] = [*(base + fwd * d + lat * side), z]
    if posture == "sitting":
        sho, hip = xy + 0.05 * d, xy
        put(0, sho, 0.08, 0, 1.25)
        put(1, sho, 0.07, -0.03, 1.29); put(2, sho, 0.07, 0.03, 1.29)
        put(3, sho, 0.0, -0.07, 1.26); put(4, sho, 0.0, 0.07, 1.26)
        put(5, sho, 0, -0.18, 1.05); put(6, sho, 0, 0.18, 1.05)
        put(7, hip, 0.05, -0.22, 0.80); put(8, hip, 0.05, 0.22, 0.80)
        put(9, hip, 0.30, -0.15, 0.75); put(10, hip, 0.30, 0.15, 0.75)
        put(11, hip, 0, -0.12, 0.48); put(12, hip, 0, 0.12, 0.48)
        put(13, hip, 0.45, -0.10, 0.50); put(14, hip, 0.45, 0.10, 0.50)
        put(15, hip, 0.50, -0.10, 0.08); put(16, hip, 0.50, 0.10, 0.08)
    else:
        put(0, xy, 0.08, 0, 1.62)
        put(1, xy, 0.07, -0.03, 1.66); put(2, xy, 0.07, 0.03, 1.66)
        put(3, xy, 0.0, -0.07, 1.62); put(4, xy, 0.0, 0.07, 1.62)
        put(5, xy, 0, -0.18, 1.40); put(6, xy, 0, 0.18, 1.40)
        put(7, xy, 0, -0.22, 1.10); put(8, xy, 0, 0.22, 1.10)
        put(9, xy, 0, -0.24, 0.85); put(10, xy, 0, 0.24, 0.85)
        put(11, xy, 0, -0.12, 0.92); put(12, xy, 0, 0.12, 0.92)
        put(13, xy, 0, -0.10, 0.50); put(14, xy, 0, 0.10, 0.50)
        put(15, xy, 0, -0.10, 0.08); put(16, xy, 0, 0.10, 0.08)
    return P


def detection(cam, xy, posture, pid=1, hide=(), desk=False, rng=None, facing=(1.0, 0.0)):
    """Camera `cam`'s pose detection of a person: keypoints (with noise), confidences
    (hidden joints at 0) and a box. With `desk`, the box stops at a 0.75 m desk edge
    in front of the person and the legs are hidden."""
    R, C = POSES[cam]
    P = skeleton(xy, posture, facing)
    px, _ = project_raw(LENS, R, C, P)
    if rng is not None:
        px = px + rng.normal(0, 1.5, px.shape)
    kconf = np.full(17, 0.9)
    hidden = set(hide) | ({11, 12, 13, 14, 15, 16} if desk else set())
    kconf[list(hidden)] = 0.0
    vis = px[kconf > 0]
    x1, y1 = vis.min(axis=0) - 15
    x2, y2 = vis.max(axis=0) + 15
    if desk:
        edge, _ = project_raw(LENS, R, C, np.array([[*(np.asarray(xy) + 0.35 * np.asarray(facing)), 0.75]]))
        y2 = max(edge[0][1], vis[:, 1].max() + 15)
    return {"bbox": (float(x1), float(y1), float(x2), float(y2)), "conf": 0.85,
            "kxy": px, "kconf": kconf, "id": pid}


def run(frames, cfg=NEW_CFG, fusion_cfg=None, dt=0.1, t0=100.0):
    """Feed a list of frames ({cam: [detections]}) through Localizer + Fusion.
    Returns (fusion, list of per-frame (t, tracks, localized detections))."""
    projs = projectors()
    loc = Localizer(cfg, _Plan())
    fus = Fusion.from_config({"use_blob": True, **(fusion_cfg or {})})
    history = []
    for k, frame in enumerate(frames):
        t = t0 + k * dt
        dets = []
        hints = fus.posture_hints()
        for cam, people in frame.items():
            people = [dict(p) for p in people]
            loc.localize(cam, people, projs[cam], (W, H), hints=hints)
            dets += people
        history.append((t, fus.update(dets, now=t), dets))
    return fus, history


SEAT = np.array([3.0, 5.2])           # chair position seen by all three cameras
SHOULDER_XY = SEAT + 0.05 * np.array([1.0, 0.0])


class TrackingTests(unittest.TestCase):

    def test_scene_is_visible_to_all_cameras(self):
        for cam in POSES:
            d = detection(cam, SEAT, "sitting")
            self.assertTrue(np.all((d["kxy"] > 0) & (d["kxy"] < [W, H])), cam)

    def test_pose_from_calibration(self):
        for n, p in projectors().items():
            _, _, C = p.pose()
            np.testing.assert_allclose(C, CAMERAS[n], atol=0.01)
        # The lab's real calibration files give plausible mounting heights.
        from tracker3d.config import list_cameras, load_config
        from tracker3d.lens import load_lens
        cfg = load_config(str(ROOT / "config.yaml"))
        for c in list_cameras(cfg):
            c = {**c, "homography_file": str(ROOT / c["homography_file"]),
                 "intrinsics_file": str(ROOT / c["intrinsics_file"])}
            lens, _ = load_lens(c)
            proj = FloorProjector.load(c["homography_file"], lens)
            proj.set_frame_size(*lens.image_size)
            self.assertTrue(2.0 < proj.pose()[2][2] < 2.9, c["name"])

    def test_seated_hidden_legs_localized_from_shoulders(self):
        """Legs hidden by a desk: the old foot guess is biased differently per camera;
        the shoulder estimate lands on the person in every camera."""
        rng = np.random.default_rng(0)
        frame = {cam: [detection(cam, SEAT, "sitting", desk=True, rng=rng)] for cam in POSES}
        _, new = run([frame], NEW_CFG)
        _, old = run([frame], OLD_CFG, OLD_FUSION)
        new_err = [np.linalg.norm(np.subtract(d["world"], SHOULDER_XY)) for d in new[0][2]]
        old_xy = [np.array(d["world"]) for d in old[0][2] if d["world"] is not None]
        old_spread = max(np.linalg.norm(a - b) for a in old_xy for b in old_xy)
        self.assertLess(max(new_err), 0.15, new_err)
        self.assertGreater(old_spread, 0.4, old_spread)
        self.assertTrue(all(d["foot_source"] == "torso" for d in new[0][2]))

    def test_triangulation_measures_height_and_posture(self):
        rng = np.random.default_rng(1)
        for posture, z in (("sitting", 1.05), ("standing", 1.40)):
            frames = [{cam: [detection(cam, SEAT, posture, desk=posture == "sitting", rng=rng)]
                       for cam in POSES} for _ in range(10)]
            fus, hist = run(frames)
            tracks = hist[-1][1]
            self.assertEqual(len(tracks), 1)
            self.assertEqual(tracks[0]["posture"], posture)
            self.assertAlmostEqual(tracks[0]["shoulder_h"], z, delta=0.05)

    def test_standing_behind_furniture_is_one_dot(self):
        """Legs hidden but standing: every camera first guesses "seated" (35 cm too
        low) and lands 1-2 m off in its own direction. Their shoulder lines of sight
        still cross, so they are grouped, measured and become one correct dot."""
        for spot in (SEAT, SEAT + [1.5, -1.5], SEAT + [-1.0, 2.0]):
            rng = np.random.default_rng(7)
            frames = [{c: [detection(c, spot, "standing", desk=True, rng=rng)] for c in POSES}
                      for _ in range(15)]
            _, hist = run(frames)
            tracks = hist[-1][1]
            self.assertEqual(len(tracks), 1, (spot, tracks))
            self.assertLess(np.linalg.norm(np.subtract(tracks[0]["world"], spot)), 0.1)
            self.assertEqual(tracks[0]["posture"], "standing")

    def test_two_people_behind_furniture_not_merged(self):
        rng = np.random.default_rng(8)
        a, b = SEAT, SEAT + np.array([0.0, 1.2])
        frames = [{c: [detection(c, a, "standing", pid=1, desk=True, rng=rng),
                       detection(c, b, "standing", pid=2, desk=True, rng=rng)] for c in POSES}
                  for _ in range(15)]
        _, hist = run(frames)
        tracks = hist[-1][1]
        self.assertEqual(len(tracks), 2, tracks)
        for spot in (a, b):
            self.assertLess(min(np.linalg.norm(np.subtract(g["world"], spot)) for g in tracks), 0.15)

    def test_seated_position_steady_with_flickering_ankle(self):
        """A seated person whose ankles under the desk flicker in and out of view in
        one camera (knees hidden): the dot must not jump between feet and chair."""
        rng = np.random.default_rng(2)
        frames = []
        for k in range(150):
            frame = {}
            for cam in POSES:
                hide = (11, 12, 13, 14) if cam == "yi04" else (11, 12, 13, 14, 15, 16)
                if cam == "yi04" and k % 2:
                    hide = (11, 12, 13, 14, 15, 16)
                frame[cam] = [detection(cam, SEAT, "sitting", hide=hide, rng=rng)]
            if 60 <= k < 90:
                frame.pop("yi05")          # one camera drops out for 3 s
            frames.append(frame)
        _, new = run(frames)
        _, old = run(frames, OLD_CFG, OLD_FUSION)
        pts = np.array([tr[0]["world"] for _, tr, _ in new[50:] if tr])
        old_pts = np.array([tr[0]["world"] for _, tr, _ in old[50:] if tr])
        self.assertLess(pts.std(axis=0).max(), 0.05, pts.std(axis=0))
        self.assertLess(np.linalg.norm(pts.mean(axis=0) - SHOULDER_XY), 0.15)
        # Measured on this scene: old ~10 cm range and ~33 cm off; new < 1 cm and on target.
        self.assertGreater(np.ptp(old_pts, axis=0).max(), 5 * np.ptp(pts, axis=0).max())
        self.assertGreater(np.linalg.norm(old_pts.mean(axis=0) - SHOULDER_XY), 0.25)
        self.assertTrue(new[-1][1][0]["settled"])

    def test_seated_hold_and_release(self):
        rng = np.random.default_rng(3)
        seated = [{cam: [detection(cam, SEAT, "sitting", desk=True, rng=rng)] for cam in POSES}
                  for _ in range(60)]
        hidden = [{} for _ in range(1000)]           # 100 s with no detections at all
        fus, hist = run(seated + hidden)
        self.assertEqual(len(hist[-1][1]), 1, "seated person must be held for 100 s")
        self.assertEqual(len(publishable(hist[-1][1])), 1)
        gid = hist[-1][1][0]["gid"]
        # They stand up and walk away (the per-camera tracker gives new IDs).
        walk = []
        for k in range(40):
            xy = SEAT + np.array([0.0, -0.08 * k])
            walk.append({cam: [detection(cam, xy, "standing", pid=7, rng=rng)] for cam in POSES})
        fus, hist = run(seated + hidden[:50] + walk)
        tracks = hist[-1][1]
        self.assertEqual(len(tracks), 1, tracks)
        self.assertEqual(tracks[0]["gid"], gid, "the seated track follows the person who got up")
        self.assertFalse(tracks[0]["settled"])
        self.assertEqual(tracks[0]["posture"], "standing")
        # Released: nothing is held at the empty desk.
        self.assertGreater(np.linalg.norm(np.subtract(tracks[0]["world"], SEAT)), 2.0)

    def test_passer_by_cannot_take_over_hidden_seated_person(self):
        rng = np.random.default_rng(4)
        seated = [{cam: [detection(cam, SEAT, "sitting", desk=True, rng=rng)] for cam in POSES}
                  for _ in range(60)]
        passing = []
        for k in range(30):
            xy = SEAT + np.array([1.2, -1.5 + 0.1 * k])
            passing.append({cam: [detection(cam, xy, "standing", pid=9, rng=rng)] for cam in POSES})
        _, hist = run(seated + passing)
        tracks = hist[-1][1]
        self.assertEqual(len(tracks), 2, tracks)
        seated_track = min(tracks, key=lambda g: np.linalg.norm(np.subtract(g["world"], SEAT)))
        self.assertLess(np.linalg.norm(np.subtract(seated_track["world"], SHOULDER_XY)), 0.15)

    def test_duplicate_box_and_ignore_zone(self):
        full = detection("yi04", SEAT + [0, 0.3], "standing")
        torso = dict(full)
        x1, y1, x2, y2 = full["bbox"]
        torso_box = (x1 + 5, y1 + 5, x2 - 5, y1 + 0.55 * (y2 - y1))
        other = detection("yi04", SEAT + [1.5, -0.5], "standing")
        xyxy = np.array([full["bbox"], torso_box, other["bbox"]], np.float32)
        conf = np.array([0.9, 0.6, 0.8], np.float32)
        kxy = np.stack([full["kxy"], full["kxy"], other["kxy"]])
        kcf = np.stack([full["kconf"], full["kconf"], other["kconf"]])
        self.assertEqual(tracker_input_mask(xyxy, conf, kxy, kcf).tolist(), [True, False, True])
        zone = [np.array([[other["bbox"][0] - 5, other["bbox"][1] - 5],
                          [other["bbox"][2] + 5, other["bbox"][1] - 5],
                          [other["bbox"][2] + 5, other["bbox"][3] + 5],
                          [other["bbox"][0] - 5, other["bbox"][3] + 5]], np.float32)]
        self.assertEqual(tracker_input_mask(xyxy, conf, kxy, kcf, zone).tolist(), [True, False, False])

    def test_identity_survives_position_jump(self):
        """A sudden 2.5 m jump in one person's estimate (beyond the 1.8 m gate) keeps
        the ID with identity matching; the old matcher leaves a ghost and a new ID."""
        def det(cam, xy):
            return {"cam": cam, "id": 4, "world": tuple(xy), "conf": 0.9, "foot_sigma_m": 0.1}
        frames = [[det("a", (2.0, 2.0)), det("b", (2.05, 2.0))]] * 6 + \
                 [[det("a", (4.5, 2.0)), det("b", (4.55, 2.0))]] * 6
        for identity, expect in ((True, 1), (False, 2)):
            f = Fusion(merge_distance_m=1.0, match_gate_m=1.8, identity_matching=identity,
                       confirm_s=0.3, max_speed_mps=0.0, smoothing=1.0)
            out = []
            for k, dets in enumerate(frames):
                out = f.update(dets, now=10 + 0.1 * k)
            self.assertEqual(len(out), expect, (identity, out))
        self.assertEqual(out[0]["gid"], 1)

    def test_vanished_walker_not_published_but_settled_is(self):
        tracks = [{"gid": 1, "world": (1, 1), "last_seen_s": 0.1, "settled": False},
                  {"gid": 2, "world": (2, 2), "last_seen_s": 0.9, "settled": False},
                  {"gid": 3, "world": (3, 3), "last_seen_s": 40.0, "settled": True}]
        self.assertEqual([t["gid"] for t in publishable(tracks, 0.5)], [1, 3])

    def test_neighbours_at_adjacent_desks_stay_apart(self):
        rng = np.random.default_rng(5)
        other_seat = SEAT + np.array([0.0, 1.0])
        frames = [{cam: [detection(cam, SEAT, "sitting", pid=1, desk=True, rng=rng),
                         detection(cam, other_seat, "sitting", pid=2, desk=True, rng=rng)]
                   for cam in POSES} for _ in range(20)]
        _, new = run(frames)
        self.assertEqual(len(new[-1][1]), 2, new[-1][1])

    def test_busy_scene_two_seated_and_a_walker(self):
        """60 s: two neighbours seated 1 m apart (each camera misses them 25% of the
        time) and a walker passing between the desks. Exactly three IDs, the right
        number of people published, seated dots steady. The old pipeline gets the
        count wrong almost all the time on this scene (6+ IDs)."""
        rng = np.random.default_rng(11)
        a, b = SEAT, SEAT + np.array([0.0, 1.0])
        frames = []
        for k in range(600):
            frame = {c: [] for c in POSES}
            for c in POSES:
                if rng.random() > 0.25:
                    frame[c].append(detection(c, a, "sitting", pid=1, desk=True, rng=rng))
                if rng.random() > 0.25:
                    frame[c].append(detection(c, b, "sitting", pid=2, desk=True, rng=rng))
                if 200 <= k < 320:
                    xy = np.array([1.5 + 0.03 * (k - 200), 5.7])
                    frame[c].append(detection(c, xy, "standing", pid=3, rng=rng))
            frames.append(frame)
        _, hist = run(frames)
        ids = {g["gid"] for _, tracks, _ in hist for g in tracks}
        self.assertEqual(len(ids), 3, ids)
        expected = [3 if 205 <= k < 320 else 2 for k in range(50, 600)]
        wrong = sum(len(publishable(hist[k][1])) != e for k, e in zip(range(50, 600), expected))
        self.assertLessEqual(wrong, 10)        # only confirm/publish delays at entry/exit
        settled = {}
        for _, tracks, _ in hist[50:]:
            for g in tracks:
                if g["settled"]:
                    settled.setdefault(g["gid"], []).append(g["world"])
        self.assertEqual(len(settled), 2)
        for pts in settled.values():
            self.assertLess(np.array(pts).std(axis=0).max(), 0.02)

    def test_crossing_walkers_keep_their_ids(self):
        rng = np.random.default_rng(21)
        frames = []
        for k in range(120):
            a = np.array([1.5 + 0.03 * k, 5.0])
            b = np.array([3.0, 3.2 + 0.03 * k])
            frames.append({c: [detection(c, a, "standing", pid=1, rng=rng),
                               detection(c, b, "standing", pid=2, rng=rng, facing=(0, 1))]
                           for c in POSES})
        _, hist = run(frames)
        self.assertEqual(len({g["gid"] for _, tr, _ in hist for g in tr}), 2)
        first = {g["gid"]: g["world"] for g in hist[10][1]}
        last = {g["gid"]: g["world"] for g in hist[-1][1]}
        walker_a = min(first, key=lambda g: np.linalg.norm(np.subtract(first[g], [1.8, 5.0])))
        self.assertLess(np.linalg.norm(np.subtract(last[walker_a], [1.5 + 0.03 * 119, 5.0])), 0.3)

    def test_replay_and_metrics_tools(self):
        """A recorded session (keypoints only) replays through the tracking with the
        fixes on and off, and the metrics tool reads both logs."""
        import importlib.util, tempfile, yaml
        from tracker3d.autocalib.session import SessionWriter, write_json
        from tracker3d.geometry import save_calibration

        def tool(name):
            spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / f"{name}.py")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
        replay, metrics = tool("replay_tracking"), tool("tracking_metrics")
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            with open(ROOT / "config.yaml") as f:
                cfg = yaml.safe_load(f)
            cfg.pop("autocalibration", None)
            cfg["floorplan"]["pointcloud_file"] = str(ROOT / cfg["floorplan"]["pointcloud_file"])
            cfg["floorplan"]["valid_area"] = []
            cfg["cameras"] = []
            for n, p in projectors().items():
                lens_path = tmp / f"{n}_intrinsics.json"
                write_json(lens_path, LENS.to_dict())
                save_calibration(str(tmp / f"{n}.npy"), p.H0, (W, H), lens=LENS)
                cfg["cameras"].append({"name": n, "source": "rtsp://x", "intrinsics_file": str(lens_path),
                                       "homography_file": str(tmp / f"{n}.npy")})
            rng = np.random.default_rng(5)
            writer = SessionWriter(tmp / "session", {n: {"lens": LENS.to_dict()} for n in POSES})
            for k in range(200):
                for ci, cam in enumerate(POSES):
                    if rng.random() < 0.25:
                        continue
                    d = detection(cam, SEAT, "sitting", desk=True, rng=rng)
                    writer.append({"kind": "frame", "camera": cam, "t": 1000 + 0.05 * k + 0.01 * ci,
                                   "image_size": [W, H], "timestamp_source": "ntp", "generation": 0,
                                   "frame_id": k, "people": [{"bbox": list(d["bbox"]), "conf": 0.85,
                                                             "id": 1, "kxy": d["kxy"].tolist(),
                                                             "kconf": d["kconf"].tolist()}]})
            writer.close()
            from tracker3d.config import load_config
            base = tmp / "cfg.yaml"
            with open(base, "w") as f:
                yaml.safe_dump(cfg, f)
            new_cfg = load_config(str(base))
            old_cfg = replay.apply_overrides(load_config(str(base)), replay.LEGACY)
            replay.replay(str(tmp / "session"), new_cfg, str(tmp / "new.csv"))
            replay.replay(str(tmp / "session"), old_cfg, str(tmp / "old.csv"))
            new = metrics.metrics(str(tmp / "new.csv"), expected=1)
            old = metrics.metrics(str(tmp / "old.csv"), expected=1, coast_s=1.5)
        self.assertEqual(new["distinct_ids"], 1)
        self.assertGreater(new["frames_with_expected_count_pct"], 95)
        self.assertTrue(new["seated_jitter_cm"])
        self.assertLess(max(new["seated_jitter_cm"].values()), 2.0)
        self.assertGreaterEqual(old["distinct_ids"], 1)

    def test_mqtt_payload_has_posture(self):
        from tracker3d.mqtt_output import MqttPositionPublisher
        pub = MqttPositionPublisher.__new__(MqttPositionPublisher)   # no broker connection
        pub._last_seen, pub._trails, pub.trail_length, pub.serial = {}, {}, 10, "test"
        payload = pub._build_payload([{"gid": 5, "world": (1.0, 2.0), "posture": "sitting"}], 0.0)
        obj = payload["object_list"]["5"]
        self.assertEqual(obj["posture"], "sitting")
        self.assertEqual(obj["object_type"], "pedestrian")


if __name__ == "__main__":
    unittest.main()
