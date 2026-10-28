"""Coverage planner tests: visibility, blocking, crossing angles, furniture, the
mounting-spot search on a synthetic room, and the real scan lining up with the map.

Run:  python -m unittest tests/test_coverage.py -v
"""
from pathlib import Path
import json
import os
import subprocess
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
from tracker3d import coverage as cov        # noqa: E402
from test_tracking import LENS               # noqa: E402

ROOM = 6.0


class _Plan:
    """A ROOM x ROOM m square room."""
    width_m = height_m = ROOM
    margin_m = 0.5
    px_per_m = 50

    def in_valid_area(self, x, y):
        return 0 <= x <= ROOM and 0 <= y <= ROOM


def box(x1, x2, y1, y2, z1, z2, step=0.05):
    g = np.mgrid[x1:x2 + 1e-9:step, y1:y2 + 1e-9:step, z1:z2 + 1e-9:step]
    return g.reshape(3, -1).T


def room_points(extra=()):
    walls = [box(-0.1, 0, 0, ROOM, 0, 3), box(ROOM, ROOM + 0.1, 0, ROOM, 0, 3),
             box(0, ROOM, -0.1, 0, 0, 3), box(0, ROOM, ROOM, ROOM + 0.1, 0, 3),
             box(0, ROOM, 0, ROOM, 0, 0.0)]
    return np.vstack(walls + list(extra))


def camera_at(C, target):
    C = np.asarray(C, float)
    d = np.asarray(target, float) - C
    yaw = np.degrees(np.arctan2(d[1], d[0]))
    tilt = np.degrees(np.arctan2(-d[2], np.hypot(d[0], d[1])))
    return cov.Camera("cam", cov.look_rotation(yaw, tilt), C, LENS)


class TestVisibility(unittest.TestCase):
    def test_in_front_seen_behind_not(self):
        cam = camera_at([0.3, 3.0, 2.5], [3.0, 3.0, 0.8])
        pts = np.array([[3.0, 3.0, 1.05], [-2.0, 3.0, 1.05], [3.0, 3.0, 4.5]])
        seen = cov.in_picture(LENS, cam.R, cam.C, pts, r_max=cam.r_max)
        self.assertEqual(seen.tolist(), [True, False, False])

    def test_wall_blocks_line_of_sight(self):
        C = np.array([0.3, 3.0, 2.5])
        target = np.array([[4.0, 3.0, 1.05]])
        open_room = cov.Occupancy(room_points())
        walled = cov.Occupancy(room_points([box(2.0, 2.1, 0, ROOM, 0, 3)]))
        self.assertTrue(open_room.clear_lines(C, target)[0])
        self.assertFalse(walled.clear_lines(C, target)[0])

    def test_monitor_hides_seated_shoulders(self):
        """A monitor top (1.0 to 1.4 m) 0.4 m in front of a seated person blocks a
        camera behind it but not one beside it."""
        monitor = box(2.95, 3.05, 2.4, 3.6, 1.0, 1.4)
        occ = cov.Occupancy(room_points([monitor]))
        person = np.array([[3.45, 3.0, 1.05]])
        self.assertFalse(occ.clear_lines(np.array([0.3, 3.0, 2.5]), person)[0])
        self.assertTrue(occ.clear_lines(np.array([3.45, 0.3, 2.5]), person)[0])


class TestClassify(unittest.TestCase):
    def test_crossing_angles(self):
        xy = np.array([[3.0, 3.0]])
        both = np.ones((2, 1), bool)
        side = [np.array([0.0, 3.0]), np.array([3.0, 0.0])]        # 90 degrees apart
        facing = [np.array([0.0, 3.0]), np.array([6.0, 3.1])]     # nearly opposite
        self.assertEqual(cov.classify(both, side, xy)[0][0], cov.GOOD)
        self.assertEqual(cov.classify(both, facing, xy)[0][0], cov.POOR)
        self.assertEqual(cov.classify(np.array([[True], [False]]), side, xy)[0][0], cov.SINGLE)
        self.assertEqual(cov.classify(np.zeros((2, 1), bool), side, xy)[0][0], cov.UNSEEN)


class TestFurnitureAndGrid(unittest.TestCase):
    def test_desk_left_out_chair_back_kept(self):
        desk = box(2.0, 3.2, 2.0, 2.6, 0.74, 0.76, step=0.02)
        chair_back = box(2.5, 2.9, 3.3, 3.32, 0.45, 1.0, step=0.02)
        pts = room_points([desk, chair_back])
        f = cov.furniture_mask(pts, _Plan())
        self.assertTrue(f(np.array([2.6]), np.array([2.3]))[0])
        self.assertFalse(f(np.array([2.7]), np.array([3.31]))[0])
        grid = cov.FloorGrid(_Plan(), 0.25, furniture=f)
        on_desk = (np.abs(grid.xy[:, 0] - 2.625) < 0.01) & (np.abs(grid.xy[:, 1] - 2.375) < 0.01)
        self.assertFalse(on_desk.any())
        self.assertGreater(len(grid.furniture_xy), 4)

    def test_focus_limits_scored_floor(self):
        full = cov.FloorGrid(_Plan(), 0.5)
        part = cov.FloorGrid(_Plan(), 0.5, focus=[(0, 0, 3, 3)])
        self.assertLess(len(part.xy), len(full.xy) / 3)
        self.assertTrue((part.xy <= 3).all())


class TestSearch(unittest.TestCase):
    def test_new_camera_adds_good_floor_and_overlaps(self):
        occ = cov.Occupancy(room_points())
        grid = cov.FloorGrid(_Plan(), 0.25)
        existing = [camera_at([0.25, 0.25, 2.5], [3.0, 3.0, 0.8])]
        planner = cov.Planner(existing, grid, occ)
        before = cov.summary(planner.status, planner.count, grid.cell_area)
        self.assertEqual(before["good_m2"], 0.0)                  # one camera: nothing measured
        spots = cov.mount_spots(_Plan(), occ, 2.5, avoid=[c.C for c in existing])
        self.assertGreater(len(spots), 20)
        near_wall = np.minimum.reduce([spots[:, 0], ROOM - spots[:, 0], spots[:, 1], ROOM - spots[:, 1]])
        self.assertTrue((near_wall < 0.45).all())
        self.assertTrue((np.hypot(*(spots - existing[0].C[:2]).T) >= 1.0).all())
        options = planner.search(LENS, spots, 2.5, top=3)
        self.assertEqual(len(options), 3)
        best = options[0]
        status, count, vis = planner.with_camera(best["camera"])
        after = cov.summary(status, count, grid.cell_area)
        self.assertGreater(after["good_m2"], 0.4 * before["floor_m2"])
        self.assertGreater(np.sum(vis & planner.seen[0]), 10)     # overlaps for calibration
        # facing the existing camera across the room gives poor angles, so the best
        # spot is well off the diagonal through the existing camera
        rel = best["C"][:2] - existing[0].C[:2]
        self.assertGreater(abs(np.sin(np.arctan2(rel[1], rel[0]) - np.pi / 4)), 0.3)
        scores = [o["score"] for o in options]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_fast_status_matches_full_classify(self):
        occ = cov.Occupancy(room_points())
        grid = cov.FloorGrid(_Plan(), 0.5)
        planner = cov.Planner([camera_at([0.25, 0.25, 2.5], [3, 3, 0.8]),
                               camera_at([5.75, 0.25, 2.5], [3, 3, 0.8])], grid, occ)
        new = camera_at([3.0, 5.75, 2.5], [3, 3, 0.8])
        status, _, vis = planner.with_camera(new)
        np.testing.assert_array_equal(planner._fast_status(new.C, vis), status)


    def test_both_postures_combine_seated_and_standing(self):
        occ = cov.Occupancy(room_points())
        grid = cov.FloorGrid(_Plan(), 0.5)
        cams = [camera_at([0.25, 0.25, 2.5], [3, 3, 0.8]), camera_at([5.75, 0.25, 2.5], [3, 3, 0.8])]
        both = cov.Planner(cams, grid, occ, "both")
        for p in ("seated", "standing"):
            one = cov.Planner(cams, grid, occ, p)
            np.testing.assert_array_equal(both.split(both.status)[p], one.status)
            self.assertEqual(both.summaries()[p], one.summaries()[p])
        worst = both.worst()
        st = both.split(both.status)
        np.testing.assert_array_equal(worst, np.minimum(st["seated"], st["standing"]))
        self.assertAlmostEqual(both.cell_area, grid.cell_area / 2)


class TestWeights(unittest.TestCase):
    def test_weights_take_largest_and_change_the_ranking(self):
        grid = cov.FloorGrid(_Plan(), 0.5, weights=[(0, 0, 3, 3, 2), (0, 0, 1, 1, 5)])
        at = lambda x, y: grid.weight[np.argmin(np.hypot(*(grid.xy - [x, y]).T))]
        self.assertEqual((at(0.75, 0.75), at(2.25, 2.25), at(5.25, 5.25)), (5.0, 2.0, 1.0))
        status = np.array([cov.GOOD, cov.SINGLE, cov.UNSEEN])
        self.assertAlmostEqual(cov.score(status, 1.0, 0.25, np.array([2.0, 4.0, 9.0])), 2 + 0.25 * 6)
        # A heavily weighted corner pulls the best new camera towards covering it.
        occ = cov.Occupancy(room_points())
        existing = [camera_at([0.25, 0.25, 2.5], [3.0, 3.0, 0.8])]
        spots = cov.mount_spots(_Plan(), occ, 2.5, spacing=0.75, avoid=[c.C for c in existing])
        corner = (4.5, 4.5, 6.0, 6.0)
        good_in_corner = []
        for weights in (None, [corner + (20,)]):
            grid = cov.FloorGrid(_Plan(), 0.5, weights=weights)
            planner = cov.Planner(existing, grid, occ)
            best = planner.search(LENS, spots, 2.5, top=1)[0]["camera"]
            status, _, _ = planner.with_camera(best)
            inside = (grid.xy[:, 0] >= 4.5) & (grid.xy[:, 1] >= 4.5)
            good_in_corner.append(np.sum(status[inside] == cov.GOOD))
        self.assertGreaterEqual(good_in_corner[1], good_in_corner[0])


class TestLayoutSearch(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.occ = cov.Occupancy(room_points())
        cls.grid = cov.FloorGrid(_Plan(), 0.5)
        cls.existing = camera_at([0.25, 0.25, 2.5], [3.0, 3.0, 0.8])
        cls.existing.name = "cam1"
        cls.spots = cov.mount_spots(_Plan(), cls.occ, 2.5, spacing=0.75)

    def search(self, **kw):
        empty = cov.Planner([], self.grid, self.occ)
        return cov.LayoutSearch(empty, LENS, self.spots, current=[self.existing], **kw)

    def test_planner_without_cameras(self):
        p = cov.Planner([], self.grid, self.occ, "both")
        self.assertEqual(p.seen.shape, (0, len(p.xy)))
        self.assertTrue((p.status == cov.UNSEEN).all())

    def test_keeping_all_matches_single_camera_search(self):
        search = self.search()
        cams, score = search.run(2, ["cam1"], restarts=3)
        self.assertEqual(len(cams), 2)
        kept = [c for c in cams if c["name"] == "cam1"][0]
        np.testing.assert_array_equal(kept["C"], self.existing.C)       # left exactly in place
        np.testing.assert_array_equal(kept["R"], self.existing.R)
        planner = cov.Planner([self.existing], self.grid, self.occ)
        single = planner.search(LENS, self.spots, 2.5, top=1)[0]["score"]
        self.assertGreater(score, 0.95 * single)

    def test_free_layout_meets_rules_and_beats_its_start(self):
        search = self.search()
        start = search.construct(3)
        cams, score = search.run(3, restarts=3, seed=1)
        self.assertEqual(len(cams), 3)
        self.assertGreaterEqual(score, search.score(start) - 1e-6)
        for a in range(3):
            for b in range(a + 1, 3):
                self.assertGreaterEqual(np.hypot(*(cams[a]["C"][:2] - cams[b]["C"][:2])), 1.0)
        status, count = search.evaluate(cams)
        good = np.mean(status == cov.GOOD)
        self.assertGreater(good, 0.5)
        shared = [[(c["vis"] & d["vis"]).sum() * search.p.cell_area for d in cams] for c in cams]
        for a in range(3):                                   # each overlaps another camera
            self.assertGreaterEqual(max(shared[a][b] for b in range(3) if b != a), 2.0)
        again, score2 = search.run(3, restarts=3, seed=1)
        self.assertAlmostEqual(score, score2)

    def test_impossible_overlap_rule_gives_no_layout(self):
        search = self.search(min_overlap_m2=1e6)
        cams, _ = search.run(2, restarts=1)
        self.assertLess(len(cams), 2)


LAS = ROOT / "data" / "smart_lab.las"


@unittest.skipUnless(LAS.exists(), "lab scan not present")
class TestRealScan(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from tracker3d.config import load_config
        from tracker3d.plan import load_plan, load_scan_points
        cwd = os.getcwd()
        os.chdir(ROOT)
        try:
            cls.cfg = load_config("config.yaml")
            cls.plan = load_plan(cls.cfg)
            cls.points = load_scan_points(cls.cfg)
        finally:
            os.chdir(cwd)

    def test_scan_lines_up_with_map(self):
        p = self.points
        self.assertLess(abs(np.percentile(p[:, 2], 5)), 0.05)     # floor at height 0
        bg = self.plan.background()
        drawn = np.max(np.abs(bg.astype(int) - 25), axis=2) > 8
        s = p[p[:, 2] < 2.5][::50]
        px = ((s[:, 0] + self.plan.margin_m) * self.plan.px_per_m).astype(int)
        py = (self.plan.canvas_h - (s[:, 1] + self.plan.margin_m) * self.plan.px_per_m).astype(int)
        ok = (px >= 0) & (px < self.plan.canvas_w) & (py >= 0) & (py < self.plan.canvas_h)
        self.assertGreater(ok.mean(), 0.99)
        self.assertGreater(drawn[py[ok], px[ok]].mean(), 0.95)

    def test_calibrated_cameras_sit_on_scanned_walls(self):
        sys.path.insert(0, str(ROOT / "tools"))
        from coverage_planner import current_cameras
        cwd = os.getcwd()
        os.chdir(ROOT)
        try:
            cams, _ = current_cameras(self.cfg)
        finally:
            os.chdir(cwd)
        occ = cov.Occupancy(self.points)
        ii, jj, kk = np.nonzero(occ.grid)
        solid = occ.origin + (np.column_stack([ii, jj, kk]) + 0.5) * occ.voxel
        for c in cams:
            self.assertLess(np.linalg.norm(solid - c.C, axis=1).min(), 0.3, c.name)

    def test_cli_writes_maps_and_report(self):
        with tempfile.TemporaryDirectory() as out:
            r = subprocess.run([sys.executable, "tools/coverage_planner.py", "--out", out,
                                "--top", "1", "--spot", "7.9,8.1"],
                               cwd=ROOT, capture_output=True, text=True, timeout=300)
            self.assertEqual(r.returncode, 0, r.stderr)
            for name in ("coverage_current.png", "coverage_candidates.png",
                         "coverage_option_1.png", "coverage_plan.json"):
                self.assertTrue(os.path.exists(os.path.join(out, name)), name)
            self.assertIn("Option 1", r.stdout)

    def test_layout_cli_writes_map_and_report(self):
        with tempfile.TemporaryDirectory() as out:
            r = subprocess.run([sys.executable, "tools/layout_planner.py", "--out", out,
                                "--restarts", "1", "--keep", "yi01", "--keep", "yi04",
                                "--keep", "yi05"],
                               cwd=ROOT, capture_output=True, text=True, timeout=300)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertTrue(os.path.exists(os.path.join(out, "layout_keep_yi01+yi04+yi05.png")))
            with open(os.path.join(out, "layout_plan.json")) as f:
                plan = json.load(f)
            cams = plan["layouts"][0]["cameras"]
            self.assertEqual(len(cams), 4)
            self.assertEqual(sum(c["kept"] for c in cams), 3)


if __name__ == "__main__":
    unittest.main()
