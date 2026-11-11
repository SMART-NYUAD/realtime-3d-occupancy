"""Coverage planner tests: visibility, blocking, crossing angles, furniture, the
mounting-spot search on a synthetic room, and the real scan lining up with the map.

Run:  python -m unittest tests/test_coverage.py -v
"""
from pathlib import Path
import base64
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

    def test_point_set_rates_points_like_the_grid(self):
        occ = cov.Occupancy(room_points())
        grid = cov.FloorGrid(_Plan(), 0.5)
        cams = [camera_at([0.25, 0.25, 2.5], [3, 3, 0.8]), camera_at([5.75, 0.25, 2.5], [3, 3, 0.8])]
        on_grid = cov.Planner(cams, grid, occ, "both")
        points = cov.Planner(cams, cov.PointSet(grid.xy[::-1]), occ, "both")
        np.testing.assert_array_equal(points.status, on_grid.status.reshape(2, -1)[:, ::-1].ravel())


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


class TestMountLines(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # A tall cabinet against the wall at x 2-3, y 0-0.6, topping out at 2.0 m.
        cls.occ = cov.Occupancy(room_points([box(2, 3, 0, 0.6, 0, 2.0)]))

    def line(self, start, end, height=2.6):
        return {"start": start, "end": end, "height_m": height}

    def test_spots_lie_exactly_on_the_line_and_face_the_room(self):
        spots, dropped = cov.spots_on_lines(_Plan(), self.occ, [self.line([0, 1], [0, 3])])
        self.assertEqual(dropped, [])
        self.assertEqual([list(s["xy"]) for s in spots],        # every 0.5 m, both ends
                         [[0.0, 1.0], [0.0, 1.5], [0.0, 2.0], [0.0, 2.5], [0.0, 3.0]])
        for s in spots:
            self.assertEqual(s["height"], 2.6)
            self.assertAlmostEqual(s["yaw"], 0.0)               # into the room (+x)
        # Drawn the other way, on the opposite wall, it still faces into the room.
        east, _ = cov.spots_on_lines(_Plan(), self.occ, [self.line([6, 3], [6, 1])])
        self.assertTrue(all(s["xy"][0] == 6.0 for s in east))
        self.assertAlmostEqual(abs(east[0]["yaw"]), 180.0)

    def test_each_line_keeps_its_own_height(self):
        lines = [self.line([0, 1], [0, 2], 2.2), self.line([6, 4], [6, 5], 3.0)]
        spots, _ = cov.spots_on_lines(_Plan(), self.occ, lines)
        self.assertEqual({s["line"]: s["height"] for s in spots}, {0: 2.2, 1: 3.0})

    def test_camera_on_a_cabinet_stays_on_its_line(self):
        spots, dropped = cov.spots_on_lines(_Plan(), self.occ, [self.line([2.1, 0.5], [2.9, 0.5], 2.1)])
        self.assertEqual(dropped, [])
        self.assertTrue(all(s["xy"][1] == 0.5 for s in spots))
        self.assertAlmostEqual(spots[0]["yaw"], 90.0)           # into the room (+y)

    def test_line_inside_furniture_is_dropped(self):
        occ = cov.Occupancy(room_points([box(3, 6, 2, 3.5, 0, 2.0)]))   # a deep free-standing block
        spots, dropped = cov.spots_on_lines(_Plan(), occ, [self.line([4.2, 2.75], [4.8, 2.75], 1.0)])
        self.assertEqual(spots, [])
        self.assertEqual({d["line"] for d in dropped}, {0})

    def test_spot_far_from_free_space_is_dropped(self):
        spots, dropped = cov.spots_on_lines(_Plan(), self.occ, [self.line([-2, 3], [-2, 3])])
        self.assertEqual(spots, [])
        self.assertEqual(dropped[0]["line"], 0)

    def test_lines_meeting_at_a_corner_share_a_spot(self):
        lines = [self.line([0, 0], [0, 1]), self.line([0, 0], [1, 0])]
        spots, _ = cov.spots_on_lines(_Plan(), self.occ, lines)
        xy = [s["xy"] for s in spots]
        gaps = [np.hypot(*(a - b)) for i, a in enumerate(xy) for b in xy[i + 1:]]
        self.assertGreaterEqual(min(gaps), 0.1)

    def test_bad_lines_are_refused(self):
        for bad in (None, [{"start": [0, 0], "end": [1, 1]}],
                    [self.line([0, 0, 0], [1, 1])], [self.line([0, 0], [1, 1], 9)],
                    [self.line([0, float("nan")], [1, 1])]):
            with self.assertRaises(ValueError):
                cov.check_mount_lines(bad)

    def test_search_only_uses_spots_on_the_lines(self):
        lines = [self.line([0, 1], [0, 5], 2.4), self.line([6, 1], [6, 5], 2.8)]
        spots, _ = cov.spots_on_lines(_Plan(), self.occ, lines)
        empty = cov.Planner([], cov.FloorGrid(_Plan(), 0.5), self.occ)
        search = cov.LayoutSearch(empty, LENS, [[*s["xy"], s["height"]] for s in spots],
                                  yaws=[s["yaw"] for s in spots])
        cams, _ = search.run(2, restarts=1)
        marked = {(round(float(s["xy"][0]), 3), round(float(s["xy"][1]), 3), s["height"]) for s in spots}
        for c in cams:
            self.assertIn((round(float(c["C"][0]), 3), round(float(c["C"][1]), 3), float(c["C"][2])), marked)

    def test_side_view_shows_the_scan_along_a_line(self):
        sys.path.insert(0, str(ROOT / "tools"))
        import mount_editor
        pts = room_points([box(2, 3, 0, 0.6, 0, 2.0)])          # cabinet against the south wall
        img, m = mount_editor.side_view(pts, None, (0, 0), (6, 0), depth=0.4)
        self.assertEqual(img.shape[:2], (m["h"], m["w"]))
        self.assertFalse(m["mirrored"])

        def pixel(im, s, z):
            return im[int((m["z1"] - z) * m["px_per_m"]), int((s - m["s0"]) * m["px_per_m"])]

        self.assertGreater(int(pixel(img, 1.0, 2.5).max()), 40)   # the wall, drawn
        self.assertEqual(int(pixel(img, 1.0, m["z1"] - 0.02).max()), 40)   # above it: empty
        # Only scan near the line counts: a box 1 m into the room shows only when the
        # depth reaches it.
        far = box(2, 3, 1.0, 1.2, 0, 1.0)
        none, _ = mount_editor.side_view(far, None, (0, 0), (6, 0), depth=0.4)
        some, _ = mount_editor.side_view(far, None, (0, 0), (6, 0), depth=1.2)
        self.assertEqual(int(none.max()), 40)
        self.assertGreater(int(some.max()), 40)
        # Seen from the room, which is on the line's left here, the view is mirrored.
        self.assertEqual(mount_editor.room_side(_Plan(), (0, 0), (6, 0)), 1)
        self.assertEqual(mount_editor.room_side(_Plan(), (6, 0), (0, 0)), -1)
        flipped, fm = mount_editor.side_view(pts, None, (0, 0), (6, 0), depth=0.4, viewer_side=1)
        self.assertTrue(fm["mirrored"])
        self.assertTrue((flipped == img[:, ::-1]).all())

    def test_editor_saves_lines_and_returns_spots(self):
        from http.server import ThreadingHTTPServer
        from threading import Thread
        from urllib.error import HTTPError
        from urllib.request import Request, urlopen
        sys.path.insert(0, str(ROOT / "tools"))
        import mount_editor

        class Plan(_Plan):
            def background(self):
                return np.zeros((350, 350, 3), np.uint8)

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "lines.json")
            server = ThreadingHTTPServer(("127.0.0.1", 0),
                                         mount_editor.make_handler(Plan(), self.occ, path,
                                                                   room_points(), None))
            Thread(target=server.serve_forever, daemon=True).start()
            base = f"http://127.0.0.1:{server.server_address[1]}"
            try:
                def post(body, kind="application/json"):
                    return urlopen(Request(base + "/lines", json.dumps(body).encode(),
                                           {"Content-Type": kind}), timeout=10)

                self.assertEqual(json.load(urlopen(base + "/state", timeout=10))["lines"], [])
                self.assertIn(b"<svg", urlopen(base + "/", timeout=10).read())
                self.assertEqual(urlopen(base + "/map.png", timeout=10).read(4), b"\x89PNG")
                lines = [self.line([0, 1], [0, 2])]
                result = json.load(post({"lines": lines}))
                self.assertEqual(len(result["spots"]), 3)
                self.assertEqual(cov.load_mount_lines(path), lines)
                self.assertEqual(json.load(urlopen(base + "/state", timeout=10))["lines"], lines)
                for body, kind, code in (({"lines": [self.line([0, 1], [0, 2], 99)]},
                                          "application/json", 400),
                                         ({"lines": []}, "text/plain", 404)):
                    with self.assertRaises(HTTPError) as err:
                        post(body, kind)
                    self.assertEqual(err.exception.code, code)
                self.assertEqual(cov.load_mount_lines(path), lines)      # refused ones not saved
                state = json.load(urlopen(base + "/state", timeout=10))
                self.assertTrue(state["side_view"])
                side = json.load(urlopen(base + "/side?x1=0&y1=0&x2=6&y2=0&depth=0.4", timeout=10))
                self.assertTrue(side["mirrored"])
                self.assertEqual(base64.b64decode(side["png"])[:4], b"\x89PNG")
                for bad in ("x1=0&y1=0&x2=6&y2=0&depth=0.3", "x1=0&y1=0&x2=99&y2=0&depth=0.4",
                            "x1=nan&y1=0&x2=6&y2=0&depth=0.4", "x1=0&y1=0&x2=6"):
                    with self.assertRaises(HTTPError) as err:
                        urlopen(base + "/side?" + bad, timeout=10)
                    self.assertEqual(err.exception.code, 400, bad)
            finally:
                server.shutdown()
                server.server_close()


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
        for c in cams:      # a gross calibration error would be metres off; brackets and
            # gaps in the scan high on the walls leave a mounted camera up to ~0.35 m away
            self.assertLess(np.linalg.norm(solid - c.C, axis=1).min(), 0.4, c.name)

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
            # --min-overlap 0 is temporary: yi04's stopgap calibration (November 2026) sits
            # inside the scanned pillar, so the planner sees it as blind. Drop this after
            # the walk that replaces it.
            r = subprocess.run([sys.executable, "tools/layout_planner.py", "--out", out,
                                "--restarts", "1", "--keep", "yi01", "--keep", "yi04",
                                "--keep", "yi05", "--min-overlap", "0"],
                               cwd=ROOT, capture_output=True, text=True, timeout=300)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertTrue(os.path.exists(os.path.join(out, "layout_keep_yi01+yi04+yi05.png")))
            with open(os.path.join(out, "layout_plan.json")) as f:
                plan = json.load(f)
            cams = plan["layouts"][0]["cameras"]
            self.assertEqual(len(cams), 4)
            self.assertEqual(sum(c["kept"] for c in cams), 3)

    def test_test_points_cli_rates_points_and_draws_maps(self):
        layout = ROOT / "docs/coverage/camera_count/2_cameras/layout_plan.json"
        with tempfile.TemporaryDirectory() as out:
            pts = os.path.join(out, "points.json")
            with open(pts, "w") as f:
                json.dump({"points": [{"id": "P01", "x": 2.5, "y": 2.75, "kind": "floor"},
                                      {"id": "P02", "x": 7.75, "y": 1.0, "kind": "floor"},
                                      {"id": "P03", "x": 3.25, "y": 1.25, "kind": "seat"}],
                           "anchors": ["P01"], "route": ["P01", "P03"]}, f)
            r = subprocess.run([sys.executable, "tools/test_points.py", "--points", pts,
                                "--layout", str(layout), "--out", out],
                               cwd=ROOT, capture_output=True, text=True, timeout=300)
            self.assertEqual(r.returncode, 0, r.stderr)
            for name in ("test_points.jpg", "test_points_2_cameras.jpg"):
                self.assertTrue(os.path.exists(os.path.join(out, name)), name)
            with open(os.path.join(out, "test_points.json")) as f:
                got = json.load(f)
            middle, corner = (got["points"][k]["predicted"]["2_cameras"] for k in (0, 1))
            for p in ("seated", "standing"):        # mid-room is seen by both; the far corner by none
                self.assertEqual(middle[p]["rating"], "good")
                self.assertEqual(len(middle[p]["cameras"]), 2)
                self.assertGreaterEqual(middle[p]["widest_crossing_deg"], 25)
                self.assertEqual(corner[p]["rating"], "unseen")
                self.assertEqual(corner[p]["cameras"], [])
            self.assertAlmostEqual(got["route_length_m"], 1.68, places=2)

    def test_layout_cli_with_mount_lines(self):
        with tempfile.TemporaryDirectory() as out:
            lines = os.path.join(out, "lines.json")
            with open(lines, "w") as f:
                json.dump({"lines": [{"start": [7.8, 7], "end": [7.8, 9], "height_m": 2.3},
                                     {"start": [3, 10.2], "end": [6, 10.2], "height_m": 2.9}]}, f)
            r = subprocess.run([sys.executable, "tools/layout_planner.py", "--out", out,
                                "--restarts", "1", "--cameras", "2", "--mount-lines", lines,
                                "--min-overlap", "0"],
                               cwd=ROOT, capture_output=True, text=True, timeout=300)
            self.assertEqual(r.returncode, 0, r.stderr)
            with open(os.path.join(out, "layout_plan.json")) as f:
                plan = json.load(f)
            cams = plan["layouts"][0]["cameras"]
            self.assertEqual(len(cams), 2)
            self.assertEqual(sorted(c["position_m"][2] for c in cams), [2.3, 2.9])
            self.assertFalse(any(c["current_camera"] for c in cams))
            self.assertEqual(len(plan["mount_lines"]), 2)
            self.assertEqual({c["mount_line"] for c in cams}, {1, 2})

    def test_layout_cli_runs_keeps_the_best(self):
        with tempfile.TemporaryDirectory() as out:
            lines = os.path.join(out, "lines.json")
            with open(lines, "w") as f:
                json.dump({"lines": [{"start": [7.8, 5], "end": [7.8, 9], "height_m": 2.4},
                                     {"start": [2, 10.2], "end": [6, 10.2], "height_m": 2.6}]}, f)
            r = subprocess.run([sys.executable, "tools/layout_planner.py", "--out", out,
                                "--restarts", "1", "--cameras", "2", "--runs", "2", "--seed", "3",
                                "--mount-lines", lines, "--min-overlap", "0",
                                "--weight", "2,2,8,10,2"],
                               cwd=ROOT, capture_output=True, text=True, timeout=300)
            self.assertEqual(r.returncode, 0, r.stderr)
            with open(os.path.join(out, "layout_plan.json")) as f:
                plan = json.load(f)
            self.assertEqual([run["seed"] for run in plan["runs"]], [3, 4])
            best = max(plan["runs"], key=lambda run: run["score"])
            self.assertEqual(plan["best_seed"], best["seed"])
            self.assertEqual(plan["layouts"][0]["score"], best["score"])
            self.assertEqual(best["largest_move_from_best_m"], 0.0)
            self.assertEqual(plan["agreement"]["runs"], 2)
            self.assertEqual(set(plan["layouts"][0]["zones"]), {"whole room", "2,2,8,10"})


if __name__ == "__main__":
    unittest.main()
