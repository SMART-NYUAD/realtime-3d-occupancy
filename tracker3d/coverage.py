"""How well the cameras cover the lab floor, and where a new camera would help most.

For a grid of floor spots, a camera "sees" a person standing or sitting there when
their head and shoulders are both inside its picture (away from the edges), within
detection range, and its lines of sight to them are not blocked by anything in the
lab scan (walls, shelves, monitors). Each spot is then:

  unseen    no camera sees it
  single    one camera: position from an assumed height, posture guessed
  poor      two or more cameras, but their lines of sight cross at less than
            min_angle_deg (the tracker treats such crossings as unsure)
  good      two cameras whose lines of sight cross well: height and posture are
            measured and neighbours are kept apart (fusion.tri_min_angle_deg)

`search` tries a new camera at every mounting spot along the walls (and tall
furniture) at a given height, over a range of directions and tilts, and ranks them
by how much floor becomes "good". Everything is in the shared world frame.
"""
import cv2
import numpy as np

UNSEEN, SINGLE, POOR, GOOD = 0, 1, 2, 3
POSTURES = {"seated": (1.05, 1.30), "standing": (1.40, 1.62)}   # shoulder, head (m)


class Occupancy:
    """The lab scan as a voxel grid: a voxel is solid if it holds >= min_points."""

    def __init__(self, points, voxel=0.1, min_points=3, z_min=0.1):
        p = np.asarray(points, float)
        p = p[p[:, 2] >= z_min]
        self.voxel = float(voxel)
        self.origin = p.min(axis=0) - self.voxel
        idx = np.floor((p - self.origin) / self.voxel).astype(np.int64)
        self.shape = tuple(int(v) for v in idx.max(axis=0) + 2)
        counts = np.bincount(np.ravel_multi_index(idx.T, self.shape),
                             minlength=int(np.prod(self.shape)))
        self.grid = (counts >= min_points).reshape(self.shape)

    def solid(self, pts):
        """True for points (..., 3) inside a solid voxel."""
        idx = np.floor((pts - self.origin) / self.voxel).astype(np.int64)
        inside = np.all((idx >= 0) & (idx < self.shape), axis=-1)
        out = np.zeros(pts.shape[:-1], bool)
        ii = idx[inside]
        out[inside] = self.grid[ii[:, 0], ii[:, 1], ii[:, 2]]
        return out

    def clear_lines(self, C, targets, skip_start=0.3, skip_end=0.15):
        """True where the straight line from C to each target misses the scan. The
        first skip_start m (the wall the camera is mounted on) and the last skip_end m
        (the person's own chair) are not checked."""
        targets = np.asarray(targets, float)
        out = np.ones(len(targets), bool)
        if not len(targets):
            return out
        d = targets - C
        length = np.linalg.norm(d, axis=1)
        step = self.voxel / 2
        ts = np.arange(int(np.ceil(length.max() / step)) + 1) * step
        for s in range(0, len(targets), 256):
            sl = slice(s, s + 256)
            L = length[sl, None]
            check = (ts[None] >= skip_start) & (ts[None] <= L - skip_end)
            pts = C + (d[sl, None, :] / L[..., None]) * ts[None, :, None]
            out[sl] = ~(self.solid(pts) & check).any(axis=1)
        return out

    def solid_band(self, z_lo, z_hi):
        """(nx, ny) bool: any solid voxel between heights z_lo and z_hi."""
        k0 = max(0, int(np.floor((z_lo - self.origin[2]) / self.voxel)))
        k1 = min(self.shape[2] - 1, int(np.floor((z_hi - self.origin[2]) / self.voxel)))
        return self.grid[:, :, k0:k1 + 1].any(axis=2)


def furniture_mask(points, plan, z_band=(0.62, 0.88), res=0.05, min_size_m=0.35):
    """Function f(x, y) -> True where a desk, table or cabinet stands: surfaces at
    desk height that survive an opening of min_size_m (thin chair backs, walls and
    monitor stands are removed). People cannot stand or sit there."""
    p = np.asarray(points)
    p = p[(p[:, 2] >= z_band[0]) & (p[:, 2] <= z_band[1])]
    nx = int(np.ceil((plan.width_m + 1) / res))
    ny = int(np.ceil((plan.height_m + 1) / res))
    ix = np.clip(((p[:, 0] + 0.5) / res).astype(int), 0, nx - 1)
    iy = np.clip(((p[:, 1] + 0.5) / res).astype(int), 0, ny - 1)
    img = np.zeros((ny, nx), np.uint8)
    img[iy, ix] = 1
    k = max(3, int(round(min_size_m / res)) | 1)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))

    def f(x, y):
        jx = np.clip(((np.asarray(x) + 0.5) / res).astype(int), 0, nx - 1)
        jy = np.clip(((np.asarray(y) + 0.5) / res).astype(int), 0, ny - 1)
        return img[jy, jx].astype(bool)
    return f


class FloorGrid:
    """Floor spots to score: cell centres inside the room (at least edge_m from its
    edge) and, unless include_furniture, not on furniture. `focus` limits scoring to
    rectangles [(x1, y1, x2, y2), ...], e.g. the desk area."""

    def __init__(self, plan, cell=0.25, edge_m=0.2, furniture=None, focus=None):
        xs = np.arange(cell / 2, plan.width_m, cell)
        ys = np.arange(cell / 2, plan.height_m, cell)
        X, Y = np.meshgrid(xs, ys)
        X, Y = X.ravel(), Y.ravel()
        ok = np.ones(len(X), bool)
        for dx, dy in ((0, 0), (edge_m, 0), (-edge_m, 0), (0, edge_m), (0, -edge_m)):
            ok &= np.array([plan.in_valid_area(x + dx, y + dy) for x, y in zip(X, Y)])
        self.on_furniture = np.zeros(len(X), bool)
        if furniture is not None:
            self.on_furniture = furniture(X, Y) & ok
            ok &= ~self.on_furniture
        if focus:
            inside = np.zeros(len(X), bool)
            for x1, y1, x2, y2 in focus:
                inside |= ((X >= min(x1, x2)) & (X <= max(x1, x2)) &
                           (Y >= min(y1, y2)) & (Y <= max(y1, y2)))
            ok &= inside
        self.cell = float(cell)
        self.xy = np.column_stack([X[ok], Y[ok]])
        self.furniture_xy = np.column_stack([X[self.on_furniture], Y[self.on_furniture]])

    def targets(self, posture="seated"):
        """(shoulder points, head points), each (N, 3)."""
        sh, hd = POSTURES[posture]
        n = len(self.xy)
        return (np.column_stack([self.xy, np.full(n, sh)]),
                np.column_stack([self.xy, np.full(n, hd)]))

    @property
    def cell_area(self):
        return self.cell ** 2


def fov_radius(lens):
    """Largest normalized image radius the picture reaches (its corners)."""
    w, h = lens.image_size
    t = np.linspace(0, 1, 50)
    border = np.vstack([np.column_stack([t * (w - 1), np.zeros_like(t)]),
                        np.column_stack([t * (w - 1), np.full_like(t, h - 1)]),
                        np.column_stack([np.zeros_like(t), t * (h - 1)]),
                        np.column_stack([np.full_like(t, w - 1), t * (h - 1)])])
    rect = lens.undistort_points(border)
    n = np.linalg.inv(lens.K_rect) @ np.column_stack([rect, np.ones(len(rect))]).T
    return float(np.hypot(n[0] / n[2], n[1] / n[2]).max())


def in_picture(lens, R, C, pts, margin_px=30, r_max=None):
    """True where world points appear inside the raw picture, margin_px from its
    edges. r_max (fov_radius) rejects points far outside the lens's field of view,
    where the distortion polynomial folds back into the picture."""
    q = (R @ (np.asarray(pts) - C).T).T
    front = q[:, 2] > 0.2
    z = np.where(front, q[:, 2], 1.0)
    x, y = q[:, 0] / z, q[:, 1] / z
    ok = front
    if r_max is not None:
        ok &= np.hypot(x, y) <= 1.02 * r_max
    xd, yd = lens._distort_norm(x, y)
    u = lens.K[0, 0] * xd + lens.K[0, 1] * yd + lens.K[0, 2]
    v = lens.K[1, 1] * yd + lens.K[1, 2]
    w, h = lens.image_size
    return ok & (u >= margin_px) & (u <= w - 1 - margin_px) & (v >= margin_px) & (v <= h - 1 - margin_px)


def look_rotation(yaw_deg, tilt_deg, mirrored=False):
    """World-to-camera rotation for a camera facing map direction yaw (degrees from
    +x towards +y) tilted tilt degrees below horizontal (x right, y down, z forward)."""
    yaw, tilt = np.radians(yaw_deg), np.radians(tilt_deg)
    fwd = np.array([np.cos(tilt) * np.cos(yaw), np.cos(tilt) * np.sin(yaw), -np.sin(tilt)])
    right = np.cross(fwd, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    if mirrored:                       # map frame is a mirror image of the room
        right = -right
    return np.vstack([right, np.cross(fwd, right), fwd])


def aim_point(C, yaw_deg, tilt_deg):
    """Floor point at the centre of the picture, or None when aimed above the floor."""
    if tilt_deg <= 0:
        return None
    dist = C[2] / np.tan(np.radians(tilt_deg))
    yaw = np.radians(yaw_deg)
    return np.array([C[0] + dist * np.cos(yaw), C[1] + dist * np.sin(yaw)])


class Camera:
    def __init__(self, name, R, C, lens):
        self.name, self.R, self.C, self.lens = name, np.asarray(R), np.asarray(C, float), lens
        self.r_max = fov_radius(lens)


def visible(cam, sh, hd, clear, max_range_m=9.0, margin_px=30):
    """Which floor spots `cam` sees a person at (bool per spot). `clear` is the
    unblocked-line test for this camera position (shoulders and head)."""
    in_range = np.linalg.norm(hd - cam.C, axis=1) <= max_range_m
    return (clear & in_range
            & in_picture(cam.lens, cam.R, cam.C, sh, margin_px, cam.r_max)
            & in_picture(cam.lens, cam.R, cam.C, hd, margin_px, cam.r_max))


def clear_from(occ, C, sh, hd):
    return occ.clear_lines(C, sh) & occ.clear_lines(C, hd)


def _bearings(C, xy):
    return np.arctan2(C[1] - xy[:, 1], C[0] - xy[:, 0])


def classify(seen, cams_xy, xy, min_angle_deg=25.0):
    """Status per spot from seen (n_cams, n_spots) and the camera positions."""
    seen = np.asarray(seen, bool)
    count = seen.sum(axis=0)
    bear = [_bearings(c, xy) for c in cams_xy]
    s_min = np.sin(np.radians(min_angle_deg))
    good = np.zeros(len(xy), bool)
    for i in range(len(seen)):
        for j in range(i + 1, len(seen)):
            good |= seen[i] & seen[j] & (np.abs(np.sin(bear[i] - bear[j])) >= s_min)
    status = np.where(count == 0, UNSEEN, np.where(count == 1, SINGLE, POOR))
    status[good] = GOOD
    return status, count


def summary(status, count, cell_area):
    n = lambda m: round(float(np.sum(m)) * cell_area, 1)
    return {"floor_m2": n(np.ones_like(status, bool)), "good_m2": n(status == GOOD),
            "poor_angle_m2": n(status == POOR), "single_m2": n(status == SINGLE),
            "unseen_m2": n(status == UNSEEN), "three_plus_m2": n(count >= 3)}


def score(status, cell_area, seen_weight=0.25):
    """Floor that is well covered, plus a little for floor that is at least seen."""
    return cell_area * (np.sum(status == GOOD) + seen_weight * np.sum(status != UNSEEN))


def mount_spots(plan, occ, height, wall_gap=(0.15, 0.35), spacing=0.5, avoid=(), avoid_m=1.0):
    """Places a camera can be fixed at `height`: free space 0.15 to 0.35 m from a
    wall, pillar or tall cabinet (in the scan) inside the room, spaced `spacing` m
    apart, and at least avoid_m from the cameras in `avoid`."""
    blocked = occ.solid_band(height - 0.25, height + 0.25)          # (nx, ny)
    nx, ny = blocked.shape
    cx = occ.origin[0] + (np.arange(nx) + 0.5) * occ.voxel
    cy = occ.origin[1] + (np.arange(ny) + 0.5) * occ.voxel
    inside = np.array([[plan.in_valid_area(x, y) for y in cy] for x in cx])
    free = (inside & ~blocked).astype(np.uint8)
    dist = cv2.distanceTransform(free, cv2.DIST_L2, 5) * occ.voxel
    ring = (free > 0) & (dist >= wall_gap[0]) & (dist <= wall_gap[1])
    ii, jj = np.nonzero(ring)
    pts = np.column_stack([cx[ii], cy[jj]])
    if not len(pts):
        return pts
    centre = pts.mean(axis=0)
    pts = pts[np.argsort(np.arctan2(pts[:, 1] - centre[1], pts[:, 0] - centre[0]))]
    chosen = []
    for p in pts:
        if any(np.hypot(*(p - np.asarray(a)[:2])) < avoid_m for a in avoid):
            continue
        if all(np.hypot(*(p - q)) >= spacing for q in chosen):
            chosen.append(p)
    return np.array(chosen)


def wall_yaw(occ, xy, height):
    """Map direction (degrees) pointing away from the nearest solid voxel at mounting
    height, i.e. into the room."""
    band = occ.solid_band(height - 0.25, height + 0.25)
    ii, jj = np.nonzero(band)
    cx = occ.origin[0] + (ii + 0.5) * occ.voxel
    cy = occ.origin[1] + (jj + 0.5) * occ.voxel
    k = int(np.argmin((cx - xy[0]) ** 2 + (cy - xy[1]) ** 2))
    return float(np.degrees(np.arctan2(xy[1] - cy[k], xy[0] - cx[k])))


class Planner:
    """Scores the current cameras and searches mounting spots for one more.

    posture "both" scores every floor spot twice, for a seated and a standing person;
    arrays then hold the seated spots followed by the standing ones (see split), and
    areas are the average of the two."""

    def __init__(self, cameras, grid, occ, posture="seated", min_angle_deg=25.0,
                 max_range_m=9.0, margin_px=30):
        self.cameras, self.grid, self.occ = cameras, grid, occ
        self.min_angle_deg, self.max_range_m, self.margin_px = min_angle_deg, max_range_m, margin_px
        self.postures = list(POSTURES) if posture == "both" else [posture]
        targets = [grid.targets(p) for p in self.postures]
        self.sh = np.vstack([sh for sh, _ in targets])
        self.hd = np.vstack([hd for _, hd in targets])
        self.xy = np.tile(grid.xy, (len(self.postures), 1))
        self.cell_area = grid.cell_area / len(self.postures)
        self.seen = np.array([self._visible(c) for c in cameras], bool).reshape(len(cameras), len(self.xy))
        self.status, self.count = classify(self.seen, [c.C for c in cameras], self.xy, min_angle_deg)

    def split(self, values):
        """{posture: per-floor-spot values} from an array over all scored spots."""
        n = len(self.grid.xy)
        return {p: values[k * n:(k + 1) * n] for k, p in enumerate(self.postures)}

    def summaries(self, status=None, count=None):
        """{posture: summary} (current cameras by default)."""
        status = self.status if status is None else status
        count = self.count if count is None else count
        st, ct = self.split(status), self.split(count)
        return {p: summary(st[p], ct[p], self.grid.cell_area) for p in self.postures}

    def worst(self, status=None):
        """Per floor spot, the worst status over the postures (for one map)."""
        st = self.split(self.status if status is None else status)
        return np.min(np.vstack(list(st.values())), axis=0)

    def _visible(self, cam, clear=None):
        if clear is None:
            clear = clear_from(self.occ, cam.C, self.sh, self.hd)
        return visible(cam, self.sh, self.hd, clear, self.max_range_m, self.margin_px)

    def with_camera(self, cam, vis=None):
        """(status, count, seen-by-new) after adding `cam`."""
        vis = self._visible(cam) if vis is None else vis
        seen = np.vstack([self.seen, vis[None]])
        status, count = classify(seen, [c.C for c in self.cameras] + [cam.C],
                                 self.xy, self.min_angle_deg)
        return status, count, vis

    def _fast_status(self, C, vis):
        """classify() for the current cameras plus one, reusing the current pairs."""
        s_min = np.sin(np.radians(self.min_angle_deg))
        good = self.status == GOOD
        bnew = _bearings(C, self.xy)
        for c, seen in zip(self.cameras, self.seen):
            good = good | (vis & seen & (np.abs(np.sin(bnew - _bearings(c.C, self.xy))) >= s_min))
        count = self.count + vis
        status = np.where(count == 0, UNSEEN, np.where(count == 1, SINGLE, POOR))
        status[good] = GOOD
        return status

    def search(self, lens, spots, height=2.5, yaws=None, tilts=(20, 28, 36, 44),
               top=5, separation_m=1.5, mirrored=False, progress=None, min_tilt_deg=15.0):
        """Best new-camera placements, best first: list of dicts with C, yaw, tilt,
        score and the camera. `spots` are (x, y) or (x, y, z) mounting points (see
        mount_spots). Without `yaws`, directions within 75 degrees of facing away
        from the nearest wall are tried. Each spot keeps its best aim; spots closer
        than separation_m to a better one are skipped; the top ones are then refined.
        Tilts stay at or above min_tilt_deg: a nearly level camera sees far, but in a
        busy room people in front hide the people behind, which is not modelled."""
        r_max = fov_radius(lens)
        tilts = [t for t in tilts if t >= min_tilt_deg] or [min_tilt_deg]
        results = []
        for n, xy in enumerate(spots):
            C = np.array([xy[0], xy[1], height if len(xy) < 3 else xy[2]], float)
            clear = clear_from(self.occ, C, self.sh, self.hd)
            in_range = np.linalg.norm(self.hd - C, axis=1) <= self.max_range_m
            clear &= in_range
            out = wall_yaw(self.occ, C[:2], C[2]) if yaws is None else None
            yaw_list = (out + np.arange(-75, 76, 15)) if yaws is None else yaws
            best = None
            for yaw in yaw_list:
                for tilt in tilts:
                    best = self._try(lens, r_max, C, clear, yaw, tilt, mirrored, best)
            if best is not None:
                best["clear"] = clear
                results.append(best)
            if progress:
                progress(n + 1, len(spots))
        results.sort(key=lambda r: -r["score"])
        picked = []
        for r in results:
            if all(np.hypot(*(r["C"][:2] - p["C"][:2])) >= separation_m for p in picked):
                picked.append(r)
            if len(picked) == top:
                break
        refined = []
        for r in picked:                                  # finer aim around the best one
            best = r
            for yaw in r["yaw"] + np.arange(-12, 13, 3):
                for tilt in np.clip(r["tilt"] + np.arange(-8, 9, 2), min_tilt_deg, 60):
                    best = self._try(lens, r_max, r["C"], r["clear"], yaw, tilt, mirrored, best)
            best = {k: v for k, v in best.items() if k != "clear"}
            best["yaw"] = float((best["yaw"] + 180) % 360 - 180)
            best["camera"] = Camera("new", look_rotation(best["yaw"], best["tilt"], mirrored),
                                    best["C"], lens)
            refined.append(best)
        refined.sort(key=lambda r: -r["score"])
        return refined

    def _try(self, lens, r_max, C, clear, yaw, tilt, mirrored, best):
        R = look_rotation(yaw, tilt, mirrored)
        vis = (clear & in_picture(lens, R, C, self.sh, self.margin_px, r_max)
               & in_picture(lens, R, C, self.hd, self.margin_px, r_max))
        if not vis.any():
            return best
        status = self._fast_status(C, vis)
        s = score(status, self.cell_area)
        if best is None or s > best["score"] + 1e-9:
            return {"C": C, "yaw": float(yaw), "tilt": float(tilt), "score": float(s), "clear": clear}
        return best


class LayoutSearch:
    """Chooses a whole set of cameras at once (all of them, or some while others stay).

    Every candidate camera (mounting spot x direction x tilt, plus each current
    camera exactly where it is) is scored once: which floor spots it sees. A layout's
    score is then cheap to compute from those masks, so many layouts can be tried:
    a greedy start (best pair, then the best addition, and so on), improved by
    swapping one camera at a time for any other candidate until no swap helps,
    repeated from randomised greedy starts, and finally a finer aim for each moved
    camera. Rules every layout must meet:
      - cameras at least min_separation_m apart (seen from above)
      - every camera shares at least min_overlap_m2 of seen floor with another, and
        all cameras link up through such overlaps (the calibration walk needs this)
    The score is the same as Planner.search: well-covered floor plus a little for
    floor that is at least seen. Moved cameras keep a tilt of at least min_tilt_deg
    (see Planner.search).
    """

    def __init__(self, planner, lens, spots, height=2.5, tilts=(20, 28, 36, 44), yaw_span=75,
                 yaw_step=15, mirrored=False, current=(), min_separation_m=1.0,
                 min_overlap_m2=2.0, seen_weight=0.25, min_tilt_deg=15.0):
        p = self.p = planner
        self.min_tilt_deg = min_tilt_deg
        tilts = [t for t in tilts if t >= min_tilt_deg] or [min_tilt_deg]
        self.lens, self.mirrored, self.seen_weight = lens, mirrored, seen_weight
        self.min_separation_m, self.min_overlap_m2 = min_separation_m, min_overlap_m2
        self.r_max = fov_radius(lens)
        positions = [np.array([xy[0], xy[1], height if len(xy) < 3 else xy[2]], float) for xy in spots]
        positions += [np.asarray(c.C, float) for c in current]
        self.P = np.array(positions).reshape(-1, 3)
        self.clear = [clear_from(p.occ, C, p.sh, p.hd) & (np.linalg.norm(p.hd - C, axis=1) <= p.max_range_m)
                      for C in self.P]
        cands, vis = [], []
        for i in range(len(spots)):
            C = self.P[i]
            out = wall_yaw(p.occ, C[:2], C[2])
            for yaw in out + np.arange(-yaw_span, yaw_span + 1, yaw_step):
                for tilt in tilts:
                    R = look_rotation(yaw, tilt, mirrored)
                    v = self._vis(R, C, self.clear[i], lens, self.r_max)
                    if v.any():
                        cands.append({"pos": i, "yaw": float(yaw), "tilt": float(tilt), "R": R,
                                      "lens": lens, "name": None})
                        vis.append(v)
        for k, cam in enumerate(current):
            i = len(spots) + k
            fwd = cam.R[2]
            cands.append({"pos": i, "yaw": float(np.degrees(np.arctan2(fwd[1], fwd[0]))),
                          "tilt": float(np.degrees(np.arcsin(-fwd[2]))), "R": cam.R,
                          "lens": cam.lens, "name": cam.name})
            vis.append(self._vis(cam.R, cam.C, self.clear[i], cam.lens, cam.r_max))
        self.cands = cands
        self.vis = np.array(vis, bool).reshape(len(cands), len(p.xy))
        self.pos = np.array([c["pos"] for c in cands], int)
        self.current_index = {c["name"]: n for n, c in enumerate(cands) if c["name"]}
        s_min = np.sin(np.radians(p.min_angle_deg))
        bear = np.array([_bearings(C, p.xy) for C in self.P]).reshape(len(self.P), len(p.xy))
        self.angle_ok = np.zeros((len(self.P), len(self.P), len(p.xy)), bool)
        for i in range(len(self.P)):
            self.angle_ok[i] = np.abs(np.sin(bear[i] - bear)) >= s_min
        self.dist = np.linalg.norm(self.P[:, None, :2] - self.P[None, :, :2], axis=2)
        v32 = self.vis.astype(np.float32)
        self.overlap = (v32 @ v32.T) * p.cell_area          # m2 of floor both cameras see

    def _vis(self, R, C, clear, lens, r_max):
        p = self.p
        return (clear & in_picture(lens, R, C, p.sh, p.margin_px, r_max)
                & in_picture(lens, R, C, p.hd, p.margin_px, r_max))

    # ---- scoring ----
    def _good_count(self, S):
        good = np.zeros(self.vis.shape[1], bool)
        count = np.zeros(self.vis.shape[1], int)
        for a, i in enumerate(S):
            count += self.vis[i]
            for j in S[a + 1:]:
                good |= self.vis[i] & self.vis[j] & self.angle_ok[self.pos[i], self.pos[j]]
        return good, count

    def _components(self, S):
        comps, left = [], list(S)
        while left:
            comp, todo = [], [left.pop()]
            while todo:
                i = todo.pop()
                comp.append(i)
                linked = [j for j in left if self.overlap[i, j] >= self.min_overlap_m2]
                left = [j for j in left if j not in linked]
                todo += linked
            comps.append(comp)
        return comps

    def valid(self, S):
        if any(self.dist[self.pos[i], self.pos[j]] < self.min_separation_m
               for a, i in enumerate(S) for j in S[a + 1:]):
            return False
        return len(S) < 2 or len(self._components(S)) == 1

    def score(self, S):
        if not self.valid(S):
            return -np.inf
        good, count = self._good_count(S)
        return float(self.p.cell_area * (good.sum() + self.seen_weight * (count > 0).sum()))

    def add_scores(self, S):
        """Score of S plus each candidate, for every candidate (-inf where a rule fails)."""
        good, count = self._good_count(S)
        V = self.vis
        g = np.repeat(good[None], len(V), axis=0)
        for k in S:
            g |= V & self.vis[k] & self.angle_ok[self.pos, self.pos[k]]
        sc = self.p.cell_area * (g.sum(axis=1) + self.seen_weight * (V | (count > 0)).sum(axis=1))
        ok = np.ones(len(V), bool)
        for k in S:
            ok &= self.dist[self.pos, self.pos[k]] >= self.min_separation_m
        for comp in self._components(S):
            ok &= (self.overlap[:, comp] >= self.min_overlap_m2).any(axis=1)
        sc[~ok] = -np.inf
        return sc

    # ---- search ----
    def construct(self, n, fixed=(), rng=None, top=8):
        """Greedy layout of n cameras containing `fixed`; with rng, each step picks at
        random among the `top` best additions."""
        S = list(fixed)
        if not S:
            if rng is None:                                   # best pair among strong singles
                singles = np.argsort(-self.vis.sum(axis=1))[:200]
                best = (-np.inf, None, None)
                for i in singles:
                    sc = self.add_scores([i])
                    j = int(np.argmax(sc))
                    if sc[j] > best[0]:
                        best = (sc[j], int(i), j)
                if best[1] is None:
                    return S
                S = [best[1], best[2]]
            else:
                S = [int(rng.choice(np.argsort(-self.vis.sum(axis=1))[:50]))]
        while len(S) < n:
            sc = self.add_scores(S)
            order = np.argsort(-sc)
            order = order[np.isfinite(sc[order])]
            if not len(order):
                break
            S.append(int(order[0] if rng is None else rng.choice(order[:top])))
        return S

    def improve(self, S, fixed=()):
        """Swap one camera at a time for the candidate that helps most, until no swap helps."""
        S, cur = list(S), self.score(S)
        while True:
            best = (cur + 1e-9, None, None)
            for slot, member in enumerate(S):
                if member in fixed:
                    continue
                others = S[:slot] + S[slot + 1:]
                sc = self.add_scores(others)
                j = int(np.argmax(sc))
                if sc[j] > best[0]:
                    best = (sc[j], slot, j)
            if best[1] is None:
                return S, cur
            S[best[1]] = best[2]
            cur = float(best[0])

    def run(self, n, fixed_names=(), restarts=20, seed=0, progress=None):
        """Best layout of n cameras keeping the named current cameras where they are.
        Returns (cameras, score): cameras are dicts with C, yaw, tilt, R, lens, name
        (a current camera's name when it stays where it is) and vis."""
        fixed = [self.current_index[name] for name in fixed_names]
        rng = np.random.default_rng(seed)
        best_S, best_sc = self.improve(self.construct(n, fixed), fixed)
        for r in range(restarts):
            S, sc = self.improve(self.construct(n, fixed, rng), fixed)
            if sc > best_sc + 1e-9 and len(S) == n:
                best_S, best_sc = S, sc
            if progress:
                progress(r + 1, restarts)
        cams = [{**self.cands[i], "C": self.P[self.cands[i]["pos"]], "vis": self.vis[i],
                 "fixed": i in fixed} for i in best_S]
        return self.refine(cams)

    def _score_cams(self, cams):
        good = np.zeros(self.vis.shape[1], bool)
        count = np.zeros(self.vis.shape[1], int)
        s_min = np.sin(np.radians(self.p.min_angle_deg))
        bear = [_bearings(c["C"], self.p.xy) for c in cams]
        for a, c in enumerate(cams):
            count += c["vis"]
            for b in range(a + 1, len(cams)):
                good |= c["vis"] & cams[b]["vis"] & (np.abs(np.sin(bear[a] - bear[b])) >= s_min)
        links = [[(c["vis"] & d["vis"]).sum() * self.p.cell_area >= self.min_overlap_m2
                  for d in cams] for c in cams]
        seen, todo = {0}, [0]
        while todo:
            a = todo.pop()
            for b in range(len(cams)):
                if links[a][b] and b not in seen:
                    seen.add(b)
                    todo.append(b)
        if len(seen) < len(cams):
            return -np.inf
        return float(self.p.cell_area * (good.sum() + self.seen_weight * (count > 0).sum()))

    def refine(self, cams, passes=2):
        """Finer direction and tilt for each camera that moves (current cameras that
        stay, and kept ones, are left exactly as they are)."""
        cur = self._score_cams(cams)
        for _ in range(passes):
            for c in cams:
                if c["fixed"] or c["name"]:
                    continue
                i = int(np.flatnonzero((self.P == c["C"]).all(axis=1))[0])
                base = dict(c)
                for yaw in base["yaw"] + np.arange(-12, 13, 3):
                    for tilt in np.clip(base["tilt"] + np.arange(-8, 9, 2), self.min_tilt_deg, 60):
                        R = look_rotation(yaw, tilt, self.mirrored)
                        trial = {**c, "yaw": float(yaw), "tilt": float(tilt), "R": R,
                                 "vis": self._vis(R, c["C"], self.clear[i], self.lens, self.r_max)}
                        others = [d if d is not c else trial for d in cams]
                        sc = self._score_cams(others)
                        if sc > cur + 1e-9:
                            cur = sc
                            c.update(trial)
        for c in cams:
            c["yaw"] = float((c["yaw"] + 180) % 360 - 180)
            c["camera"] = Camera(c["name"] or "new", c["R"], c["C"], c["lens"])
        return cams, cur

    def evaluate(self, cams):
        """(status, count) over the planner's floor spots for a list of layout cameras."""
        seen = np.array([c["vis"] for c in cams], bool).reshape(len(cams), len(self.p.xy))
        return classify(seen, [c["C"] for c in cams], self.p.xy, self.p.min_angle_deg)
