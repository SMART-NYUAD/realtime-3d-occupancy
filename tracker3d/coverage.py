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
    """Scores the current cameras and searches mounting spots for one more."""

    def __init__(self, cameras, grid, occ, posture="seated", min_angle_deg=25.0,
                 max_range_m=9.0, margin_px=30):
        self.cameras, self.grid, self.occ = cameras, grid, occ
        self.min_angle_deg, self.max_range_m, self.margin_px = min_angle_deg, max_range_m, margin_px
        self.sh, self.hd = grid.targets(posture)
        self.seen = np.array([self._visible(c) for c in cameras]).reshape(len(cameras), -1)
        self.status, self.count = classify(self.seen, [c.C for c in cameras], grid.xy, min_angle_deg)

    def _visible(self, cam, clear=None):
        if clear is None:
            clear = clear_from(self.occ, cam.C, self.sh, self.hd)
        return visible(cam, self.sh, self.hd, clear, self.max_range_m, self.margin_px)

    def with_camera(self, cam, vis=None):
        """(status, count, seen-by-new) after adding `cam`."""
        vis = self._visible(cam) if vis is None else vis
        seen = np.vstack([self.seen, vis[None]])
        status, count = classify(seen, [c.C for c in self.cameras] + [cam.C],
                                 self.grid.xy, self.min_angle_deg)
        return status, count, vis

    def _fast_status(self, C, vis):
        """classify() for the current cameras plus one, reusing the current pairs."""
        s_min = np.sin(np.radians(self.min_angle_deg))
        good = self.status == GOOD
        bnew = _bearings(C, self.grid.xy)
        for c, seen in zip(self.cameras, self.seen):
            good = good | (vis & seen & (np.abs(np.sin(bnew - _bearings(c.C, self.grid.xy))) >= s_min))
        count = self.count + vis
        status = np.where(count == 0, UNSEEN, np.where(count == 1, SINGLE, POOR))
        status[good] = GOOD
        return status

    def search(self, lens, spots, height=2.5, yaws=None, tilts=(20, 28, 36, 44),
               top=5, separation_m=1.5, mirrored=False, progress=None):
        """Best new-camera placements, best first: list of dicts with C, yaw, tilt,
        score and the camera. `spots` are (x, y) or (x, y, z) mounting points (see
        mount_spots). Without `yaws`, directions within 75 degrees of facing away
        from the nearest wall are tried. Each spot keeps its best aim; spots closer
        than separation_m to a better one are skipped; the top ones are then refined."""
        r_max = fov_radius(lens)
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
                for tilt in np.clip(r["tilt"] + np.arange(-8, 9, 2), 10, 60):
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
        s = score(status, self.grid.cell_area)
        if best is None or s > best["score"] + 1e-9:
            return {"C": C, "yaw": float(yaw), "tilt": float(tilt), "score": float(s), "clear": clear}
        return best
