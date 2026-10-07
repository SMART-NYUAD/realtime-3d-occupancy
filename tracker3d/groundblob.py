"""Ground-plane "blob" intersection: turn each detection into a floor *region*
of foot-point uncertainty (a trapezoid), then intersect those regions across
cameras to collapse the uncertainty onto the true position.

Why
---
`geometry.foot_point` projects a single pixel (bbox bottom-center) through the
ground homography. That is exact only when the feet are visible and the box is
tight. When the feet are occluded (behind a desk) or clipped (below the frame),
the box bottom is *above* the real feet, so the single point lands too far away
and two cameras disagree -> one person becomes two dots.

A bounding box actually tells us more than one point:
  - its *columns* [x1, x2] bound the person's left/right (azimuth) extent -> a
    reliable horizontal slice on the floor;
  - its bottom *row* y2 is an estimate of the foot depth, but an uncertain one.

So we project a trapezoid: the columns [x1, x2] swept over a depth *band* around
the foot row. The naive foot point sits in the middle of that band, so several
cameras' trapezoids intersect symmetrically around the true foot point.

  a camera that sees the feet  -> tight band  -> short trapezoid at the feet
  a camera that can't (clipped) -> wide band  -> long trapezoid; the person is
                                                  "somewhere along that slice"

The intersection of the trapezoids is where every camera agrees the feet can be;
its centroid is the fused position. If the trapezoids don't overlap (bad
calibration), the caller keeps its point-based estimate.

Everything here is pure numpy + the ground homography from `geometry`; no Shapely.
"""
import numpy as np

_EPS = 1e-9


def _project(H, u, v, ref_sign=0.0):
    """Map image pixel (u, v) to floor (x, y) meters, or None if it lies on/beyond
    the horizon.

    The homogeneous scale w[2] keeps a constant sign across the whole floor and
    flips past the horizon -- but that constant sign is whatever the calibration's
    overall sign of H happened to be (it can be negative). So "beyond the horizon"
    is "w[2] has the opposite sign to the foot point", not "w[2] <= 0". Pass the
    foot's sign as `ref_sign`; with the default 0.0 any non-degenerate point maps
    (used to read the foot's own sign)."""
    w = H @ np.array([u, v, 1.0])
    if abs(w[2]) <= _EPS:
        return None
    if ref_sign != 0.0 and np.sign(w[2]) != ref_sign:
        return None
    return np.array([w[0] / w[2], w[1] / w[2]])


def _floor_sign(H, u, v):
    """Sign of the homogeneous scale at a known-floor pixel (its horizon side)."""
    w2 = float((H @ np.array([u, v, 1.0]))[2])
    return np.sign(w2) if abs(w2) > _EPS else 0.0


def _project_row(H, x1, x2, v, ref_sign):
    """Project the two ends (x1, v) and (x2, v) of an image row to the floor.
    Returns (left, right) floor points, or None if the row is beyond the horizon."""
    a = _project(H, x1, v, ref_sign)
    b = _project(H, x2, v, ref_sign)
    if a is None or b is None:
        return None
    return a, b


def bbox_floor_polygon(H, bbox, frame_wh, far_frac=0.1, near_frac=1.0,
                       clip_inflate=2.5, body_aspect=2.5, short_aspect=1.3, clipped=None):
    """Project a detection's bbox to a floor trapezoid of foot-point uncertainty.

    The uncertainty is *one-sided*: occlusion and clipping can only hide the
    lower part of a person, which lifts the visible box bottom y2 *above* the
    real feet -> the naive foot point lands too far away, never too near. So the
    trapezoid reaches generously toward the camera (rows below y2) and only a
    little past it (rows above y2):

        v_far  = y2 - far_frac  * box_height   (feet barely-farther margin)
        v_near = y2 + near_frac * box_height   (feet could be much nearer)

    Args:
        H:          3x3 ground homography (image px -> floor meters).
        bbox:       (x1, y1, x2, y2) detection box in pixels.
        frame_wh:   (width, height) of the source frame, for clip detection.
        far_frac:   far margin as a fraction of box height (small; box bottom is
                    rarely below the true feet).
        near_frac:  near reach as a fraction of box height (large; the feet may
                    be this much nearer when occluded by furniture).
        clip_inflate: if the box bottom touches the frame bottom (feet off-frame
                    below), reach this many box-heights toward the camera instead
                    -- a "very tall trapezoid", feet anywhere along that slice.
        body_aspect: a standing person is taller than wide. When the lower body is
                    occluded the box gets *short* but keeps its width, so a
                    height-based reach falls short. So also reach down to where a
                    full body of height `body_aspect * box_width` would put the
                    feet (0 disables this width-based term).
        short_aspect: aspect ratio (box_height / box_width) below which the box is
                    treated as a partial (torso-only) detection -- well under the
                    ~2.2 of a full standing person -- whose bottom edge sits at the
                    waist/chest rather than the feet. The shorter it is, the
                    further past the box bottom the real feet lie, so `near_frac`
                    is scaled up by `short_aspect / aspect`, reaching more
                    aggressively toward the camera (0 disables this gate). The
                    scale is 1.0 exactly at the threshold, so detections crossing
                    it frame-to-frame don't jump.
        clipped:    whether the box touches the frame bottom. None infers it from
                    `bbox` and `frame_wh`; pass it explicitly when `bbox` is in
                    rectified (undistorted) pixels, where the frame edge is curved.

    Returns an (N, 2) float array of floor points (a convex quad, ordered), or
    None if the band is entirely above the horizon (can't form a floor region).
    """
    x1, y1, x2, y2 = (float(b) for b in bbox)
    _w, h_img = frame_wh
    box_h = max(y2 - y1, 1.0)
    box_w = max(x2 - x1, 1.0)

    # The foot pixel is known-floor; its horizon side governs every other row.
    ref = _floor_sign(H, 0.5 * (x1 + x2), y2)
    if ref == 0.0:
        return None

    # Partial (torso-only) detection: when the lower body is occluded or clipped
    # the box gets short *relative to its width* and its bottom edge sits at the
    # waist/chest, not the feet -- so projecting that bottom point would place the
    # person too far away. The shorter the box, the more of the lower body (and
    # thus depth toward the camera) is missing, so reach more aggressively: scale
    # the near reach up by how far the aspect ratio falls short of `short_aspect`.
    aspect = box_h / box_w
    if short_aspect > 0.0 and aspect < short_aspect:
        near_frac *= short_aspect / aspect

    v_far = y2 - far_frac * box_h          # feet barely farther than box bottom
    v_near = y2 + near_frac * box_h        # feet possibly much nearer (occlusion)
    # Width-robust reach: feet of a full standing body (head row y1 + expected
    # height). Catches heavy bottom-occlusion that shrinks the box height.
    if body_aspect > 0.0:
        v_near = max(v_near, y1 + body_aspect * box_w)

    # "Can't see the feet": box clipped at the bottom edge -> feet are off-frame,
    # nearer than y2 and very uncertain. Reach much further toward the camera.
    if clipped is None:
        clipped = y2 >= h_img - 2.0
    if clipped:
        v_near = max(v_near, y2 + clip_inflate * box_h)

    # The far edge may sit beyond the horizon; walk it down toward the foot row
    # until it projects to real floor points (or give up if even y2 is invalid).
    far = None
    for v in np.linspace(v_far, y2, 6):
        far = _project_row(H, x1, x2, v, ref)
        if far is not None:
            break
    near = _project_row(H, x1, x2, v_near, ref)
    if near is None:
        near = _project_row(H, x1, x2, y2, ref)
    if far is None or near is None:
        return None

    fl, fr = far
    nl, nr = near
    # CCW-ish quad: far-left, far-right, near-right, near-left.
    return np.array([fl, fr, nr, nl], dtype=np.float64)


# --- convex polygon geometry ------------------------------------------------

def _signed_area(poly):
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _ensure_ccw(poly):
    return poly if _signed_area(poly) >= 0 else poly[::-1].copy()


def polygon_centroid(poly):
    """Area-weighted centroid of a simple polygon (vertex mean if degenerate)."""
    a = _signed_area(poly)
    if abs(a) < 1e-12:
        return poly.mean(axis=0)
    x, y = poly[:, 0], poly[:, 1]
    xn, yn = np.roll(x, -1), np.roll(y, -1)
    cross = x * yn - xn * y
    cx = np.dot(x + xn, cross) / (6 * a)
    cy = np.dot(y + yn, cross) / (6 * a)
    return np.array([cx, cy])


def _line_intersect(a, b, p, q):
    """Intersection of infinite line a->b with segment p->q (assumed to cross)."""
    r = b - a
    s = q - p
    denom = r[0] * s[1] - r[1] * s[0]
    if abs(denom) < 1e-12:
        return q
    t = ((p[0] - a[0]) * s[1] - (p[1] - a[1]) * s[0]) / denom
    return a + t * r


def convex_intersection(subject, clip):
    """Sutherland-Hodgman clip of `subject` against convex `clip`. Both are
    (N, 2) arrays. Returns the clipped polygon, or None if they don't overlap."""
    out = list(_ensure_ccw(np.asarray(subject, float)))
    cp = _ensure_ccw(np.asarray(clip, float))
    n = len(cp)
    for i in range(n):
        a, b = cp[i], cp[(i + 1) % n]
        edge = b - a
        if not out:
            return None
        inp, out = out, []
        s = inp[-1]
        s_in = edge[0] * (s[1] - a[1]) - edge[1] * (s[0] - a[0]) >= 0
        for e in inp:
            e_in = edge[0] * (e[1] - a[1]) - edge[1] * (e[0] - a[0]) >= 0
            if e_in:
                if not s_in:
                    out.append(_line_intersect(a, b, s, e))
                out.append(e)
            elif s_in:
                out.append(_line_intersect(a, b, s, e))
            s, s_in = e, e_in
    if len(out) < 3:
        return None
    return np.array(out)


def intersect_blobs(polys):
    """Intersect a list of convex floor polygons.

    Returns (centroid, intersection_polygon). `centroid` is an (x, y) array, or
    None if fewer than two valid polygons were given or they don't all overlap.
    """
    valid = [np.asarray(p, float) for p in polys if p is not None and len(p) >= 3]
    if len(valid) < 2:
        return None, None
    inter = valid[0]
    for p in valid[1:]:
        inter = convex_intersection(inter, p)
        if inter is None:
            return None, None
    return polygon_centroid(inter), inter
