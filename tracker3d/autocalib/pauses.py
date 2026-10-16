"""Find long pauses in a recorded walk, so a person walking alone needs no keypresses.

Visit the floor marks in a fixed order and stand on each one longer than at
ordinary pauses (about 5 s versus 1-2 s). This lists every long pause in time
order with its approximate position, and writes a points template whose `t`
values `autocalib.py anchor --points` and `validate --checkpoints` accept in
place of SPACE markers. Fill in each pause's measured map position.
"""
import numpy as np

from .observations import prepare
from .session import check_artifact
from .solve import evaluate, rig_geometry

TYPICAL_NOSE_HEIGHT_M = 1.55   # only to turn distances into approximate meters before anchoring


def find_pauses(trajectory, meters_per_unit, min_seconds=4.0, radius_m=0.15, max_gap_s=0.6):
    """Group trajectory samples where the person stayed within `radius_m` of the
    group's median position with no gap over `max_gap_s`; keep groups lasting at
    least `min_seconds`. Returns pauses in time order."""
    points = sorted(trajectory, key=lambda p: p["t"])
    radius = radius_m / meters_per_unit
    groups, current = [], []
    for p in points:
        xy = np.asarray(p["xy"], float)
        if current:
            centre = np.median([q["xy"] for q in current], axis=0)
            if p["t"] - current[-1]["t"] > max_gap_s or np.linalg.norm(xy - centre) > radius:
                groups.append(current)
                current = []
        current.append(p)
    if current:
        groups.append(current)
    pauses = []
    for g in groups:
        start, end = g[0]["t"], g[-1]["t"]
        if end - start >= min_seconds:
            pauses.append({"t": float((start + end) / 2), "start": float(start), "end": float(end),
                           "seconds": float(end - start),
                           "rig_xy": np.median([q["xy"] for q in g], axis=0).tolist()})
    return pauses


def session_pauses(rig_or_anchored, session, min_seconds=4.0):
    """Long pauses of `session`, reconstructed with a solved (or anchored) rig.
    With an anchored rig, positions are also given in map meters."""
    anchored = rig_or_anchored if rig_or_anchored.get("kind") == "anchored_rig" else None
    rig = check_artifact(anchored["rig"] if anchored else rig_or_anchored, "rig")
    if anchored:
        check_artifact(anchored, "anchored_rig")
    poses, lenses = rig_geometry(rig)
    samples, _, counts = prepare(session, **rig["sampling"])
    _, trajectory = evaluate(samples, poses, lenses, rig["body"]["ankle_z"])
    meters_per_unit = (anchored["scale_m"] if anchored
                       else TYPICAL_NOSE_HEIGHT_M / rig["body"]["nose_height"])
    pauses = find_pauses(trajectory, meters_per_unit, min_seconds)
    if anchored:
        from .anchor import apply_transform
        S = np.asarray(anchored["transform"])
        for p in pauses:
            p["map_xy"] = apply_transform(S, [p["rig_xy"]])[0].tolist()
    t0 = min(r["t"] for r in session["records"] if r["kind"] == "frame")
    for i, p in enumerate(pauses, 1):
        p["number"] = i
        p["seconds_into_walk"] = float(p["t"] - t0)
    return pauses, counts


def points_template(pauses, role):
    """A --points / --checkpoints file to complete: one entry per long pause,
    `world` left empty for the measured map position of that floor mark."""
    return [{"t": p["t"], "world": None, "role": role,
             "note": f"pause {p['number']}: {p['seconds']:.1f} s, "
                     f"{p['seconds_into_walk']:.0f} s into the walk"
                     + (f", near map ({p['map_xy'][0]:.2f}, {p['map_xy'][1]:.2f})" if "map_xy" in p else "")}
            for p in pauses]
