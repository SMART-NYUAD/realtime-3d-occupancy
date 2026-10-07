"""Lens calibration (intrinsics + distortion) from a ChArUco board, once per camera.

    # 1. record ~30 views of the board moved across the WHOLE image
    python tools/calib_intrinsics.py record --camera yi01
    python tools/calib_intrinsics.py record --camera webcam --source usb:0   # home test

    # 2. solve -> calib/<cam>_intrinsics.json
    python tools/calib_intrinsics.py solve --camera yi01 [--square-mm 54.8]

    # 3. check: raw | corrected side by side, door frames should be straight
    python tools/calib_intrinsics.py view --camera yi01

Print the board with tools/charuco_board.py. Then add
`intrinsics_file: "calib/<cam>_intrinsics.json"` to the camera in config.yaml and
re-click its floor calibration with calibrate.py (the old homography was fitted on
raw pixels; track.py keeps using it uncorrected until you do).

record keys:  SPACE force-keep | u undo last | q finish
Frames are saved to calib/intrinsics_frames/<cam>/ (gitignored: they show people).
"""
import argparse
import datetime
import glob
import math
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tracker3d.capture import CameraStream                                 # noqa: E402
from tracker3d.config import list_cameras, load_config                     # noqa: E402
from tracker3d.lens import (BOARD_DEFAULTS, MIN_CORNERS, LensModel, detect_board,  # noqa: E402
                            make_board, make_detector, solve_intrinsics, solve_with_outliers)

GRID = (8, 6)               # coverage cells across x, y
TARGET_VIEWS = 30
STILL_PX = 2.0              # mean corner motion between frames below this = not blurred
MIN_VIEWS = 10


# --- shared helpers -----------------------------------------------------------

def _camera(args):
    """Camera dict from config.yaml, or an ad-hoc one (e.g. --camera webcam --source usb:0)."""
    cams = {c["name"]: c for c in list_cameras(load_config(args.config))}
    cam = dict(cams.get(args.camera, {"name": args.camera}))
    if getattr(args, "source", None):
        cam["source"] = args.source
    return cam


def _board_params(args):
    return {"cols": args.cols, "rows": args.rows, "square_mm": args.square_mm,
            "marker_mm": args.marker_mm, "dictionary": args.dict}


def _frames_dir(args, cam):
    return args.frames or os.path.join("calib", "intrinsics_frames", cam["name"])


def _intrinsics_path(args, cam):
    return args.out or cam.get("intrinsics_file") or os.path.join(
        "calib", f"{cam['name']}_intrinsics.json")


def _open_stream(cam):
    if "source" not in cam:
        raise SystemExit(f"camera '{cam['name']}' is not in config.yaml — pass --source "
                         "(rtsp://..., usb:0 or a video file)")
    print(f"connecting {cam['source']} ...")
    stream = CameraStream(cam)
    t0 = time.time()
    while True:
        _, f = stream.read()
        if f is not None:
            return stream, f
        if time.time() - t0 > 15:
            raise SystemExit("no frame from the camera after 15 s")
        time.sleep(0.05)


def _fit(img, max_w=1280, max_h=720):
    h, w = img.shape[:2]
    s = min(max_w / w, max_h / h, 1.0)
    return (cv2.resize(img, (int(w * s), int(h * s))) if s < 1.0 else img.copy()), s


def _coverage(points_list, size):
    """(cells with >= 3 corners, total cells, border cells covered, border total)."""
    w, h = size
    counts = np.zeros((GRID[1], GRID[0]), int)
    for pts in points_list:
        cx = np.clip((pts[:, 0] / w * GRID[0]).astype(int), 0, GRID[0] - 1)
        cy = np.clip((pts[:, 1] / h * GRID[1]).astype(int), 0, GRID[1] - 1)
        np.add.at(counts, (cy, cx), 1)
    covered = counts >= 3
    border = np.zeros_like(covered)
    border[0, :] = border[-1, :] = border[:, 0] = border[:, -1] = True
    return counts, int(covered.sum()), covered.size, int((covered & border).sum()), int(border.sum())


def _descriptor(pts, size):
    """Where/how big/how foreshortened the board looks, to tell new views from repeats."""
    w, h = size
    hull = cv2.convexHull(pts.astype(np.float32))
    x, y, bw, bh = cv2.boundingRect(hull)
    return (float(pts[:, 0].mean() / w), float(pts[:, 1].mean() / h),
            math.sqrt(max(cv2.contourArea(hull), 1.0)) / w, bw / max(bh, 1))


def _is_novel(desc, kept):
    cx, cy, s, a = desc
    for kx, ky, ks, ka in kept:
        if (abs(cx - kx) < 0.08 and abs(cy - ky) < 0.08 and abs(math.log(s / ks)) < 0.15
                and abs(math.log(a / ka)) < 0.15):
            return False
    return True


# --- record -----------------------------------------------------------------------

def cmd_record(args):
    cam = _camera(args)
    out = _frames_dir(args, cam)
    os.makedirs(out, exist_ok=True)
    if args.fresh:
        for p in glob.glob(os.path.join(out, "*.png")):
            os.remove(p)
    board = make_board(**_board_params(args))
    detector = make_detector(board)

    stream, frame = _open_stream(cam)
    size = (frame.shape[1], frame.shape[0])
    kept = []          # (path, img_pts, descriptor)
    for p in sorted(glob.glob(os.path.join(out, "*.png"))):   # resume an earlier session
        img = cv2.imread(p)
        det = detect_board(img, detector, board) if img is not None else None
        if det is not None:
            kept.append((p, det[1], _descriptor(det[1], size)))
    if kept:
        print(f"{len(kept)} usable frames already in {out} (use --fresh to start over)")
    print(f"Recording {cam['name']} at {size[0]}x{size[1]}. Move the board SLOWLY over the "
          "whole image — corners and edges matter most — tilted ~30-45 deg in different "
          "directions. SPACE force-keep | u undo | q finish")

    prev = None        # (ids, img_pts) of the previous frame's detection
    last_seq, idx = None, len(kept)
    flash_until = 0.0
    while True:
        seq, frame = stream.read()
        if frame is None or seq == last_seq:
            k = cv2.waitKey(5) & 0xFF
            if k == ord("q"):
                break
            continue
        last_seq = seq
        if (frame.shape[1], frame.shape[0]) != size:
            print("frame size changed mid-recording — ignoring frame")
            continue
        det = detect_board(frame, detector, board)
        still = False
        status = "no board"
        if det is not None:
            _, pts, ids = det
            if prev is not None:
                common, a, b = np.intersect1d(prev[0], ids, return_indices=True)
                if len(common) >= MIN_CORNERS:
                    still = float(np.linalg.norm(prev[1][a] - pts[b], axis=1).mean()) < STILL_PX
            desc = _descriptor(pts, size)
            novel = _is_novel(desc, [k[2] for k in kept])
            status = f"{len(ids)} corners | {'still' if still else 'moving'} | " \
                     f"{'new view' if novel else 'already have this view'}"
            prev = (ids, pts)
        else:
            prev = None

        k = cv2.waitKey(1) & 0xFF
        force = k == ord(" ")
        if det is not None and ((still and novel) or force):
            path = os.path.join(out, f"{idx:03d}.png")
            cv2.imwrite(path, frame)
            kept.append((path, det[1], desc))
            idx += 1
            flash_until = time.time() + 0.3
            print(f"  kept {path} ({len(det[2])} corners)  [{len(kept)}/{TARGET_VIEWS}]")
        elif force:
            print("  no board detected in this frame — not kept")
        if k == ord("u") and kept:
            p, _, _ = kept.pop()
            if os.path.exists(p):
                os.remove(p)
            print(f"  removed {p}")
        if k == ord("q"):
            break

        disp, s = _fit(frame)
        counts, cov, tot, bcov, btot = _coverage([kk[1] for kk in kept], size)
        overlay = disp.copy()
        cw, ch = disp.shape[1] / GRID[0], disp.shape[0] / GRID[1]
        for gy in range(GRID[1]):
            for gx in range(GRID[0]):
                p1 = (int(gx * cw), int(gy * ch))
                p2 = (int((gx + 1) * cw), int((gy + 1) * ch))
                if counts[gy, gx] >= 3:
                    cv2.rectangle(overlay, p1, p2, (0, 160, 0), -1)
                cv2.rectangle(disp, p1, p2, (90, 90, 90), 1)
        disp = cv2.addWeighted(overlay, 0.35, disp, 0.65, 0)
        if det is not None:
            for x, y in det[1]:
                cv2.circle(disp, (int(x * s), int(y * s)), 3, (0, 255, 255), -1)
        if time.time() < flash_until:
            cv2.rectangle(disp, (0, 0), (disp.shape[1] - 1, disp.shape[0] - 1), (0, 255, 0), 8)
        cv2.putText(disp, f"kept {len(kept)}/{TARGET_VIEWS}  coverage {cov}/{tot}  "
                    f"edges {bcov}/{btot}", (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 255), 2)
        cv2.putText(disp, status, (10, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imshow("record intrinsics", disp)

    stream.release()
    cv2.destroyAllWindows()
    _, cov, tot, bcov, btot = _coverage([kk[1] for kk in kept], size)
    print(f"\n{len(kept)} frames in {out}; coverage {cov}/{tot} cells, edges {bcov}/{btot}.")
    if len(kept) < MIN_VIEWS:
        print(f"Need at least {MIN_VIEWS} (ideally ~{TARGET_VIEWS}) — record more.")
    else:
        print(f"Next: python tools/calib_intrinsics.py solve --camera {cam['name']}")


# --- solve ------------------------------------------------------------------------

def cmd_solve(args):
    cam = _camera(args)
    frames = _frames_dir(args, cam)
    out = _intrinsics_path(args, cam)
    bp = _board_params(args)
    board = make_board(**bp)
    detector = make_detector(board)

    paths = sorted(glob.glob(os.path.join(frames, "*.png")))
    if not paths:
        raise SystemExit(f"no frames in {frames} — run `record` first")
    names, objs, imgs, size = [], [], [], None
    for p in paths:
        img = cv2.imread(p)
        if img is None:
            continue
        s = (img.shape[1], img.shape[0])
        if size is None:
            size = s
        elif s != size:
            raise SystemExit(f"{p} is {s[0]}x{s[1]} but earlier frames are {size[0]}x{size[1]}")
        det = detect_board(img, detector, board)
        if det is not None:
            names.append(os.path.basename(p))
            objs.append(det[0])
            imgs.append(det[1])
    print(f"{len(objs)}/{len(paths)} frames usable ({size[0]}x{size[1]})")
    if len(objs) < MIN_VIEWS:
        raise SystemExit(f"need at least {MIN_VIEWS} usable frames — record more")

    candidates = ["standard", "rational"] if args.model == "auto" else [args.model]
    results = {}
    for m in candidates:
        try:
            results[m] = solve_with_outliers(objs, imgs, size, m, MIN_VIEWS)
            print(f"  {m:9s} rms {results[m][0]['rms']:.3f} px "
                  f"({int(results[m][1].sum())} views)")
        except cv2.error as e:
            print(f"  {m:9s} failed: {str(e).strip().splitlines()[-1]}")
    if not results:
        raise SystemExit("calibration failed for every model — check the frames")
    model = min(results, key=lambda m: results[m][0]["rms"])
    if "standard" in results and model != "standard" and \
            results[model][0]["rms"] > 0.9 * results["standard"][0]["rms"]:
        model = "standard"      # the richer model must earn its extra parameters (>10 %)
    res, keep = results[model]
    dropped = [n for n, k in zip(names, keep) if not k]

    used_imgs = [i for i, k in zip(imgs, keep) if k]
    used_objs = [o for o, k in zip(objs, keep) if k]
    _, cov, tot, bcov, btot = _coverage(used_imgs, size)

    # Split-half stability: solve on even and odd views separately.
    stab = None
    if len(used_objs) >= 2 * MIN_VIEWS:
        try:
            a = solve_intrinsics(used_objs[0::2], used_imgs[0::2], size, model)
            b = solve_intrinsics(used_objs[1::2], used_imgs[1::2], size, model)
            stab = {"fx_diff_pct": float(abs(a["K"][0, 0] - b["K"][0, 0]) / res["K"][0, 0] * 100),
                    "k1_diff": float(abs(a["dist"][0] - b["dist"][0]))}
        except cv2.error:
            pass

    lens = LensModel(res["K"], res["dist"], size, model, path=out)
    d = {"camera": cam["name"], **lens.to_dict(),
         "rms_px": round(res["rms"], 4), "n_views": int(keep.sum()), "dropped_views": dropped,
         "per_view_rms_px": {n: round(float(e), 3)
                             for n, e in zip([n for n, k in zip(names, keep) if k],
                                             res["per_view"])},
         "coverage": {"cells": f"{cov}/{tot}", "edge_cells": f"{bcov}/{btot}"},
         "stability": stab, "board": bp,
         "created": datetime.datetime.now().isoformat(timespec="seconds"),
         "opencv": cv2.__version__}
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    import json
    with open(out, "w") as f:
        json.dump(d, f, indent=2)

    # Side-by-side preview of one frame for a quick visual check.
    sample = cv2.imread(os.path.join(frames, names[0]))
    prev = np.hstack([sample, lens.undistort_image(sample)])
    prev_path = os.path.join(frames, "undistort_preview.jpg")
    cv2.imwrite(prev_path, _fit(prev, 2400, 800)[0])

    K = res["K"]
    print(f"\nModel: {model}   RMS reprojection error: {res['rms']:.3f} px "
          f"over {int(keep.sum())} views" + (f" (dropped {len(dropped)} outliers)" if dropped else ""))
    print(f"fx {K[0, 0]:.1f}  fy {K[1, 1]:.1f}  cx {K[0, 2]:.1f}  cy {K[1, 2]:.1f}")
    print("dist " + " ".join(f"{v:+.4f}" for v in res["dist"]))
    print(f"Coverage: {cov}/{tot} cells, edges {bcov}/{btot}")
    if stab:
        print(f"Stability (even vs odd views): fx differs {stab['fx_diff_pct']:.2f} %, "
              f"k1 differs {stab['k1_diff']:.4f}")
    verdict = ("GOOD" if res["rms"] < 0.5 else "OK" if res["rms"] < 1.0 else
               "POOR — re-record (blur? bent board? too few views?)")
    print(f"Verdict: {verdict}")
    if bcov < 0.6 * btot:
        print("WARNING: image edges poorly covered — that's where distortion is largest. "
              "Record more views near the edges/corners.")
    if stab and (stab["fx_diff_pct"] > 1.0 or stab["k1_diff"] > 0.02):
        print("WARNING: unstable between halves — record more varied views.")
    print(f"Saved {out}  (fingerprint {lens.fingerprint()})")
    print(f"Preview: {prev_path}")
    print(f"Next: python tools/calib_intrinsics.py view --camera {cam['name']}, then add "
          f"intrinsics_file: \"{out}\" to {cam['name']} in config.yaml and re-run calibrate.py")


# --- view -------------------------------------------------------------------------

def _grid_lines(img, n=8, color=(0, 255, 255)):
    h, w = img.shape[:2]
    for i in range(1, n):
        x, y = int(w * i / n), int(h * i / n)
        cv2.line(img, (x, 0), (x, h - 1), color, 1)
        cv2.line(img, (0, y), (w - 1, y), color, 1)


def cmd_view(args):
    cam = _camera(args)
    path = args.intrinsics or _intrinsics_path(args, cam)
    if not os.path.exists(path):
        raise SystemExit(f"{path} not found — run `solve` first")
    lens = LensModel.load(path)
    stream, frame = _open_stream(cam)
    h, w = frame.shape[:2]
    if lens.image_size != (w, h):
        print(f"note: lens calibrated at {lens.image_size}, stream is {w}x{h} — scaling")
    tile_w = 640
    tile_h = int(round(h * tile_w / w))
    print("Left: raw   Right: lens-corrected. Straight edges near the borders (door "
          "frames, ceiling, desks) should be straight on the right. s=snapshot q=quit")
    last = None
    while True:
        seq, frame = stream.read()
        if frame is not None and seq != last:
            last = seq
            raw = cv2.resize(frame, (tile_w, tile_h))
            rect = lens.undistort_image(frame, (tile_w, tile_h))
            _grid_lines(raw)
            _grid_lines(rect)
            view = np.hstack([raw, rect])
            cv2.putText(view, "raw", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(view, "lens-corrected", (tile_w + 10, 24), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 0), 2)
            cv2.imshow("lens check", view)
        k = cv2.waitKey(5) & 0xFF
        if k == ord("q"):
            break
        if k == ord("s") and frame is not None:
            p = f"lens_check_{cam['name']}.jpg"
            cv2.imwrite(p, np.hstack([frame, lens.undistort_image(frame)]))
            print(f"saved {p}")
    stream.release()
    cv2.destroyAllWindows()


def main():
    d = BOARD_DEFAULTS
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", default="config.yaml")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("record", "solve", "view"):
        p = sub.add_parser(name)
        p.add_argument("--camera", required=True, help="camera name (config.yaml or ad-hoc)")
        p.add_argument("--frames", default=None, help="frames dir (default calib/intrinsics_frames/<cam>)")
        p.add_argument("--out", default=None, help="intrinsics JSON (default: camera's "
                       "intrinsics_file or calib/<cam>_intrinsics.json)")
        p.add_argument("--cols", type=int, default=d["cols"])
        p.add_argument("--rows", type=int, default=d["rows"])
        p.add_argument("--square-mm", type=float, default=d["square_mm"],
                       help="measured printed square size (recorded only; doesn't change the lens)")
        p.add_argument("--marker-mm", type=float, default=d["marker_mm"])
        p.add_argument("--dict", default=d["dictionary"])
        if name in ("record", "view"):
            p.add_argument("--source", default=None, help="override: rtsp://..., usb:0, video file")
        if name == "record":
            p.add_argument("--fresh", action="store_true", help="delete earlier frames first")
        if name == "solve":
            p.add_argument("--model", default="auto",
                           choices=["auto", "standard", "rational", "fisheye"])
        if name == "view":
            p.add_argument("--intrinsics", default=None, help="intrinsics JSON to view")
    args = ap.parse_args()
    {"record": cmd_record, "solve": cmd_solve, "view": cmd_view}[args.cmd](args)


if __name__ == "__main__":
    main()
