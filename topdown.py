"""Preview / export the top-down map used for calibration.

Renders the configured world plan (point cloud by default) to a PNG so you can
eyeball the "bottom-up" view before calibrating, and to warm the render cache.

Usage:
    python topdown.py [--config config.yaml] [--out topdown.png] [--show]
                      [--source pointcloud|dxf] [--mode height|rgb]
Keys (with --show): any key / q to close.
"""
import argparse
import sys

import cv2
import yaml

sys.path.insert(0, "src")
from plan import load_plan        # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--out", default="topdown.png")
    ap.add_argument("--show", action="store_true", help="open a window too")
    ap.add_argument("--source", default=None, help="override floorplan.source")
    ap.add_argument("--mode", default=None, help="override color_mode (height|rgb)")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.source:
        cfg["floorplan"]["source"] = args.source
    if args.mode:
        cfg["floorplan"]["color_mode"] = args.mode

    plan = load_plan(cfg)
    img = plan.background()
    cv2.imwrite(args.out, img)
    print(f"Saved {args.out}  ({plan.width_m:.2f} x {plan.height_m:.2f} m, "
          f"{img.shape[1]}x{img.shape[0]}px)")

    if args.show:
        cv2.imshow("top-down map (q to close)", img)
        while True:
            k = cv2.waitKey(50) & 0xFF
            if k in (ord("q"), 27) or k != 255:
                break
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
