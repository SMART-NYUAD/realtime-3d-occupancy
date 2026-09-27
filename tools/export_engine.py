"""Build the TensorRT FP16 engine for the configured pose model (run once per
machine / model / Ultralytics+TensorRT version; takes a few minutes).

    python tools/export_engine.py [--config config.yaml] [--batch 4] [--imgsz 640]

Writes e.g. yolo11m-pose.engine next to the .pt. The engine is built with a
dynamic batch up to --batch so all cameras run in one inference call; keep
`detector.max_batch` in config.yaml <= this value.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tracker3d.config import load_config   # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--batch", type=int, default=None, help="max batch (default: detector.max_batch)")
    ap.add_argument("--imgsz", type=int, default=None, help="input size (default: detector.imgsz)")
    args = ap.parse_args()
    d = load_config(args.config)["detector"]
    model = d["model"]
    pt = model[:-len(".engine")] + ".pt" if model.endswith(".engine") else model
    batch = args.batch or int(d.get("max_batch", 4))
    imgsz = args.imgsz or int(d.get("imgsz", 640))

    from ultralytics import YOLO
    print(f"Exporting {pt} -> TensorRT FP16 (dynamic batch <= {batch}, imgsz {imgsz}) ...")
    out = YOLO(pt).export(format="engine", half=True, dynamic=True, batch=batch,
                          imgsz=imgsz, device=0, workspace=4)
    print(f"Done: {out}")
    if not model.endswith(".engine"):
        print(f"Set detector.model: \"{os.path.basename(out)}\" in {args.config} to use it.")


if __name__ == "__main__":
    main()
