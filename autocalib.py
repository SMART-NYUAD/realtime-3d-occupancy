"""Record a calibration walk, solve the rig, anchor to the map, validate and export.

All output paths are exclusive: existing sessions, solutions and candidates are
never overwritten. Offline commands do not import YOLO, CUDA or GStreamer.
Run `python autocalib.py <command> --help` for each stage.
"""
import argparse
import json

from tracker3d.config import load_config
from tracker3d.autocalib.session import load_session, read_json, write_json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("record", "solve", "anchor", "validate", "export"):
        p = sub.add_parser(name)
        p.add_argument("--config", default="config.yaml")
        p.add_argument("--out", required=True, help="new output file or directory (never overwritten)")
        if name in ("solve", "validate"):
            p.add_argument("--session", required=True, help="recorded session directory")
            p.add_argument("--allow-incomplete", action="store_true", help="recover intact records after a crash")
        if name in ("anchor", "validate", "export"):
            p.add_argument("--rig", required=True, help="solve output for anchor; anchored output otherwise")
        if name == "record":
            p.add_argument("--seconds", type=float, default=120)
            p.add_argument("--headless", action="store_true")
        elif name == "solve":
            p.add_argument("--kp-conf", type=float, default=0.65)
            p.add_argument("--max-gap-ms", type=float, default=120)
            p.add_argument("--sample-hz", type=float, default=5)
            p.add_argument("--max-ankle-speed", type=float, default=0.30,
                           help="maximum motion in person box heights/second; prefers pauses")
            p.add_argument("--ankle-height-ratio", type=float, default=0.05,
                           help="ankle height / nose height above floor; anthropometric prior, not measured")
            p.add_argument("--refine-focal", action="store_true", help="try bounded +/-3%% focal refinement")
            p.add_argument("--max-nfev", type=int, default=1500,
                           help="optimizer evaluation limit; reaching it is recorded, not fatal")
        elif name == "anchor":
            p.add_argument("--points", help="JSON list of {marker or t, world:[x,y], role:anchor|check}; otherwise GUI")
        elif name == "validate":
            p.add_argument("--checkpoints", help="JSON list of measured points in this validation session")
        elif name == "export":
            p.add_argument("--report", required=True, help="passing validate output")
    args = parser.parse_args(argv)
    try:
        if args.command == "record":
            from tracker3d.autocalib.record import record_walk
            record_walk(load_config(args.config), args.out, args.seconds, args.headless)
            return 0
        if args.command == "solve":
            from tracker3d.autocalib.solve import solve_session
            session = load_session(args.session, args.allow_incomplete)
            result = solve_session(session, ankle_ratio=args.ankle_height_ratio,
                                   refine_focal=args.refine_focal, max_nfev=args.max_nfev,
                                   kp_conf=args.kp_conf, max_gap_s=args.max_gap_ms / 1000,
                                   sample_hz=args.sample_hz, max_ankle_speed=args.max_ankle_speed)
            write_json(args.out, result)
            print(json.dumps({"saved": args.out, "counts": result["counts"],
                              "heldout": result["heldout"], "focal_refinement": result["focal_refinement"]}, indent=2))
        elif args.command == "anchor":
            from tracker3d.autocalib.anchor import anchor_rig, pick_map_points
            rig, cfg = read_json(args.rig), load_config(args.config)
            points = read_json(args.points) if args.points else pick_map_points(rig, cfg)
            result = anchor_rig(rig, points, cfg)
            write_json(args.out, result)
            print(f"Saved {args.out}; scale {result['scale_m']:.4f} m/unit; anchor errors {result['fit_error_m']}")
            print(f"Walker's reconstructed nose height: {result['nose_height_m']:.2f} m "
                  "(compare with theirs: about 10 cm below standing height)")
            for w in result["warnings"]:
                print(f"WARNING: {w}")
        elif args.command == "validate":
            from tracker3d.autocalib.validation import validate
            report = validate(read_json(args.rig), load_session(args.session, args.allow_incomplete),
                              load_config(args.config), read_json(args.checkpoints) if args.checkpoints else None)
            write_json(args.out, report)
            print(json.dumps(report, indent=2))
            return 0 if report["passed"] else 2
        else:
            from tracker3d.autocalib.validation import export_candidate
            path = export_candidate(read_json(args.rig), read_json(args.report), load_config(args.config), args.out)
            print(f"Candidate ready: python track.py --config {path}\nMQTT is disabled in the candidate config.")
    except (ValueError, OSError, KeyError, ImportError) as e:
        parser.exit(2, f"autocalib: {e}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
