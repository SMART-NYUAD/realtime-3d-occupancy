"""Config + secrets loading, and per-camera config merging."""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


def load_env():
    """Load optional secrets (MQTT credentials) from <repo>/.env."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(ROOT / ".env")


def load_config(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def list_cameras(cfg):
    """Per-camera dicts with `camera_defaults` merged in."""
    defaults = cfg.get("camera_defaults", {})
    return [{**defaults, **cam} for cam in cfg["cameras"]]


def get_camera(cfg, name):
    cams = list_cameras(cfg)
    for cam in cams:
        if cam["name"] == name:
            return cam
    raise SystemExit(f"camera '{name}' not found. Available: {[c['name'] for c in cams]}")


