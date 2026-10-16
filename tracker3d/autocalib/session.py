"""Versioned, append-only calibration recordings and immutable JSON artifacts."""
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
from datetime import datetime, timezone

import numpy as np

from tracker3d.lens import LensModel

VERSION = 1


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                     separators=(",", ":")).encode()).hexdigest()


def write_json(path, value, overwrite=False):
    """Install complete JSON only; exclusive by default, including against races."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".autocalib-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(value, f, indent=2, allow_nan=False)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        if overwrite:
            os.replace(tmp, path)
        else:
            os.link(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def read_json(path):
    with open(path) as f:
        return json.load(f)


def lens_from_dict(data):
    lens = LensModel(data["camera_matrix"], data["dist_coeffs"], data["image_size"],
                     data.get("model", "standard"), data["rectified_camera_matrix"])
    if (lens.model not in ("standard", "rational", "fisheye")
            or (lens.model == "fisheye" and len(lens.dist) != 4)
            or (lens.model != "fisheye" and len(lens.dist) not in (4, 5, 8))
            or lens.K.shape != (3, 3) or lens.K_rect.shape != (3, 3)
            or not np.isfinite(lens.K).all() or not np.isfinite(lens.K_rect).all()
            or not np.isfinite(lens.dist).all() or min(lens.image_size) < 2
            or lens.K[0, 0] <= 0 or lens.K[1, 1] <= 0):
        raise ValueError("invalid lens snapshot")
    if lens.fingerprint() != data.get("fingerprint"):
        raise ValueError("lens snapshot fingerprint does not match its parameters")
    return lens


def validate_cameras(cameras):
    if len(cameras) < 2:
        raise ValueError("autocalibration needs at least two cameras")
    for name, data in cameras.items():
        if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
            raise ValueError(f"unsafe camera name: {name!r}")
        lens_from_dict(data["lens"])


class SessionWriter:
    def __init__(self, path, cameras, detector=None):
        validate_cameras(cameras)
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=False)
        self.manifest = {"version": VERSION, "kind": "walk", "complete": False,
                         "created_utc": datetime.now(timezone.utc).isoformat(),
                         "cameras": cameras, "detector": detector or {}}
        write_json(self.path / "manifest.json", self.manifest)
        self.file = open(self.path / "observations.jsonl", "x")
        self.count = 0

    def append(self, record):
        self.file.write(json.dumps(record, allow_nan=False, separators=(",", ":")) + "\n")
        self.file.flush()
        self.count += 1

    def close(self, complete=True):
        if self.file.closed:
            return
        self.file.flush()
        os.fsync(self.file.fileno())
        self.file.close()
        self.manifest.update(complete=bool(complete), records=self.count)
        write_json(self.path / "manifest.json", self.manifest, overwrite=True)


def load_session(path, allow_incomplete=False):
    path = Path(path)
    manifest = read_json(path / "manifest.json")
    if manifest.get("version") != VERSION or manifest.get("kind") != "walk":
        raise ValueError("unsupported calibration session format")
    validate_cameras(manifest["cameras"])
    if not manifest.get("complete") and not allow_incomplete:
        raise ValueError("recording is incomplete; inspect it, then use --allow-incomplete")
    raw = (path / "observations.jsonl").read_bytes()
    lines = raw.splitlines(keepends=True)
    records = []
    for i, line in enumerate(lines):
        try:
            record = json.loads(line)
        except (ValueError, UnicodeError):
            if allow_incomplete and i == len(lines) - 1 and not line.endswith(b"\n"):
                break
            raise ValueError(f"invalid observation at line {i + 1}") from None
        if record.get("kind") not in ("frame", "marker"):
            raise ValueError(f"unknown observation kind at line {i + 1}")
        timestamp = record.get("t")
        if not isinstance(timestamp, (float, int)) or not math.isfinite(timestamp):
            raise ValueError(f"invalid timestamp at line {i + 1}")
        if record["kind"] == "frame":
            name = record.get("camera")
            if name not in manifest["cameras"]:
                raise ValueError(f"unknown camera at line {i + 1}")
            if record.get("image_size") != manifest["cameras"][name]["lens"]["image_size"]:
                raise ValueError(f"{name}: frame size changed during recording")
            if not isinstance(record.get("people"), list):
                raise ValueError(f"invalid detections at line {i + 1}")
        records.append(record)
    if manifest.get("complete") and len(records) != manifest.get("records"):
        raise ValueError("record count does not match manifest")
    return {"manifest": manifest, "records": records,
            "observations_id": hashlib.sha256(raw).hexdigest(),
            "id": hashlib.sha256(json.dumps(manifest, sort_keys=True).encode() + raw).hexdigest()}


def check_artifact(value, kind):
    if value.get("version") != VERSION or value.get("kind") != kind:
        raise ValueError(f"expected a version {VERSION} {kind} artifact")
    expected = value.get("id")
    if expected != digest({k: v for k, v in value.items() if k != "id"}):
        raise ValueError(f"{kind} artifact was modified or is incomplete")
    return value


def seal(value):
    value = {k: v for k, v in value.items() if k != "id"}
    return {**value, "id": digest(value)}
