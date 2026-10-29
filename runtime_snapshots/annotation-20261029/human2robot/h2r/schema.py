from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import hashlib
import json
import numpy as np
from .geometry import check_transform

SCHEMA_VERSION = "human-reference/0.1"
SCALE_STATUSES = {"metric_calibrated", "metric_estimated", "relative_only", "unknown", "synthetic_metric"}


def json_ready(x):
    if isinstance(x, np.ndarray):
        return json_ready(x.tolist())
    if isinstance(x, (np.integer, np.floating, np.bool_)):
        return json_ready(x.item())
    if isinstance(x, float) and not np.isfinite(x):
        return None
    if isinstance(x, dict):
        return {str(k): json_ready(v) for k, v in x.items()}
    if isinstance(x, (tuple, list)):
        return [json_ready(v) for v in x]
    return x


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(value), ensure_ascii=False, indent=2, allow_nan=False))


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


@dataclass
class Episode:
    episode_id: str
    timestamp: np.ndarray
    joints_camera: np.ndarray
    observed: np.ndarray
    confidence: np.ndarray
    camera_world: np.ndarray
    track_id: np.ndarray
    phase: np.ndarray
    metadata: dict = field(default_factory=dict)

    def validate(self):
        n = len(self.timestamp)
        if n < 2 or not np.isfinite(self.timestamp).all() or np.any(np.diff(self.timestamp) <= 0):
            raise ValueError("Timestamps must be finite, strictly increasing, and have >=2 samples")
        for name, shape in [("joints_camera", (n, 21, 3)), ("observed", (n,)),
                            ("confidence", (n,)), ("camera_world", (n, 4, 4)),
                            ("track_id", (n,)), ("phase", (n,))]:
            if np.shape(getattr(self, name)) != shape:
                raise ValueError(f"{name}: expected {shape}, got {np.shape(getattr(self,name))}")
        if self.metadata.get("units") != "m":
            raise ValueError("Explicitly normalize units to metres before processing")
        if self.metadata.get("scale_status") not in SCALE_STATUSES:
            raise ValueError("Unknown scale_status")
        if not np.isfinite(self.confidence).all() or np.any((self.confidence < 0) | (self.confidence > 1)):
            raise ValueError("Confidence weights must be finite in [0,1]; not calibrated probabilities")
        if not np.isfinite(self.joints_camera[self.observed]).all():
            raise ValueError("Observed frames contain non-finite landmarks")
        for T in self.camera_world:
            check_transform(T)
        for key in ["source_kind", "source_uri", "split_group", "camera_status"]:
            if key not in self.metadata:
                raise ValueError(f"Missing provenance: {key}")
        return self

    def save(self, path):
        self.validate()
        write_json(path, {"schema_version": SCHEMA_VERSION, "episode_id": self.episode_id,
                         **{k: getattr(self,k) for k in ["timestamp","joints_camera","observed",
                            "confidence","camera_world","track_id","phase","metadata"]}})

    @classmethod
    def load(cls, path):
        d = json.loads(Path(path).read_text())
        if d.pop("schema_version", None) != SCHEMA_VERSION:
            raise ValueError("Unsupported input schema")
        for key in ["timestamp","joints_camera","confidence","camera_world"]:
            d[key] = np.asarray(d[key], float)
        d["observed"] = np.asarray(d["observed"], bool)
        for key in ["track_id","phase"]:
            d[key] = np.asarray(d[key])
        return cls(**d).validate()


def split_for_group(group, seed=17):
    """All derived clips from a source/session inherit the same split."""
    v = int(hashlib.sha256(f"{seed}:{group}".encode()).hexdigest()[:8], 16) % 100
    return "train" if v < 80 else "validation" if v < 90 else "test"
