"""Qt-free frame load/save helpers for Observatory JSONL sessions."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict, List, Union

import numpy as np

from phca.monitoring.observability import ObservabilityFrame, normalize_observability_json

PathLike = Union[str, Path]


def frame_from_json(obj: Dict[str, Any]) -> ObservabilityFrame:
    """Reconstruct an ObservabilityFrame from a JSONL object."""
    obj = normalize_observability_json(obj)
    f = ObservabilityFrame()
    array_fields = {
        "predicted_state", "obs_vector", "goal_ref", "continuous_action",
        "sanitized_state", "state_precision", "goal_target",
        "prediction_precision", "gprime_uncertainty", "per_dim_peu",
        "attention_weights", "last_action_vector",
    }
    for k, v in obj.items():
        if k == "grid" and v is not None:
            setattr(f, k, np.asarray(v, dtype=np.int32))
        elif k in array_fields and v is not None:
            setattr(f, k, np.asarray(v, dtype=np.float32))
        elif isinstance(v, (dict, list)):
            setattr(f, k, copy.deepcopy(v))
        else:
            setattr(f, k, v)
    return f


def frame_to_json(frame: ObservabilityFrame) -> Dict[str, Any]:
    """Serialise an ObservabilityFrame to a JSON-friendly dict."""
    return frame.to_json()


def load_jsonl_lines(lines: List[str]) -> List[ObservabilityFrame]:
    """Parse JSONL text lines into ObservabilityFrame instances."""
    frames: List[ObservabilityFrame] = []
    for ln in lines:
        ln = ln.strip()
        if not ln:
            continue
        frames.append(frame_from_json(json.loads(ln)))
    return frames


def load_jsonl_path(path: PathLike) -> List[ObservabilityFrame]:
    """Load frames from a JSONL file path."""
    text = Path(path).read_text()
    return load_jsonl_lines(text.splitlines())


def load_session_frames(session_dir: PathLike) -> List[ObservabilityFrame]:
    """Load ``timeseries.jsonl`` from a session directory."""
    jsonl_p = Path(session_dir) / "timeseries.jsonl"
    if not jsonl_p.exists():
        raise FileNotFoundError(f"not a valid session dir (missing timeseries.jsonl): {jsonl_p.parent}")
    return load_jsonl_path(jsonl_p)
