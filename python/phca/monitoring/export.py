"""CSV export of per-cycle scalar observability metrics."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from phca.monitoring.cognitive_panels import (
    build_moment_series,
    classify_action_mechanism,
    flow_bottleneck_module,
)
from phca.monitoring.session_io import load_session_frames

PathLike = Union[str, Path]

SESSION_SCALAR_COLUMNS = [
    "cycle_id",
    "agent_id",
    "agent_label",
    "timeline_step",
    "schema_version",
    "env_kind",
    "prediction_error",
    "prediction_confidence",
    "latency_ms",
    "active_drive_id",
    "violations_count",
    "goal_reached",
    "rss_bytes",
    "rbta_action",
    "explored",
    "best_score",
    "mechanism",
    "spike",
    "learn_burst",
    "decision_shift",
    "violation",
    "learn_ms",
    "module_ms_total",
    "gprime_learn_ms",
    "bottleneck_module",
]


def _scalar_row(frame, moment: Dict[str, Any]) -> Dict[str, Any]:
    r = frame.action_rationale or {}
    timings = dict(getattr(frame, "module_timings", {}) or {})
    bs = r.get("best_score")
    return {
        "cycle_id": int(getattr(frame, "cycle_id", 0) or 0),
        "agent_id": int(getattr(frame, "agent_id", 0) or 0),
        "agent_label": str(getattr(frame, "agent_label", "") or ""),
        "timeline_step": int(
            -1 if getattr(frame, "timeline_step", -1) is None
            else getattr(frame, "timeline_step", -1)
        ),
        "schema_version": int(getattr(frame, "schema_version", 0) or 0),
        "env_kind": str(getattr(frame, "env_kind", "") or ""),
        "prediction_error": float(getattr(frame, "prediction_error", 0.0) or 0.0),
        "prediction_confidence": float(getattr(frame, "prediction_confidence", 0.0) or 0.0),
        "latency_ms": float(getattr(frame, "latency_ms", 0.0) or 0.0),
        "active_drive_id": int(getattr(frame, "active_drive_id", 0) or 0),
        "violations_count": int(getattr(frame, "violations_count", 0) or 0),
        "goal_reached": bool(getattr(frame, "goal_reached", False)),
        "rss_bytes": int(getattr(frame, "rss_bytes", 0) or 0),
        "rbta_action": str(getattr(frame, "rbta_action", "") or ""),
        "explored": bool(r.get("explored", False)),
        "best_score": float(bs) if isinstance(bs, (int, float)) else "",
        "mechanism": classify_action_mechanism(r),
        "spike": bool(moment.get("spike", False)),
        "learn_burst": bool(moment.get("learn_burst", False)),
        "decision_shift": bool(moment.get("decision_shift", False)),
        "violation": bool(moment.get("violation", False)),
        "learn_ms": float(moment.get("learn_ms", 0.0) or 0.0),
        "module_ms_total": float(sum(float(v or 0.0) for v in timings.values())),
        "gprime_learn_ms": float(timings.get("gprime_learn", 0.0) or 0.0),
        "bottleneck_module": flow_bottleneck_module(frame),
    }


def export_session_csv(session_dir: PathLike, out_path: Optional[PathLike] = None) -> Path:
    """Export per-cycle scalar metrics from a session directory to CSV."""
    d = Path(session_dir)
    frames = load_session_frames(d)
    moments = build_moment_series(frames)
    rows: List[Dict[str, Any]] = []
    for f, m in zip(frames, moments):
        rows.append(_scalar_row(f, m))
    dest = Path(out_path) if out_path is not None else d / "scalars.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=SESSION_SCALAR_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return dest
