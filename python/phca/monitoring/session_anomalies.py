"""Session-level anomaly detection (Phase 13): spike, drift, leak, goal instability."""
from __future__ import annotations

import sys
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

from phca.monitoring.cognitive_panels import build_moment_series, goal_id_from_frame
from phca.monitoring.observability import ObservabilityFrame
from phca.monitoring.retention_slope import compute_rss_slopes, rss_leak_flagged

AnomalyKind = Literal["spike", "drift", "leak", "goal_instability"]
Severity = Literal["info", "warn", "critical"]

ANOMALY_KINDS: Tuple[AnomalyKind, ...] = (
    "spike",
    "drift",
    "leak",
    "goal_instability",
)

ANOMALY_SEVERITY: Dict[AnomalyKind, Severity] = {
    "spike": "info",
    "drift": "warn",
    "leak": "critical",
    "goal_instability": "warn",
}

DEFAULT_THRESHOLDS: Dict[str, Any] = {
    "spike_rate": 0.20,
    "spike_rate_min_cycles": 30,
    "spike_count_min_cycles": 50,
    "spike_count_min": 10,
    "drift_error_ratio": 1.20,
    "drift_error_abs_delta": 0.5,
    "drift_dist_ratio": 1.15,
    "drift_min_cycles": 30,
    "leak_min_rss_samples": 50,
    "leak_min_cycles": 200,
    "goal_switch_rate": 0.25,
    "goal_rolling_window": 20,
    "goal_rolling_min_switches": 3,
    "cycle_markers_cap": 50,
}

_CYCLE_MARKER_CAP = 50


def _thresholds(thresholds: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    out = dict(DEFAULT_THRESHOLDS)
    if thresholds:
        out.update(thresholds)
    return out


def _median(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    s = sorted(float(v) for v in values)
    n = len(s)
    mid = n // 2
    if n % 2:
        return s[mid]
    return (s[mid - 1] + s[mid]) / 2.0


def _slice_early_late(values: Sequence[float], frac: float = 0.10) -> Tuple[List[float], List[float]]:
    n = len(values)
    if n == 0:
        return [], []
    k = max(1, int(n * frac))
    return list(values[:k]), list(values[-k:])


def _worsening_drift(
    early: Optional[float],
    late: Optional[float],
    *,
    ratio: float,
    abs_delta: float,
) -> bool:
    if early is None or late is None:
        return False
    if early <= 0:
        return late > abs_delta
    return late > early * ratio and (late - early) > abs_delta


def _drive_switch_indices(frames: Sequence[ObservabilityFrame]) -> List[int]:
    """Cycle ids where goal/drive id changes vs previous frame."""
    out: List[int] = []
    prev_gid: Optional[int] = None
    for idx, f in enumerate(frames):
        gid = goal_id_from_frame(f)
        if gid and prev_gid and gid != prev_gid:
            out.append(int(getattr(f, "cycle_id", idx) or idx))
        if gid:
            prev_gid = gid
    return out


def _rolling_goal_instability_at_index(
    switch_indices: Sequence[int],
    frame_idx: int,
    frames: Sequence[ObservabilityFrame],
    th: Dict[str, Any],
) -> bool:
    window = int(th["goal_rolling_window"])
    min_sw = int(th["goal_rolling_min_switches"])
    cur_cid = int(getattr(frames[frame_idx], "cycle_id", frame_idx) or frame_idx)
    lo = cur_cid - window + 1
    count = sum(1 for cid in switch_indices if lo <= cid <= cur_cid)
    return count >= min_sw


def detect_cycle_anomalies(
    frames: Sequence[ObservabilityFrame],
    moment_series: Sequence[Dict[str, Any]],
    *,
    thresholds: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Per-cycle anomaly flags aligned with cognitive moment series."""
    th = _thresholds(thresholds)
    switch_indices = _drive_switch_indices(frames)
    markers: List[Dict[str, Any]] = []

    for idx, (f, m) in enumerate(zip(frames, moment_series)):
        flags: List[AnomalyKind] = []
        if m.get("spike"):
            flags.append("spike")
        if _rolling_goal_instability_at_index(switch_indices, idx, frames, th):
            flags.append("goal_instability")
        if flags:
            markers.append({
                "cycle_id": int(getattr(f, "cycle_id", idx) or idx),
                "flags": flags,
            })
    return markers


def _session_spike_flag(
    spike_count: int,
    cycles: int,
    th: Dict[str, Any],
) -> bool:
    if cycles <= 0:
        return False
    rate = spike_count / cycles
    min_rate_cycles = int(th.get("spike_rate_min_cycles", 0))
    if cycles >= min_rate_cycles and rate > float(th["spike_rate"]):
        return True
    if cycles >= int(th["spike_count_min_cycles"]) and spike_count >= int(th["spike_count_min"]):
        return True
    return False


def _session_drift_flag(
    *,
    error_early: Optional[float],
    error_late: Optional[float],
    dist_early: Optional[float],
    dist_late: Optional[float],
    th: Dict[str, Any],
) -> bool:
    if _worsening_drift(
        error_early,
        error_late,
        ratio=float(th["drift_error_ratio"]),
        abs_delta=float(th["drift_error_abs_delta"]),
    ):
        return True
    if dist_early is not None or dist_late is not None:
        return _worsening_drift(
            dist_early,
            dist_late,
            ratio=float(th["drift_dist_ratio"]),
            abs_delta=float(th["drift_error_abs_delta"]),
        )
    return False


def _rolling_goal_instability(
    frames: Sequence[ObservabilityFrame],
    th: Dict[str, Any],
) -> bool:
    switch_indices = _drive_switch_indices(frames)
    for idx in range(len(frames)):
        if _rolling_goal_instability_at_index(switch_indices, idx, frames, th):
            return True
    return False


def _session_leak_flag(
    frames: Sequence[ObservabilityFrame],
    cycles: int,
    th: Dict[str, Any],
) -> Tuple[bool, Optional[float], Optional[float], Optional[str]]:
    cycle_ids: List[int] = []
    rss_vals: List[float] = []
    for idx, f in enumerate(frames):
        rss = float(getattr(f, "rss_bytes", 0.0) or 0.0)
        if rss > 0:
            cycle_ids.append(int(getattr(f, "cycle_id", idx) or idx))
            rss_vals.append(rss)
    sample_count = len(cycle_ids)
    if sample_count < 2:
        return False, None, None, None
    slopes = compute_rss_slopes(cycle_ids, rss_vals, total_cycles=cycles)
    late_slope = float(slopes["late_slope"])
    min_samples = int(th["leak_min_rss_samples"])
    min_cycles = int(th["leak_min_cycles"])
    if cycles <= min_cycles or sample_count < min_samples:
        # A short soak cannot distinguish normal allocator/fill behaviour from a
        # sustained leak. Preserve slopes for inspection without raising CRITICAL.
        return False, late_slope, float(slopes["full_slope"]), "insufficient_short_soak"
    flagged = rss_leak_flagged(late_slope, cycles)
    return flagged, late_slope, float(slopes["full_slope"]), str(slopes["leak_gate_mode"])


def detect_session_anomalies(
    frames: Sequence[ObservabilityFrame],
    report: Optional[Dict[str, Any]] = None,
    *,
    thresholds: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Detect session-level anomalies; optionally reuse report aggregates."""
    th = _thresholds(thresholds)
    n = len(frames)
    moment_series = build_moment_series(list(frames))

    if report is not None:
        spike_count = int(report.get("spike_count", 0) or 0)
        error_early = report.get("error_early_median")
        error_late = report.get("error_late_median")
        dist_early = report.get("dist_early_median")
        dist_late = report.get("dist_late_median")
        drive_switch_count = int(report.get("drive_switch_count", 0) or 0)
        goals_m = report.get("goals_metrics") or {}
        active_drive_switch_count = int(goals_m.get("active_drive_switch_count", 0) or 0)
    else:
        spike_count = sum(1 for m in moment_series if m.get("spike"))
        all_errors = [
            float(getattr(f, "prediction_error", 0.0) or 0.0) for f in frames
        ]
        err_early, err_late = _slice_early_late(all_errors)
        error_early = _median(err_early)
        error_late = _median(err_late)
        dist_early = dist_late = None
        drive_switch_count = 0
        active_drive_switch_count = 0
        prev_gid: Optional[int] = None
        prev_active: Optional[int] = None
        for f in frames:
            gid = goal_id_from_frame(f)
            if gid and prev_gid and gid != prev_gid:
                drive_switch_count += 1
            if gid:
                prev_gid = gid
            ad = int(getattr(f, "active_drive_id", 0) or 0)
            if prev_active is not None and ad and ad != prev_active:
                active_drive_switch_count += 1
            if ad:
                prev_active = ad

    cycles = int(report.get("cycles", n) if report else n)
    spike_rate = round(spike_count / cycles, 4) if cycles else 0.0
    # Single switch source — prefer MDIM active_drive; do not sum overlapping counters.
    switch_total = (
        active_drive_switch_count
        if active_drive_switch_count > 0
        else drive_switch_count
    )
    drive_switch_rate = round(switch_total / cycles, 4) if cycles else 0.0

    leak_flag, late_slope, full_slope, leak_mode = _session_leak_flag(frames, cycles, th)
    goal_rolling = _rolling_goal_instability(frames, th)
    drift_evaluated = cycles >= int(th["drift_min_cycles"])

    flags: Dict[str, bool] = {
        "spike": _session_spike_flag(spike_count, cycles, th),
        "drift": (
            drift_evaluated
            and _session_drift_flag(
                error_early=error_early if isinstance(error_early, (int, float)) else None,
                error_late=error_late if isinstance(error_late, (int, float)) else None,
                dist_early=dist_early if isinstance(dist_early, (int, float)) else None,
                dist_late=dist_late if isinstance(dist_late, (int, float)) else None,
                th=th,
            )
        ),
        "leak": leak_flag,
        "goal_instability": (
            goal_rolling
            or (cycles > 0 and drive_switch_rate > float(th["goal_switch_rate"]))
        ),
    }

    cycle_markers = detect_cycle_anomalies(frames, moment_series, thresholds=th)
    cap = int(th.get("cycle_markers_cap", _CYCLE_MARKER_CAP))
    if len(cycle_markers) > cap:
        cycle_markers = cycle_markers[:cap]

    active = [k for k in ANOMALY_KINDS if flags.get(k)]

    return {
        "flags": flags,
        "active": active,
        "severity": dict(ANOMALY_SEVERITY),
        "metrics": {
            "spike_count": spike_count,
            "spike_rate": spike_rate,
            "error_early_median": error_early,
            "error_late_median": error_late,
            "dist_early_median": dist_early,
            "dist_late_median": dist_late,
            "rss_full_slope_bytes_per_cycle": full_slope,
            "rss_late_slope_bytes_per_cycle": late_slope,
            "rss_leak_gate_mode": leak_mode,
            "drive_switch_count": drive_switch_count,
            "active_drive_switch_count": active_drive_switch_count,
            "drive_switch_rate": drive_switch_rate,
            "goal_rolling_instability": bool(goal_rolling),
            "drift_evaluated": drift_evaluated,
        },
        "thresholds": dict(th),
        "cycle_markers": cycle_markers,
    }


def anomalies_from_report(report: Dict[str, Any]) -> Dict[str, Any]:
    """Re-evaluate anomaly flags from an existing session report (no frames).

    Leak detection requires ``anomalies.metrics.rss_late_slope_bytes_per_cycle``
    in the report; it cannot be recomputed without the original frame sequence.
    Goal instability can use ``goal_rolling_instability`` when present.
    """
    th = _thresholds(None)
    cycles = int(report.get("cycles", 0) or 0)
    spike_count = int(report.get("spike_count", 0) or 0)
    goals_m = report.get("goals_metrics") or {}
    active_drive_switch_count = int(goals_m.get("active_drive_switch_count", 0) or 0)
    drive_switch_count = int(report.get("drive_switch_count", 0) or 0)
    # Single switch source — prefer MDIM active_drive; do not sum overlapping counters.
    switch_total = (
        active_drive_switch_count
        if active_drive_switch_count > 0
        else drive_switch_count
    )
    drive_switch_rate = round(switch_total / cycles, 4) if cycles else 0.0

    error_early = report.get("error_early_median")
    error_late = report.get("error_late_median")
    dist_early = report.get("dist_early_median")
    dist_late = report.get("dist_late_median")

    metrics_block = (report.get("anomalies") or {}).get("metrics") or {}
    late_slope = metrics_block.get("rss_late_slope_bytes_per_cycle")
    goal_rolling = metrics_block.get("goal_rolling_instability")
    leak_flag = False
    if isinstance(late_slope, (int, float)):
        leak_flag = rss_leak_flagged(float(late_slope), cycles)

    drift_evaluated = cycles >= int(th["drift_min_cycles"])
    flags: Dict[str, bool] = {
        "spike": _session_spike_flag(spike_count, cycles, th),
        "drift": (
            drift_evaluated
            and _session_drift_flag(
                error_early=error_early if isinstance(error_early, (int, float)) else None,
                error_late=error_late if isinstance(error_late, (int, float)) else None,
                dist_early=dist_early if isinstance(dist_early, (int, float)) else None,
                dist_late=dist_late if isinstance(dist_late, (int, float)) else None,
                th=th,
            )
        ),
        "leak": leak_flag,
        "goal_instability": (
            bool(goal_rolling)
            if isinstance(goal_rolling, bool)
            else (cycles > 0 and drive_switch_rate > float(th["goal_switch_rate"]))
        ),
    }
    active = [k for k in ANOMALY_KINDS if flags.get(k)]
    return {
        "flags": flags,
        "active": active,
        "severity": dict(ANOMALY_SEVERITY),
        "metrics": {
            "spike_count": spike_count,
            "spike_rate": round(spike_count / cycles, 4) if cycles else 0.0,
            "error_early_median": error_early,
            "error_late_median": error_late,
            "dist_early_median": dist_early,
            "dist_late_median": dist_late,
            "rss_late_slope_bytes_per_cycle": late_slope,
            "drive_switch_count": drive_switch_count,
            "active_drive_switch_count": active_drive_switch_count,
            "drive_switch_rate": drive_switch_rate,
            "goal_rolling_instability": (
                metrics_block.get("goal_rolling_instability")
                if isinstance(metrics_block.get("goal_rolling_instability"), bool)
                else None
            ),
            "drift_evaluated": drift_evaluated,
        },
        "thresholds": dict(th),
        "cycle_markers": (report.get("anomalies") or {}).get("cycle_markers") or [],
    }


def anomaly_strict_fail(result: Dict[str, Any]) -> bool:
    """True when any critical-severity anomaly flag is active."""
    flags = result.get("flags") or {}
    severity = result.get("severity") or ANOMALY_SEVERITY
    for kind, active in flags.items():
        if active and severity.get(kind) == "critical":
            return True
    return False


def anomaly_overall_pass(result: Dict[str, Any]) -> bool:
    """True when no anomaly flags are active."""
    return not (result.get("active") or [])


def print_anomaly_summary(result: Dict[str, Any], *, stream=None) -> None:
    """Human-readable PASS/FAIL per anomaly kind."""
    out = stream or sys.stderr
    flags = result.get("flags") or {}
    metrics = result.get("metrics") or {}
    parts = []
    for kind in ANOMALY_KINDS:
        if kind == "drift" and not metrics.get("drift_evaluated", True):
            parts.append(f"{kind}=SKIP (short session)")
            continue
        status = "FAIL" if flags.get(kind) else "PASS"
        parts.append(f"{kind}={status}")
    print(f"  anomalies  : {' '.join(parts)}", file=out)
    overall = "PASS" if anomaly_overall_pass(result) else "FAIL"
    print(f"  Anomaly overall: {overall}", file=out)
