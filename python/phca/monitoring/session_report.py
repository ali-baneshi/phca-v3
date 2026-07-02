"""Offline session report — reuses Overview narrative helpers on JSONL ground truth."""
from __future__ import annotations

import json
import statistics
import sys
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

from phca.monitoring.observability import ObservabilityFrame
from phca.monitoring.qt_dashboard import (
    TREND_WINDOW,
    _OVERVIEW_PHASE_STEPS,
    _apply_decision_shift,
    _overview_evidence_line,
    _overview_goal_id,
    _overview_goal_intent_line,
    _overview_moment_flags,
    _overview_new_events,
    _overview_outcome_line,
    _overview_phase_ms,
    _reacher_kinematics_from_obs,
)
from phca.monitoring.render import frame_from_json


def _track_drive_change(
    last_active: Optional[int],
    goal_id: Optional[int],
) -> Tuple[Optional[Tuple[int, int]], Optional[int]]:
    """Pure drive-transition helper (mirrors OverviewAgentView._track_drive_change)."""
    drive_change = None
    if goal_id and last_active is not None and goal_id != last_active:
        drive_change = (last_active, goal_id)
    new_last = goal_id if goal_id else last_active
    return drive_change, new_last


def _median(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return float(statistics.median(values))


def _slice_early_late(values: List[float], frac: float = 0.10) -> Tuple[List[float], List[float]]:
    n = len(values)
    if n == 0:
        return [], []
    k = max(1, int(n * frac))
    return values[:k], values[-k:]


def _anchor_cycle_ids(n: int) -> Dict[str, int]:
    if n <= 0:
        return {}
    mid = n // 2
    anchors = {"0": 0, "99": 99, "999": 999, "mid": mid, "last": n - 1}
    return {k: min(v, n - 1) for k, v in anchors.items()}


def _narrative_bundle(
    f: ObservabilityFrame,
    err_hist: Deque[float],
    dist_hist: Deque[float],
) -> Dict[str, str]:
    flags = _overview_moment_flags(f, err_hist)
    return {
        "cycle_id": int(getattr(f, "cycle_id", 0) or 0),
        "intent": _overview_goal_intent_line(f, flags),
        "evidence": _overview_evidence_line(f, flags, err_hist, dist_hist),
        "outcome": _overview_outcome_line(f, flags, err_hist, dist_hist),
    }


def build_session_report(meta: Dict[str, Any], lines: List[str]) -> Dict[str, Any]:
    """Walk JSONL lines and aggregate Overview-aligned session metrics."""
    frames: List[ObservabilityFrame] = []
    for ln in lines:
        ln = ln.strip()
        if not ln:
            continue
        frames.append(frame_from_json(json.loads(ln)))

    n = len(frames)
    err_hist: Deque[float] = deque(maxlen=TREND_WINDOW)
    dist_hist: Deque[float] = deque(maxlen=TREND_WINDOW)
    prev_best_score: Optional[float] = None
    last_active_drive: Optional[int] = None
    prev_explored: Optional[bool] = None

    explore_count = 0
    spike_count = 0
    learn_burst_count = 0
    decision_shift_count = 0
    drive_switch_count = 0
    goal_reached_count = 0

    all_errors: List[float] = []
    all_dists: List[float] = []
    phase_totals: Dict[str, float] = {label: 0.0 for _, label in _OVERVIEW_PHASE_STEPS}
    phase_grand_total = 0.0

    anchor_targets = _anchor_cycle_ids(n)
    anchor_narratives: Dict[str, Dict[str, Any]] = {}
    notable_cycles: List[Dict[str, Any]] = []

    for idx, f in enumerate(frames):
        err_hist.append(float(getattr(f, "prediction_error", 0.0) or 0.0))
        all_errors.append(float(getattr(f, "prediction_error", 0.0) or 0.0))

        obs = f.obs_vector if f.obs_vector is not None else f.sanitized_state
        kin = _reacher_kinematics_from_obs(obs)
        if kin is not None:
            dist_hist.append(float(kin["dist"]))
            all_dists.append(float(kin["dist"]))

        r = dict(getattr(f, "action_rationale", {}) or {})
        bs = r.get("best_score")
        cur_score = float(bs) if isinstance(bs, (int, float)) else None
        shifted = _apply_decision_shift(prev_best_score, cur_score)
        if shifted:
            decision_shift_count += 1
        if cur_score is not None:
            prev_best_score = cur_score
        r["decision_shift"] = shifted
        f.action_rationale = r

        goal_id = _overview_goal_id(f)
        drive_change, last_active_drive = _track_drive_change(last_active_drive, goal_id)
        if drive_change is not None:
            drive_switch_count += 1

        flags = _overview_moment_flags(f, err_hist)
        explored = bool(flags.get("explored", False))
        if explored:
            explore_count += 1
        if flags.get("spike"):
            spike_count += 1
        if flags.get("learn_burst"):
            learn_burst_count += 1
        if getattr(f, "goal_reached", False):
            goal_reached_count += 1

        explore_entered = explored and not bool(prev_explored)
        prev_explored = explored

        timings = dict(getattr(f, "module_timings", {}) or {})
        for key, label in _OVERVIEW_PHASE_STEPS:
            ms = _overview_phase_ms(timings, key)
            phase_totals[label] += ms
            phase_grand_total += ms

        for anchor_key, target_idx in anchor_targets.items():
            if idx == target_idx and anchor_key not in anchor_narratives:
                anchor_narratives[anchor_key] = _narrative_bundle(f, err_hist, dist_hist)

        notable = False
        reasons: List[str] = []
        if flags.get("spike") and flags.get("learn_burst"):
            notable = True
            reasons.append("spike+learn")
        if drive_change is not None:
            notable = True
            reasons.append("drive_change")
        if notable and len(notable_cycles) < 20:
            events = _overview_new_events(f, flags, drive_change, explore_entered=explore_entered)
            notable_cycles.append({
                "cycle_id": int(getattr(f, "cycle_id", idx) or idx),
                "reasons": reasons,
                "events": events,
            })

    err_early, err_late = _slice_early_late(all_errors)
    dist_early, dist_late = _slice_early_late(all_dists)
    err_early_med = _median(err_early)
    err_late_med = _median(err_late)
    dist_early_med = _median(dist_early)
    dist_late_med = _median(dist_late)

    phase_budget_pct: Dict[str, float] = {}
    if phase_grand_total > 0:
        for label, total in phase_totals.items():
            phase_budget_pct[label] = round(100.0 * total / phase_grand_total, 2)

    return {
        "meta": dict(meta),
        "cycles": n,
        "explore_ratio": round(explore_count / n, 4) if n else 0.0,
        "spike_count": spike_count,
        "learn_burst_count": learn_burst_count,
        "decision_shift_count": decision_shift_count,
        "drive_switch_count": drive_switch_count,
        "goal_reached_count": goal_reached_count,
        "error_early_median": err_early_med,
        "error_late_median": err_late_med,
        "error_improved": (
            err_early_med is not None and err_late_med is not None
            and err_late_med < err_early_med
        ),
        "dist_early_median": dist_early_med,
        "dist_late_median": dist_late_med,
        "dist_improved": (
            dist_early_med is not None and dist_late_med is not None
            and dist_late_med < dist_early_med
        ),
        "phase_budget_pct": phase_budget_pct,
        "anchor_narratives": anchor_narratives,
        "notable_cycles": notable_cycles,
    }


def write_session_report(session_dir: str | Path) -> Dict[str, Any]:
    """Build and write session_report.json into a session directory."""
    d = Path(session_dir)
    meta_p = d / "meta.json"
    jsonl_p = d / "timeseries.jsonl"
    if not meta_p.exists() or not jsonl_p.exists():
        raise FileNotFoundError(f"not a valid session dir: {d}")
    meta = json.loads(meta_p.read_text())
    lines = [ln for ln in jsonl_p.read_text().splitlines() if ln.strip()]
    report = build_session_report(meta, lines)
    out_p = d / "session_report.json"
    out_p.write_text(json.dumps(report, indent=2))
    return report


def print_report_summary(report: Dict[str, Any], *, stream=None) -> None:
    """Human-readable stderr summary of a session report."""
    out = stream or sys.stderr
    meta = report.get("meta", {})
    print(f"Session report — {meta.get('env', '?')} · {report.get('cycles', 0)} cycles",
          file=out)
    print(f"  explore_ratio={report.get('explore_ratio')}  spikes={report.get('spike_count')}"
          f"  learn_bursts={report.get('learn_burst_count')}"
          f"  decision_shifts={report.get('decision_shift_count')}"
          f"  drive_switches={report.get('drive_switch_count')}"
          f"  goals={report.get('goal_reached_count')}",
          file=out)
    print(f"  error median early/late: {report.get('error_early_median')}"
          f" → {report.get('error_late_median')}"
          f" ({'improved' if report.get('error_improved') else 'flat/worse'})",
          file=out)
    dist_e, dist_l = report.get("dist_early_median"), report.get("dist_late_median")
    if dist_e is not None or dist_l is not None:
        print(f"  dist median early/late: {dist_e} → {dist_l}"
              f" ({'improved' if report.get('dist_improved') else 'flat/worse'})",
              file=out)
    budget = report.get("phase_budget_pct") or {}
    if budget:
        top = sorted(budget.items(), key=lambda kv: kv[1], reverse=True)[:3]
        print("  phase budget: " + ", ".join(f"{k}={v}%" for k, v in top), file=out)
    anchors = report.get("anchor_narratives") or {}
    for key in ("0", "99", "999", "mid", "last"):
        bundle = anchors.get(key)
        if not bundle:
            continue
        print(f"  anchor [{key}] cycle {bundle.get('cycle_id')}:", file=out)
        print(f"    {bundle.get('intent')}", file=out)
        print(f"    {bundle.get('evidence')}", file=out)
        print(f"    {bundle.get('outcome')}", file=out)
    notable = report.get("notable_cycles") or []
    if notable:
        print(f"  notable cycles ({len(notable)}):", file=out)
        for item in notable[:5]:
            print(f"    cycle {item.get('cycle_id')}: {', '.join(item.get('reasons', []))}",
                  file=out)
