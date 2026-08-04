"""Offline session report — reuses Overview narrative helpers on JSONL ground truth."""
from __future__ import annotations

import json
import statistics
import sys
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np

from phca.monitoring.observability import ObservabilityFrame, normalize_observability_json
from phca.monitoring.cognitive_panels import (
    build_moment_series,
    classify_action_mechanism,
    count_moments,
    flow_near_bound_modules,
    goal_id_from_frame,
)
from phca.monitoring.session_anomalies import detect_session_anomalies
from phca.monitoring.action_explain import explain_anchor_bundle, infer_decision_reason
from phca.monitoring.belief_projection import BeliefProjection
from phca.monitoring.overview_narrative import (
    TREND_WINDOW,
    _OVERVIEW_PHASE_STEPS,
    _action_score_margin,
    _action_status_line,
    _flow_bottleneck_key,
    _flow_status_line,
    _phase_frame_is_grid,
    _phase_status_line,
    _overview_evidence_line,
    _overview_goal_id,
    _overview_goal_intent_line,
    _overview_moment_flags,
    _overview_new_events,
    _overview_outcome_line,
    _overview_phase_ms,
    _reacher_kinematics_from_obs,
)
from phca.monitoring.session_io import frame_from_json


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


def _anchor_cycle_ids(n: int, frames: List[ObservabilityFrame]) -> Dict[str, int]:
    """Map report anchor keys to frame-list indices.

    Keys ``0`` / ``99`` / ``999`` resolve by ``cycle_id`` when present; otherwise
    fall back to ``min(target, n-1)``. ``mid`` / ``last`` use list indices.
    """
    if n <= 0:
        return {}
    mid = n // 2
    by_cycle: Dict[int, int] = {}
    for i, f in enumerate(frames):
        cid = int(getattr(f, "cycle_id", i) or i)
        if cid not in by_cycle:
            by_cycle[cid] = i

    def _resolve(key: str, default: int) -> int:
        if key in ("mid", "last"):
            return min(default, n - 1)
        target = int(key)
        return by_cycle.get(target, min(target, n - 1))

    return {
        "0": _resolve("0", 0),
        "99": _resolve("99", 99),
        "999": _resolve("999", 999),
        "mid": _resolve("mid", mid),
        "last": _resolve("last", n - 1),
    }


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
        frames.append(frame_from_json(normalize_observability_json(json.loads(ln))))

    from phca.monitoring.multi_agent import (
        frames_for_agent,
        is_multi_agent_session,
        session_agent_ids,
    )
    if is_multi_agent_session(meta, frames=frames):
        aids = session_agent_ids(frames)
        agent_reports = {
            str(aid): _build_agent_report_from_frames(
                meta, frames_for_agent(frames, aid), agent_id=aid,
            )
            for aid in aids
        }
        primary = agent_reports.get("0") or agent_reports[str(aids[0])]
        return {**primary, "agents": agent_reports}
    return _build_agent_report_from_frames(meta, frames)


def _build_agent_report_from_frames(
    meta: Dict[str, Any],
    frames: List[ObservabilityFrame],
    *,
    agent_id: Optional[int] = None,
) -> Dict[str, Any]:
    """Aggregate Overview-aligned metrics for one agent's frame sequence."""
    n = len(frames)
    err_hist: Deque[float] = deque(maxlen=TREND_WINDOW)
    dist_hist: Deque[float] = deque(maxlen=TREND_WINDOW)
    prev_best_score: Optional[float] = None
    last_active_drive: Optional[int] = None
    prev_drive_id: Optional[int] = None
    prev_explored: Optional[bool] = None
    prev_m3_count: Optional[int] = None
    prev_m4_count: Optional[int] = None
    prev_active_drive: Optional[int] = None

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

    anchor_targets = _anchor_cycle_ids(n, frames)
    anchor_narratives: Dict[str, Dict[str, Any]] = {}
    anchor_flow_status: Dict[str, Dict[str, Any]] = {}
    anchor_action_status: Dict[str, Dict[str, Any]] = {}
    anchor_phase_status: Dict[str, Dict[str, Any]] = {}
    anchor_retention: Dict[str, Dict[str, Any]] = {}
    anchor_memory: Dict[str, Dict[str, Any]] = {}
    anchor_goals: Dict[str, Dict[str, Any]] = {}
    notable_cycles: List[Dict[str, Any]] = []

    m3_prune_count = 0
    m4_prune_count = 0
    envelope_over_count = 0
    cycles_with_m3 = 0
    cycles_with_m4 = 0
    active_drive_switch_count = 0

    module_time_totals: Dict[str, float] = {}
    bottleneck_counts: Dict[str, int] = {}
    violation_cycle_count = 0
    near_bound_cycle_count = 0

    cycles_with_scores = 0
    cycles_explore_empty_scores = 0
    cycles_with_chosen_idx = 0
    score_margins: List[float] = []
    all_best_scores: List[float] = []
    env_kind_counts: Dict[str, int] = {}
    mean_abs_pred_errors: List[float] = []
    pca_variances: List[float] = []
    max_pred_errors: List[float] = []
    pca_proj = BeliefProjection(window=16)
    mechanism_counts: Dict[str, int] = {
        "greedy_fallback": 0,
        "prediction": 0,
        "explore": 0,
        "stay": 0,
        "continuous": 0,
        "rbta_safe": 0,
        "other": 0,
    }
    selector_mode_counts: Dict[str, int] = {}
    decision_reason_counts: Dict[str, int] = {}
    anchor_explain: Dict[str, Dict[str, Any]] = {}
    terminate_cycle_count = 0
    interrupt_cycle_count = 0

    for idx, f in enumerate(frames):
        err_hist.append(float(getattr(f, "prediction_error", 0.0) or 0.0))
        all_errors.append(float(getattr(f, "prediction_error", 0.0) or 0.0))

        obs = f.obs_vector if f.obs_vector is not None else f.sanitized_state
        kin = _reacher_kinematics_from_obs(obs)
        if kin is not None:
            dist_hist.append(float(kin["dist"]))
            all_dists.append(float(kin["dist"]))
        else:
            ap = getattr(f, "agent_pos", None)
            gp = getattr(f, "goal_pos", None)
            if ap is not None and gp is not None and len(ap) >= 2 and len(gp) >= 2:
                manhattan = float(abs(int(ap[0]) - int(gp[0])) + abs(int(ap[1]) - int(gp[1])))
                dist_hist.append(manhattan)
                all_dists.append(manhattan)

        r = dict(getattr(f, "action_rationale", {}) or {})
        mechanism_counts[classify_action_mechanism(r)] += 1
        selector_mode = str(r.get("selector_mode") or "")
        if selector_mode:
            selector_mode_counts[selector_mode] = selector_mode_counts.get(selector_mode, 0) + 1
        dr = infer_decision_reason(r)
        decision_reason_counts[dr] = decision_reason_counts.get(dr, 0) + 1
        rbta_act = str(getattr(f, "rbta_action", "") or "").upper()
        if rbta_act == "TERMINATE":
            terminate_cycle_count += 1
        elif rbta_act == "INTERRUPT":
            interrupt_cycle_count += 1
        bs = r.get("best_score")
        cur_score = float(bs) if isinstance(bs, (int, float)) else None

        goal_id = _overview_goal_id(f)
        drive_change, last_active_drive = _track_drive_change(last_active_drive, goal_id)
        if drive_change is not None:
            drive_switch_count += 1

        flags = _overview_moment_flags(
            f, err_hist, prev_drive_id=prev_drive_id, prev_best_score=prev_best_score)
        if flags.get("decision_shift"):
            decision_shift_count += 1
        if cur_score is not None:
            prev_best_score = cur_score
        gid = goal_id_from_frame(f)
        if gid:
            prev_drive_id = gid
        explored = bool(flags.get("explored", False))
        if explored:
            explore_count += 1
        if flags.get("spike"):
            spike_count += 1
        if flags.get("learn_burst"):
            learn_burst_count += 1
        if getattr(f, "goal_reached", False):
            goal_reached_count += 1

        # Prefer explicit m3_count; legacy JSONL stuffed M3 size into episode_count.
        m3_count = int(getattr(f, "m3_count", 0) or 0)
        ep_count = int(getattr(f, "episode_count", 0) or 0)
        if m3_count <= 0 and ep_count > 0:
            m3_count = ep_count
        fact_count = int(getattr(f, "fact_count", 0) or 0)
        if prev_m3_count is not None and m3_count < prev_m3_count:
            m3_prune_count += 1
        if prev_m4_count is not None and fact_count < prev_m4_count:
            m4_prune_count += 1
        prev_m3_count = m3_count
        prev_m4_count = fact_count
        m3_cap = int(getattr(f, "m3_cap", 0) or 0)
        m4_cap = int(getattr(f, "m4_cap", 0) or 0)
        if (m3_cap and m3_count > m3_cap) or (m4_cap and fact_count > m4_cap):
            envelope_over_count += 1
        if getattr(f, "m3_recent", None) or getattr(f, "m3_top_error", None):
            cycles_with_m3 += 1
        if getattr(f, "m4_relevant", None) or getattr(f, "m4_top", None):
            cycles_with_m4 += 1
        ad = int(getattr(f, "active_drive_id", 0) or 0)
        if prev_active_drive is not None and ad and ad != prev_active_drive:
            active_drive_switch_count += 1
        if ad:
            prev_active_drive = ad

        explore_entered = explored and not bool(prev_explored)
        prev_explored = explored

        ek = str(getattr(f, "env_kind", "") or "unknown")
        env_kind_counts[ek] = env_kind_counts.get(ek, 0) + 1
        if getattr(f, "predicted_state", None) is not None:
            ref = f.obs_vector if f.obs_vector is not None else f.goal_ref
            if ref is not None:
                pred = np.asarray(f.predicted_state, dtype=np.float32).reshape(-1)
                ref_a = np.asarray(ref, dtype=np.float32).reshape(-1)
                dlen = min(len(pred), len(ref_a))
                if dlen > 0:
                    abs_err = np.abs(pred[:dlen] - ref_a[:dlen])
                    mean_abs_pred_errors.append(float(np.mean(abs_err)))
                    max_pred_errors.append(float(np.max(abs_err)))
        elif getattr(f, "prediction_error", None) is not None:
            mean_abs_pred_errors.append(float(f.prediction_error))
            max_pred_errors.append(float(f.prediction_error))

        is_grid = _phase_frame_is_grid(f)

        if not is_grid and f.obs_vector is not None:
            pca_proj.update(f)
            ve = pca_proj.variance_explained()
            if ve is not None:
                pca_variances.append(float(ve))

        timings = dict(getattr(f, "module_timings", {}) or {})
        for key, label in _OVERVIEW_PHASE_STEPS:
            ms = _overview_phase_ms(timings, key)
            phase_totals[label] += ms
            phase_grand_total += ms
        for mod, ms in timings.items():
            module_time_totals[str(mod)] = module_time_totals.get(str(mod), 0.0) + float(ms or 0.0)

        bn = _flow_bottleneck_key(f)
        if bn:
            bottleneck_counts[bn] = bottleneck_counts.get(bn, 0) + 1
        if int(getattr(f, "violations_count", 0) or 0) > 0 or bool(getattr(f, "rbta_violations", None)):
            violation_cycle_count += 1
        if flow_near_bound_modules(f, top_k=1):
            near_bound_cycle_count += 1

        scores = [float(x) for x in (getattr(f, "candidate_scores", []) or [])]
        if scores:
            cycles_with_scores += 1
            margin = _action_score_margin(scores)
            if margin is not None and not explored:
                score_margins.append(margin)
        elif explored:
            cycles_explore_empty_scores += 1
        bs = r.get("best_score")
        if isinstance(bs, (int, float)):
            all_best_scores.append(float(bs))
        chosen = -1
        ci = r.get("chosen_idx")
        if isinstance(ci, (int, float)):
            chosen = int(ci)
        elif scores:
            chosen = int(np.argmax(scores))
        if isinstance(r.get("chosen_idx"), (int, float)):
            cycles_with_chosen_idx += 1

        for anchor_key, target_idx in anchor_targets.items():
            if idx == target_idx and anchor_key not in anchor_narratives:
                anchor_narratives[anchor_key] = _narrative_bundle(f, err_hist, dist_hist)
            if idx == target_idx and anchor_key not in anchor_flow_status:
                anchor_flow_status[anchor_key] = {
                    "cycle_id": int(getattr(f, "cycle_id", idx) or idx),
                    "flow_status": _flow_status_line(f),
                }
            if idx == target_idx and anchor_key not in anchor_action_status:
                anchor_action_status[anchor_key] = {
                    "cycle_id": int(getattr(f, "cycle_id", idx) or idx),
                    "action_status": _action_status_line(f, scores, chosen),
                }
            if idx == target_idx and anchor_key not in anchor_explain:
                anchor_explain[anchor_key] = explain_anchor_bundle(f)
            if idx == target_idx and anchor_key not in anchor_phase_status:
                anchor_phase_status[anchor_key] = {
                    "cycle_id": int(getattr(f, "cycle_id", idx) or idx),
                    "phase_status": _phase_status_line(f, replay=True, is_grid=is_grid),
                }
            if idx == target_idx and anchor_key not in anchor_retention:
                anchor_retention[anchor_key] = {
                    "cycle_id": int(getattr(f, "cycle_id", idx) or idx),
                    "episode_count": ep_count,
                    "fact_count": fact_count,
                    "m3_cap": m3_cap,
                    "m4_cap": m4_cap,
                    "rss_bytes": float(getattr(f, "rss_bytes", 0.0) or 0.0),
                    "latency_ms": float(getattr(f, "latency_ms", 0.0) or 0.0),
                }
            if idx == target_idx and anchor_key not in anchor_memory:
                anchor_memory[anchor_key] = {
                    "cycle_id": int(getattr(f, "cycle_id", idx) or idx),
                    "m3_items": len(list(getattr(f, "m3_recent", None) or [])
                                    + list(getattr(f, "m3_top_error", None) or [])),
                    "m4_items": len(list(getattr(f, "m4_relevant", None) or [])
                                    + list(getattr(f, "m4_top", None) or [])),
                    "fact_count": fact_count,
                }
            if idx == target_idx and anchor_key not in anchor_goals:
                pareto = list(getattr(f, "pareto_front", None) or [])
                anchor_goals[anchor_key] = {
                    "cycle_id": int(getattr(f, "cycle_id", idx) or idx),
                    "active_drive_id": ad,
                    "pareto_size": len(pareto),
                    "goal_reached": bool(getattr(f, "goal_reached", False)),
                }

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

    mod_grand = sum(module_time_totals.values()) or 1.0
    module_time_share_pct = {
        k: round(100.0 * v / mod_grand, 2) for k, v in module_time_totals.items()
    }
    bottleneck_module = max(bottleneck_counts, key=bottleneck_counts.get) if bottleneck_counts else ""
    bs_early, bs_late = _slice_early_late(all_best_scores)
    dominant_env = max(env_kind_counts, key=env_kind_counts.get) if env_kind_counts else ""
    moment_series = build_moment_series(frames)
    moment_totals = count_moments(moment_series)
    anchor_moment_flags: Dict[str, Dict[str, Any]] = {}
    mechanism_pct: Dict[str, float] = {}
    selector_mode_pct: Dict[str, float] = {}
    if n:
        mechanism_pct = {
            k: round(100.0 * v / n, 2) for k, v in mechanism_counts.items()
        }
        selector_mode_pct = {
            k: round(100.0 * v / n, 2) for k, v in selector_mode_counts.items()
        }
    for anchor_key, target_idx in anchor_targets.items():
        if 0 <= target_idx < len(moment_series):
            m = moment_series[target_idx]
            anchor_moment_flags[anchor_key] = {
                "cycle_id": int(getattr(frames[target_idx], "cycle_id", target_idx) or target_idx),
                "spike": bool(m.get("spike")),
                "learn_burst": bool(m.get("learn_burst")),
                "decision_shift": bool(m.get("decision_shift")),
                "dominant_phase": m.get("dominant_phase", ""),
            }

    partial_report = {
        "cycles": n,
        "spike_count": spike_count,
        "drive_switch_count": drive_switch_count,
        "error_early_median": err_early_med,
        "error_late_median": err_late_med,
        "dist_early_median": dist_early_med,
        "dist_late_median": dist_late_med,
        "goals_metrics": {"active_drive_switch_count": active_drive_switch_count},
    }
    anomalies = detect_session_anomalies(frames, partial_report)
    report_classification = {
        "evidence_quality": (
            "forensic_only"
            if str(meta.get("status", "") or "").lower() in {"incomplete", "crashed", "corrupt", "empty"}
            else "trusted"
        ),
        "core_runtime_signals": list(anomalies.get("active") or []),
        "observability_limitations": [],
        "expected_limitations": [],
    }
    geometry_dominant_pct = (
        float(selector_mode_pct.get("task_lock_planner", 0.0))
        + float(selector_mode_pct.get("pure_geometry_ablation", 0.0))
        + float(selector_mode_pct.get("adaptive_geometry_fallback", 0.0))
    )
    if float(selector_mode_pct.get("task_lock_planner", 0.0)) >= 50.0:
        report_classification["expected_limitations"].append(
            "discrete_task_lock_planner_dominant"
        )
        report_classification["observability_limitations"].append(
            "goal_success_can_mask_prediction_path_quality"
        )
    # Default GridWorld path (D-156/D-161) uses pure_geometry_ablation; treat it
    # like task_lock for honesty — goal success can mask prediction-path quality.
    if float(selector_mode_pct.get("pure_geometry_ablation", 0.0)) >= 50.0:
        report_classification["expected_limitations"].append(
            "discrete_pure_geometry_ablation_dominant"
        )
        if "goal_success_can_mask_prediction_path_quality" not in report_classification[
            "observability_limitations"
        ]:
            report_classification["observability_limitations"].append(
                "goal_success_can_mask_prediction_path_quality"
            )
    if geometry_dominant_pct >= 50.0:
        report_classification["expected_limitations"].append(
            "geometry_dominated_action_selection"
        )
    if str(meta.get("status", "") or "").lower() == "incomplete":
        report_classification["observability_limitations"].append(
            "incomplete_session_trend_claims_are_limited"
        )

    return {
        "meta": dict(meta),
        "agent_id": agent_id if agent_id is not None else int(
            getattr(frames[0], "agent_id", 0) if frames else 0
        ),
        "cycles": n,
        "explore_ratio": round(explore_count / n, 4) if n else 0.0,
        "rbta_safe_count": int(mechanism_counts.get("rbta_safe", 0)),
        "rbta_safe_ratio": (
            round(mechanism_counts.get("rbta_safe", 0) / n, 4) if n else 0.0
        ),
        "terminate_cycle_count": terminate_cycle_count,
        "terminate_ratio": round(terminate_cycle_count / n, 4) if n else 0.0,
        "interrupt_cycle_count": interrupt_cycle_count,
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
        "flow_metrics": {
            "module_time_share_pct": module_time_share_pct,
            "violation_cycle_count": violation_cycle_count,
            "near_bound_cycle_count": near_bound_cycle_count,
            "bottleneck_module": bottleneck_module,
            "anchor_flow_status": anchor_flow_status,
            "anchor_flow_moments": anchor_moment_flags,
        },
        "action_metrics": {
            "cycles_with_scores": cycles_with_scores,
            "cycles_explore_empty_scores": cycles_explore_empty_scores,
            "cycles_with_chosen_idx": cycles_with_chosen_idx,
            "score_margin_median": _median(score_margins),
            "best_score_early_median": _median(bs_early),
            "best_score_late_median": _median(bs_late),
            "mechanism_histogram": mechanism_counts,
            "mechanism_pct": mechanism_pct,
            "selector_mode_histogram": selector_mode_counts,
            "selector_mode_pct": selector_mode_pct,
            "task_lock_planner_dominant": (
                float(selector_mode_pct.get("task_lock_planner", 0.0)) >= 50.0
            ),
            "pure_geometry_ablation_dominant": (
                float(selector_mode_pct.get("pure_geometry_ablation", 0.0)) >= 50.0
            ),
            "geometry_dominated_action_selection": (
                float(selector_mode_pct.get("task_lock_planner", 0.0))
                + float(selector_mode_pct.get("pure_geometry_ablation", 0.0))
                + float(selector_mode_pct.get("adaptive_geometry_fallback", 0.0))
                >= 50.0
            ),
            "anchor_action_status": anchor_action_status,
            "anchor_action_moments": anchor_moment_flags,
            "explain_metrics": {
                "decision_reason_counts": decision_reason_counts,
                "anchor_explain": anchor_explain,
            },
        },
        "phase_space_metrics": {
            "dominant_env_kind": dominant_env,
            "env_kind_counts": env_kind_counts,
            "mean_abs_pred_error_median": _median(mean_abs_pred_errors),
            "pca_variance_explained_median": _median(pca_variances),
            "max_pred_error_dim_median": _median(max_pred_errors),
            "anchor_phase_status": anchor_phase_status,
        },
        "retention_metrics": {
            "m3_prune_count": m3_prune_count,
            "m4_prune_count": m4_prune_count,
            "envelope_over_count": envelope_over_count,
            "anchor_retention": anchor_retention,
        },
        "memory_metrics": {
            "cycles_with_m3": cycles_with_m3,
            "cycles_with_m4": cycles_with_m4,
            "anchor_memory": anchor_memory,
        },
        "goals_metrics": {
            "active_drive_switch_count": active_drive_switch_count,
            "anchor_goals": anchor_goals,
        },
        "cognitive_panels_metrics": {
            **moment_totals,
            "anchor_moment_flags": anchor_moment_flags,
        },
        "anchor_narratives": anchor_narratives,
        "notable_cycles": notable_cycles,
        "anomalies": anomalies,
        "report_classification": report_classification,
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
    anom = report.get("anomalies") or {}
    aflags = anom.get("flags") or {}
    if aflags:
        print(
            f"  anomalies: spike={'FAIL' if aflags.get('spike') else 'PASS'}"
            f" drift={'FAIL' if aflags.get('drift') else 'PASS'}"
            f" leak={'FAIL' if aflags.get('leak') else 'PASS'}"
            f" goal={'FAIL' if aflags.get('goal_instability') else 'PASS'}",
            file=out,
        )
    flow_m = report.get("flow_metrics") or {}
    if flow_m:
        print(f"  flow: bottleneck={flow_m.get('bottleneck_module')}"
              f"  violations={flow_m.get('violation_cycle_count')}"
              f"  near_bound={flow_m.get('near_bound_cycle_count')}",
              file=out)
        for key in ("0", "mid", "last"):
            item = (flow_m.get("anchor_flow_status") or {}).get(key)
            if item:
                print(f"    flow [{key}] cycle {item.get('cycle_id')}: {item.get('flow_status')}",
                      file=out)
    action_m = report.get("action_metrics") or {}
    if action_m:
        print(f"  action: with_scores={action_m.get('cycles_with_scores')}"
              f"  explore_empty={action_m.get('cycles_explore_empty_scores')}"
              f"  margin_med={action_m.get('score_margin_median')}",
              file=out)
        explain_m = action_m.get("explain_metrics") or {}
        dr_counts = explain_m.get("decision_reason_counts") or {}
        if dr_counts:
            top = max(dr_counts.items(), key=lambda kv: kv[1])
            print(f"  explain: top reason {top[0]} ({top[1]} cycles)", file=out)
        for key in ("0", "mid", "last"):
            item = (action_m.get("anchor_action_status") or {}).get(key)
            if item:
                print(f"    action [{key}] cycle {item.get('cycle_id')}: {item.get('action_status')}",
                      file=out)


def load_session_report(session_dir: str | Path) -> Dict[str, Any]:
    """Load or build session_report.json for a session directory."""
    d = Path(session_dir)
    report_p = d / "session_report.json"
    if report_p.exists():
        return json.loads(report_p.read_text())
    meta_p = d / "meta.json"
    jsonl_p = d / "timeseries.jsonl"
    if not meta_p.exists() or not jsonl_p.exists():
        raise FileNotFoundError(f"not a valid session dir: {d}")
    meta = json.loads(meta_p.read_text())
    lines = [ln for ln in jsonl_p.read_text().splitlines() if ln.strip()]
    return build_session_report(meta, lines)


def _metric_delta(current: Any, baseline: Any) -> Optional[float]:
    if current is None or baseline is None:
        return None
    try:
        return float(current) - float(baseline)
    except (TypeError, ValueError):
        return None


def compare_session_reports(
    current: Dict[str, Any],
    baseline: Dict[str, Any],
    *,
    agent_id: int = 0,
) -> Dict[str, Any]:
    """Phase 12: structured deltas between two session reports."""
    cur_agents = current.get("agents") or {}
    base_agents = baseline.get("agents") or {}
    aid = str(agent_id)
    if aid in cur_agents and aid in base_agents:
        current = cur_agents[aid]
        baseline = base_agents[aid]
    elif aid in cur_agents or aid in base_agents:
        current = cur_agents.get(aid, current)
        baseline = base_agents.get(aid, baseline)

    cur_meta = current.get("meta") or {}
    base_meta = baseline.get("meta") or {}
    cur_flow = current.get("flow_metrics") or {}
    base_flow = baseline.get("flow_metrics") or {}
    cur_action = current.get("action_metrics") or {}
    base_action = baseline.get("action_metrics") or {}

    return {
        "current": {
            "session": cur_meta.get("session_id") or cur_meta.get("env"),
            "env": cur_meta.get("env"),
            "cycles": current.get("cycles"),
            "agent_id": current.get("agent_id", agent_id),
        },
        "baseline": {
            "session": base_meta.get("session_id") or base_meta.get("env"),
            "env": base_meta.get("env"),
            "cycles": baseline.get("cycles"),
            "agent_id": baseline.get("agent_id", agent_id),
        },
        "deltas": {
            "error_late_median": _metric_delta(
                current.get("error_late_median"), baseline.get("error_late_median")),
            "explore_ratio": _metric_delta(
                current.get("explore_ratio"), baseline.get("explore_ratio")),
            "goal_reached_count": _metric_delta(
                current.get("goal_reached_count"), baseline.get("goal_reached_count")),
            "spike_count": _metric_delta(
                current.get("spike_count"), baseline.get("spike_count")),
            "violation_cycle_count": _metric_delta(
                cur_flow.get("violation_cycle_count"),
                base_flow.get("violation_cycle_count")),
            "score_margin_median": _metric_delta(
                cur_action.get("score_margin_median"),
                base_action.get("score_margin_median")),
        },
        "regression_flags": {
            "error_late_worse": (
                _metric_delta(current.get("error_late_median"), baseline.get("error_late_median"))
                is not None
                and _metric_delta(current.get("error_late_median"), baseline.get("error_late_median")) > 0
            ),
            "violations_increased": (
                _metric_delta(
                    cur_flow.get("violation_cycle_count"),
                    base_flow.get("violation_cycle_count"),
                )
                is not None
                and _metric_delta(
                    cur_flow.get("violation_cycle_count"),
                    base_flow.get("violation_cycle_count"),
                ) > 0
            ),
        },
        "agent_id": agent_id,
    }


def compare_all_agents(
    current: Dict[str, Any],
    baseline: Dict[str, Any],
) -> Dict[str, Any]:
    """Per-agent deltas when both reports include ``agents`` buckets."""
    cur_agents = current.get("agents") or {}
    base_agents = baseline.get("agents") or {}
    if not cur_agents and not base_agents:
        return {"agents": {}, "summary": compare_session_reports(current, baseline)}
    aids = sorted(set(cur_agents) | set(base_agents), key=lambda x: int(x))
    per_agent: Dict[str, Any] = {}
    for aid in aids:
        cur = cur_agents.get(aid, current)
        base = base_agents.get(aid, baseline)
        per_agent[aid] = compare_session_reports(cur, base, agent_id=int(aid))
    return {"agents": per_agent}


def print_session_compare(compare: Dict[str, Any], *, stream=None) -> None:
    """Human-readable multi-session comparison summary."""
    out = stream or sys.stderr
    cur = compare.get("current") or {}
    base = compare.get("baseline") or {}
    print(
        f"Session compare — current={cur.get('session')} ({cur.get('env')}, {cur.get('cycles')} cyc)"
        f" vs baseline={base.get('session')} ({base.get('env')}, {base.get('cycles')} cyc)",
        file=out,
    )
    deltas = compare.get("deltas") or {}
    for key, delta in deltas.items():
        if delta is None:
            continue
        print(f"  Δ {key}: {delta:+.4g}", file=out)
    flags = compare.get("regression_flags") or {}
    if flags:
        active = [k for k, v in flags.items() if v]
        if active:
            print(f"  regression flags: {', '.join(active)}", file=out)
        else:
            print("  regression flags: none", file=out)
