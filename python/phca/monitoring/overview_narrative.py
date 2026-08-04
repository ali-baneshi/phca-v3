"""Qt-free Overview / Flow / Action / Phase narrative helpers for reports and UI."""
from __future__ import annotations

import math
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np

from phca.monitoring.belief_projection import BeliefProjection
from phca.monitoring.cognitive_panels import (
    EXECUTION_PHASE_STEPS,
    FLOW_ALL_MODULES,
    PIPELINE_LABEL,
    PIPELINE_MODULES,
    RBTA_TO_FLOW,
    action_status_extras,
    cognitive_moment,
    flow_status_extras,
    frame_is_geometry_control,
    geometry_score_label,
    learn_phase_label,
)
from phca.monitoring.observability import ObservabilityFrame

TREND_WINDOW = 200
_OVERVIEW_PHASE_STEPS = EXECUTION_PHASE_STEPS

DRIVE_SHORT = {1: "D1", 2: "D2", 3: "D3", 4: "D4", 5: "D5", 6: "D6"}

_REACHER_L1 = 0.42
_REACHER_L2 = 0.38


def _drive_short(did: int) -> str:
    return DRIVE_SHORT.get(did, f"D{did}")


def _flow_pipe_ms(f: ObservabilityFrame, timings: Optional[Dict[str, Any]] = None) -> float:
    if timings is None:
        timings = dict(getattr(f, "module_timings", {}) or {})
    return sum(float(timings.get(m, 0.0) or 0.0) for m in PIPELINE_MODULES)


def _flow_bottleneck_key(f: ObservabilityFrame, timings: Optional[Dict[str, Any]] = None) -> str:
    if timings is None:
        timings = dict(getattr(f, "module_timings", {}) or {})
    if not timings:
        return ""
    best = max(FLOW_ALL_MODULES, key=lambda m: float(timings.get(m, 0.0) or 0.0))
    return best if float(timings.get(best, 0.0) or 0.0) > 0 else ""


def _flow_status_line(f: ObservabilityFrame, *, active_idx: Optional[int] = None) -> str:
    timings = dict(getattr(f, "module_timings", {}) or {})
    bn_key = _flow_bottleneck_key(f, timings=timings)
    bn_lbl = PIPELINE_LABEL.get(bn_key, bn_key) if bn_key else "—"
    pipe_ms = _flow_pipe_ms(f, timings=timings)
    vcount = int(getattr(f, "violations_count", 0) or 0)
    viol_mods: set = set()
    for v in getattr(f, "rbta_violations", []) or []:
        mid = RBTA_TO_FLOW.get(v.get("module_id", v.get("module", "")),
                               v.get("module_id", v.get("module", "")))
        if mid:
            viol_mods.add(PIPELINE_LABEL.get(mid, mid))
    if active_idx is not None and 0 <= active_idx < len(PIPELINE_MODULES):
        active_lbl = PIPELINE_LABEL.get(PIPELINE_MODULES[active_idx], PIPELINE_MODULES[active_idx])
    elif timings:
        dom = max(PIPELINE_MODULES, key=lambda m: float(timings.get(m, 0.0) or 0.0))
        active_lbl = PIPELINE_LABEL.get(dom, dom)
    else:
        active_lbl = "—"
    viol_s = f"{vcount}V" if vcount else "OK"
    if viol_mods:
        viol_s += f" ({','.join(sorted(viol_mods))})"
    base = (f"Flow: bottleneck={bn_lbl} · Σpipe={pipe_ms:.1f}ms · "
            f"violations={viol_s} · Δ={active_lbl}")
    extras = flow_status_extras(f)
    return f"{base} · {extras}" if extras else base


def _action_score_margin(scores: List[float]) -> Optional[float]:
    if len(scores) < 2:
        return None
    ordered = sorted(scores, reverse=True)
    return float(ordered[0] - ordered[1])


def _action_status_line(
    f: ObservabilityFrame,
    scores: List[float],
    chosen: int,
    *,
    replay: bool = False,
    review: bool = False,
    prefix_len: int = 0,
    moment: Optional[Dict[str, Any]] = None,
) -> str:
    r = f.action_rationale or {}
    mode = "EXPLORE" if r.get("explored") else "EXPLOIT"
    eps = r.get("eps")
    eps_s = f"{float(eps):.3f}" if isinstance(eps, (int, float)) else "—"
    k = r.get("k_candidates")
    k_s = str(int(k)) if isinstance(k, (int, float)) else "—"
    bs = r.get("best_score")
    bs_s = f"{float(bs):.3f}" if isinstance(bs, (int, float)) else "—"
    margin = _action_score_margin(scores)
    margin_s = f"{margin:+.3f}" if margin is not None else "—"
    rollouts = list(getattr(f, "candidate_rollouts", []) or [])
    score_label = geometry_score_label(f)
    base = (f"Action: {mode} · ε={eps_s} · k={k_s} · {score_label}={bs_s} · "
            f"Δ2nd={margin_s} · rollouts={len(rollouts)}")
    extras = action_status_extras(
        f, scores, chosen, replay=replay, review=review,
        prefix_len=prefix_len, moment=moment)
    return f"{base} · {extras}" if extras else base


def _phase_frame_is_grid(f: ObservabilityFrame) -> bool:
    return f.grid is not None and f.agent_pos is not None


def _phase_status_line(
    f: ObservabilityFrame,
    *,
    replay: bool = False,
    review: bool = False,
    prefix_len: int = 0,
    is_grid: bool = False,
    proj: Optional[BeliefProjection] = None,
    show_radar: bool = True,
) -> str:
    if is_grid:
        mode = "grid"
    elif proj is not None and getattr(proj, "_raw2d", False):
        mode = "raw-2D"
    else:
        mode = "PCA"
    pe = float(getattr(f, "prediction_error", 0.0) or 0.0)
    gk = getattr(f, "gprime_kind", None) or "g′"
    parts = [f"Phase: {mode}", f"err={pe:.2f}", gk]
    if review and prefix_len > 0:
        parts.append(f"cycle={int(f.cycle_id)} · prefix={prefix_len}")
    if proj is not None and not is_grid:
        ve = proj.variance_explained()
        if ve is not None:
            parts.append(f"PCA={ve:.0f}%")
    rollouts = list(getattr(f, "candidate_rollouts", []) or [])
    if replay:
        parts.append("rollouts=0(replay)")
        if f.goal_target is None:
            parts.append("goal=—(replay)")
        if f.sanitized_state is None:
            parts.append("anchor=obs_vector")
    elif rollouts:
        parts.append(f"rollouts={len(rollouts)}")
    did = int(getattr(f, "active_drive_id", 0) or 0)
    if did:
        parts.append(f"drive={_drive_short(did)}")
    if not show_radar:
        lv = list(getattr(f, "drive_levels", []) or [])
        if lv:
            levels_str = ",".join(f"{v:.2f}" for v in lv[:6])
            parts.append(f"drives=[{levels_str}]")
    return " · ".join(parts)


def _reacher_kinematics_from_obs(obs: Any) -> Optional[dict]:
    if obs is None:
        return None
    arr = np.asarray(obs, dtype=np.float32).reshape(-1)
    if arr.size < 4 or not np.all(np.isfinite(arr)):
        return None
    a0 = math.atan2(float(arr[1]), float(arr[0]))
    a1 = math.atan2(float(arr[3]), float(arr[2]))
    ex = _REACHER_L1 * math.cos(a0)
    ey = _REACHER_L1 * math.sin(a0)
    fx = ex + _REACHER_L2 * math.cos(a0 + a1)
    fy = ey + _REACHER_L2 * math.sin(a0 + a1)
    rel_x = float(arr[-2]) if arr.size >= 2 else 0.0
    rel_y = float(arr[-1]) if arr.size >= 1 else 0.0
    tx = fx - rel_x
    ty = fy - rel_y
    return {
        "ex": ex, "ey": ey, "fx": fx, "fy": fy, "tx": tx, "ty": ty,
        "dist": math.hypot(rel_x, rel_y),
    }


def _overview_goal_id(f: ObservabilityFrame) -> Optional[int]:
    r = f.action_rationale or {}
    gid = r.get("goal_id")
    if gid is not None:
        try:
            return int(gid)
        except (TypeError, ValueError):
            pass
    active = int(getattr(f, "active_drive_id", 0) or 0)
    return active if active > 0 else None


def _overview_spike(err_hist: Deque[float], cur_err: float,
                    env_kind: str = "") -> bool:
    errs = list(err_hist)
    prev_err = float(errs[-2]) if len(errs) >= 2 else None
    if prev_err is None or prev_err <= 1e-6:
        return False
    kind = (env_kind or "").lower()
    if kind == "mujoco_rgb":
        if cur_err > 3.0 * prev_err:
            return True
        return abs(cur_err - prev_err) > 5.0
    return cur_err > 2.0 * prev_err


def _overview_moment_flags(
    f: ObservabilityFrame,
    err_hist: Deque[float],
    *,
    moment: Optional[Dict[str, Any]] = None,
    prev_drive_id: Optional[int] = None,
    prev_best_score: Optional[float] = None,
) -> Dict[str, Any]:
    if moment is None:
        moment = cognitive_moment(
            f, err_hist, prev_drive_id=prev_drive_id, prev_best_score=prev_best_score)
    r = f.action_rationale or {}
    score = r.get("best_score")
    score_v = float(score) if isinstance(score, (int, float)) else None
    return {
        **moment,
        "score": score_v,
        "explored": bool(r.get("explored", False)),
    }


def _phase_ms(timings: Dict[str, Any], key) -> float:
    if isinstance(key, tuple):
        return sum(float(timings.get(k, 0.0) or 0.0) for k in key)
    return float(timings.get(key, 0.0) or 0.0)


def _overview_phase_ms(timings: Dict[str, Any], key) -> float:
    return _phase_ms(timings, key)


def _overview_goal_intent_line(f: ObservabilityFrame, flags: Dict[str, Any]) -> str:
    r = f.action_rationale or {}
    mode = "EXPLORE" if flags.get("explored") else "EXPLOIT"
    gid = _overview_goal_id(f)
    goal_s = _drive_short(gid) if gid else "—"
    score_s = f"{flags['score']:.2f}" if flags["score"] is not None else "—"
    score_label = geometry_score_label(f)
    selector = str(r.get("selector_mode") or "")
    if flags.get("explored") or r.get("at_goal_explore"):
        why = "sampling alternatives"
    elif "geometry" in selector or r.get("greedy_fallback") or str(
        r.get("decision_reason") or ""
    ).startswith("ablation_pure_geometry"):
        why = "geometry planner"
    else:
        why = "best-score selection"
    if r.get("note"):
        why = str(r.get("note"))
    extras: List[str] = []
    eps = r.get("eps")
    if isinstance(eps, (int, float)):
        extras.append(f"eps={float(eps):.2f}")
    k_cand = r.get("k_candidates")
    if isinstance(k_cand, (int, float)):
        extras.append(f"k={int(k_cand)}")
    extra_s = (" · " + " · ".join(extras)) if extras else ""
    return f"Intent: {mode} · goal={goal_s} · {score_label}={score_s}{extra_s} · why={why}"


def _overview_evidence_line(f: ObservabilityFrame, flags: Dict[str, Any],
                            err_hist: Deque[float],
                            dist_hist: Optional[Deque[float]] = None) -> str:
    errs = list(err_hist)
    err_arrow = ""
    if len(errs) >= 2:
        err_arrow = "↘" if errs[-1] <= errs[-2] else "↗"
    timings = dict(getattr(f, "module_timings", {}) or {})
    dominant = max(
        [(label, _phase_ms(timings, key))
         for key, label in EXECUTION_PHASE_STEPS],
        key=lambda s: s[1],
        default=("", 0.0),
    )[0] or "—"
    learn_s = f"{flags['learn_ms']:.1f}ms" if flags["learn_ms"] > 0.0 else "—"
    spike_s = "yes" if flags.get("spike") else "no"
    decision_s = "yes" if flags.get("decision_shift") else "no"
    dist_s = "—"
    if dist_hist is not None and len(dist_hist) >= 2:
        dist_s = "toward" if dist_hist[-1] <= dist_hist[-2] else "away"
    peu_s = f" · PEŪ={flags['peu_mean']:.2f}" if flags.get("peu_mean") is not None else ""
    return (f"Evidence: phase={dominant} · err={f.prediction_error:.2f}{err_arrow}"
            f" · learn={learn_s} · spike={spike_s} · decision_shift={decision_s}"
            f" · motion={dist_s}{peu_s}")


def _overview_outcome_line(f: ObservabilityFrame, flags: Dict[str, Any],
                           err_hist: Deque[float],
                           dist_hist: Optional[Deque[float]] = None) -> str:
    parts: List[str] = []
    kin = _reacher_kinematics_from_obs(
        f.obs_vector if f.obs_vector is not None else f.sanitized_state)
    if kin is not None:
        dist_s = f"{kin['dist']:.3f}"
        if dist_hist is not None and len(dist_hist) >= 2:
            dist_s += "↘" if dist_hist[-1] <= dist_hist[-2] else "↗"
        parts.append(f"dist={dist_s}")
    elif dist_hist is not None and len(dist_hist) >= 1:
        dist_s = f"{dist_hist[-1]:.3f}"
        if len(dist_hist) >= 2:
            dist_s += "↘" if dist_hist[-1] <= dist_hist[-2] else "↗"
        parts.append(f"dist={dist_s}")
    ca = getattr(f, "continuous_action", None)
    if ca is not None:
        vec = np.asarray(ca, dtype=np.float32).reshape(-1)
        if vec.size == 2:
            parts.append(f"action=τ=[{vec[0]:.2f},{vec[1]:.2f}]")
        elif vec.size:
            parts.append(f"action|τ|={float(np.linalg.norm(vec)):.2f}")
    elif getattr(f, "action_name", None):
        parts.append(f"action={f.action_name}")
    parts.append("goal=yes" if getattr(f, "goal_reached", False) else "goal=no")
    rbta = getattr(f, "rbta_action", None) or "CONTINUE"
    vcount = int(getattr(f, "violations_count", 0) or 0)
    if rbta != "CONTINUE" or vcount > 0:
        rbta_s = f"rbta={rbta}"
        if vcount:
            rbta_s += f"({vcount}V)"
        parts.append(rbta_s)
    else:
        parts.append("rbta=OK")
    return "Outcome: " + " · ".join(parts) if parts else "Outcome: —"


def _overview_new_events(f: ObservabilityFrame, flags: Dict[str, Any],
                         drive_change: Optional[Tuple[int, int]],
                         *, explore_entered: bool = False) -> List[str]:
    events: List[str] = []
    if flags.get("spike"):
        events.append("SPIKE: prediction error jumped")
    if flags.get("learn_burst"):
        events.append(f"{learn_phase_label(f, learn_burst=True)} ({flags['learn_ms']:.1f}ms)")
    if getattr(f, "goal_reached", False):
        events.append("GOAL: target reached")
    if drive_change is not None:
        old_d, new_d = drive_change
        events.append(f"DRIVE: {_drive_short(old_d)}→{_drive_short(new_d)}")
    if flags.get("decision_shift"):
        events.append("DECISION: best learned-score action changed")
    if explore_entered:
        events.append("EXPLORE: sampling candidates")
    if flags.get("near_bound"):
        nb = flags["near_bound"]
        lbl = PIPELINE_LABEL.get(nb, str(nb))
        events.append(f"NEAR-BOUND: {lbl}")
    if flags.get("violation"):
        events.append("VIOLATION: RBTA bound exceeded")
    return events
