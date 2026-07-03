"""Pure helpers for Flow / Action cognitive panels."""
from __future__ import annotations

from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np

from phca.monitoring.observability import ObservabilityFrame

LEARN_MS_MIN = 5.0

MOMENT_COLORS = {
    "spike": "#e74c3c",
    "learn_burst": "#9b59b6",
    "decision_shift": "#f1c40f",
    "near_bound": "#f39c12",
    "drive_change": "#3498db",
}

EXECUTION_PHASE_STEPS = (
    ("prediction", "Predict"),
    ("action_selection", "Act"),
    ("peu", "Error"),
    ("gprime_learn", "Learn"),
    (("mdim", "tspl"), "Decide"),
    ("rbta", "Safe"),
)

RBTA_TO_FLOW = {
    "ASI": "sanitize", "WM": "memory_write", "G'": "prediction", "PEU": "peu",
    "PE": "peu", "TSPL": "tspl", "TSPL-P": "tspl", "TSPLP": "tspl",
    "ACTION": "action_selection", "RBTA": "rbta",
    "G'LEARN": "gprime_learn", "GPRIME_LEARN": "gprime_learn", "MDIM": "mdim",
    "ATTN": "attn", "HPM": "hpm", "CR": "cr",
    "CONSOL": "consolidation", "CONSOLID": "consolidation", "CONS": "consolidation",
}

FLOW_ALL_MODULES = [
    "sanitize", "memory_write", "prediction", "peu", "tspl",
    "action_selection", "rbta",
    "gprime_learn", "mdim", "attn", "hpm", "cr", "consolidation",
]

PIPELINE_LABEL = {
    "sanitize": "ASI", "memory_write": "M2", "prediction": "G′",
    "peu": "PEU", "tspl": "TSPL", "action_selection": "Act", "rbta": "RBTA",
    "gprime_learn": "G′lrn", "mdim": "MDIM", "attn": "Att", "hpm": "HPM",
    "cr": "CR", "consolidation": "Cons",
}

DECISION_SHIFT_THRESHOLD = 0.20

PIPELINE_MODULES = [
    "sanitize", "memory_write", "prediction", "peu", "tspl",
    "action_selection", "rbta",
]


def _phase_ms(timings: Dict[str, Any], key) -> float:
    if isinstance(key, tuple):
        return sum(float(timings.get(k, 0.0) or 0.0) for k in key)
    return float(timings.get(key, 0.0) or 0.0)


def execution_dominant_phase(f: ObservabilityFrame) -> str:
    timings = dict(getattr(f, "module_timings", {}) or {})
    segs = [(label, _phase_ms(timings, key)) for key, label in EXECUTION_PHASE_STEPS]
    if not segs:
        return ""
    return max(segs, key=lambda s: s[1])[0]


def flow_bottleneck_module(f: ObservabilityFrame) -> str:
    timings = dict(getattr(f, "module_timings", {}) or {})
    if not timings:
        return ""
    best = max(FLOW_ALL_MODULES, key=lambda m: float(timings.get(m, 0.0) or 0.0))
    return best if float(timings.get(best, 0.0) or 0.0) > 0 else ""


def _flow_pipe_ms(f: ObservabilityFrame) -> float:
    timings = dict(getattr(f, "module_timings", {}) or {})
    return sum(float(timings.get(m, 0.0) or 0.0) for m in PIPELINE_MODULES)


def apply_decision_shift(prev_score: Optional[float],
                         cur_score: Optional[float]) -> bool:
    """True when best_score jumps more than the Overview decision threshold."""
    if prev_score is None or cur_score is None:
        return False
    return abs(cur_score - prev_score) > DECISION_SHIFT_THRESHOLD


def _best_score(f: ObservabilityFrame) -> Optional[float]:
    r = f.action_rationale or {}
    bs = r.get("best_score")
    return float(bs) if isinstance(bs, (int, float)) else None


def _pipeline_label(mod: str) -> str:
    return PIPELINE_LABEL.get(mod, mod)


def _goal_id(f: ObservabilityFrame) -> Optional[int]:
    r = f.action_rationale or {}
    gid = r.get("goal_id")
    if gid is not None:
        return int(gid)
    did = int(getattr(f, "active_drive_id", 0) or 0)
    return did if did else None


def overview_spike(err_hist: Deque[float], cur_err: float, env_kind: str = "") -> bool:
    """Same spike logic as Overview _overview_spike."""
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


def cognitive_moment(
    f: ObservabilityFrame,
    err_hist: Deque[float],
    *,
    prev_drive_id: Optional[int] = None,
    prev_best_score: Optional[float] = None,
) -> Dict[str, Any]:
    """Extended moment flags for Flow / Action panels (Overview-aligned spike)."""
    cur_err = float(getattr(f, "prediction_error", 0.0) or 0.0)
    env_kind = (getattr(f, "env_kind", "") or "").lower()
    timings = dict(getattr(f, "module_timings", {}) or {})
    learn_ms = float(timings.get("gprime_learn", 0.0) or 0.0)
    learn_burst = learn_ms >= LEARN_MS_MIN
    r = f.action_rationale or {}
    cur_score = _best_score(f)
    decision_shift = apply_decision_shift(prev_best_score, cur_score)
    if not decision_shift:
        decision_shift = bool(r.get("decision_shift", False))
    peu = getattr(f, "per_dim_peu", None)
    peu_mean = None
    if peu is not None:
        arr = np.asarray(peu, dtype=np.float32).reshape(-1)
        if arr.size:
            peu_mean = float(np.mean(arr))
    gid = _goal_id(f)
    drive_change = None
    if gid and prev_drive_id is not None and gid != prev_drive_id:
        drive_change = (prev_drive_id, gid)
    near_bound = flow_near_bound_module(f)
    vcount = int(getattr(f, "violations_count", 0) or 0)
    return {
        "spike": overview_spike(err_hist, cur_err, env_kind),
        "learn_ms": learn_ms,
        "learn_burst": learn_burst,
        "decision_shift": decision_shift,
        "peu_mean": peu_mean,
        "explored": bool(r.get("explored", False)),
        "drive_change": drive_change,
        "dominant_phase": execution_dominant_phase(f),
        "near_bound": near_bound,
        "violation": vcount > 0,
    }


def build_moment_series(frames: List[ObservabilityFrame]) -> List[Dict[str, Any]]:
    err_hist: Deque[float] = deque(maxlen=200)
    prev_drive: Optional[int] = None
    prev_best_score: Optional[float] = None
    out: List[Dict[str, Any]] = []
    for f in frames:
        err_hist.append(float(getattr(f, "prediction_error", 0.0) or 0.0))
        m = cognitive_moment(
            f, err_hist, prev_drive_id=prev_drive, prev_best_score=prev_best_score)
        out.append(m)
        gid = _goal_id(f)
        if gid:
            prev_drive = gid
        cur_score = _best_score(f)
        if cur_score is not None:
            prev_best_score = cur_score
    return out


def append_cognitive_moment(
    series: List[Dict[str, Any]],
    f: ObservabilityFrame,
    err_hist: Deque[float],
    *,
    prev_drive_id: Optional[int],
    prev_best_score: Optional[float],
    maxlen: int,
) -> Tuple[List[Dict[str, Any]], Optional[int], Optional[float]]:
    """Append one live moment; trim to maxlen; return updated prev state."""
    err_hist.append(float(getattr(f, "prediction_error", 0.0) or 0.0))
    m = cognitive_moment(
        f, err_hist, prev_drive_id=prev_drive_id, prev_best_score=prev_best_score)
    series = list(series)
    series.append(m)
    if len(series) > maxlen:
        series = series[-maxlen:]
    gid = _goal_id(f)
    new_prev = gid if gid else prev_drive_id
    cur_score = _best_score(f)
    new_best = cur_score if cur_score is not None else prev_best_score
    return series, new_prev, new_best


def goal_id_from_frame(f: ObservabilityFrame) -> Optional[int]:
    return _goal_id(f)


def flow_near_bound_module(f: ObservabilityFrame) -> str:
    mods = flow_near_bound_modules(f, top_k=1)
    return mods[0][0] if mods else ""


def flow_near_bound_modules(
    f: ObservabilityFrame,
    *,
    top_k: int = 3,
) -> List[Tuple[str, float]]:
    """Return top-K modules by measured/bound ratio (>0.5)."""
    timings = dict(getattr(f, "module_timings", {}) or {})
    bounds = dict(getattr(f, "rbta_bounds", {}) or {})
    ranked: List[Tuple[str, float]] = []
    for mod in FLOW_ALL_MODULES:
        measured = float(timings.get(mod, 0.0) or 0.0)
        if measured <= 0:
            continue
        bound = None
        for k, v in bounds.items():
            if RBTA_TO_FLOW.get(k, "") == mod and isinstance(v, dict):
                t = v.get("time")
                if t is not None and float(t) > 0:
                    bound = float(t)
                    break
        if bound and bound > 0:
            ratio = measured / bound
            if ratio > 0.5:
                ranked.append((mod, ratio))
    ranked.sort(key=lambda x: -x[1])
    return ranked[:top_k]


def belief_reference(
    f: ObservabilityFrame,
    *,
    prefer_goal: bool = False,
) -> Tuple[Optional[np.ndarray], str]:
    """Unified reference vector + label for phase portrait / PCA."""
    if prefer_goal and getattr(f, "goal_ref", None) is not None:
        ref = np.asarray(f.goal_ref, dtype=np.float32).reshape(-1)
        return ref, "goal_ref"
    if f.sanitized_state is not None:
        return np.asarray(f.sanitized_state, dtype=np.float32).reshape(-1), "sanitized_state"
    if f.obs_vector is not None:
        return np.asarray(f.obs_vector, dtype=np.float32).reshape(-1), "obs_vector"
    if getattr(f, "goal_ref", None) is not None:
        return np.asarray(f.goal_ref, dtype=np.float32).reshape(-1), "goal_ref"
    if getattr(f, "goal_target", None) is not None:
        return np.asarray(f.goal_target, dtype=np.float32).reshape(-1), "goal_target"
    return None, "—"


def flow_status_extras(f: ObservabilityFrame) -> str:
    parts: List[str] = []
    parts.append(f"cycle={int(getattr(f, 'cycle_id', 0) or 0)}")
    dom = execution_dominant_phase(f)
    if dom:
        parts.append(f"phase={dom}")
    timings = dict(getattr(f, "module_timings", {}) or {})
    learn_ms = float(timings.get("gprime_learn", 0.0) or 0.0)
    if learn_ms >= LEARN_MS_MIN:
        parts.append(f"learn={learn_ms:.1f}ms")
    nb = flow_near_bound_module(f)
    if nb:
        parts.append(f"near_bound={_pipeline_label(nb)}")
    rbta_act = str(getattr(f, "rbta_action", "") or "").strip()
    if rbta_act and rbta_act != "CONTINUE":
        parts.append(f"rbta={rbta_act}")
    lat = float(getattr(f, "latency_ms", 0.0) or 0.0)
    pipe = _flow_pipe_ms(f)
    if lat > 0:
        parts.append(f"latency={lat:.1f}ms")
        if pipe > 0 and abs(lat - pipe) / max(lat, 1e-6) > 0.2:
            parts.append(f"Δpipe={lat - pipe:+.1f}ms")
    err = float(getattr(f, "prediction_error", 0.0) or 0.0)
    if err > 0:
        parts.append(f"err={err:.2f}")
    return " · ".join(parts)


def action_status_extras(
    f: ObservabilityFrame,
    scores: List[float],
    chosen: int,
    *,
    replay: bool = False,
    moment: Optional[Dict[str, Any]] = None,
) -> str:
    parts: List[str] = []
    parts.append(f"cycle={int(getattr(f, 'cycle_id', 0) or 0)}")
    names = list(getattr(f, "action_names", []) or [])
    r = f.action_rationale or {}
    is_cont = bool(r.get("continuous", f.continuous_action is not None))
    if chosen >= 0:
        lbl = names[chosen] if chosen < len(names) and names[chosen] else f"#{chosen}"
        if is_cont and str(lbl).startswith("MOVE_"):
            lbl = f"τ#{chosen}"
        parts.append(f"chosen={lbl}")
    gid = r.get("goal_id")
    if gid is not None:
        parts.append(f"goal={int(gid)}")
    if moment and moment.get("decision_shift"):
        parts.append("shift=1")
    elif r.get("decision_shift"):
        parts.append("shift=1")
    cr_t = float(getattr(f, "cr_temperature", 0.0) or 0.0)
    if cr_t > 0:
        parts.append(f"cr_T={cr_t:.2f}")
    emp = getattr(f, "empowerment", None)
    if isinstance(emp, (int, float)) and float(emp) > 0:
        parts.append(f"emp={float(emp):.2f}")
    err = float(getattr(f, "prediction_error", 0.0) or 0.0)
    if err > 0:
        parts.append(f"err={err:.2f}")
    peu_mean = (moment or {}).get("peu_mean")
    if peu_mean is None and getattr(f, "per_dim_peu", None) is not None:
        arr = np.asarray(f.per_dim_peu, dtype=np.float32).reshape(-1)
        if arr.size:
            peu_mean = float(np.mean(arr))
    if peu_mean is not None:
        parts.append(f"PEŪ={peu_mean:.2f}")
    rollouts = list(getattr(f, "candidate_rollouts", []) or [])
    if replay and scores and not rollouts:
        parts.append("rollouts=replay")
    return " · ".join(parts)


def flow_action_link_line(f: ObservabilityFrame) -> str:
    """Short Flow→Action context for the Action tab."""
    bn = flow_bottleneck_module(f)
    dom = execution_dominant_phase(f)
    r = f.action_rationale or {}
    mode = "EXPLORE" if r.get("explored") else "EXPLOIT"
    parts: List[str] = []
    if bn:
        parts.append(f"Flow bottleneck={_pipeline_label(bn)}")
    if dom:
        parts.append(f"phase={dom}")
    parts.append(mode)
    nb = flow_near_bound_module(f)
    if nb:
        parts.append(f"near_bound={_pipeline_label(nb)}")
    return " · ".join(parts)


def count_moments(series: List[Dict[str, Any]]) -> Dict[str, int]:
    return {
        "spike_count": sum(1 for m in series if m.get("spike")),
        "learn_burst_count": sum(1 for m in series if m.get("learn_burst")),
        "decision_shift_count": sum(1 for m in series if m.get("decision_shift")),
        "drive_change_count": sum(1 for m in series if m.get("drive_change")),
        "violation_count": sum(1 for m in series if m.get("violation")),
    }
