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


def rbta_bound_for_module(mod: str, bounds: Dict[str, Any]) -> Optional[float]:
    """RBTA time bound for a flow module in **seconds** (canonical storage)."""
    for k, v in (bounds or {}).items():
        if RBTA_TO_FLOW.get(k, "") == mod and isinstance(v, dict):
            t = v.get("time")
            if t is not None and float(t) > 0:
                return float(t)
    return None


def rbta_time_bound_ms(mod: str, bounds: Dict[str, Any]) -> Optional[float]:
    """RBTA time bound converted to milliseconds for module_timings comparison."""
    b = rbta_bound_for_module(mod, bounds)
    return b * 1000.0 if b is not None else None


def flow_timing_ratio(
    mod: str,
    timings: Dict[str, Any],
    bounds: Dict[str, Any],
) -> Optional[float]:
    """Measured module time (ms) / RBTA time bound (ms), or None if unavailable."""
    measured = float(timings.get(mod, 0.0) or 0.0)
    if measured <= 0:
        return None
    bound_ms = rbta_time_bound_ms(mod, bounds)
    if bound_ms is None or bound_ms <= 0:
        return None
    return measured / bound_ms


def pipeline_time_budget_ms(
    bounds: Dict[str, Any],
    modules: List[str],
) -> float:
    """Sum of RBTA time bounds (ms) for modules present in bounds."""
    total = 0.0
    for mod in modules:
        bms = rbta_time_bound_ms(mod, bounds)
        if bms is not None:
            total += bms
    return total


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
        ratio = flow_timing_ratio(mod, timings, bounds)
        if ratio is not None and ratio > 0.5:
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
    review: bool = False,
    prefix_len: int = 0,
    moment: Optional[Dict[str, Any]] = None,
) -> str:
    parts: List[str] = []
    parts.append(f"cycle={int(getattr(f, 'cycle_id', 0) or 0)}")
    if review and prefix_len > 0:
        parts.append(f"prefix={prefix_len}")
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


def phase_tab_status_line(
    f: ObservabilityFrame,
    proj: Optional[Any] = None,
    *,
    review: bool = False,
    prefix_len: int = 0,
) -> str:
    """Unified Phase Space tab strip (traj + radar + perdim context)."""
    parts = [f"Phase · cycle={int(getattr(f, 'cycle_id', 0) or 0)}"]
    if review and prefix_len > 0:
        parts.append(f"prefix={prefix_len}")
    if proj is not None:
        ve = proj.variance_explained()
        if ve is not None:
            parts.append(f"PCA={ve:.0f}%")
    r = f.action_rationale or {}
    ci = r.get("chosen_idx")
    rollouts = list(getattr(f, "candidate_rollouts", []) or [])
    if rollouts and isinstance(ci, (int, float)):
        parts.append(f"chosen=#{int(ci)}")
    ref_lbl = "sanitized" if f.sanitized_state is not None else "obs_vector"
    if getattr(f, "goal_ref", None) is not None:
        parts.append(f"ref={ref_lbl}|goal_ref")
    did = int(getattr(f, "active_drive_id", 0) or 0)
    if did:
        parts.append(f"drive=D{did}")
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


# ----- data-contract banners (live + replay, shared across tabs) ------------

DATA_CONTRACT_REPLAY: Dict[str, str] = {
    "overview": "grid/state/metrics from JSONL; camera and bulk memory are live-only",
    "flow": "module_timings + rbta_bounds recorded; PEU details are live-only",
    "action": "scores/rationale recorded · rollouts live-only (candidate_rollouts)",
    "phase": "state/prediction/goal_ref recorded; rollouts/live goal vectors unavailable",
    "retention": "counts/caps/RSS/latency recorded; bulk M3/M4 lists live-only",
    "rbta": "rbta_bounds + module_timings recorded in JSONL",
    "memory": "G′ uncertainty + M3 top-error recorded; bulk memory/diff live-only",
    "goals": "drive_goals radial inset is live-only in JSONL",
}

DATA_CONTRACT_LIVE: Dict[str, str] = {
    "overview": "camera frame live-only; JSONL has obs_vector + continuous_action",
    "flow": "module_timings + rbta recorded each cycle",
    "action": "candidate_scores recorded; rollouts live-only",
    "phase": "obs_vector/goal_ref recorded; bulk rollouts N/A",
    "retention": "RSS/latency/episode_count recorded",
    "rbta": "rbta_bounds + module_timings recorded each cycle",
    "memory": "fact_count + top-error recorded; bulk M3/M4 lists live-only",
    "goals": "drive_levels + active_drive recorded",
}

DATA_CONTRACT_REVIEW: Dict[str, str] = {
    "overview": "session complete · scrub prefix 0..cursor · results panel active",
    "flow": "full-session prefix rebuild on scrub · module_timings from JSONL",
    "action": "mechanism mix · prefix 0..cursor · scores window ≤200",
    "phase": "belief projection rebuilt from prefix 0..cursor · radar/perdim aligned to scrub cursor · traj window ≤256",
    "retention": "M3/M4/RSS/latency series aligned to scrub cursor",
    "rbta": "bound envelope sparklines aligned to scrub cursor",
    "memory": "belief geography at cursor · M3/M4 lists from recorded frame",
    "goals": "tanks at cursor · deficit heatmap from prefix 0..cursor · heatmap window ≤200",
}

DATA_CONTRACT_REVIEW_INCOMPLETE: Dict[str, str] = {
    "overview": "SESSION ABORTED/INCOMPLETE · scrub recorded prefix only · live-only panels empty",
    "flow": "partial session · module_timings from recorded frames only",
    "action": "partial session · scores/rationale from recorded frames only",
    "phase": "partial session · belief projection from recorded prefix",
    "retention": "partial session · series end at last recorded cycle",
    "rbta": "partial session · bounds from recorded frames only",
    "memory": "partial session · memory samples from recorded frames only",
    "goals": "partial session · drives/goals from recorded frames only",
}


def data_contract_text(
    panel_key: str,
    *,
    replay: bool,
    review: bool = False,
    multi_agent: bool = False,
    incomplete: bool = False,
) -> str:
    """Return the data-contract banner string for a tab panel."""
    key = str(panel_key)
    if review and incomplete:
        body = DATA_CONTRACT_REVIEW_INCOMPLETE.get(key, "")
        prefix = "ABORTED — "
    elif review:
        body = DATA_CONTRACT_REVIEW.get(key, "")
        prefix = "REVIEW — "
    elif replay:
        body = DATA_CONTRACT_REPLAY.get(key, "")
        prefix = "REPLAY — "
    else:
        body = DATA_CONTRACT_LIVE.get(key, "")
        prefix = "LIVE — "
    if not body:
        return ""
    text = f"{prefix}{body}"
    if multi_agent:
        text += " · multi-agent: camera/bulk memory are per-agent live-only"
    return text


def classify_action_mechanism(rationale: Dict[str, Any]) -> str:
    """Classify one cycle's action mechanism (mirrors session_report)."""
    if rationale.get("rbta_safe_mode"):
        return "rbta_safe"
    mech = rationale.get("mechanism")
    if isinstance(mech, str) and mech:
        return mech
    if rationale.get("explored"):
        return "explore"
    if rationale.get("greedy_fallback"):
        return "greedy_fallback"
    if rationale.get("continuous"):
        return "continuous"
    if rationale.get("goal_id") == 5 or rationale.get("note") == "D5 energy: STAY":
        return "stay"
    if rationale.get("best_score") is not None or rationale.get("k_candidates"):
        return "prediction"
    return "other"


def mechanism_histogram(
    frames: List[Any],
    *,
    window: int = 256,
) -> Dict[str, int]:
    """Rolling mechanism counts from action_rationale (live rollup parity)."""
    counts = {
        "greedy_fallback": 0,
        "prediction": 0,
        "explore": 0,
        "stay": 0,
        "continuous": 0,
        "rbta_safe": 0,
        "other": 0,
    }
    tail = frames[-window:] if window > 0 else frames
    for f in tail:
        r = dict(getattr(f, "action_rationale", {}) or {})
        counts[classify_action_mechanism(r)] += 1
    return counts


def mechanism_pct(counts: Dict[str, int]) -> Dict[str, float]:
    total = sum(counts.values()) or 1
    return {k: round(100.0 * v / total, 1) for k, v in counts.items() if v}


def decimate_frames_for_history(
    frames: List[Any],
    max_points: int = 2000,
) -> List[Any]:
    """Evenly downsample frame lists for scrub history rebuild (Phase 11)."""
    n = len(frames)
    if max_points <= 0 or n <= max_points:
        return frames
    step = n / float(max_points)
    out: List[Any] = []
    i = 0.0
    while int(i) < n and len(out) < max_points:
        out.append(frames[int(i)])
        i += step
    if out and out[-1] is not frames[-1]:
        out.append(frames[-1])
    return out


def rolling_prefix(
    frames: List[Any],
    cursor: int,
    *,
    review_mode: bool = False,
    window: int = 200,
    max_points: int = 2000,
) -> List[Any]:
    """Return the rolling prefix for scrub/review history rebuild.

    Review mode uses frames[0:cursor+1] (full session prefix).
    Replay/live scrub uses a trailing window of ``window`` frames.
    Long prefixes are decimated to ``max_points``."""
    if not frames:
        return []
    i = int(max(0, min(cursor, len(frames) - 1)))
    lo = 0 if review_mode else max(0, i - window)
    rolling = frames[lo:i + 1]
    if len(rolling) > max_points:
        rolling = decimate_frames_for_history(rolling, max_points)
    return rolling


OBSERVATORY_TAB_LABELS: Tuple[str, ...] = (
    "Overview",
    "Cognitive Flow",
    "Action Selection",
    "Phase Space & Trajectory",
    "Retention & Resources",
    "Memory & Belief",
    "Goals & Motivation",
)


def moment_tab_badge(flags: Dict[str, Any]) -> str:
    """Short badge suffix for tab titles when a cognitive moment is active."""
    if flags.get("spike"):
        return "SPIKE"
    if flags.get("violation"):
        return "VIOL"
    if flags.get("decision_shift"):
        return "DECISION"
    if flags.get("learn_burst"):
        return "G′lrn"
    if flags.get("drive_change"):
        return "DRIVE"
    return ""


def session_status_text(
    *,
    env: str = "",
    env_kind: str = "",
    camera: str = "",
    cycle_id: int = 0,
    total: int = 0,
    live_lag: int = 0,
    schema_version: int = 0,
    jsonl_count: int = 0,
    recording: bool = True,
    verify_status: str = "",
    agent_id: int = 0,
    agent_count: int = 1,
    agent_label: str = "",
    playback_error: str = "",
    cycle_error: str = "",
    incomplete: bool = False,
    rbta_safe_ratio: float = 0.0,
    rbta_safe_warn_threshold: float = 0.10,
) -> str:
    """One-line global session strip (all tabs share this context)."""
    parts: List[str] = []
    if cycle_error:
        parts.append(f"ABORTED {cycle_error[:64]}")
    elif incomplete and total > 0 and jsonl_count < total:
        parts.append(f"INCOMPLETE {jsonl_count}/{total}")
    if playback_error:
        parts.append(f"ERR {playback_error[:72]}")
    if rbta_safe_ratio >= rbta_safe_warn_threshold:
        parts.append(f"SAFE-MODE {100.0 * rbta_safe_ratio:.0f}%")
    if agent_count > 1:
        lbl = f" {agent_label}" if agent_label else ""
        parts.append(f"agent {agent_id + 1}/{agent_count}{lbl}")
    if env:
        ek = env_kind or "?"
        parts.append(f"{env} · {ek}")
    if camera:
        parts.append(f"cam={camera}")
    if total > 0:
        parts.append(f"cycle {cycle_id}/{total}")
    if live_lag > 0:
        parts.append(f"lag={live_lag}")
    if schema_version:
        parts.append(f"schema v{schema_version}")
    if recording:
        parts.append(f"JSONL {jsonl_count}/{total or '?'}")
    if verify_status:
        parts.append(f"verify={verify_status}")
    return "  ·  ".join(parts)


def format_early_late(
    label: str,
    early: Optional[float],
    late: Optional[float],
) -> str:
    """Compact early→late metric with trend arrow."""
    if early is None and late is None:
        return f"{label} —"
    if early is None:
        return f"{label} —→{late:.2g}"
    if late is None:
        return f"{label} {early:.2g}→—"
    if late < early:
        arrow = "↘"
    elif late > early:
        arrow = "↗"
    else:
        arrow = "→"
    return f"{label} {early:.2g}→{late:.2g} {arrow}"


def _top_pct_items(pct: Dict[str, Any], n: int = 3) -> List[Tuple[str, float]]:
    items: List[Tuple[str, float]] = []
    for k, v in (pct or {}).items():
        try:
            fv = float(v)
        except (TypeError, ValueError):
            continue
        if fv > 0:
            items.append((str(k), fv))
    items.sort(key=lambda kv: kv[1], reverse=True)
    return items[:n]


def _dominant_mechanism(pct: Dict[str, Any]) -> str:
    top = _top_pct_items(pct, 1)
    return top[0][0] if top else ""


def format_session_results_lines(
    report: Dict[str, Any],
    *,
    compare: Optional[Dict[str, Any]] = None,
    benchmark: Optional[Dict[str, Any]] = None,
    verify_status: str = "",
) -> List[str]:
    """Human-readable session performance lines for Overview results panel."""
    lines: List[str] = []
    n = int(report.get("cycles") or 0)
    explore = report.get("explore_ratio")
    explore_s = f"{100.0 * float(explore):.1f}%" if explore is not None else "—"
    flow_m = report.get("flow_metrics") or {}
    viol_n = flow_m.get("violation_cycle_count", 0)
    lines.append(
        f"session {n} cycles · explore {explore_s} · "
        f"goals {report.get('goal_reached_count', 0)} · "
        f"spikes {report.get('spike_count', 0)} · "
        f"viol {viol_n}"
    )
    safe_ratio = report.get("rbta_safe_ratio")
    term_n = int(report.get("terminate_cycle_count") or 0)
    if safe_ratio is not None and float(safe_ratio) >= 0.10:
        lines.append(
            f"SAFE-MODE {100.0 * float(safe_ratio):.0f}% "
            f"(rbta_safe · TERMINATE cycles {term_n})"
        )
    err_line = format_early_late(
        "error",
        report.get("error_early_median"),
        report.get("error_late_median"),
    )
    dist_e = report.get("dist_early_median")
    dist_l = report.get("dist_late_median")
    if dist_e is not None or dist_l is not None:
        lines.append(format_early_late("dist", dist_e, dist_l))
    lines.append(err_line)

    anom = report.get("anomalies") or {}
    active = anom.get("active") or []
    if active:
        lines.append("anomalies: " + ", ".join(active))
    cls = report.get("report_classification") or {}
    expected_limits = cls.get("expected_limitations") or []
    if expected_limits:
        lines.append("limits: " + ", ".join(expected_limits[:2]))

    action_m = report.get("action_metrics") or {}
    mech_pct = action_m.get("mechanism_pct") or {}
    selector_pct = action_m.get("selector_mode_pct") or {}
    selector_top = _top_pct_items(selector_pct, 2)
    if selector_top:
        lines.append(
            "selector: " + " · ".join(f"{k} {v:.0f}%" for k, v in selector_top)
        )
    if action_m.get("task_lock_planner_dominant"):
        lines.append("mode: task-lock planner dominated (discrete fallback path)")
    mech_top = _top_pct_items(mech_pct, 3)
    if mech_top:
        lines.append(
            "mechanism: " + " · ".join(f"{k} {v:.0f}%" for k, v in mech_top)
        )
    explain_m = action_m.get("explain_metrics") or {}
    dr_counts = explain_m.get("decision_reason_counts") or {}
    if dr_counts and report.get("cycles"):
        n = int(report["cycles"])
        dr_top = sorted(dr_counts.items(), key=lambda kv: kv[1], reverse=True)[:3]
        lines.append(
            "explain: "
            + " · ".join(
                f"{k} {100.0 * v / n:.0f}%" for k, v in dr_top if v > 0
            )
        )

    phase_pct = report.get("phase_budget_pct") or {}
    phase_top = _top_pct_items(phase_pct, 3)
    if phase_top:
        lines.append(
            "phase: " + " · ".join(f"{k} {v:.0f}%" for k, v in phase_top)
        )

    if compare:
        cmp_bits: List[str] = []
        prev_err = compare.get("error_late_median")
        cur_err = report.get("error_late_median")
        if prev_err is not None and cur_err is not None:
            delta = float(cur_err) - float(prev_err)
            cmp_bits.append(f"Δerr {delta:+.2g}")
        prev_ex = compare.get("explore_ratio")
        if prev_ex is not None and explore is not None:
            cmp_bits.append(f"Δexplore {100.0 * (float(explore) - float(prev_ex)):+.1f}pp")
        prev_m = _dominant_mechanism((compare.get("action_metrics") or {}).get("mechanism_pct") or {})
        cur_m = _dominant_mechanism(mech_pct)
        if prev_m and cur_m and prev_m != cur_m:
            cmp_bits.append(f"mech {prev_m}→{cur_m}")
        if cmp_bits:
            lines.append("vs prev: " + " · ".join(cmp_bits))

    if benchmark:
        overall = benchmark.get("overall_phi_iq")
        if overall is not None:
            lines.append(f"Φ-IQ benchmark overall {float(overall):.3f}")
        results = benchmark.get("results") or []
        lvl = []
        for r in results[:4]:
            lvl.append(f"L{r.get('level', '?')}={float(r.get('phi_iq', 0.0)):.2f}")
        if lvl:
            lines.append("  " + " · ".join(lvl))

    if verify_status:
        lines.append(f"verify: {verify_status} · close window to exit")
    else:
        lines.append("review mode · scrub transport · close window to exit")
    return lines
