"""PHCA v3.0 — Visual Observability Layer (Phase 7 extension).

A non-invasive, opt-in observability buffer for the cognitive cycle. The
cycle thread is the only writer; a separate visualiser thread is the only
reader. The buffer is a lock-guarded ring buffer (mirrors MetricsStore).

Design:
  - ObservabilityFrame: an immutable snapshot of one cycle's world + mind
    state (agent pos, goal, grid, predicted next state, MDIM D1-D6 levels,
    attention saliences, RSS, + CycleMetrics scalars).
  - ObservabilityFrame.from_cycle(cycle): the ONLY place that reads live
    cycle/env/mdim/attention objects; builds copies so the reader never sees
    a mutating object (G-009 thread-safe monitoring).
  - ObservabilityStore: lock-guarded deque + latest()/snapshot().
  - SessionRecorder: persists each frame to logs/sessions/<ts>/ as
    timeseries.jsonl + frames/<id>.png + meta.json for replay/analysis.

Zero-overhead when not attached: CognitiveCycle builds a frame only when
self.observability_store is not None (same opt-in pattern as metrics_store).

v3.0 trace: A1 (frame build ~microsec, recording off hot path), A4 (the
frame carries the prediction -> visualised as a heatmap), A5 (error/conf
trends show feedback adaptation live).
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_logger = logging.getLogger(__name__)

# Retention caps (mirrored from consolidation.scheduler for the dashboard).
# Imported lazily/defensively so the monitoring package never hard-fails if the
# consolidation module is refactored.
try:
    from phca.consolidation.scheduler import M4_MAX_FACTS as _M4_MAX_FACTS, \
        M4_PRUNE_TARGET as _M4_PRUNE_TARGET
except Exception as exc:
    _logger.debug("consolidation scheduler import failed, using defaults: %s", exc)
    _M4_MAX_FACTS = 1_000
    _M4_PRUNE_TARGET = 500

try:
    import psutil
    _HAVE_PSUTIL = True
except Exception as exc:
    _logger.debug("psutil import failed, RSS monitoring disabled: %s", exc)
    _HAVE_PSUTIL = False

_PROC = psutil.Process() if _HAVE_PSUTIL else None

# RSS changes slowly; throttle the syscall so the per-cycle frame build stays
# well under the 5% latency-overhead gate. Refresh at most every 0.25s.
_RSS_CACHE: Dict[str, Tuple[float, int]] = {}
_RSS_TTL_S = 0.25

# Heavy memory samples (M3 episodes + M4 facts) are read off the cycle thread
# at most every 0.5s so the per-cycle frame build stays cheap. These too are
# live-only (excluded from JSONL).
_MEM_CACHE: Dict[str, Tuple[float, Dict[str, Any]]] = {}
_MEM_TTL_S = 0.5

OBSERVABILITY_SCHEMA_VERSION = 1
SUPPORTED_OBSERVABILITY_SCHEMA_VERSIONS = {0, OBSERVABILITY_SCHEMA_VERSION}


def _cached_rss() -> int:
    if _PROC is None:
        return 0
    now = time.monotonic()
    ts, val = _RSS_CACHE.get("rss", (0.0, 0))
    if now - ts > _RSS_TTL_S:
        try:
            val = int(_PROC.memory_info().rss)
            _RSS_CACHE["rss"] = (now, val)
        except Exception as exc:
            _logger.debug("RSS read failed: %s", exc)
            val = 0
    return val


def _normalize_rgb_frame(frame: Any) -> Optional[np.ndarray]:
    """Coerce MuJoCo/gym camera output to contiguous (H,W,3) uint8 RGB."""
    if frame is None:
        return None
    try:
        arr = np.asarray(frame)
    except Exception:
        return None
    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.ndim != 3 or arr.shape[0] < 1 or arr.shape[1] < 1:
        return None
    if arr.shape[-1] == 4:
        arr = arr[:, :, :3]
    elif arr.shape[-1] != 3:
        return None
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


def _fact_to_dict(f: Any) -> Dict[str, Any]:
    return {
        "fact_type": getattr(f, "fact_type", "?"),
        "confidence": float(getattr(f, "confidence", 0.0)),
        "frequency": int(getattr(f, "frequency", getattr(f, "support", 0)) or 0),
        "summary": str(getattr(f, "summary", getattr(f, "description", "")))[:80],
        "timestamp": int(getattr(f, "timestamp", getattr(f, "created_at", 0)) or 0),
    }


def _episode_to_dict(ep: Any) -> Dict[str, Any]:
    """Lightweight M3 episode record for the Memory tab (scalars + small preview)."""
    sb = getattr(ep, "state_before", None)
    sa = getattr(ep, "state_after", None)
    sb_vals = getattr(sb, "values", None) if sb is not None else None
    sa_vals = getattr(sa, "values", None) if sa is not None else None
    return {
        "episode_id": int(getattr(ep, "episode_id", 0)),
        "timestamp": int(getattr(ep, "timestamp", 0)),
        "prediction_error": float(getattr(ep, "prediction_error", 0.0)),
        "confidence": float(getattr(ep, "confidence", 0.0)),
        "drive_id": getattr(ep, "drive_id", None),
        "sb_head": (np.asarray(sb_vals, dtype=np.float32)[:8].tolist()
                    if sb_vals is not None else None),
        "sa_head": (np.asarray(sa_vals, dtype=np.float32)[:8].tolist()
                    if sa_vals is not None else None),
    }


def _cached_memory(cycle: Any, m3: Any) -> Dict[str, Any]:
    """Throttled M3/M4 samples for the Memory & Belief tab (live-only)."""
    now = time.monotonic()
    ts, val = _MEM_CACHE.get("mem", (0.0, {}))
    if val and now - ts <= _MEM_TTL_S:
        return val
    out: Dict[str, Any] = {"m3_recent": [], "m3_top_error": [],
                           "m4_relevant": [], "m4_top": []}
    try:
        if m3 is not None:
            if hasattr(m3, "recent_episodes"):
                out["m3_recent"] = [_episode_to_dict(e)
                                    for e in m3.recent_episodes(5)]
            if hasattr(m3, "top_error_episodes"):
                out["m3_top_error"] = [_episode_to_dict(e)
                                       for e in m3.top_error_episodes(5)]
    except Exception as exc:
        _logger.debug("M3 memory sample failed: %s", exc)
    try:
        consol = getattr(cycle, "consolidation", None)
        cur = getattr(cycle, "current_state", None)
        if consol is not None:
            if cur is not None and hasattr(consol, "get_relevant_facts"):
                facts = consol.get_relevant_facts(cur, n=5, min_confidence=0.3)
                out["m4_relevant"] = [_fact_to_dict(f) for f in facts]
            if hasattr(consol, "get_semantic_facts"):
                facts = consol.get_semantic_facts(min_confidence=0.3,
                                                  max_results=20)
                ranked = sorted(facts, key=lambda f: getattr(f, "frequency",
                                 getattr(f, "support", 0)), reverse=True)[:8]
                out["m4_top"] = [_fact_to_dict(f) for f in ranked]
    except Exception as exc:
        _logger.debug("M4 memory sample failed: %s", exc)
    _MEM_CACHE["mem"] = (now, out)
    return out


_SNAP_CACHE_MAX = 200


def _snap(obj: Any, method: str) -> Dict[str, Any]:
    """Defensively call an additive snapshot() accessor; {} if absent/raises."""
    fn = getattr(obj, method, None)
    if not callable(fn):
        return {}
    try:
        return dict(fn() or {})
    except Exception as exc:
        _logger.debug("snapshot %s.%s failed: %s", type(obj).__name__, method, exc)
        return {}


# TTL cache for the *expensive* additive snapshots (tspl theta-norms, mdim
# goal-stack/pareto/drive-history). These change slowly (only after learning /
# goal transitions), so recomputing them every cycle is pure overhead. Cheap,
# fast-changing signals (deficits, scalars) are still read fresh below by
# overwriting the cached dict's deficits. Keeps the ≤5% overhead gate in reach.
# Bounded LRU with type-qualified key to prevent unbounded growth.
_SNAP_CACHE: OrderedDict = OrderedDict()


def _cached_snap(key: str, obj: Any, method: str, ttl: float) -> Dict[str, Any]:
    now = time.monotonic()
    cache_key = f"{key}:{id(obj)}:{type(obj).__name__}"
    ent = _SNAP_CACHE.get(cache_key)
    if ent is not None and now - ent[0] < ttl:
        _SNAP_CACHE.move_to_end(cache_key)
        return dict(ent[1])  # shallow copy so callers can't mutate the cache
    val = _snap(obj, method)
    _SNAP_CACHE[cache_key] = (now, val)
    _SNAP_CACHE.move_to_end(cache_key)
    while len(_SNAP_CACHE) > _SNAP_CACHE_MAX:
        _SNAP_CACHE.popitem(last=False)
    return dict(val)


def clear_snap_cache() -> None:
    """Clear the TTL snapshot cache (tests / session boundaries)."""
    _SNAP_CACHE.clear()


def _recursive_json(x: Any) -> Any:
    """Recursively convert numpy types to JSON-friendly Python types."""
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, dict):
        return {str(k): _recursive_json(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_recursive_json(v) for v in x]
    return x


@dataclass
class ObservabilityFrame:
    """Immutable snapshot of one cognitive cycle's world + mind state."""
    schema_version: int = OBSERVABILITY_SCHEMA_VERSION
    cycle_id: int = 0
    agent_id: int = 0
    agent_label: str = ""
    timeline_step: int = -1  # shared step when aligned; -1 = legacy/single-agent
    # External world
    agent_pos: Optional[Tuple[int, int]] = None
    goal_pos: Optional[Tuple[int, int]] = None
    grid: Optional[np.ndarray] = None        # GridWorld only
    predicted_state: Optional[np.ndarray] = None  # G' predicted next-state values
    obs_vector: Optional[np.ndarray] = None  # MuJoCo fallback (no grid)
    goal_ref: Optional[np.ndarray] = None    # MuJoCo goal reference (MPC alignment)
    continuous_action: Optional[np.ndarray] = None  # chosen continuous action vector
    # Internal mind
    drive_levels: List[float] = field(default_factory=list)   # D1-D6 values
    drive_targets: List[float] = field(default_factory=list)  # D1-D6 setpoints
    active_drive_id: int = 1
    attention_saliences: List[float] = field(default_factory=list)
    attention_indices: List[int] = field(default_factory=list)
    attention_selected_saliences: List[float] = field(default_factory=list)
    rbta_violations: List[Dict[str, Any]] = field(default_factory=list)  # module/bound_type/measured/allowed
    action_rationale: Dict[str, Any] = field(default_factory=dict)  # explored/eps/goal_id/best_score/k_candidates
    candidate_scores: List[float] = field(default_factory=list)  # per-action / per-candidate scores
    module_timings: Dict[str, float] = field(default_factory=dict)  # per-module ms (cognitive flow)
    # CycleMetrics scalar mirrors
    latency_ms: float = 0.0
    prediction_error: float = 0.0
    prediction_confidence: float = 0.0  # epistemic
    prediction_quality: Optional[float] = None  # post-PEU; None = not recorded
    display_confidence: Optional[float] = None  # min(epistemic, quality); None = derive
    per_dim_peu_top: List[Dict[str, Any]] = field(default_factory=list)  # compact top-K
    per_dim_peu_sum: float = 0.0
    rbta_action: str = "CONTINUE"
    violations_count: int = 0
    goal_reached: bool = False
    action_name: str = ""
    drive_id: int = 1
    episode_count: int = 0  # env episode index
    m3_count: int = 0  # M3 store size
    fact_count: int = 0
    # Continual learning + cognitive resilience (Level-4-lite / observability)
    task_id: int = -1
    failure_events: List[str] = field(default_factory=list)
    recovery_active: bool = False
    # Retention caps (so the dashboard can show cap engagement)
    m3_cap: int = 0
    m4_cap: int = 0
    m4_prune_target: int = 0
    # Resource
    rss_bytes: int = 0
    # ── Observability v4: env-agnostic + cognitive-portrait fields ──
    # All default-empty so old JSONL replays stay backward-compatible.
    env_kind: str = ""                       # "grid" | "mujoco_rgb" | "continuous"
    state_dim: int = 0
    action_dim: int = 0
    action_count: int = 0
    action_kind: str = ""                    # "discrete" | "continuous"
    env_frame: Optional[np.ndarray] = None   # (H,W,3) uint8 — LIVE-ONLY, NOT serialized
    sanitized_state: Optional[np.ndarray] = None
    state_precision: Optional[np.ndarray] = None
    goal_target: Optional[np.ndarray] = None
    goal_tolerance: float = 0.0
    goal_priority: float = 0.0
    goal_creation_cycle: int = 0
    prediction_precision: Optional[np.ndarray] = None
    gprime_kind: str = ""                    # "gaussian" | "mlp" | "discrete"
    gprime_uncertainty: Optional[np.ndarray] = None   # per-dim std
    gprime_mutual_info: Optional[float] = None
    drive_deficits: List[float] = field(default_factory=list)
    drive_goals: List[Optional[np.ndarray]] = field(default_factory=list)
    drive_goal_norms: List[float] = field(default_factory=list)  # compact ‖drive_goals[i]‖
    goal_stack: List[Dict[str, Any]] = field(default_factory=list)
    pareto_front: List[int] = field(default_factory=list)
    meta_stable: Dict[str, Any] = field(default_factory=dict)
    drive_history: List[List[float]] = field(default_factory=list)
    goal_history: List[int] = field(default_factory=list)
    cr_temperature: float = 0.0
    empowerment: float = 0.0
    tspl_skill_accuracy: float = 0.0
    tspl_skill_compiled: bool = False
    tspl_compiled_skill_ids: List[str] = field(default_factory=list)
    candidate_rollouts: List[Dict[str, Any]] = field(default_factory=list)
    per_dim_peu: Optional[np.ndarray] = None
    attention_weights: Optional[np.ndarray] = None
    attention_precisions: List[float] = field(default_factory=list)
    rbta_bounds: Dict[str, Any] = field(default_factory=dict)
    runtime_log: Dict[str, float] = field(default_factory=dict)
    memory_log: Dict[str, float] = field(default_factory=dict)
    energy_log: Dict[str, float] = field(default_factory=dict)
    belief_entropies: Dict[str, float] = field(default_factory=dict)
    m3_recent: List[Dict[str, Any]] = field(default_factory=list)
    m3_top_error: List[Dict[str, Any]] = field(default_factory=list)
    m4_relevant: List[Dict[str, Any]] = field(default_factory=list)
    m4_top: List[Dict[str, Any]] = field(default_factory=list)
    last_action_vector: Optional[np.ndarray] = None
    # v5: named labels for dimension-adaptive views (default empty → d{i} fallback)
    dim_names: List[str] = field(default_factory=list)
    action_names: List[str] = field(default_factory=list)
    # v5.1: Criticality Φ (gradient-norm criticality, 0..1). Populated only in
    # live mode; defaults to 0.0 on replay/legacy JSONL.
    phi_criticality: float = 0.0

    @classmethod
    def from_cycle(cls, cycle: Any) -> "ObservabilityFrame":
        """Build a snapshot from a live CognitiveCycle (writer side, lock-free)."""
        try:
            return cls._build_from_cycle(cycle)
        except Exception as exc:
            _logger.warning("from_cycle failed; returning empty frame: %s", exc)
            return cls()

    @classmethod
    def _build_from_cycle(cls, cycle: Any) -> "ObservabilityFrame":
        """Inner build — separated so :meth:`from_cycle` can catch & return a
        default-valued frame on partial failure rather than losing the cycle."""
        env = cycle.env
        agent_pos = getattr(env, "agent_pos", None)
        goal_pos = env.get_goal_position() if hasattr(env, "get_goal_position") else None
        grid = None
        if hasattr(env, "grid") and getattr(env, "size", 0) and env.size > 1:
            grid = np.asarray(env.grid).copy()
        pred = getattr(cycle, "last_prediction", None)
        predicted_state = pred.values.copy() if pred is not None else None
        obs_vector = getattr(env, "_last_obs", None)
        if obs_vector is None:
            # GridWorld (and others) often omit _last_obs; use live cycle state.
            cur = getattr(cycle, "current_state", None)
            if cur is not None and getattr(cur, "values", None) is not None:
                obs_vector = cur.values
            else:
                getter = getattr(env, "get_observation", None)
                if callable(getter):
                    try:
                        obs_vector = getter()
                    except Exception as exc:
                        _logger.debug("obs_vector get_observation failed: %s", exc)
                        obs_vector = None
        if obs_vector is not None:
            obs_vector = np.asarray(obs_vector).copy()
        # MuJoCo goal reference (MPC alignment target)
        goal_ref = None
        getter = getattr(env, "get_goal_reference", None)
        if callable(getter):
            try:
                gr = getter()
                if gr is not None:
                    goal_ref = np.asarray(gr, dtype=np.float32).copy()
            except Exception as exc:
                _logger.debug("goal_ref read failed: %s", exc)
                goal_ref = None
        # Chosen continuous action (None for discrete)
        continuous_action = None
        last_act = getattr(cycle, "last_action", None)
        space = getattr(cycle, "action_space", None)
        if getattr(cycle, "_is_continuous", False) and last_act is not None:
            try:
                continuous_action = np.asarray(last_act, dtype=np.float32).copy()
            except Exception as exc:
                _logger.debug("continuous_action read failed: %s", exc)
                continuous_action = None
        mdim = cycle.mdim
        mdim_drives = getattr(mdim, "drives", {})
        drive_levels = [float(mdim_drives[d].value) for d in range(1, 7) if d in mdim_drives]
        targets = getattr(mdim, "_targets", {})
        drive_targets = [float(targets[d]) for d in range(1, 7) if d in targets]
        active_drive_id = int(getattr(getattr(mdim, "current_goal", None), "drive_id", 1)
                              or 1)
        att = cycle.attention
        attention_saliences = list(getattr(att, "_last_saliences", []))
        attention_indices = [int(x) for x in (getattr(att, "_last_selected_indices", []) or [])]
        # Aligned 1:1 with indices for honest Overview bars / replay.
        attention_selected_saliences: List[float] = []
        if attention_indices and attention_saliences:
            max_i = max(attention_indices)
            if len(attention_saliences) > max_i:
                attention_selected_saliences = [
                    float(attention_saliences[i]) for i in attention_indices
                ]
            else:
                attention_selected_saliences = [
                    float(attention_saliences[i]) if i < len(attention_saliences) else 0.0
                    for i in range(len(attention_indices))
                ]
        # RBTA violation details (Observability v2)
        rbta_violations = []
        for v in getattr(cycle, "last_violations", []) or []:
            rbta_violations.append({
                "module_id": getattr(v, "module_id", "?"),
                "bound_type": getattr(v, "bound_type", "?"),
                "measured": float(getattr(v, "measured", 0.0)),
                "allowed": float(getattr(v, "allowed", 0.0)),
            })
        action_rationale = dict(getattr(cycle, "last_action_rationale", {}) or {})
        # Latest CycleMetrics (last appended, not yet pushed to observability)
        m = cycle.metrics_history[-1] if getattr(cycle, "metrics_history", None) else None
        candidate_scores = [float(x) for x in getattr(cycle, "last_candidate_scores", []) or []]
        module_timings = {k: float(v) for k, v in
                          dict(getattr(m, "module_timings", {}) or {}).items()}
        # Retention caps
        m3_cap = 0
        try:
            m3 = getattr(getattr(cycle, "consolidation", None), "m3", None)
            if m3 is not None:
                m3_cap = int(getattr(m3, "_max_episodes", 0))
        except Exception as exc:
            _logger.debug("m3 cap read failed: %s", exc)
            m3_cap = 0
        m4_cap = _M4_MAX_FACTS
        m4_prune_target = _M4_PRUNE_TARGET
        rss = _cached_rss()

        # ── Observability v4: env-agnostic + cognitive portrait ──
        # Environment classification (grid / mujoco_rgb / continuous).
        has_grid = grid is not None
        has_rgb = hasattr(env, "render_rgb")
        env_kind = "grid" if has_grid else ("mujoco_rgb" if has_rgb else "continuous")
        if env_kind != "grid":
            agent_pos = None
            goal_pos = None
        is_cont = bool(getattr(cycle, "_is_continuous", False))
        action_kind = "continuous" if is_cont else "discrete"
        space = getattr(cycle, "action_space", None)
        if is_cont:
            action_dim_v = int(getattr(space, "dim", 0))
            action_count = 0
        else:
            action_dim_v = 0
            action_count = int(getattr(env, "action_space_size", 0))
        state_dim_v = int(getattr(cycle, "state_dim", 0))

        # v5: named dimension + action labels (additive; default empty).
        dim_names: List[str] = []
        try:
            gdn = getattr(env, "get_dim_names", None)
            if callable(gdn):
                dim_names = [str(x) for x in (gdn() or [])][:state_dim_v]
        except Exception as exc:
            _logger.debug("dim_names read failed: %s", exc)
            dim_names = []
        action_names: List[str] = []
        try:
            gan = getattr(env, "get_action_names", None)
            if callable(gan):
                action_names = [str(x) for x in (gan() or [])]
        except Exception as exc:
            _logger.debug("action_names read failed: %s", exc)
            action_names = []

        # RGB camera is captured on the Qt main thread (ObservatoryWindow camera_provider).
        env_frame = None

        # Sanitized state + precision (live-only; obs_vector already serialised).
        sanitized_state = None
        state_precision = None
        cur = getattr(cycle, "current_state", None)
        if cur is not None:
            try:
                sanitized_state = np.asarray(cur.values, dtype=np.float32).copy()
                state_precision = np.asarray(cur.precision, dtype=np.float32).copy()
            except Exception as exc:
                _logger.debug("sanitized_state read failed: %s", exc)
                sanitized_state = None
                state_precision = None

        # Goal portrait.
        goal_target = goal_tol = goal_pri = goal_cc = None
        g = getattr(cycle, "current_goal", None)
        if g is not None:
            tgt = getattr(g, "target_state", None)
            if tgt is not None:
                try:
                    goal_target = np.asarray(tgt.values, dtype=np.float32).copy()
                except Exception as exc:
                    _logger.debug("goal_target read failed: %s", exc)
                    goal_target = None
            goal_tol = float(getattr(g, "tolerance", 0.0))
            goal_pri = float(getattr(g, "priority", 0.0))
            goal_cc = int(getattr(g, "creation_cycle", 0))

        # Prediction precision.
        prediction_precision = None
        if pred is not None:
            try:
                prediction_precision = np.asarray(pred.precision,
                                                  dtype=np.float32).copy()
            except Exception as exc:
                _logger.debug("prediction_precision read failed: %s", exc)
                prediction_precision = None

        # Additive module snapshots (defensive — default {} if absent).
        # tspl theta-norms and mdim goal-stack/drive-history are expensive and
        # slow-changing → TTL-cached. Deficits are fast-changing and cheap, so
        # they are refreshed inline below to keep the radar live.
        mdim_snap = _cached_snap("mdim", cycle.mdim, "snapshot", 0.20)
        gprime_snap = _cached_snap("gprime", cycle.gprime, "uncertainty_snapshot", 0.10)
        tspl_snap = _cached_snap("tspl", cycle.tspl, "snapshot", 0.50)
        att_snap = _cached_snap("attention", cycle.attention, "snapshot", 0.20)
        rbta_bounds = _cached_snap("rbta_bounds", cycle.rbta, "bounds_snapshot", 0.50)

        # Fresh deficits (fast-changing) straight from MDIM drives — 6 attribute
        # reads, cheap; keeps the radar live despite the rest being TTL-cached.
        try:
            mdim_drives = getattr(cycle.mdim, "drives", {}) or {}
            mdim_snap["deficits"] = [float(getattr(mdim_drives.get(d),
                                                   "deficit", 0.0))
                                     for d in range(1, 7)]
        except Exception as exc:
            _logger.debug("deficits read failed: %s", exc)

        drive_deficits = [float(x) for x in mdim_snap.get("deficits", [])]
        drive_goals = [np.asarray(t, dtype=np.float32).copy()
                       if t is not None else None
                       for t in mdim_snap.get("goals", [])]
        drive_goal_norms = [
            float(np.linalg.norm(g)) if g is not None else 0.0
            for g in drive_goals
        ]
        goal_stack = list(mdim_snap.get("goal_stack", []))
        pareto_front = [int(x) for x in mdim_snap.get("pareto_front", [])]
        meta_stable = dict(mdim_snap.get("meta_stable", {}))
        drive_history = [[float(v) for v in row]
                         for row in mdim_snap.get("drive_history", [])]
        goal_history = [int(x) for x in mdim_snap.get("goal_history", [])]
        cr_temperature = float(getattr(cycle.mdim, "temperature", 0.0))

        gprime_kind = str(gprime_snap.get("kind", ""))
        gprime_uncertainty = gprime_snap.get("per_dim_std")
        if gprime_uncertainty is not None:
            # G' builds a fresh array each predict(); reference is safe and the
            # to_json path .tolist()s it, so no copy needed here.
            gprime_uncertainty = np.asarray(gprime_uncertainty, dtype=np.float32)
        gprime_mutual_info = gprime_snap.get("mutual_info")
        if gprime_mutual_info is not None:
            gprime_mutual_info = float(gprime_mutual_info)

        tspl_skill_accuracy = float(tspl_snap.get("skill_accuracy",
                                   getattr(cycle.tspl, "skill_accuracy", 0.0)))
        tspl_skill_compiled = bool(tspl_snap.get("skill_compiled",
                                   getattr(cycle.tspl, "skill_compiled", False)))
        tspl_compiled_skill_ids = [str(x) for x in
                                   tspl_snap.get("compiled_skill_ids", [])]

        # Rollouts / per-dim PEU / empowerment: captured on the hot path
        # behind the observability guard (cycle.py). Read defensively.
        rollouts = []
        for r in getattr(cycle, "last_candidate_rollouts", []) or []:
            try:
                # Live-only (excluded from JSONL); rollouts are rebuilt every
                # cycle in cycle.py so a reference (no copy) is safe and cheaper.
                rollouts.append({
                    "action": np.asarray(r["action"], dtype=np.float32),
                    "predicted": np.asarray(r["predicted"], dtype=np.float32),
                    "score": float(r.get("score", 0.0)),
                    "chosen": bool(r.get("chosen", False)),
                })
            except Exception as exc:
                _logger.debug("rollout read failed: %s", exc)
                continue
        per_dim_peu = getattr(cycle, "last_per_dim_peu", None)
        per_dim_peu_top: List[Dict[str, Any]] = []
        per_dim_peu_sum = 0.0
        if per_dim_peu is not None:
            per_dim_peu = np.asarray(per_dim_peu, dtype=np.float32)
            flat = per_dim_peu.reshape(-1)
            per_dim_peu_sum = float(np.sum(flat)) if flat.size else 0.0
            if flat.size:
                k = min(16, int(flat.size))
                top_idx = np.argpartition(flat, -k)[-k:]
                top_idx = top_idx[np.argsort(-flat[top_idx])]
                per_dim_peu_top = [
                    {"idx": int(i), "value": float(flat[i])} for i in top_idx
                ]
        empowerment = float(getattr(cycle, "last_empowerment", 0.0) or 0.0)

        attention_weights = getattr(cycle, "_attention_weights", None)
        if attention_weights is not None:
            attention_weights = np.asarray(attention_weights, dtype=np.float32)
        attention_precisions = [float(x) for x in att_snap.get("precisions", [])]

        # Aggregate logs change slowly → TTL-cached (avoids 4 dict-comprehensions
        # every cycle). Belief entropies similarly.
        def _cached_log(key: str, obj: Any, attr: str, ttl: float = 0.30):
            now = time.monotonic()
            cache_key = f"{key}:{id(obj)}:{type(obj).__name__}"
            ent = _SNAP_CACHE.get(cache_key)
            if ent is not None and now - ent[0] < ttl:
                _SNAP_CACHE.move_to_end(cache_key)
                return dict(ent[1])
            try:
                d = {str(k): float(v) for k, v in
                     dict(getattr(obj, attr, {}) or {}).items()}
            except Exception as exc:
                _logger.debug("cached_log %s.%s failed: %s", type(obj).__name__, attr, exc)
                d = {}
            _SNAP_CACHE[cache_key] = (now, d)
            _SNAP_CACHE.move_to_end(cache_key)
            return dict(d)
        runtime_log = _cached_log("runtime_log", cycle, "runtime_log")
        memory_log = _cached_log("memory_log", cycle, "memory_log")
        energy_log = _cached_log("energy_log", cycle, "energy_log")
        belief_entropies = _cached_log("belief_entropies", cycle, "belief_entropies")

        last_action_vector = None
        la = getattr(cycle, "last_action", None)
        if la is not None:
            try:
                last_action_vector = np.asarray(la, dtype=np.float32)
            except Exception as exc:
                _logger.debug("last_action_vector read failed: %s", exc)
                last_action_vector = None

        # Throttled memory samples (live-only).
        mem = _cached_memory(cycle, m3)
        m3_recent = mem.get("m3_recent", [])
        m3_top_error = mem.get("m3_top_error", [])
        m4_relevant = mem.get("m4_relevant", [])
        m4_top = mem.get("m4_top", [])

        agent_id = int(getattr(cycle, "observability_agent_id", 0) or 0)
        agent_label = str(getattr(cycle, "observability_agent_label", "") or "")
        timeline_step = int(getattr(cycle, "observability_timeline_step", -1) or -1)
        phi_criticality = float(getattr(cycle, "last_error_volatility", 0.0) or 0.0)

        return cls(
            cycle_id=cycle.cycle_count,
            agent_id=agent_id,
            agent_label=agent_label,
            timeline_step=timeline_step,
            agent_pos=tuple(agent_pos) if agent_pos is not None else None,
            goal_pos=tuple(goal_pos) if goal_pos is not None else None,
            grid=grid,
            predicted_state=predicted_state,
            obs_vector=obs_vector,
            goal_ref=goal_ref,
            continuous_action=continuous_action,
            drive_levels=drive_levels,
            drive_targets=drive_targets,
            active_drive_id=active_drive_id,
            attention_saliences=attention_saliences,
            attention_indices=attention_indices,
            attention_selected_saliences=attention_selected_saliences,
            rbta_violations=rbta_violations,
            action_rationale=action_rationale,
            candidate_scores=candidate_scores,
            module_timings=module_timings,
            latency_ms=getattr(m, "latency_ms", 0.0),
            prediction_error=getattr(m, "prediction_error", 0.0),
            prediction_confidence=getattr(m, "prediction_confidence", 0.0),
            prediction_quality=float(getattr(m, "prediction_quality", 1.0) or 1.0),
            display_confidence=float(
                getattr(m, "display_confidence", 0.0)
                or min(
                    float(getattr(m, "prediction_confidence", 0.0) or 0.0),
                    float(getattr(m, "prediction_quality", 1.0) or 1.0),
                )
            ),
            per_dim_peu_top=per_dim_peu_top,
            per_dim_peu_sum=per_dim_peu_sum,
            rbta_action=getattr(m, "rbta_action", "CONTINUE"),
            violations_count=getattr(m, "violations_count", 0),
            goal_reached=getattr(m, "goal_reached", False),
            action_name=getattr(m, "action_name", ""),
            drive_id=getattr(m, "drive_id", 1),
            episode_count=getattr(m, "episode_count", 0),
            m3_count=getattr(m, "m3_count", 0),
            fact_count=getattr(m, "fact_count", 0),
            task_id=getattr(m, "task_id", -1),
            failure_events=list(getattr(m, "failure_events", []) or []),
            recovery_active=bool(getattr(m, "recovery_active", False)),
            m3_cap=m3_cap,
            m4_cap=m4_cap,
            m4_prune_target=m4_prune_target,
            rss_bytes=rss,
            env_kind=env_kind,
            state_dim=state_dim_v,
            action_dim=action_dim_v,
            action_count=action_count,
            action_kind=action_kind,
            env_frame=env_frame,
            sanitized_state=sanitized_state,
            state_precision=state_precision,
            goal_target=goal_target,
            goal_tolerance=goal_tol or 0.0,
            goal_priority=goal_pri or 0.0,
            goal_creation_cycle=goal_cc or 0,
            prediction_precision=prediction_precision,
            gprime_kind=gprime_kind,
            gprime_uncertainty=gprime_uncertainty,
            gprime_mutual_info=gprime_mutual_info,
            drive_deficits=drive_deficits,
            drive_goals=drive_goals,
            drive_goal_norms=drive_goal_norms,
            goal_stack=goal_stack,
            pareto_front=pareto_front,
            meta_stable=meta_stable,
            drive_history=drive_history,
            goal_history=goal_history,
            cr_temperature=cr_temperature,
            empowerment=empowerment,
            tspl_skill_accuracy=tspl_skill_accuracy,
            tspl_skill_compiled=tspl_skill_compiled,
            tspl_compiled_skill_ids=tspl_compiled_skill_ids,
            candidate_rollouts=rollouts,
            per_dim_peu=per_dim_peu,
            attention_weights=attention_weights,
            attention_precisions=attention_precisions,
            rbta_bounds=rbta_bounds,
            runtime_log=runtime_log,
            memory_log=memory_log,
            energy_log=energy_log,
            belief_entropies=belief_entropies,
            m3_recent=m3_recent,
            m3_top_error=m3_top_error,
            m4_relevant=m4_relevant,
            m4_top=m4_top,
            last_action_vector=last_action_vector,
            dim_names=dim_names,
            action_names=action_names,
            phi_criticality=phi_criticality,
        )

    def to_json(self) -> Dict[str, Any]:
        """Single-pass JSON-friendly serialisation.

        Live-only fields (RGB camera frame, sanitized state, full drive_goals
        vectors, candidate rollouts, bulk M3/M4 lists) are EXCLUDED. Compact
        samples remain: ``m3_top_error``, ``m4_top``, ``drive_goal_norms``,
        ``per_dim_peu_top``, ``attention_selected_saliences``.
        """
        d: Dict[str, Any] = {}
        # ── Scalar integers ──
        for k in ("cycle_id", "episode_count", "m3_count", "fact_count", "task_id",
                   "violations_count", "drive_id", "active_drive_id",
                   "rss_bytes", "m3_cap", "m4_cap", "m4_prune_target",
                   "state_dim", "action_dim", "action_count",
                   "goal_creation_cycle", "schema_version"):
            v = getattr(self, k, None)
            if v is not None:
                d[k] = int(v)
        # ── String labels ──
        for k in ("agent_label", "env_kind", "action_kind", "gprime_kind",
                  "rbta_action", "action_name"):
            v = getattr(self, k, None)
            if v:
                d[k] = str(v)
        # ── Float scalars ──
        for k in ("latency_ms", "prediction_error", "prediction_confidence",
                  "prediction_quality", "display_confidence", "per_dim_peu_sum",
                  "goal_tolerance", "goal_priority", "cr_temperature",
                  "empowerment", "tspl_skill_accuracy", "gprime_mutual_info",
                  "phi_criticality"):
            v = getattr(self, k, None)
            if v is not None:
                d[k] = float(v)
        # ── Numpy array fields → list ──
        for arr_field in ("grid", "predicted_state", "obs_vector", "goal_ref",
                          "continuous_action", "gprime_uncertainty"):
            v = getattr(self, arr_field, None)
            if v is not None:
                d[arr_field] = np.asarray(v).tolist()
        # ── Float-list fields ──
        for lst in ("drive_levels", "drive_targets", "attention_saliences",
                    "attention_selected_saliences", "candidate_scores",
                    "drive_deficits", "drive_goal_norms", "attention_precisions"):
            d[lst] = [float(v) for v in getattr(self, lst, [])]
        # Compact PEU top-K for Flow/Phase replay (full per_dim_peu stays live-only)
        d["per_dim_peu_top"] = [
            {"idx": int(x.get("idx", 0)), "value": float(x.get("value", 0.0))}
            for x in (getattr(self, "per_dim_peu_top", None) or [])
            if isinstance(x, dict)
        ]
        # Named labels for replay/UI
        d["dim_names"] = [str(x) for x in (getattr(self, "dim_names", None) or [])]
        d["action_names"] = [str(x) for x in (getattr(self, "action_names", None) or [])]
        # ── String-list fields ──
        for lst in ("failure_events",):
            d[lst] = list(getattr(self, lst, []))
        # ── Bool fields (always emit goal_reached so replay/report stay honest) ──
        d["goal_reached"] = bool(getattr(self, "goal_reached", False))
        for k in ("recovery_active",):
            v = getattr(self, k, None)
            if v is not None:
                d[k] = bool(v)
        # ── Int-list fields ──
        for lst in ("attention_indices", "pareto_front", "goal_history"):
            d[lst] = [int(v) for v in getattr(self, lst, [])]
        # ── Grid position tuples (int list, cleared for non-grid envs) ──
        if self.agent_pos is not None:
            d["agent_pos"] = [int(v) for v in self.agent_pos]
        if self.goal_pos is not None:
            d["goal_pos"] = [int(v) for v in self.goal_pos]
        if str(getattr(self, "env_kind", "") or "") != "grid":
            d["agent_pos"] = None
            d["goal_pos"] = None
        # ── string-keyed float dicts ──
        for dct in ("module_timings", "runtime_log", "memory_log",
                    "energy_log", "belief_entropies"):
            d[dct] = {str(k): float(v) for k, v in getattr(self, dct, {}).items()}
        # ── Downsampled drive-history ──
        dh = self.drive_history[-24:] if self.drive_history else []
        d["drive_history"] = [[round(float(v), 3) for v in row] for row in dh]
        # ── Nested complex structures (recursive helper, only 3 fields) ──
        for nested in ("rbta_bounds", "goal_stack", "meta_stable"):
            v = getattr(self, nested, None)
            if v is not None and v:
                d[nested] = _recursive_json(v)
        if self.action_rationale:
            d["action_rationale"] = _recursive_json(self.action_rationale)
        if self.m3_top_error:
            d["m3_top_error"] = _recursive_json(self.m3_top_error)
        # Compact M4 top-8 for Memory tab scrub (full m4_relevant stays live-only)
        m4 = getattr(self, "m4_top", None) or []
        if m4:
            d["m4_top"] = [
                {
                    "fact_type": str(x.get("fact_type", "?")),
                    "summary": str(x.get("summary", ""))[:80],
                    "confidence": float(x.get("confidence", 0.0) or 0.0),
                    "timestamp": int(x.get("timestamp", 0) or 0),
                    "frequency": int(x.get("frequency", 0) or 0),
                }
                for x in m4[:8]
                if isinstance(x, dict)
            ]
        # ── Skill IDs ──
        d["tspl_compiled_skill_ids"] = [str(x) for x in self.tspl_compiled_skill_ids]
        # ── Agent ──
        d["agent_id"] = int(self.agent_id)
        d["timeline_step"] = int(self.timeline_step)
        # ── Live-only: never serialised ──
        for drop in ("env_frame", "drive_goals", "candidate_rollouts",
                     "m3_recent", "m4_relevant",
                     "sanitized_state", "state_precision", "goal_target",
                     "prediction_precision", "per_dim_peu",
                     "attention_weights", "attention_precisions",
                     "last_action_vector"):
            d.pop(drop, None)
        return d


def observability_schema_version(obj: Dict[str, Any]) -> int:
    """Return the Observatory JSON schema version; missing means legacy v0."""
    raw = obj.get("schema_version", 0)
    if isinstance(raw, bool):
        raise ValueError("schema_version must be an integer")
    try:
        version = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"schema_version must be an integer, got {raw!r}") from exc
    if version < 0:
        raise ValueError(f"schema_version must be >= 0, got {version}")
    return version


def normalize_observability_json(obj: Dict[str, Any]) -> Dict[str, Any]:
    """Copy and normalize a JSONL object before replay/report reconstruction.

    v0 is the pre-Phase-9 legacy shape with no explicit schema marker. v1 is
    the current schema. Unknown future versions fail closed so replay/check
    does not silently misrepresent data.
    """
    if not isinstance(obj, dict):
        raise ValueError("observability JSON record must be an object")
    version = observability_schema_version(obj)
    if version not in SUPPORTED_OBSERVABILITY_SCHEMA_VERSIONS:
        supported = sorted(SUPPORTED_OBSERVABILITY_SCHEMA_VERSIONS)
        raise ValueError(
            f"unsupported observability schema_version={version} "
            f"(supported: {supported})"
        )
    out = json.loads(json.dumps(obj))
    out["schema_version"] = version
    if "agent_id" not in out:
        out["agent_id"] = 0
    if "agent_label" not in out:
        out["agent_label"] = ""
    if "timeline_step" not in out:
        out["timeline_step"] = -1
    # Legacy JSONL omitted goal_reached; default False so replay stays explicit.
    if "goal_reached" not in out:
        out["goal_reached"] = False
    else:
        out["goal_reached"] = bool(out["goal_reached"])
    return out


def slim_frame_for_ui_history(frame: ObservabilityFrame) -> ObservabilityFrame:
    """Shrink heavy arrays on a frame already serialized to JSONL.

    Mutates in place. Keeps compact fields (peu_top, uncertainty as float16,
    obs/predicted as float16) so scrub UI still works with less RSS.
    """
    for name in ("predicted_state", "obs_vector", "gprime_uncertainty", "goal_ref"):
        v = getattr(frame, name, None)
        if v is not None:
            try:
                setattr(frame, name, np.asarray(v, dtype=np.float16))
            except Exception:
                pass
    frame.candidate_rollouts = []
    frame.sanitized_state = None
    frame.state_precision = None
    frame.env_frame = None
    frame.drive_goals = []
    frame.per_dim_peu = None
    frame.attention_weights = None
    frame.prediction_precision = None
    frame.last_action_vector = None
    frame.m3_recent = []
    frame.m4_relevant = []
    return frame


class ObservabilityStore:
    """Thread-safe ring buffer of ObservabilityFrame (mirrors MetricsStore)."""

    def __init__(self, maxlen: int = 1000):
        self._deque: deque = deque(maxlen=maxlen)
        self._lock = threading.Lock()

    def push(self, frame: ObservabilityFrame) -> None:
        with self._lock:
            self._deque.append(frame)

    def latest(self) -> Optional[ObservabilityFrame]:
        with self._lock:
            return self._deque[-1] if self._deque else None

    def snapshot(self) -> List[ObservabilityFrame]:
        with self._lock:
            return list(self._deque)

    def latest_n(self, k: int) -> List[ObservabilityFrame]:
        """Bounded tail copy (k most recent) — avoids full-deque copy per tick."""
        with self._lock:
            n = len(self._deque)
            if n <= k:
                return list(self._deque)
            it = iter(self._deque)
            for _ in range(n - k):
                next(it)
            return [next(it) for _ in range(k)]

    def frames_after(self, cycle_id: int, agent_id: int = 0) -> List[ObservabilityFrame]:
        """Return retained frames newer than ``cycle_id`` for ``agent_id``."""
        with self._lock:
            return [
                f for f in self._deque
                if int(getattr(f, "agent_id", 0)) == int(agent_id)
                and int(getattr(f, "cycle_id", -1)) > int(cycle_id)
            ]

    def __len__(self) -> int:
        with self._lock:
            return len(self._deque)


class SessionRecorder:
    """Persists the per-cycle time-series to logs/sessions/<ts>/ as JSONL only.

    Observability v2: PNG frames were a storage firehose (thousands of ~100KB
    images per run). Visual frames are now captured by VideoRecorder into a
    single encoded video file; this recorder keeps only the cheap, full-fidelity
    scalar/vector JSONL (~2KB/cycle) for analytics and replay-time reconstruction.
    """

    def __init__(self, root: str = "logs/sessions", fps: float = 5.0,
                 record: bool = True):
        self.root = Path(root)
        self.fps = float(fps)
        self.enabled = bool(record)
        self.session_dir: Optional[Path] = None
        self._jsonl = None
        self._count = 0
        self._error: Optional[str] = None
        self._last_flush_t: float = 0.0

    @staticmethod
    def _utc_now() -> str:
        from datetime import datetime, timezone
        return datetime.now(timezone.utc).replace(microsecond=0).isoformat()

    def _latest_pointer_path(self) -> Path:
        return self.root / ".latest"

    def _write_latest_pointer(self) -> None:
        if self.session_dir is None:
            return
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            self._latest_pointer_path().write_text(str(self.session_dir.resolve()))
        except OSError:
            pass

    def _clear_latest_pointer(self) -> None:
        try:
            p = self._latest_pointer_path()
            if p.exists():
                p.unlink()
        except OSError:
            pass

    def _patch_meta(self, patch: Dict[str, Any]) -> None:
        if self.session_dir is None or not self.enabled:
            return
        meta_p = self.session_dir / "meta.json"
        try:
            meta = json.loads(meta_p.read_text()) if meta_p.exists() else {}
            meta.update(patch)
            meta_p.write_text(json.dumps(meta, indent=2))
        except (OSError, json.JSONDecodeError):
            pass

    def start(self, meta: Dict[str, Any]) -> Optional[Path]:
        if not self.enabled:
            return None
        from datetime import datetime
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = self.root / ts
        self.session_dir.mkdir(parents=True, exist_ok=True)
        meta = {
            **meta,
            "fps": self.fps,
            "observability_schema_version": OBSERVABILITY_SCHEMA_VERSION,
            "status": "running",
            "started_at": self._utc_now(),
        }
        (self.session_dir / "meta.json").write_text(json.dumps(meta, indent=2))
        self._jsonl = open(self.session_dir / "timeseries.jsonl", "w", encoding="utf-8")
        self._write_latest_pointer()
        return self.session_dir

    def record(self, frame: ObservabilityFrame, fig: Any = None) -> None:
        """Append one JSONL line. ``fig`` is ignored (video is handled by VideoRecorder)."""
        if not self.enabled or self._jsonl is None:
            return
        try:
            self._jsonl.write(json.dumps(frame.to_json()) + "\n")
            self._count += 1
            if time.monotonic() - self._last_flush_t >= 5.0:
                self._jsonl.flush()
                self._last_flush_t = time.monotonic()
        except Exception as e:
            # Record the first failure so the caller can surface it; keep going.
            if self._error is None:
                self._error = str(e)

    def flush(self) -> None:
        if self._jsonl is not None:
            try:
                self._jsonl.flush()
            except OSError:
                pass

    @property
    def count(self) -> int:
        return self._count

    @property
    def error(self) -> Optional[str]:
        return self._error

    def _finalize_jsonl(self) -> None:
        if self._jsonl is not None:
            try:
                self._jsonl.close()
            except OSError:
                pass
            self._jsonl = None

    def close(self, *, incomplete: bool = False, reason: str = "") -> None:
        """Finalize a normal session.

        When ``incomplete=True`` (short run / cycle error), status is
        ``incomplete`` or ``empty`` — never claim ``complete``.
        """
        if incomplete:
            self.abort(reason or "short_run")
            return
        self.flush()
        self._finalize_jsonl()
        if self.session_dir is not None and self.enabled:
            self._patch_meta({
                "status": "complete",
                "recorded_cycles": self._count,
                "closed_at": self._utc_now(),
            })
            self._clear_latest_pointer()

    def abort(self, reason: str = "abnormal") -> None:
        """Best-effort finalize after crash or early exit (main thread only)."""
        self.flush()
        self._finalize_jsonl()
        if self.session_dir is None or not self.enabled:
            return
        if self._count > 0:
            status = "incomplete"
        else:
            status = "empty"
        self._patch_meta({
            "status": status,
            "recorded_cycles": self._count,
            "closed_at": self._utc_now(),
            "recovery_reason": reason,
        })
        self._clear_latest_pointer()


class VideoRecorder:
    """Captures the live matplotlib figure into ONE encoded video file.

    DEPRECATED for PyQt Observatory: use ``phca_observatory.py`` QPixmap capture
    instead. Retained for legacy ``phca_visualise.py`` (matplotlib dashboard).

    Replaces the per-frame PNG firehose. Uses FFMpegWriter (mp4) when ffmpeg is
    available, falling back to PillowWriter (gif). grab_frame() reuses the
    figure's already-drawn canvas (the visualiser calls plt.pause to render to
    screen first), so there is no second full rasterization to disk.
    """

    def __init__(self, fps: float = 5.0, dpi: int = 90, format: str = "auto"):
        self.fps = float(fps)
        self.dpi = int(dpi)
        self.format = format  # "auto" | "mp4" | "gif"
        self.path: Optional[Path] = None
        self._writer: Any = None
        self._grab_ctx: Any = None
        self._frames = 0
        self.enabled = True

    @staticmethod
    def _resolve_writer(format: str) -> Tuple[Any, str]:
        from matplotlib.animation import FFMpegWriter, PillowWriter
        if format in ("mp4", "auto"):
            try:
                w = FFMpegWriter(fps=5)  # fps set properly in start(); just probing
                if w.bin_path() is not None:
                    return FFMpegWriter, ".mp4"
            except Exception:
                pass
        if format in ("gif", "auto"):
            return PillowWriter, ".gif"
        raise ValueError(f"unsupported video format: {format}")

    def start(self, session_dir: Path, fig: Any) -> Tuple[Optional[Path], str]:
        if not self.enabled:
            return None, "disabled"
        try:
            WriterCls, ext = self._resolve_writer(self.format)
        except Exception as e:
            self.enabled = False
            return None, f"no writer: {e}"
        self._writer = WriterCls(fps=self.fps)
        self.path = Path(session_dir) / f"session{ext}"
        try:
            self._grab_ctx = self._writer.saving(fig, str(self.path), self.dpi)
            self._grab_ctx.__enter__()
        except Exception as e:
            self.enabled = False
            self._writer = None
            return None, f"writer.open failed: {e}"
        return self.path, ext.lstrip(".")

    def grab(self, fig: Any) -> None:
        if not self.enabled or self._writer is None:
            return
        try:
            self._writer.grab_frame()
            self._frames += 1
        except Exception:
            # A single dropped frame must never crash the live run.
            pass

    @property
    def frames(self) -> int:
        return self._frames

    def close(self) -> None:
        if self._grab_ctx is not None:
            try:
                self._grab_ctx.__exit__(None, None, None)
            except Exception:
                pass
            self._grab_ctx = None
        self._writer = None
