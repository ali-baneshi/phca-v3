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
import os
import threading
import time
from collections import deque
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Retention caps (mirrored from consolidation.scheduler for the dashboard).
# Imported lazily/defensively so the monitoring package never hard-fails if the
# consolidation module is refactored.
try:
    from phca.consolidation.scheduler import M4_MAX_FACTS as _M4_MAX_FACTS, \
        M4_PRUNE_TARGET as _M4_PRUNE_TARGET
except Exception:
    _M4_MAX_FACTS = 1_000
    _M4_PRUNE_TARGET = 500

try:
    import psutil
    _HAVE_PSUTIL = True
except Exception:
    _HAVE_PSUTIL = False

_PROC = psutil.Process() if _HAVE_PSUTIL else None

# RSS changes slowly; throttle the syscall so the per-cycle frame build stays
# well under the 5% latency-overhead gate. Refresh at most every 0.25s.
_RSS_CACHE: Dict[int, Tuple[float, int]] = {}
_RSS_TTL_S = 0.25

# Live-only RGB camera frame (MuJoCo). Rendering is ~5-10ms so we throttle to
# ~2.5 Hz; the dashboard reuses the cached frame between refreshes. The frame
# is NEVER serialised to JSONL (image firehose) — it reaches mp4 via the
# dashboard's own QPixmap.grab during --record-video.
_ENV_FRAME_CACHE: Dict[int, Tuple[float, Optional[np.ndarray]]] = {}
_ENV_FRAME_TTL_S = 0.4

# Heavy memory samples (M3 episodes + M4 facts) are read off the cycle thread
# at most every 0.5s so the per-cycle frame build stays cheap. These too are
# live-only (excluded from JSONL).
_MEM_CACHE: Dict[int, Tuple[float, Dict[str, Any]]] = {}
_MEM_TTL_S = 0.5


def _cached_rss() -> int:
    if _PROC is None:
        return 0
    pid = _PROC.pid
    now = time.monotonic()
    ts, val = _RSS_CACHE.get(pid, (0.0, 0))
    if now - ts > _RSS_TTL_S:
        try:
            val = int(_PROC.memory_info().rss)
            _RSS_CACHE[pid] = (now, val)
        except Exception:
            val = 0
    return val


def _cached_env_frame(env: Any) -> Optional[np.ndarray]:
    """Throttled live RGB camera frame (MuJoCo rgb_array). None if unavailable."""
    getter = getattr(env, "render_rgb", None)
    if not callable(getter):
        return None
    pid = os.getpid()
    now = time.monotonic()
    ts, val = _ENV_FRAME_CACHE.get(pid, (0.0, None))
    if val is None or now - ts > _ENV_FRAME_TTL_S:
        try:
            frame = getter()
            val = np.asarray(frame).copy() if frame is not None else None
            _ENV_FRAME_CACHE[pid] = (now, val)
        except Exception:
            val = None
    return val


def _fact_to_dict(f: Any) -> Dict[str, Any]:
    return {
        "fact_type": getattr(f, "fact_type", "?"),
        "confidence": float(getattr(f, "confidence", 0.0)),
        "frequency": int(getattr(f, "frequency", getattr(f, "support", 0)) or 0),
        "summary": str(getattr(f, "summary", getattr(f, "description", "")))[:80],
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
    pid = os.getpid()
    now = time.monotonic()
    ts, val = _MEM_CACHE.get(pid, (0.0, {}))
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
    except Exception:
        pass
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
    except Exception:
        pass
    _MEM_CACHE[pid] = (now, out)
    return out


def _snap(obj: Any, method: str) -> Dict[str, Any]:
    """Defensively call an additive snapshot() accessor; {} if absent/raises."""
    fn = getattr(obj, method, None)
    if not callable(fn):
        return {}
    try:
        return dict(fn() or {})
    except Exception:
        return {}


# TTL cache for the *expensive* additive snapshots (tspl theta-norms, mdim
# goal-stack/pareto/drive-history). These change slowly (only after learning /
# goal transitions), so recomputing them every cycle is pure overhead. Cheap,
# fast-changing signals (deficits, scalars) are still read fresh below by
# overwriting the cached dict's deficits. Keeps the ≤5% overhead gate in reach.
_SNAP_CACHE: Dict[str, Tuple[float, Dict[str, Any]]] = {}


def _cached_snap(key: str, obj: Any, method: str, ttl: float) -> Dict[str, Any]:
    now = time.monotonic()
    ent = _SNAP_CACHE.get(key)
    if ent is not None and now - ent[0] < ttl:
        return dict(ent[1])  # shallow copy so callers can't mutate the cache
    val = _snap(obj, method)
    _SNAP_CACHE[key] = (now, val)
    return dict(val)


@dataclass
class ObservabilityFrame:
    """Immutable snapshot of one cognitive cycle's world + mind state."""
    cycle_id: int = 0
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
    rbta_violations: List[Dict[str, Any]] = field(default_factory=list)  # module/bound_type/measured/allowed
    action_rationale: Dict[str, Any] = field(default_factory=dict)  # explored/eps/goal_id/best_score/k_candidates
    candidate_scores: List[float] = field(default_factory=list)  # per-action / per-candidate scores
    module_timings: Dict[str, float] = field(default_factory=dict)  # per-module ms (cognitive flow)
    # CycleMetrics scalar mirrors
    latency_ms: float = 0.0
    prediction_error: float = 0.0
    prediction_confidence: float = 0.0
    rbta_action: str = "CONTINUE"
    violations_count: int = 0
    goal_reached: bool = False
    action_name: str = ""
    drive_id: int = 1
    episode_count: int = 0
    fact_count: int = 0
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
    m4_relevant: List[Dict[str, Any]] = field(default_factory=list)
    m4_top: List[Dict[str, Any]] = field(default_factory=list)
    last_action_vector: Optional[np.ndarray] = None
    # v5: named labels for dimension-adaptive views (default empty → d{i} fallback)
    dim_names: List[str] = field(default_factory=list)
    action_names: List[str] = field(default_factory=list)

    @classmethod
    def from_cycle(cls, cycle: Any) -> "ObservabilityFrame":
        """Build a snapshot from a live CognitiveCycle (writer side, lock-free)."""
        env = cycle.env
        agent_pos = getattr(env, "agent_pos", None)
        goal_pos = env.get_goal_position() if hasattr(env, "get_goal_position") else None
        grid = None
        if hasattr(env, "grid") and getattr(env, "size", 0) and env.size > 1:
            grid = np.asarray(env.grid).copy()
        pred = getattr(cycle, "last_prediction", None)
        predicted_state = pred.values.copy() if pred is not None else None
        obs_vector = getattr(env, "_last_obs", None)
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
            except Exception:
                goal_ref = None
        # Chosen continuous action (None for discrete)
        continuous_action = None
        last_act = getattr(cycle, "last_action", None)
        space = getattr(cycle, "action_space", None)
        if getattr(cycle, "_is_continuous", False) and last_act is not None:
            try:
                continuous_action = np.asarray(last_act, dtype=np.float32).copy()
            except Exception:
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
        attention_indices = list(getattr(att, "_last_selected_indices", []))
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
        except Exception:
            m3_cap = 0
        m4_cap = _M4_MAX_FACTS
        m4_prune_target = _M4_PRUNE_TARGET
        rss = _cached_rss()

        # ── Observability v4: env-agnostic + cognitive portrait ──
        # Environment classification (grid / mujoco_rgb / continuous).
        has_grid = grid is not None
        has_rgb = hasattr(env, "render_rgb")
        env_kind = "grid" if has_grid else ("mujoco_rgb" if has_rgb else "continuous")
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
        except Exception:
            dim_names = []
        action_names: List[str] = []
        try:
            gan = getattr(env, "get_action_names", None)
            if callable(gan):
                action_names = [str(x) for x in (gan() or [])]
        except Exception:
            action_names = []

        # Live RGB camera frame (throttled, live-only).
        env_frame = _cached_env_frame(env)

        # Sanitized state + precision (live-only; obs_vector already serialised).
        sanitized_state = None
        state_precision = None
        cur = getattr(cycle, "current_state", None)
        if cur is not None:
            try:
                sanitized_state = np.asarray(cur.values, dtype=np.float32).copy()
                state_precision = np.asarray(cur.precision, dtype=np.float32).copy()
            except Exception:
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
                except Exception:
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
            except Exception:
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
        except Exception:
            pass

        drive_deficits = [float(x) for x in mdim_snap.get("deficits", [])]
        drive_goals = [np.asarray(t, dtype=np.float32).copy()
                       if t is not None else None
                       for t in mdim_snap.get("goals", [])]
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
            except Exception:
                continue
        per_dim_peu = getattr(cycle, "last_per_dim_peu", None)
        if per_dim_peu is not None:
            per_dim_peu = np.asarray(per_dim_peu, dtype=np.float32)
        empowerment = float(getattr(cycle, "last_empowerment", 0.0) or 0.0)

        attention_weights = getattr(cycle, "_attention_weights", None)
        if attention_weights is not None:
            attention_weights = np.asarray(attention_weights, dtype=np.float32)
        attention_precisions = [float(x) for x in att_snap.get("precisions", [])]

        # Aggregate logs change slowly → TTL-cached (avoids 4 dict-comprehensions
        # every cycle). Belief entropies similarly.
        def _cached_log(key: str, obj: Any, attr: str, ttl: float = 0.30):
            now = time.monotonic()
            ent = _SNAP_CACHE.get(key)
            if ent is not None and now - ent[0] < ttl:
                return dict(ent[1])
            try:
                d = {str(k): float(v) for k, v in
                     dict(getattr(obj, attr, {}) or {}).items()}
            except Exception:
                d = {}
            _SNAP_CACHE[key] = (now, d)
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
            except Exception:
                last_action_vector = None

        # Throttled memory samples (live-only).
        mem = _cached_memory(cycle, m3)
        m3_recent = mem.get("m3_recent", [])
        m4_relevant = mem.get("m4_relevant", [])
        m4_top = mem.get("m4_top", [])

        return cls(
            cycle_id=cycle.cycle_count,
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
            rbta_violations=rbta_violations,
            action_rationale=action_rationale,
            candidate_scores=candidate_scores,
            module_timings=module_timings,
            latency_ms=getattr(m, "latency_ms", 0.0),
            prediction_error=getattr(m, "prediction_error", 0.0),
            prediction_confidence=getattr(m, "prediction_confidence", 0.0),
            rbta_action=getattr(m, "rbta_action", "CONTINUE"),
            violations_count=getattr(m, "violations_count", 0),
            goal_reached=getattr(m, "goal_reached", False),
            action_name=getattr(m, "action_name", ""),
            drive_id=getattr(m, "drive_id", 1),
            episode_count=getattr(m, "episode_count", 0),
            fact_count=getattr(m, "fact_count", 0),
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
            m4_relevant=m4_relevant,
            m4_top=m4_top,
            last_action_vector=last_action_vector,
            dim_names=dim_names,
            action_names=action_names,
        )

    def to_json(self) -> Dict[str, Any]:
        """Serialise to a JSON-friendly dict for the time-series log.

        Live-only fields (the RGB camera frame, sanitized state, per-drive goal
        vectors, candidate rollouts, and the throttled M3/M4 memory samples)
        are deliberately EXCLUDED: they are heavy / nested-array payloads that
        would turn the ~lean JSONL into an image+vector firehose. They reach the
        recorded mp4 via the dashboard's own QPixmap.grab during --record-video
        and are reconstructed live from the cycle on the dashboard side.
        """
        def _s(x):
            if isinstance(x, (np.integer,)):
                return int(x)
            if isinstance(x, (np.floating,)):
                return float(x)
            if isinstance(x, np.ndarray):
                return x.tolist()
            if isinstance(x, dict):
                return {str(k): _s(v) for k, v in x.items()}
            if isinstance(x, (list, tuple)):
                return [_s(v) for v in x]
            return x
        d = {k: _s(v) for k, v in asdict(self).items()}
        # Belt-and-braces explicit casts for the structured fields.
        d["agent_pos"] = ([int(v) for v in self.agent_pos]
                          if self.agent_pos is not None else None)
        d["goal_pos"] = ([int(v) for v in self.goal_pos]
                         if self.goal_pos is not None else None)
        d["grid"] = self.grid.tolist() if self.grid is not None else None
        d["predicted_state"] = (self.predicted_state.tolist()
                                if self.predicted_state is not None else None)
        d["obs_vector"] = (self.obs_vector.tolist()
                           if self.obs_vector is not None else None)
        d["goal_ref"] = (self.goal_ref.tolist()
                         if self.goal_ref is not None else None)
        d["continuous_action"] = (self.continuous_action.tolist()
                                  if self.continuous_action is not None else None)
        d["attention_indices"] = [int(i) for i in self.attention_indices]
        d["drive_levels"] = [float(v) for v in self.drive_levels]
        d["drive_targets"] = [float(v) for v in self.drive_targets]
        d["attention_saliences"] = [float(v) for v in self.attention_saliences]
        d["candidate_scores"] = [float(v) for v in self.candidate_scores]
        d["module_timings"] = {str(k): float(v) for k, v in self.module_timings.items()}
        # ── v4 field casts ──
        for arr_field in ("sanitized_state", "state_precision", "goal_target",
                          "prediction_precision", "gprime_uncertainty",
                          "per_dim_peu", "attention_weights",
                          "last_action_vector"):
            v = getattr(self, arr_field)
            d[arr_field] = v.tolist() if v is not None else None
        d["drive_deficits"] = [float(v) for v in self.drive_deficits]
        d["attention_precisions"] = [float(v) for v in self.attention_precisions]
        d["pareto_front"] = [int(v) for v in self.pareto_front]
        d["goal_history"] = [int(v) for v in self.goal_history]
        # downsample the drive-history heatmap to the last 24 cycles, rounded —
        # keeps the Goals-tab heatmap meaningful on replay without bloating JSONL.
        dh = self.drive_history[-24:] if self.drive_history else []
        d["drive_history"] = [[round(float(v), 3) for v in row] for row in dh]
        d["runtime_log"] = {str(k): float(v) for k, v in self.runtime_log.items()}
        d["memory_log"] = {str(k): float(v) for k, v in self.memory_log.items()}
        d["energy_log"] = {str(k): float(v) for k, v in self.energy_log.items()}
        d["belief_entropies"] = {str(k): float(v) for k, v in
                                 self.belief_entropies.items()}
        d["rbta_bounds"] = _s(self.rbta_bounds)
        d["goal_stack"] = _s(self.goal_stack)
        d["meta_stable"] = _s(self.meta_stable)
        d["tspl_compiled_skill_ids"] = [str(x) for x in self.tspl_compiled_skill_ids]
        for k in ("cycle_id", "episode_count", "fact_count", "violations_count",
                  "drive_id", "active_drive_id", "rss_bytes", "m3_cap",
                  "m4_cap", "m4_prune_target", "state_dim", "action_dim",
                  "action_count", "goal_creation_cycle"):
            if k in d and d[k] is not None:
                d[k] = int(d[k])
        for k in ("latency_ms", "prediction_error", "prediction_confidence",
                  "goal_tolerance", "goal_priority", "cr_temperature",
                  "empowerment", "tspl_skill_accuracy", "gprime_mutual_info"):
            if k in d and d[k] is not None:
                d[k] = float(d[k])
        # Live-only: never serialised (kept off the JSONL firehose). The heavy
        # cognitive-portrait arrays are reconstructed live from the cycle; on
        # replay the affected sub-views degrade gracefully (empty/None).
        for drop in ("env_frame", "drive_goals", "candidate_rollouts",
                     "m3_recent", "m4_relevant", "m4_top",
                     "sanitized_state", "state_precision", "goal_target",
                     "prediction_precision", "per_dim_peu",
                     "attention_weights", "attention_precisions",
                     "last_action_vector"):
            d.pop(drop, None)
        return d


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

    def start(self, meta: Dict[str, Any]) -> Optional[Path]:
        if not self.enabled:
            return None
        from datetime import datetime
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = self.root / ts
        self.session_dir.mkdir(parents=True, exist_ok=True)
        meta = {**meta, "fps": self.fps}
        (self.session_dir / "meta.json").write_text(json.dumps(meta, indent=2))
        self._jsonl = open(self.session_dir / "timeseries.jsonl", "w", encoding="utf-8")
        return self.session_dir

    def record(self, frame: ObservabilityFrame, fig: Any = None) -> None:
        """Append one JSONL line. ``fig`` is ignored (video is handled by VideoRecorder)."""
        if not self.enabled or self._jsonl is None:
            return
        try:
            self._jsonl.write(json.dumps(frame.to_json()) + "\n")
            self._count += 1
        except Exception as e:
            # Record the first failure so the caller can surface it; keep going.
            if self._error is None:
                self._error = str(e)

    def flush(self) -> None:
        if self._jsonl is not None:
            try:
                self._jsonl.flush()
            except Exception:
                pass

    @property
    def count(self) -> int:
        return self._count

    @property
    def error(self) -> Optional[str]:
        return self._error

    def close(self) -> None:
        if self._jsonl is not None:
            try:
                self._jsonl.close()
            except Exception:
                pass
            self._jsonl = None


class VideoRecorder:
    """Captures the live matplotlib figure into ONE encoded video file.

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
