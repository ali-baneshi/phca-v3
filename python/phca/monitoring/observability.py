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
        )

    def to_json(self) -> Dict[str, Any]:
        """Serialise to a JSON-friendly dict for the time-series log."""
        def _s(x):
            if isinstance(x, (np.integer,)):
                return int(x)
            if isinstance(x, (np.floating,)):
                return float(x)
            if isinstance(x, np.ndarray):
                return x.tolist()
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
        for k in ("cycle_id", "episode_count", "fact_count", "violations_count",
                  "drive_id", "active_drive_id", "rss_bytes", "m3_cap",
                  "m4_cap", "m4_prune_target"):
            d[k] = int(d[k])
        for k in ("latency_ms", "prediction_error", "prediction_confidence"):
            d[k] = float(d[k])
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
