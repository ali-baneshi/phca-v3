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
    # Internal mind
    drive_levels: List[float] = field(default_factory=list)   # D1-D6 values
    attention_saliences: List[float] = field(default_factory=list)
    attention_indices: List[int] = field(default_factory=list)
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
        mdim_drives = getattr(cycle.mdim, "drives", {})
        drive_levels = [float(mdim_drives[d].value) for d in range(1, 7) if d in mdim_drives]
        att = cycle.attention
        attention_saliences = list(getattr(att, "_last_saliences", []))
        attention_indices = list(getattr(att, "_last_selected_indices", []))
        # Latest CycleMetrics (last appended, not yet pushed to observability)
        m = cycle.metrics_history[-1] if getattr(cycle, "metrics_history", None) else None
        rss = _cached_rss()
        return cls(
            cycle_id=cycle.cycle_count,
            agent_pos=tuple(agent_pos) if agent_pos is not None else None,
            goal_pos=tuple(goal_pos) if goal_pos is not None else None,
            grid=grid,
            predicted_state=predicted_state,
            obs_vector=obs_vector,
            drive_levels=drive_levels,
            attention_saliences=attention_saliences,
            attention_indices=attention_indices,
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
        d = asdict(self)
        d["agent_pos"] = [list(map(int, self.agent_pos))] if False else (
            [int(v) for v in self.agent_pos] if self.agent_pos is not None else None)
        d["goal_pos"] = ([int(v) for v in self.goal_pos]
                         if self.goal_pos is not None else None)
        d["grid"] = self.grid.tolist() if self.grid is not None else None
        d["predicted_state"] = (self.predicted_state.tolist()
                                if self.predicted_state is not None else None)
        d["obs_vector"] = (self.obs_vector.tolist()
                           if self.obs_vector is not None else None)
        d["attention_indices"] = [int(i) for i in self.attention_indices]
        d["drive_levels"] = [float(v) for v in self.drive_levels]
        d["attention_saliences"] = [float(v) for v in self.attention_saliences]
        d["cycle_id"] = int(self.cycle_id)
        d["episode_count"] = int(self.episode_count)
        d["fact_count"] = int(self.fact_count)
        d["violations_count"] = int(self.violations_count)
        d["drive_id"] = int(self.drive_id)
        d["rss_bytes"] = int(self.rss_bytes)
        d["latency_ms"] = float(self.latency_ms)
        d["prediction_error"] = float(self.prediction_error)
        d["prediction_confidence"] = float(self.prediction_confidence)
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

    def __len__(self) -> int:
        with self._lock:
            return len(self._deque)


class SessionRecorder:
    """Persists frames to logs/sessions/<ts>/ as JSONL + PNGs + meta.json.

    Recording is driven by the visualiser render tick (off the cycle hot path).
    """

    def __init__(self, root: str = "logs/sessions", fps: float = 5.0,
                 record: bool = True):
        self.root = Path(root)
        self.fps = float(fps)
        self.enabled = bool(record)   # NOTE: do NOT name this `record` — it would shadow record()
        self.session_dir: Optional[Path] = None
        self._jsonl = None
        self._frames_dir: Optional[Path] = None

    def start(self, meta: Dict[str, Any]) -> Optional[Path]:
        if not self.enabled:
            return None
        from datetime import datetime
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = self.root / ts
        self._frames_dir = self.session_dir / "frames"
        self._frames_dir.mkdir(parents=True, exist_ok=True)
        meta = {**meta, "fps": self.fps}
        (self.session_dir / "meta.json").write_text(json.dumps(meta, indent=2))
        self._jsonl = open(self.session_dir / "timeseries.jsonl", "a")
        return self.session_dir

    def record(self, frame: ObservabilityFrame, fig: Any = None) -> None:
        if not self.enabled or self._jsonl is None:
            return
        try:
            self._jsonl.write(json.dumps(frame.to_json()) + "\n")
            self._jsonl.flush()
        except Exception:
            pass  # never let a serialisation glitch crash the render loop
        if fig is not None and self._frames_dir is not None:
            try:
                fig.savefig(self._frames_dir / f"{frame.cycle_id:05d}.png")
            except Exception:
                pass  # never let a render glitch crash the cycle

    def close(self) -> None:
        if self._jsonl is not None:
            self._jsonl.close()
            self._jsonl = None
