"""Science trace schema and opt-in collector."""

from __future__ import annotations

import copy
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional


@dataclass
class CycleTraceRecord:
    """Minimal per-cycle trace for metric computation."""
    cycle_id: int
    action: int
    reward: float = 0.0
    prediction_error: float = 0.0
    prediction_confidence: float = 0.0
    goal_drive: int = 1
    goal_switched: bool = False
    env_goal_relocated: bool = False
    rbta_action: str = "CONTINUE"
    violations: int = 0
    latency_ms: float = 0.0
    goal_reached: bool = False
    module_timings: Dict[str, float] = field(default_factory=dict)
    attention_weights: Optional[List[float]] = None
    predicted_state_hash: Optional[str] = None
    # Additive action-selection evidence for causal audit and replay.
    selector_mode: str = ""
    decision_reason: str = ""
    action_rationale: Dict[str, object] = field(default_factory=dict)
    candidate_scores: List[float] = field(default_factory=list)


class TraceCollector:
    """Lock-guarded ring buffer for science traces (mirrors MetricsStore pattern)."""

    def __init__(self, capacity: int = 5000, verbose: bool = False) -> None:
        self.capacity = capacity
        self.verbose = verbose
        self._lock = threading.Lock()
        self._records: Deque[CycleTraceRecord] = deque(maxlen=capacity)
        self._last_goal_drive: Optional[int] = None

    def record(
        self,
        *,
        cycle_id: int,
        action: int,
        reward: float,
        prediction_error: float,
        prediction_confidence: float,
        goal_drive: int,
        rbta_action: str,
        violations: int,
        latency_ms: float,
        goal_reached: bool,
        module_timings: Dict[str, float],
        attention_weights: Optional[List[float]] = None,
        predicted_state_hash: Optional[str] = None,
        env_goal_relocated: bool = False,
        action_rationale: Optional[Dict[str, object]] = None,
        candidate_scores: Optional[List[float]] = None,
    ) -> None:
        goal_switched = (
            self._last_goal_drive is not None and self._last_goal_drive != goal_drive
        )
        self._last_goal_drive = goal_drive
        rationale = copy.deepcopy(action_rationale or {})
        rec = CycleTraceRecord(
            cycle_id=cycle_id,
            action=action,
            reward=reward,
            prediction_error=prediction_error,
            prediction_confidence=prediction_confidence,
            goal_drive=goal_drive,
            goal_switched=goal_switched,
            env_goal_relocated=env_goal_relocated,
            rbta_action=rbta_action,
            violations=violations,
            latency_ms=latency_ms,
            goal_reached=goal_reached,
            module_timings=dict(module_timings),
            attention_weights=attention_weights if self.verbose else None,
            predicted_state_hash=predicted_state_hash if self.verbose else None,
            selector_mode=str(rationale.get("selector_mode") or ""),
            decision_reason=str(rationale.get("decision_reason") or ""),
            action_rationale=rationale,
            candidate_scores=[float(x) for x in (candidate_scores or [])],
        )
        with self._lock:
            self._records.append(rec)

    def snapshot(self) -> List[CycleTraceRecord]:
        with self._lock:
            return list(self._records)

    def clear(self) -> None:
        with self._lock:
            self._records.clear()
            self._last_goal_drive = None

    def __len__(self) -> int:
        with self._lock:
            return len(self._records)


def summarize_action_selection(
    records: List[CycleTraceRecord],
) -> Dict[str, object]:
    """Summarize persisted action-selection evidence without inferring intent."""
    mode_counts: Dict[str, int] = {}
    reason_counts: Dict[str, int] = {}
    rationale_count = 0
    score_count = 0
    for record in records:
        mode = str(record.selector_mode or "unknown")
        reason = str(record.decision_reason or "unknown")
        mode_counts[mode] = mode_counts.get(mode, 0) + 1
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
        rationale_count += int(bool(record.action_rationale))
        score_count += int(bool(record.candidate_scores))
    total = len(records)
    return {
        "n_records": total,
        "selector_mode_counts": mode_counts,
        "decision_reason_counts": reason_counts,
        "rationale_coverage": rationale_count / max(total, 1),
        "candidate_score_coverage": score_count / max(total, 1),
    }
