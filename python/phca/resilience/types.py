"""Cognitive resilience types — failure events and recovery results."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class FailureCategory(str, Enum):
    """High-level failure families (whitepaper §4.1 subset)."""

    DISTRIBUTION_SHIFT = "B1"
    CATASTROPHIC_FORGETTING = "B4"
    MODE_COLLAPSE = "B5"
    FEEDBACK_INSTABILITY = "C1"
    CONSOLIDATION_FAILURE = "F5"


@dataclass
class FailureEvent:
    """Detected failure with measured signal and threshold."""

    mode_id: str
    category: FailureCategory
    severity: float
    cycle_id: int
    measured: float
    threshold: float
    detail: str = ""


class RecoveryAction(str, Enum):
    """Recovery protocol identifiers."""

    BOOST_ETA = "boost_eta"
    REPLAY_BOOST = "replay_boost"
    BOOST_TEMPERATURE = "boost_temperature"
    PID_C1_RECOVERY = "pid_c1_recovery"
    FORCE_CONSOLIDATION = "force_consolidation"


@dataclass
class RecoveryResult:
    """Outcome of applying recovery protocols for one cycle."""

    events_handled: List[str] = field(default_factory=list)
    actions_applied: List[RecoveryAction] = field(default_factory=list)
    mitigated: List[str] = field(default_factory=list)
    still_active: List[str] = field(default_factory=list)


@dataclass
class CycleSnapshot:
    """Signals collected each cycle for failure detection."""

    cycle_id: int = 0
    prediction_error: float = 0.0
    prediction_confidence: float = 0.0
    wm_entropy_proxy: float = 0.5
    mdim_deficits: Dict[str, float] = field(default_factory=dict)
    rbta_action_history: List[str] = field(default_factory=list)
    m3_fill_ratio: float = 0.0
    m4_fact_count: int = 0
    m4_fact_count_delta: int = 0
    module_timings: Dict[str, float] = field(default_factory=dict)
    unique_actions_recent: int = 5
    per_task_goal_rate: Optional[float] = None
    per_task_baseline_goal_rate: Optional[float] = None
    recent_prediction_errors: List[float] = field(default_factory=list)
    recent_actions: List[int] = field(default_factory=list)
    recent_confidences: List[float] = field(default_factory=list)
    fact_count_stagnant_cycles: int = 0
    extra: Dict[str, Any] = field(default_factory=dict)
