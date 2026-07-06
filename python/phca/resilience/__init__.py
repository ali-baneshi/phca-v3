"""Cognitive failure detection and recovery (distinct from Observatory session recovery)."""

from phca.resilience.detector import FailureDetector
from phca.resilience.metrics import recovery_rate
from phca.resilience.recovery import RecoveryManager
from phca.resilience.types import (
    CycleSnapshot,
    FailureCategory,
    FailureEvent,
    RecoveryAction,
    RecoveryResult,
)

__all__ = [
    "CycleSnapshot",
    "FailureCategory",
    "FailureDetector",
    "FailureEvent",
    "RecoveryAction",
    "RecoveryManager",
    "RecoveryResult",
    "recovery_rate",
]
