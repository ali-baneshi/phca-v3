"""Cognitive failure detection, recovery, and emergency fallback."""

from phca.resilience.detector import FailureDetector
from phca.resilience.metrics import recovery_rate
from phca.resilience.recovery import RecoveryManager
from phca.resilience.types import (
    CycleSnapshot,
    EmergencyEvent,
    FailureCategory,
    FailureEvent,
    FallbackAction,
    RecoveryAction,
    RecoveryResult,
)
from phca.resilience.fallback_controller import FallbackController

__all__ = [
    "CycleSnapshot",
    "EmergencyEvent",
    "FailureCategory",
    "FailureDetector",
    "FailureEvent",
    "FallbackAction",
    "FallbackController",
    "RecoveryAction",
    "RecoveryManager",
    "RecoveryResult",
    "recovery_rate",
]
