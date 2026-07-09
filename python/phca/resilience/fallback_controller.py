"""Emergency fallback controller — instinctive behavior under critical conditions.

Monitors BOTH belief entropy AND failure detector events to trigger
simple instinctive behavior (STAY in GridWorld, zero-torque in continuous
environments) when the cognitive cycle enters a critical state.

Dual-signal approach:
  1. Entropy signal: EMA-smoothed belief entropy crossing threshold.
  2. Failure cascade: any FailureDetector event (B1/B4/B5/C1/F5/E1)
     repeated within a window triggers emergency.

This elevates the architecture from research-grade to industrial-grade
by ensuring the agent never acts arbitrarily under high uncertainty OR
when the failure detection pipeline detects imminent collapse.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional


class FallbackAction(str, Enum):
    STOP = "stop"
    NOOP = "noop"


class EmergencyTrigger(str, Enum):
    ENTROPY = "entropy"
    FAILURE_CASCADE = "failure_cascade"


@dataclass
class EmergencyEvent:
    entropy: float
    threshold: float
    action: FallbackAction
    cycle_id: int
    trigger: EmergencyTrigger = EmergencyTrigger.ENTROPY
    trigger_detail: str = ""


class FallbackController:
    """Monitors entropy + failure events and triggers instinctive fallback.

    Dual-signal detection:
      1. Entropy: EMA-smoothed belief entropy above threshold for N consecutive cycles.
      2. Failure cascade: M distinct failure events within a rolling window.
    """

    def __init__(
        self,
        entropy_threshold: float = 0.85,
        cooldown: int = 10,
        ema_alpha: float = 0.3,
        consecutive_threshold: int = 3,
        failure_streak_threshold: int = 3,
        failure_window: int = 15,
    ):
        self.entropy_threshold = entropy_threshold
        self.cooldown = cooldown
        self.ema_alpha = ema_alpha
        self.consecutive_threshold = consecutive_threshold
        self.failure_streak_threshold = failure_streak_threshold
        self.failure_window = failure_window

        self._smoothed_entropy: float = 0.0
        self._consecutive_high: int = 0
        self._cooldown_remaining: int = 0
        self._active: bool = False
        self._last_event: Optional[EmergencyEvent] = None
        self._total_emergencies: int = 0

        # Failure cascade tracking
        self._failure_history: List[tuple[int, str, float]] = []

    def notify_failure(self, mode_id: str, severity: float, cycle_id: int) -> None:
        """Register a failure detector event for cascade detection."""
        self._failure_history.append((cycle_id, mode_id, severity))
        # Prune old entries outside the window
        cutoff = cycle_id - self.failure_window
        self._failure_history = [
            e for e in self._failure_history if e[0] >= cutoff
        ]

    def check(self, entropy: float, cycle_id: int) -> Optional[EmergencyEvent]:
        # --- Entropy signal ---
        if self._smoothed_entropy == 0.0:
            self._smoothed_entropy = entropy
        else:
            self._smoothed_entropy = (
                self.ema_alpha * entropy
                + (1.0 - self.ema_alpha) * self._smoothed_entropy
            )

        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1
            if self._cooldown_remaining <= 0:
                self._active = False
            return None

        # --- Failure cascade signal ---
        unique_failures = len(set(e[1] for e in self._failure_history))
        cascade_trigger = (
            len(self._failure_history) >= self.failure_streak_threshold
            and unique_failures >= 2
        )

        # --- Entropy trigger ---
        entropy_trigger = self._smoothed_entropy >= self.entropy_threshold
        if entropy_trigger:
            self._consecutive_high += 1
        else:
            self._consecutive_high = 0

        entropy_ready = self._consecutive_high >= self.consecutive_threshold

        if entropy_ready or cascade_trigger:
            self._active = True
            self._cooldown_remaining = self.cooldown
            self._consecutive_high = 0
            self._total_emergencies += 1

            if entropy_ready:
                trigger = EmergencyTrigger.ENTROPY
                detail = f"entropy={self._smoothed_entropy:.3f} >= {self.entropy_threshold}"
            else:
                trigger = EmergencyTrigger.FAILURE_CASCADE
                modes = set(e[1] for e in self._failure_history)
                detail = f"failure_cascade: {','.join(sorted(modes))}"

            self._failure_history.clear()

            self._last_event = EmergencyEvent(
                entropy=self._smoothed_entropy,
                threshold=self.entropy_threshold,
                action=FallbackAction.STOP,
                cycle_id=cycle_id,
                trigger=trigger,
                trigger_detail=detail,
            )
            return self._last_event

        return None

    def active(self) -> bool:
        return self._active

    def active_trigger(self) -> str:
        if self._last_event is None:
            return "none"
        return self._last_event.trigger.value

    def reset(self) -> None:
        self._smoothed_entropy = 0.0
        self._consecutive_high = 0
        self._cooldown_remaining = 0
        self._active = False
        self._last_event = None
        self._failure_history.clear()

    @property
    def last_event(self) -> Optional[EmergencyEvent]:
        return self._last_event

    @property
    def total_emergencies(self) -> int:
        return self._total_emergencies

    @property
    def smoothed_entropy(self) -> float:
        return self._smoothed_entropy
