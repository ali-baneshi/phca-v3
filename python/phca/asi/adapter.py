"""
Minimal grounding level adapter for ASI.

Tracks a simple grounding level (0=raw, 1=feature, 2=semantic)
that adapts based on sensor health. When sensors degrade, the
level steps down (conservative mode). When sensors recover,
the level steps back up.

Phase 3.2 stepping-stone — not a full L0-L4 hierarchy.
"""

from __future__ import annotations


class GroundingAdapter:
    """Adaptive grounding level based on sensor health.

    Attributes:
        current_level: Current grounding level (0, 1, or 2).
        _healthy_cycles: Consecutive cycles with no sensor failure.
    """

    MIN_LEVEL: int = 0
    MAX_LEVEL: int = 2
    HEALTHY_THRESHOLD: int = 10  # cycles of health before level-up

    def __init__(self):
        self.current_level: int = 1
        self._healthy_cycles: int = 0

    def update(self, sensor_failure_count: int, prediction_confidence: float) -> int:
        """Update grounding level based on sensor and model health.

        Args:
            sensor_failure_count: Consecutive ASI sensor failures (0 = healthy).
            prediction_confidence: Current prediction confidence [0, 1].

        Returns:
            New grounding level.
        """
        if sensor_failure_count > 0:
            self._healthy_cycles = 0
            if sensor_failure_count >= 3 and self.current_level > self.MIN_LEVEL:
                self.current_level = max(self.MIN_LEVEL, self.current_level - 1)
            elif sensor_failure_count >= 5 and self.current_level > self.MIN_LEVEL:
                self.current_level = max(self.MIN_LEVEL, self.current_level - 1)
        else:
            self._healthy_cycles += 1
            if (self._healthy_cycles >= self.HEALTHY_THRESHOLD
                    and prediction_confidence > 0.6
                    and self.current_level < self.MAX_LEVEL):
                self.current_level = min(self.MAX_LEVEL, self.current_level + 1)
                self._healthy_cycles = 0

        return self.current_level

    def reset(self) -> None:
        self.current_level = 1
        self._healthy_cycles = 0