"""
PHCA v3.0 - MDIM (Multi-Drive Intrinsic Motivation) Stub.

Phase 3.1: Stub - always returns default GoalVector(D1: prediction error minimization).
Phase 3.2+: Full Pareto front with D1-D6 drives and meta-stable state selection.

v3.0 Reference: xa7.3.3 Definition 3.5, xa7.3.3 Definition 3.6
"""

from __future__ import annotations

from typing import Any

import numpy as np

from phca.config import GoalVector, StateVector


class MDIM:
    """Multi-Drive Intrinsic Motivation - Phase 3.1 stub."""

    def __init__(self, state_dim: int = 4):
        self.state_dim = state_dim
        self._cycle = 0

    def generate_goal(self, context: Any = None) -> GoalVector:
        """Generate a default goal (Phase 3.1 stub)."""
        self._cycle += 1
        return GoalVector(
            drive_id=1,
            target_state=StateVector(
                values=np.zeros(self.state_dim, dtype=np.float32),
                precision=np.ones(self.state_dim, dtype=np.float32),
            ),
            tolerance=0.1,
            creation_cycle=self._cycle,
            priority=1.0,
        )

    def reset(self) -> None:
        """Reset MDIM state for a new training run."""
        self._cycle = 0
