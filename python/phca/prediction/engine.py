"""
PHCA v3.0 — Prediction Engine.

Phase 3.1: Single-model ensemble (G' only).
Phase 3.2+: Dual-model ensemble (G' + V) with meta-gradient weights.

v3.0 References:
    - §2.2 Definition 2.5 (prediction via G' only, Phase 3.1)
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from phca.config import StateVector
from phca.world_model.graph import WorldModelGPrime


class PredictionEngine:
    """Prediction Engine — predicts next state(s) given current state and action.

    Phase 3.1: Single-model ensemble (G' Bayesian network).
        - Calls WorldModelGPrime.predict() for each step.
        - Confidence decays with horizon (1/horizon heuristic).
        - grounding_level 0 (raw) and 1 (feature) supported.

    Phase 3.2+: Dual-model (G' + V) with meta-gradient ensemble weights.
        - grounding_level 2 (semantic) prediction.
        - Confidence via ensemble disagreement.
    """

    def __init__(self, gprime: WorldModelGPrime):
        """Initialize the prediction engine.

        Args:
            gprime: The G' probabilistic world model.
        """
        self.gprime = gprime
        self.last_action: np.ndarray = np.zeros(gprime.action_dim, dtype=np.float32)
        self.ensemble_weights: list[float] = [1.0]  # Phase 3.1: single model

    def predict(
        self,
        state: StateVector,
        horizon: int = 1,
        grounding_level: int = 1,
    ) -> Tuple[StateVector, float]:
        """Predict future state(s) given current state and last action.

        Phase 3.1: Single-step or horizon-rolled prediction via G' only.

        Args:
            state: Current state vector.
            horizon: Number of steps ahead to predict (1-10).
            grounding_level: 0 (raw), 1 (feature), 2 (semantic — Phase 3.2+).

        Returns:
            Tuple of (predicted_state, confidence):
                predicted_state: StateVector with predicted values at horizon.
                confidence: Prediction confidence (0.0 = uncertain, 1.0 = certain).

        Raises:
            ValueError: If horizon is out of range.
            NotImplementedError: If grounding_level=2 (deferred to Phase 3.2).
        """
        if horizon < 1 or horizon > 100:
            raise ValueError(
                f"horizon must be in [1, 100], got {horizon}. "
                "Horizons > 10 may have degraded confidence."
            )
        if grounding_level not in (0, 1):
            raise NotImplementedError(
                f"grounding_level={grounding_level} prediction deferred to Phase 3.2. "
                "Phase 3.1 supports levels 0 (raw) and 1 (feature)."
            )

        # Roll prediction horizon steps
        current_state = state
        for step in range(horizon):
            next_state, step_confidence = self.gprime.predict(
                current_state, self.last_action
            )
            current_state = next_state

        # Confidence decays with horizon (1/h heuristic)
        confidence = step_confidence / max(horizon, 1)

        return current_state, min(confidence, 1.0)

    def update_action(self, action: np.ndarray) -> None:
        """Record the last action taken (used for next prediction).

        Args:
            action: Action vector from the last cycle step.
        """
        self.last_action = action.copy()
