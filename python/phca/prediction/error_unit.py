"""
PHCA v3.0 — Prediction Error Unit (PEU).

Computes the prediction error δ_t between observed and predicted states.

Phase 3.1: Simple δ = ‖observed - predicted‖₂² (unweighted).
           Precision-weighted: Σ p_i × (o_i - p_i)².
Phase 3.2+: Hierarchical error decomposition (per grounding level).

v3.0 Reference: §2.2 Definition 2.5 (prediction error via PEU)
"""

from __future__ import annotations

import numpy as np

from phca.config import StateVector


class PredictionErrorUnit:
    """Prediction Error Unit — computes δ_t for TSPL learning.

    The PEU is called after every environment step to quantify the
    discrepancy between what G' predicted and what was observed.
    This error signal drives learning in the P-Stream (§3.1).
    """

    def compute_precision_weighted(
        self,
        observed: StateVector,
        predicted: StateVector,
        precision: np.ndarray,
    ) -> float:
        """Compute precision-weighted error: Σ p_i × (o_i - p_i)².

        Weighs each dimension's error by its precision. This accounts
        for sensor reliability: high-precision dimensions contribute
        more to the error signal.

        Args:
            observed: The actual state observed from the environment.
            predicted: The state predicted by the Prediction Engine.
            precision: Per-dimension precision weights (same shape as values).

        Returns:
            Scalar precision-weighted error value ≥ 0.

        Raises:
            ValueError: If precision shape doesn't match values shape.
        """
        if precision.shape != observed.values.shape:
            raise ValueError(
                f"precision shape {precision.shape} != values shape "
                f"{observed.values.shape}"
            )
        diff = observed.values - predicted.values
        return float(np.sum(precision * diff ** 2))


