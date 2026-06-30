"""Prediction Engine + Prediction Error Unit + Ensemble Prediction.

Phase 3.1:
    - PredictionEngine: Single-model (G') prediction with horizon rolling.
    - PredictionErrorUnit: Delta computation (L2, precision-weighted, RMSE).

Phase 3.2+:
    - Dual-model ensemble (G' + V) with meta-gradient weights.
    - Hierarchical error decomposition (per grounding level).

v3.0 Reference: §2.2 Definition 2.5
"""

from phca.prediction.engine import PredictionEngine
from phca.prediction.error_unit import PredictionErrorUnit

__all__ = [
    "PredictionEngine",
    "PredictionErrorUnit",
]
