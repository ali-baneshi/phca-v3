"""
Minimal hybrid ensemble: Graph G' + MLP G'.

Wraps both WorldModelGPrime and WorldModelMLP behind a single interface.
predict() runs both models and blends outputs by confidence-weighted mean.
Disagreement between the two models is exposed as epistemic uncertainty
(mutual_info) used by RBTA.

Phase 3.2 stepping-stone — not a full ensemble with meta-gradients.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

if TYPE_CHECKING:
    from phca.world_model.graph import WorldModelGPrime

import numpy as np

from phca.config import StateVector
from phca.world_model.mlp import WorldModelMLP


_EPS = 1e-8


class HybridGraphMLP:
    """Ensemble wrapper: discrete/gaussian G' + MLP G'.

    Exposes the same predict/learn/reset interface as each model.
    predict() blends outputs: result = w_graph * pred_graph + w_mlp * pred_mlp
    where weights are proportional to each model's confidence.

    Attributes mirror both models for compatibility with isinstance checks
    and cycle.py attribute access (hidden_dim, replay_capacity, etc.).
    """

    def __init__(
        self,
        graph_model: "WorldModelGPrime",
        mlp_model: WorldModelMLP,
    ):
        self.graph = graph_model
        self.mlp = mlp_model
        self.state_dim = mlp_model.state_dim
        self.action_dim = mlp_model.action_dim
        self.hidden_dim = mlp_model.hidden_dim
        self.replay_capacity = mlp_model.replay_capacity
        self.batch_size = mlp_model.batch_size
        self.dropout_rate = mlp_model.dropout_rate
        self._last_mc_per_dim_std: Optional[np.ndarray] = None
        self._last_mutual_info: float = 0.0
        self.replay_boost: bool = False
        self._attention_weights: np.ndarray = np.ones(self.state_dim, dtype=np.float32)

    def predict(
        self, state: StateVector, action: np.ndarray
    ) -> Tuple[StateVector, float]:
        p_graph, c_graph = self.graph.predict(state, action)
        p_mlp, c_mlp = self.mlp.predict(state, action)

        w_graph = float(c_graph) / (float(c_graph) + float(c_mlp) + _EPS)
        w_mlp = 1.0 - w_graph

        blended = w_graph * p_graph.values + w_mlp * p_mlp.values
        blended_conf = max(c_graph, c_mlp) * 0.5 + min(c_graph, c_mlp) * 0.5
        disagreement = float(np.linalg.norm(p_graph.values - p_mlp.values)) / max(
            np.sqrt(float(self.state_dim)), 1.0
        )
        mlp_mi = float(getattr(self.mlp, "_last_mutual_info", 0.0))
        self._last_mutual_info = min(1.0, mlp_mi + disagreement * 0.3)

        self._last_mc_per_dim_std = self.mlp._last_mc_per_dim_std

        precision = np.full(self.state_dim, blended_conf, dtype=np.float32)
        return (
            StateVector(
                values=blended.astype(np.float32),
                precision=precision,
                timestamp=state.timestamp + 1.0,
                grounding_level=state.grounding_level,
            ),
            blended_conf,
        )

    def learn(
        self,
        state_t: StateVector,
        action: np.ndarray,
        state_t1: StateVector,
        error: float,
    ) -> None:
        self.graph._attention_weights = self._attention_weights
        self.mlp._attention_weights = self._attention_weights
        self.graph.learn(state_t, action, state_t1, error)
        self.mlp.learn(state_t, action, state_t1, error)

    def reset(self) -> None:
        self.graph.reset()
        self.mlp.reset()

    def estimate_empowerment(self, state: Any) -> float:
        g = self.graph.estimate_empowerment(state)
        m = self.mlp.estimate_empowerment(state)
        return float(np.clip((g + m) * 0.5, 0.0, 1.0))

    def uncertainty_snapshot(self) -> Dict[str, Any]:
        g_snap = self.graph.uncertainty_snapshot()
        m_snap = self.mlp.uncertainty_snapshot()
        return {
            "kind": "hybrid",
            "graph": g_snap,
            "mlp": m_snap,
            "mutual_info": float(self._last_mutual_info),
        }

    def get_prediction_accuracy(
        self, state: Any, action: np.ndarray, target: np.ndarray
    ) -> float:
        return self.mlp.get_prediction_accuracy(state, action, target)

    def set_tspl_bias(self, bias: Optional[np.ndarray]) -> None:
        self.mlp.set_tspl_bias(bias)

    def last_input_sensitivity(self, state_dim: int) -> float:
        return self.mlp.last_input_sensitivity(state_dim)