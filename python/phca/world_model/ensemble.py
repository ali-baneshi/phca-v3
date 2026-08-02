"""Probabilistic MLP ensemble for G'.

The wrapper keeps the existing G' interface while exposing epistemic
uncertainty from disagreement across independently seeded MLP members.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from phca.config import StateVector
from phca.world_model.mlp import WorldModelMLP


_EPS = 1e-8


class WorldModelMLPEnsemble:
    """Deep-ensemble style world model built from independent NumPy MLPs."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        *,
        n_members: int = 3,
        hidden_dim: int = 128,
        seed: int = 42,
        lr: float = 0.2,
        position_dim: Optional[int] = None,
    ):
        self.members: List[WorldModelMLP] = [
            WorldModelMLP(
                state_dim=state_dim,
                action_dim=action_dim,
                hidden_dim=hidden_dim,
                seed=seed + i * 997,
                lr=lr,
                position_dim=position_dim,
            )
            for i in range(max(2, int(n_members)))
        ]
        head = self.members[0]
        self.state_dim = head.state_dim
        self.action_dim = head.action_dim
        self.hidden_dim = head.hidden_dim
        self.replay_capacity = head.replay_capacity
        self.batch_size = head.batch_size
        self.dropout_rate = head.dropout_rate
        self.position_dim = head.position_dim
        self.replay_boost: bool = False
        self._attention_weights: np.ndarray = np.ones(self.state_dim, dtype=np.float32)
        self._last_mc_per_dim_std: Optional[np.ndarray] = None
        self._last_mutual_info: float = 0.0
        self.last_loss_components: Dict[str, float] = {}

    @property
    def mlp(self) -> WorldModelMLP:
        """Compatibility handle for code that expects an MLP-like child."""
        return self.members[0]

    def predict(
        self, state: StateVector, action: np.ndarray
    ) -> Tuple[StateVector, float]:
        preds: List[np.ndarray] = []
        confs: List[float] = []
        member_stds: List[np.ndarray] = []
        for member in self.members:
            pred, conf = member.predict(state, action)
            preds.append(pred.values.astype(np.float32))
            confs.append(float(conf))
            std = getattr(member, "_last_mc_per_dim_std", None)
            if std is not None:
                member_stds.append(std.astype(np.float32))

        pred_arr = np.stack(preds, axis=0)
        mean = pred_arr.mean(axis=0).astype(np.float32)
        disagreement_var = pred_arr.var(axis=0)
        dropout_var = (
            np.mean(np.stack(member_stds, axis=0) ** 2, axis=0)
            if member_stds else np.zeros(self.state_dim, dtype=np.float32)
        )
        total_var = disagreement_var + dropout_var
        self._last_mc_per_dim_std = np.sqrt(total_var).astype(np.float32)
        variance = float(np.mean(total_var))
        self._last_mutual_info = float(np.log1p(min(max(variance, 0.0), 2.0)))

        confidence = float(np.clip(np.mean(confs) - 0.5 * self._last_mutual_info, 0.01, 1.0))
        return (
            StateVector(
                values=mean,
                precision=np.full(self.state_dim, confidence, dtype=np.float32),
                timestamp=state.timestamp + 1.0,
                grounding_level=state.grounding_level,
            ),
            confidence,
        )

    def learn(
        self,
        state_t: StateVector,
        action: np.ndarray,
        state_t1: StateVector,
        error: float,
    ) -> None:
        for member in self.members:
            member.replay_boost = self.replay_boost
            member._attention_weights = self._attention_weights.copy()
            member.learn(state_t, action, state_t1, error)
        self._sync_loss_components()

    def learn_m3_episodes(
        self, episodes: list, lr_scale: float = 1.0,
    ) -> Tuple[int, List[Tuple[int, float]]]:
        max_steps = 0
        priority_updates: Dict[int, float] = {}
        for member in self.members:
            steps, updates = member.learn_m3_episodes(episodes, lr_scale=lr_scale)
            max_steps = max(max_steps, steps)
            for episode_id, err in updates:
                priority_updates[int(episode_id)] = float(err)
        self._sync_loss_components()
        return max_steps, sorted(priority_updates.items())

    def _sync_loss_components(self) -> None:
        keys = {"weighted_mse", "position_ce", "grad_norm"}
        self.last_loss_components = {
            key: float(np.mean([
                getattr(m, "last_loss_components", {}).get(key, 0.0)
                for m in self.members
            ]))
            for key in keys
        }

    def reset(self) -> None:
        for member in self.members:
            member.reset()

    def estimate_empowerment(self, state: Any) -> float:
        values = [member.estimate_empowerment(state) for member in self.members]
        return float(np.clip(np.mean(values), 0.0, 1.0))

    def uncertainty_snapshot(self) -> Dict[str, Any]:
        return {
            "kind": "mlp_ensemble",
            "n_members": len(self.members),
            "per_dim_std": (
                self._last_mc_per_dim_std.copy()
                if self._last_mc_per_dim_std is not None else None
            ),
            "mutual_info": float(self._last_mutual_info),
        }

    def get_prediction_accuracy(
        self, state: Any, action: np.ndarray, target: np.ndarray
    ) -> float:
        values = [
            member.get_prediction_accuracy(state, action, target)
            for member in self.members
        ]
        return float(np.mean(values))

    def set_tspl_bias(self, bias: Optional[np.ndarray]) -> None:
        for member in self.members:
            member.set_tspl_bias(bias)

    def last_input_sensitivity(self, state_dim: int) -> float:
        values = [member.last_input_sensitivity(state_dim) for member in self.members]
        return float(np.mean(values))

    def __repr__(self) -> str:
        return (
            "WorldModelMLPEnsemble("
            f"state_dim={self.state_dim}, action_dim={self.action_dim}, "
            f"members={len(self.members)}, hidden_dim={self.hidden_dim})"
        )
