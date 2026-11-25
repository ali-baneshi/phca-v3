"""Tests for probabilistic MLP ensemble G'."""

from __future__ import annotations

import numpy as np

from phca.config import StateVector
from phca.world_model.ensemble import WorldModelMLPEnsemble


def test_mlp_ensemble_predict_exposes_variance_uncertainty():
    model = WorldModelMLPEnsemble(
        state_dim=12, action_dim=3, hidden_dim=8, n_members=3, seed=5, position_dim=9,
    )
    state = StateVector(
        values=np.eye(1, 12, 0, dtype=np.float32).reshape(-1),
        precision=np.ones(12, dtype=np.float32),
        timestamp=0.0,
    )
    action = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    pred, conf = model.predict(state, action)
    snap = model.uncertainty_snapshot()

    assert pred.values.shape == (12,)
    assert 0.0 < conf <= 1.0
    assert snap["kind"] == "mlp_ensemble"
    assert snap["n_members"] == 3
    assert snap["per_dim_std"].shape == (12,)
    assert snap["mutual_info"] >= 0.0


def test_mlp_ensemble_replays_m3_episodes():
    model = WorldModelMLPEnsemble(
        state_dim=12, action_dim=3, hidden_dim=8, n_members=2, seed=9, position_dim=9,
    )
    state = StateVector(
        values=np.zeros(12, dtype=np.float32),
        precision=np.ones(12, dtype=np.float32),
        timestamp=0.0,
    )
    next_state = StateVector(
        values=np.eye(1, 12, 8, dtype=np.float32).reshape(-1),
        precision=np.ones(12, dtype=np.float32),
        timestamp=1.0,
    )
    episode = type("Episode", (), {
        "episode_id": 7,
        "state_before": state,
        "state_after": next_state,
        "action_taken": np.array([0.0, 1.0, 0.0], dtype=np.float32),
    })()

    steps, updates = model.learn_m3_episodes([episode], lr_scale=0.1)

    assert steps == 1
    assert updates and updates[0][0] == 7
    assert model.last_loss_components["position_ce"] >= 0.0
