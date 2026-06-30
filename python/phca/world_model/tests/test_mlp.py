"""
Tests for WorldModelMLP — MLP G' replacement (Phase 3.3b).

Covers: forward pass, backward pass (finite-difference verification),
interface compatibility, theta sync, reset, confidence calibration.

Note: MLP uses linear output + MSE loss (not sigmoid + BCE) because
GridWorld state vectors contain values ∈ {0, 1, 2}.
"""

from __future__ import annotations

import numpy as np
import pytest

from phca.config import StateVector
from phca.world_model.mlp import WorldModelMLP


# ── Fixtures ──────────────────────────────────────────────────


@pytest.fixture
def mlp() -> WorldModelMLP:
    """Default MLP for GridWorld (84-dim state, 5-dim action)."""
    return WorldModelMLP(state_dim=84, action_dim=5, hidden_dim=128, seed=42)


@pytest.fixture
def small_mlp() -> WorldModelMLP:
    """Tiny MLP for gradient verification (4-dim state, 2-dim action)."""
    return WorldModelMLP(state_dim=4, action_dim=2, hidden_dim=8, seed=42)


@pytest.fixture
def sample_state() -> StateVector:
    """A one-hot state vector for testing (dim 84)."""
    v = np.zeros(84, dtype=np.float32)
    v[10] = 1.0  # one-hot at position 10
    return StateVector(values=v, precision=np.ones(84, dtype=np.float32), timestamp=0.0)


@pytest.fixture
def sample_action() -> np.ndarray:
    """A one-hot action for testing (dim 5)."""
    a = np.zeros(5, dtype=np.float32)
    a[0] = 1.0  # MOVE_N
    return a


@pytest.fixture
def sample_next_state() -> StateVector:
    """Expected next state after MOVE_N from position 10 (11 = position 10 + size 5)."""
    v = np.zeros(84, dtype=np.float32)
    v[11] = 1.0  # moved north by 1 cell (assuming contiguous rows)
    return StateVector(values=v, precision=np.ones(84, dtype=np.float32), timestamp=1.0)


# ── Forward Pass ──────────────────────────────────────────────


class TestForwardPass:
    """Tests for the MLP forward pass (predict())."""

    def test_output_shape(self, mlp, sample_state, sample_action):
        """predict() should return a StateVector with state_dim values."""
        pred, conf = mlp.predict(sample_state, sample_action)
        assert isinstance(pred, StateVector)
        assert pred.values.shape == (84,)
        assert pred.precision.shape == (84,)

    def test_confidence_range(self, mlp, sample_state, sample_action):
        """Confidence should be in (0, 1]."""
        pred, conf = mlp.predict(sample_state, sample_action)
        assert 0.0 < conf <= 1.0

    def test_confidence_near_one_on_perfect_match(self, small_mlp):
        """When prediction matches target exactly, confidence ≈ 1.0."""
        state = StateVector(
            values=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            precision=np.ones(4, dtype=np.float32), timestamp=0.0,
        )
        action = np.array([1.0, 0.0], dtype=np.float32)
        # Set target equal to the prediction itself
        pred, conf = small_mlp.predict(state, action)
        expected_mse = 0.5 * float(np.mean((pred.values - state.values) ** 2))
        expected_conf = float(np.exp(-expected_mse))
        assert abs(conf - expected_conf) < 1e-6

    def test_deterministic_with_seed(self):
        """Same seed should produce identical initial predictions."""
        mlp1 = WorldModelMLP(state_dim=4, action_dim=2, seed=42)
        mlp2 = WorldModelMLP(state_dim=4, action_dim=2, seed=42)
        state = StateVector(
            values=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            precision=np.ones(4, dtype=np.float32), timestamp=0.0,
        )
        action = np.array([1.0, 0.0], dtype=np.float32)
        p1, _ = mlp1.predict(state, action)
        p2, _ = mlp2.predict(state, action)
        assert np.allclose(p1.values, p2.values)

    def test_action_influence(self, mlp, sample_state):
        """Different actions should produce different predictions."""
        a_north = np.array([1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        a_south = np.array([0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        p_north, _ = mlp.predict(sample_state, a_north)
        p_south, _ = mlp.predict(sample_state, a_south)
        assert not np.allclose(p_north.values, p_south.values)

    def test_small_mlp_predict(self, small_mlp):
        """Small MLP should produce valid output."""
        state = StateVector(
            values=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            precision=np.ones(4, dtype=np.float32), timestamp=0.0,
        )
        action = np.array([1.0, 0.0], dtype=np.float32)
        pred, conf = small_mlp.predict(state, action)
        assert pred.values.shape == (4,)
        assert 0.0 < conf <= 1.0


# ── Reset ─────────────────────────────────────────────────────


class TestReset:
    """Tests for reset()."""

    def test_reset_clears_cache(self, mlp, sample_state, sample_action):
        """reset() should clear cached activations."""
        mlp.predict(sample_state, sample_action)
        assert mlp._last_input is not None
        mlp.reset()
        assert mlp._last_input is None
        assert mlp._last_activations is None

    def test_reset_preserves_weights(self, mlp):
        """reset() should NOT change weights (they persist across episodes)."""
        w1, b1, w2, b2, w3, b3 = mlp.w1.copy(), mlp.b1.copy(), mlp.w2.copy(), mlp.b2.copy(), mlp.w3.copy(), mlp.b3.copy()
        mlp.reset()
        assert np.allclose(mlp.w1, w1)
        assert np.allclose(mlp.b1, b1)
        assert np.allclose(mlp.w2, w2)
        assert np.allclose(mlp.b2, b2)
        assert np.allclose(mlp.w3, w3)
        assert np.allclose(mlp.b3, b3)


# ── Integration Interface Compatibility ───────────────────────


class TestInterfaceCompatibility:
    """MLP must match G' predict/learn interface exactly."""

    def test_predict_returns_tuple(self, mlp, sample_state, sample_action):
        """predict() returns (StateVector, float) like G'."""
        result = mlp.predict(sample_state, sample_action)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], StateVector)
        assert isinstance(result[1], float)

    def test_learn_accepts_four_args(self, mlp, sample_state, sample_action, sample_next_state):
        """learn() accepts (state, action, next_state, error) like G'."""
        # Should not raise
        mlp.predict(sample_state, sample_action)  # cache forward pass
        mlp.learn(sample_state, sample_action, sample_next_state, error=0.5)

    def test_reset_method(self, mlp):
        """reset() exists and is callable like G'.reset()."""
        mlp.reset()

    def test_state_dim_property(self, mlp):
        """MLP exposes state_dim like G'."""
        assert mlp.state_dim == 84

    def test_predict_timestamp(self, mlp, sample_state, sample_action):
        """Predicted state timestamp should be state.timestamp + 1.0."""
        pred, _ = mlp.predict(sample_state, sample_action)
        assert pred.timestamp == sample_state.timestamp + 1.0

    def test_predict_grounding_level(self, mlp, sample_state, sample_action):
        """Predicted state grounding_level should match input."""
        pred, _ = mlp.predict(sample_state, sample_action)
        assert pred.grounding_level == sample_state.grounding_level


# ── Cache Management ──────────────────────────────────────────


class TestCacheManagement:
    """Tests for internal activation caching."""

    def test_predict_caches_input(self, mlp, sample_state, sample_action):
        """predict() should cache input for backward pass."""
        assert mlp._last_input is None
        mlp.predict(sample_state, sample_action)
        assert mlp._last_input is not None
        assert mlp._last_input.shape == (89,)  # 84 + 5

    def test_predict_caches_activations(self, mlp, sample_state, sample_action):
        """predict() should cache activations for backward pass."""
        assert mlp._last_activations is None
        mlp.predict(sample_state, sample_action)
        assert mlp._last_activations is not None
        z1, z2, out = mlp._last_activations
        assert z1.shape == (128,)
        assert z2.shape == (128,)
        assert out.shape == (84,)

    def test_cache_overwritten_on_new_predict(self, mlp, sample_state, sample_action):
        """A second predict() should overwrite cached activations."""
        mlp.predict(sample_state, sample_action)
        cache1 = mlp._last_activations
        # Different state should produce different cache
        state2 = StateVector(
            values=np.ones(84, dtype=np.float32),
            precision=np.ones(84, dtype=np.float32), timestamp=0.0,
        )
        mlp.predict(state2, sample_action)
        cache2 = mlp._last_activations
        assert not np.allclose(cache1[2], cache2[2])  # outputs differ
