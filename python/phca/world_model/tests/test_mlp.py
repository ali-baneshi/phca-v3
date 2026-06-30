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


# ── Backward Pass (Gradient) ──────────────────────────────────


class TestGradient:
    """Tests for backward pass and gradient computation."""

    def test_compute_gradient_returns_dict(self, small_mlp):
        """compute_gradient should return dict with all 6 param keys."""
        state = StateVector(
            values=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            precision=np.ones(4, dtype=np.float32), timestamp=0.0,
        )
        action = np.array([1.0, 0.0], dtype=np.float32)
        target = StateVector(
            values=np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32),
            precision=np.ones(4, dtype=np.float32), timestamp=1.0,
        )
        # First call predict to cache activations
        small_mlp.predict(state, action)
        grad = small_mlp.compute_gradient(state, action, target)
        expected_keys = {"gprime_w1", "gprime_b1", "gprime_w2", "gprime_b2", "gprime_w3", "gprime_b3"}
        assert set(grad.keys()) == expected_keys

    def test_gradient_shapes(self, small_mlp):
        """Gradient shapes should match parameter shapes."""
        state = StateVector(
            values=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            precision=np.ones(4, dtype=np.float32), timestamp=0.0,
        )
        action = np.array([1.0, 0.0], dtype=np.float32)
        target = StateVector(
            values=np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32),
            precision=np.ones(4, dtype=np.float32), timestamp=1.0,
        )
        small_mlp.predict(state, action)
        grad = small_mlp.compute_gradient(state, action, target)

        assert grad["gprime_w1"].shape == (6, 8)   # (4+2, 8)
        assert grad["gprime_b1"].shape == (8,)
        assert grad["gprime_w2"].shape == (8, 8)
        assert grad["gprime_b2"].shape == (8,)
        assert grad["gprime_w3"].shape == (8, 4)
        assert grad["gprime_b3"].shape == (4,)

    def test_gradient_is_nonzero(self, small_mlp):
        """Gradient should be non-zero when prediction differs from target."""
        state = StateVector(
            values=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            precision=np.ones(4, dtype=np.float32), timestamp=0.0,
        )
        action = np.array([1.0, 0.0], dtype=np.float32)
        target = StateVector(
            values=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),  # different from initial pred
            precision=np.ones(4, dtype=np.float32), timestamp=1.0,
        )
        small_mlp.predict(state, action)
        grad = small_mlp.compute_gradient(state, action, target)

        # At least some gradients should be non-zero
        total_norm = sum(np.sum(g**2) for g in grad.values())
        assert total_norm > 1e-10

    def test_gradient_zero_on_perfect_match(self, small_mlp):
        """Gradient should be near-zero when pred matches target."""
        state = StateVector(
            values=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            precision=np.ones(4, dtype=np.float32), timestamp=0.0,
        )
        action = np.array([1.0, 0.0], dtype=np.float32)
        # Set target equal to the initial forward pass
        small_mlp.predict(state, action)
        # The prediction is stored; use it as target
        target = small_mlp._last_activations[2]  # out
        target_sv = StateVector(
            values=target.astype(np.float32),
            precision=np.ones(4, dtype=np.float32), timestamp=1.0,
        )
        grad = small_mlp.compute_gradient(state, action, target_sv)

        # All gradient norms should be very small
        for key, g in grad.items():
            norm = float(np.sqrt(np.sum(g**2)))
            assert norm < 1e-4, f"{key} gradient norm {norm} should be near zero"

    def test_gradient_clipped(self, small_mlp):
        """Gradient values should be clipped to [-1, 1]."""
        state = StateVector(
            values=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            precision=np.ones(4, dtype=np.float32), timestamp=0.0,
        )
        action = np.array([1.0, 0.0], dtype=np.float32)
        target = StateVector(
            values=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
            precision=np.ones(4, dtype=np.float32), timestamp=1.0,
        )
        small_mlp.predict(state, action)
        grad = small_mlp.compute_gradient(state, action, target)

        for g in grad.values():
            assert np.all(g >= -1.0 - 1e-6), f"gradient below -1: {g.min()}"
            assert np.all(g <= 1.0 + 1e-6), f"gradient above 1: {g.max()}"

    def test_finite_difference_verification(self, small_mlp):
        """Verify gradients numerically: (L(θ+h) - L(θ-h)) / 2h ≈ dL/dθ."""
        state_np = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        action_np = np.array([1.0, 0.0], dtype=np.float32)
        target_np = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)

        def mse_loss(out: np.ndarray, tgt: np.ndarray) -> float:
            return 0.5 * float(np.mean((out - tgt) ** 2))

        state = StateVector(values=state_np, precision=np.ones(4, dtype=np.float32), timestamp=0.0)
        target = StateVector(values=target_np, precision=np.ones(4, dtype=np.float32), timestamp=1.0)
        action = action_np

        small_mlp.predict(state, action)
        grad = small_mlp.compute_gradient(state, action, target)

        # Pick one parameter (b3[0]) for finite-difference check
        h = 1e-4
        idx = 0
        orig = small_mlp.b3[idx].copy()

        # L(θ + h)
        small_mlp.b3[idx] = orig + h
        _, _, out_plus = small_mlp._forward(
            np.concatenate([state_np, action_np])
        )
        mse_plus = mse_loss(out_plus, target_np)

        # L(θ - h)
        small_mlp.b3[idx] = orig - h
        _, _, out_minus = small_mlp._forward(
            np.concatenate([state_np, action_np])
        )
        mse_minus = mse_loss(out_minus, target_np)

        small_mlp.b3[idx] = orig  # restore

        numerical_grad = (mse_plus - mse_minus) / (2.0 * h)
        analytical_grad = grad["gprime_b3"][idx]

        # Relative error should be < 1%
        denom = max(abs(numerical_grad), 1e-8)
        rel_error = abs(analytical_grad - numerical_grad) / denom
        assert rel_error < 0.01, (
            f"b3[{idx}]: numerical={numerical_grad:.8f}, "
            f"analytical={analytical_grad:.8f}, rel_err={rel_error:.6f}"
        )

    def test_gradient_without_forward_cache(self, mlp, sample_state, sample_action, sample_next_state):
        """Without a cached forward pass, compute_gradient returns zeros."""
        grad = mlp.compute_gradient(sample_state, sample_action, sample_next_state)
        for g in grad.values():
            assert np.all(g == 0.0), "Gradient should be zero without cached forward pass"


# ── Theta Sync ────────────────────────────────────────────────


class TestThetaSync:
    """Tests for get_theta() / set_theta() round-trip."""

    def test_get_theta_keys(self, mlp):
        """get_theta() should return all 6 parameter keys."""
        theta = mlp.get_theta()
        expected_keys = {"gprime_w1", "gprime_b1", "gprime_w2", "gprime_b2", "gprime_w3", "gprime_b3"}
        assert set(theta.keys()) == expected_keys

    def test_get_theta_shapes(self, mlp):
        """get_theta() shapes should match architecture."""
        theta = mlp.get_theta()
        assert theta["gprime_w1"].shape == (89, 128)
        assert theta["gprime_b1"].shape == (128,)
        assert theta["gprime_w2"].shape == (128, 128)
        assert theta["gprime_b2"].shape == (128,)
        assert theta["gprime_w3"].shape == (128, 84)
        assert theta["gprime_b3"].shape == (84,)

    def test_round_trip_preserves_values(self, mlp):
        """set_theta(get_theta()) should preserve all weights."""
        theta = mlp.get_theta()
        # Modify a copy
        theta["gprime_w1"][0, 0] = 99.0
        theta["gprime_b3"][-1] = -5.0
        mlp.set_theta(theta)
        theta2 = mlp.get_theta()
        assert theta2["gprime_w1"][0, 0] == 99.0
        assert theta2["gprime_b3"][-1] == -5.0

    def test_set_theta_partial(self, mlp):
        """set_theta should ignore unknown keys and not crash on partial dict."""
        partial = {"gprime_w1": np.zeros((89, 128), dtype=np.float32)}
        mlp.set_theta(partial)
        assert np.all(mlp.w1 == 0.0)
        # Other params should be unchanged
        assert np.any(mlp.w2 != 0.0)

    def test_theta_affects_prediction(self, mlp, sample_state, sample_action):
        """Changing theta should produce different predictions."""
        pred_before, _ = mlp.predict(sample_state, sample_action)

        # Zero out all weights
        zero_theta = {
            "gprime_w1": np.zeros((89, 128), dtype=np.float32),
            "gprime_b1": np.zeros(128, dtype=np.float32),
            "gprime_w2": np.zeros((128, 128), dtype=np.float32),
            "gprime_b2": np.zeros(128, dtype=np.float32),
            "gprime_w3": np.zeros((128, 84), dtype=np.float32),
            "gprime_b3": np.zeros(84, dtype=np.float32),
        }
        mlp.set_theta(zero_theta)

        pred_after, _ = mlp.predict(sample_state, sample_action)
        # With zero weights and zero biases, output = 0.0
        assert np.allclose(pred_after.values, 0.0)


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
        theta_before = mlp.get_theta()
        mlp.reset()
        theta_after = mlp.get_theta()
        for k in theta_before:
            assert np.allclose(theta_before[k], theta_after[k]), (
                f"{k} changed after reset()"
            )


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
