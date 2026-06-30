"""Tests for PHCA-3.1-010: P-Stream TSPL + Skill Compilation."""

from __future__ import annotations

import numpy as np
import pytest

from phca.config import StateVector, StreamID
from phca.learning.tspl import TSPL



# ── Fixtures ──────────────────────────────────────────────────


@pytest.fixture
def tspl() -> TSPL:
    model = TSPL(seed=42)
    model.init_parameters("gprime_cpd_transition", (4, 4))
    model.init_parameters("gprime_cpd_bias", (4,))
    return model


@pytest.fixture
def sample_state() -> StateVector:
    return StateVector(
        values=np.array([0.5, -0.3], dtype=np.float32),
        precision=np.array([0.9, 0.9], dtype=np.float32),
    )


@pytest.fixture
def sample_prediction() -> StateVector:
    return StateVector(
        values=np.array([0.48, -0.28], dtype=np.float32),
        precision=np.array([0.9, 0.9], dtype=np.float32),
    )


# ── TSPL Tests ───────────────────────────────────────────────


class TestTSPLInit:
    """Tests for TSPL initialization."""

    def test_default_configs(self):
        """Default stream configs should have correct ordering."""
        tspl = TSPL()
        # P-Stream: highest alpha, lowest lambda, highest eta, enabled by default
        assert tspl.configs[StreamID.P_STREAM].alpha == 0.08
        assert tspl.configs[StreamID.P_STREAM].lambda_ == 0.01
        assert tspl.configs[StreamID.P_STREAM].eta == 0.1
        assert tspl.configs[StreamID.P_STREAM].accuracy_threshold == 0.95
        assert tspl.configs[StreamID.P_STREAM].enabled is True

    def test_initial_state(self):
        """Fresh TSPL should have no parameters."""
        tspl = TSPL()
        assert tspl.theta == {}
        assert not tspl.skill_compiled
        assert tspl.skill_accuracy == 0.0
        assert tspl.compiled_skill_ids == []

    def test_init_parameters_creates_array(self, tspl):
        """init_parameters should create float32 arrays."""
        assert "gprime_cpd_transition" in tspl.theta
        assert tspl.theta["gprime_cpd_transition"].shape == (4, 4)
        assert tspl.theta["gprime_cpd_transition"].dtype == np.float32

class TestTSPLUpdate:
    """Tests for TSPL.update()."""

    def test_p_stream_update_changes_theta(self, tspl, sample_state, sample_prediction):
        """P-Stream update should modify theta."""
        original = {k: v.copy() for k, v in tspl.theta.items()}
        theta_new, compiled = tspl.update(
            StreamID.P_STREAM, 0.01, sample_state, sample_prediction,
        )
        # Theta should be updated
        for key in original:
            assert not np.allclose(original[key], theta_new[key]), \
                f"{key} should change after P-Stream update"

    def test_p_stream_update_returns_theta(self, tspl, sample_state, sample_prediction):
        """P-Stream update should return theta dict."""
        theta_new, compiled = tspl.update(
            StreamID.P_STREAM, 0.01, sample_state, sample_prediction,
        )
        assert isinstance(theta_new, dict)
        assert "gprime_cpd_transition" in theta_new
        assert theta_new["gprime_cpd_transition"].shape == (4, 4)

    def test_update_with_no_theta_returns_empty(self, sample_state, sample_prediction):
        """Update with no initialized parameters should return empty."""
        tspl = TSPL()
        theta_new, compiled = tspl.update(
            StreamID.P_STREAM, 0.01, sample_state, sample_prediction,
        )
        assert theta_new == {}
        assert not compiled

    def test_update_with_gradient(self, tspl, sample_state, sample_prediction):
        """Providing explicit gradient should be used instead of auto-computation."""
        gradient = {
            "gprime_cpd_transition": np.zeros((4, 4), dtype=np.float32),
            "gprime_cpd_bias": np.zeros((4,), dtype=np.float32),
        }
        theta_new, compiled = tspl.update(
            StreamID.P_STREAM, 0.01, sample_state, sample_prediction,
            gradient=gradient,
        )
        assert isinstance(theta_new, dict)

    def test_skill_not_compiled_on_large_error(self, tspl):
        """Large prediction error should not trigger skill compilation."""
        state = StateVector(
            values=np.array([0.5, -0.3], dtype=np.float32),
            precision=np.array([0.9, 0.9], dtype=np.float32),
        )
        prediction = StateVector(
            values=np.array([-0.5, 0.3], dtype=np.float32),  # large diff
            precision=np.array([0.9, 0.9], dtype=np.float32),
        )
        theta_new, compiled = tspl.update(
            StreamID.P_STREAM, 5.0, state, prediction,
        )
        assert not compiled
        assert tspl.skill_accuracy < 0.5

    def test_accuracy_estimation(self, tspl):
        """_estimate_accuracy should return 0 for large error, 1 for zero error."""
        state = StateVector(
            values=np.array([0.5, -0.3], dtype=np.float32),
            precision=np.array([0.9, 0.9], dtype=np.float32),
        )

        # Zero error → accuracy = 1.0
        acc1 = tspl._estimate_accuracy(state, state, 0.0)
        assert acc1 == pytest.approx(1.0)

        # Large error → accuracy near 0
        large_error = 10.0  # sqrt(10/2) ≈ 2.236 → max(0, 1-2.236) = 0
        acc2 = tspl._estimate_accuracy(state, state, large_error)
        assert acc2 == 0.0
