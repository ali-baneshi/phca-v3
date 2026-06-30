"""Tests for PHCA-3.1-009: Prediction Engine + Prediction Error Unit."""

from __future__ import annotations

import numpy as np
import pytest

from phca.config import StateVector
from phca.prediction.engine import PredictionEngine
from phca.prediction.error_unit import PredictionErrorUnit


# ── Fixtures ──────────────────────────────────────────────────


@pytest.fixture
def sample_state() -> StateVector:
    return StateVector(
        values=np.array([0.5, -0.3], dtype=np.float32),
        precision=np.array([0.9, 0.9], dtype=np.float32),
        timestamp=1.0,
    )


@pytest.fixture
def predicted_state() -> StateVector:
    return StateVector(
        values=np.array([0.48, -0.28], dtype=np.float32),
        precision=np.array([0.9, 0.9], dtype=np.float32),
        timestamp=2.0,
    )


@pytest.fixture
def distant_prediction() -> StateVector:
    return StateVector(
        values=np.array([0.1, 0.8], dtype=np.float32),
        precision=np.array([0.9, 0.9], dtype=np.float32),
        timestamp=2.0,
    )


# ── PredictionEngine Tests ───────────────────────────────────


class TestPredictionEngine:
    """Tests for PredictionEngine class."""

    def test_horizon_1(self, mocker):
        """Predict horizon=1 should produce a valid result."""
        gprime = mocker.MagicMock()
        gprime.action_dim = 0
        gprime.predict.return_value = (
            StateVector(
                values=np.array([0.48, -0.29], dtype=np.float32),
                precision=np.array([0.85, 0.85], dtype=np.float32),
                timestamp=2.0,
            ),
            0.85,
        )

        engine = PredictionEngine(gprime)
        state = StateVector(
            values=np.array([0.5, -0.3], dtype=np.float32),
            precision=np.array([0.9, 0.9], dtype=np.float32),
        )
        result, confidence = engine.predict(state, horizon=1)

        assert isinstance(result, StateVector)
        assert 0.0 <= confidence <= 1.0
        assert confidence == 0.85  # horizon=1: no decay
        # Verify call args, handling empty array safely
        call_args = gprime.predict.call_args
        assert call_args is not None
        called_state, called_action = call_args[0]
        np.testing.assert_array_equal(called_state.values, state.values)
        assert called_action.size == 0

    def test_horizon_3_confidence_decay(self, mocker):
        """Horizon > 1 should decay confidence by 1/horizon."""
        gprime = mocker.MagicMock()
        gprime.action_dim = 0

        def fake_predict(state, action):
            return (
                StateVector(
                    values=state.values + 0.01,
                    precision=np.ones_like(state.values, dtype=np.float32),
                    timestamp=state.timestamp + 1.0,
                ),
                0.9,
            )

        gprime.predict.side_effect = fake_predict

        engine = PredictionEngine(gprime)
        state = StateVector(
            values=np.array([0.5, -0.3], dtype=np.float32),
            precision=np.array([0.9, 0.9], dtype=np.float32),
        )
        result, confidence = engine.predict(state, horizon=3)

        assert engine.gprime.predict.call_count == 3
        # Confidence = 0.9 / 3 = 0.3
        assert confidence == pytest.approx(0.30, abs=1e-6)
        assert 0.0 <= confidence <= 1.0

    def test_horizon_100(self, mocker):
        """Horizon=100 (max) should not raise."""
        gprime = mocker.MagicMock()
        gprime.predict.return_value = (
            StateVector(
                values=np.array([0.48, -0.29], dtype=np.float32),
                precision=np.array([0.85, 0.85], dtype=np.float32),
                timestamp=2.0,
            ),
            0.9,
        )

        engine = PredictionEngine(gprime)
        state = StateVector(
            values=np.array([0.5, -0.3], dtype=np.float32),
            precision=np.array([0.9, 0.9], dtype=np.float32),
        )
        result, confidence = engine.predict(state, horizon=100)
        assert isinstance(result, StateVector)
        assert 0.0 <= confidence <= 1.0

    def test_invalid_horizon(self, mocker):
        """Horizon < 1 should raise ValueError."""
        gprime = mocker.MagicMock()
        engine = PredictionEngine(gprime)
        state = StateVector(
            values=np.array([0.5], dtype=np.float32),
            precision=np.array([0.9], dtype=np.float32),
        )
        with pytest.raises(ValueError, match="horizon must be in"):
            engine.predict(state, horizon=0)

    def test_invalid_horizon_101(self, mocker):
        """Horizon > 100 should raise ValueError."""
        gprime = mocker.MagicMock()
        engine = PredictionEngine(gprime)
        state = StateVector(
            values=np.array([0.5], dtype=np.float32),
            precision=np.array([0.9], dtype=np.float32),
        )
        with pytest.raises(ValueError, match="horizon must be in"):
            engine.predict(state, horizon=101)

    def test_grounding_level_2_not_implemented(self, mocker):
        """grounding_level=2 should raise NotImplementedError."""
        gprime = mocker.MagicMock()
        engine = PredictionEngine(gprime)
        state = StateVector(
            values=np.array([0.5], dtype=np.float32),
            precision=np.array([0.9], dtype=np.float32),
        )
        with pytest.raises(NotImplementedError, match="deferred to Phase 3.2"):
            engine.predict(state, horizon=1, grounding_level=2)

    def test_update_action(self, mocker):
        """update_action should store a copy of the action."""
        gprime = mocker.MagicMock()
        gprime.action_dim = 2
        engine = PredictionEngine(gprime)
        action = np.array([0.1, -0.5], dtype=np.float32)
        engine.update_action(action)

        np.testing.assert_array_equal(engine.last_action, action)

        # Verify it's a copy, not a reference
        action[0] = 99.0
        assert engine.last_action[0] == pytest.approx(0.1)


# ── PredictionErrorUnit Tests ────────────────────────────────


class TestPredictionErrorUnit:
    """Tests for PredictionErrorUnit class."""

    def test_compute_zero_error(self, sample_state):
        """Identical states should produce zero error."""
        peu = PredictionErrorUnit()
        error = peu.compute(sample_state, sample_state)
        assert error == 0.0

    def test_compute_small_error(self, sample_state, predicted_state):
        """Close states should produce small error."""
        peu = PredictionErrorUnit()
        error = peu.compute(sample_state, predicted_state)
        # diff = [0.02, -0.02], dot = 0.0004 + 0.0004 = 0.0008
        assert error == pytest.approx(0.0008, abs=1e-6)

    def test_compute_large_error(self, sample_state, distant_prediction):
        """Distant states should produce larger error."""
        peu = PredictionErrorUnit()
        error = peu.compute(sample_state, distant_prediction)
        # diff = [0.4, -1.1], dot = 0.16 + 1.21 = 1.37
        assert error == pytest.approx(1.37, abs=1e-6)
        assert error > 1.0

    def test_compute_precision_weighted(self, sample_state, predicted_state):
        """Precision-weighted error should weigh dimensions differently."""
        peu = PredictionErrorUnit()
        precision = np.array([0.5, 2.0], dtype=np.float32)
        error = peu.compute_precision_weighted(sample_state, predicted_state, precision)
        # diff = [0.02, -0.02], precision * diff^2 = [0.5*0.0004, 2.0*0.0004] = [0.0002, 0.0008]
        # sum = 0.0010
        assert error == pytest.approx(0.0010, abs=1e-6)

    def test_compute_precision_weighted_shape_mismatch(self, sample_state, predicted_state):
        """Precision with wrong shape should raise ValueError."""
        peu = PredictionErrorUnit()
        with pytest.raises(ValueError, match="precision shape"):
            peu.compute_precision_weighted(
                sample_state, predicted_state, np.array([0.5], dtype=np.float32)
            )

    def test_compute_rmse(self, sample_state, predicted_state):
        """RMSE should return sqrt of mean squared error."""
        peu = PredictionErrorUnit()
        rmse = peu.compute_rmse(sample_state, predicted_state)
        # mse = (0.0004 + 0.0004) / 2 = 0.0004
        # rmse = sqrt(0.0004) = 0.02
        assert rmse == pytest.approx(0.02, abs=1e-6)

    def test_compute_rmse_zero(self, sample_state):
        """RMSE of identical states should be zero."""
        peu = PredictionErrorUnit()
        rmse = peu.compute_rmse(sample_state, sample_state)
        assert rmse == 0.0

    def test_multiple_errors_ordering(self):
        """Errors should be ordered: distant > close > zero."""
        peu = PredictionErrorUnit()
        state = StateVector(
            values=np.zeros(3, dtype=np.float32),
            precision=np.ones(3, dtype=np.float32),
        )
        close = StateVector(
            values=np.array([0.01, 0.01, 0.01], dtype=np.float32),
            precision=np.ones(3, dtype=np.float32),
        )
        far = StateVector(
            values=np.array([1.0, -1.0, 0.5], dtype=np.float32),
            precision=np.ones(3, dtype=np.float32),
        )

        assert peu.compute(far, state) > peu.compute(close, state) > peu.compute(state, state)
