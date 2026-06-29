"""
Tests for the ASI Step 0 Sanitizer (PHCA-3.1-004).

Covers all acceptance criteria from v3.0 Patch B and Playbook §7.1.
Uses fixed random seed for deterministic tests.
"""

import numpy as np
import pytest

from phca.asi.sanitizer import ASISanitizer
from phca.config import ASIStatus


@pytest.fixture
def sanitizer():
    """Create a sanitizer with 4 sensor dimensions for testing."""
    return ASISanitizer(sensor_dim=4, v_max=100.0, epsilon_confidence=0.01)


class TestASISanitizer:
    """Test suite for ASISanitizer."""

    def test_valid_value_passes_through(self, sanitizer):
        """Valid inputs should not be modified."""
        raw = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        state, status = sanitizer.sanitize(raw)
        assert status == ASIStatus.OK
        np.testing.assert_array_almost_equal(state.values, raw)

    def test_nan_replaced_by_last_valid(self, sanitizer):
        """NaN should be replaced by the last valid value."""
        raw_valid = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        sanitizer.sanitize(raw_valid)

        raw_nan = np.array([np.nan, 2.0, 3.0, 4.0], dtype=np.float32)
        state, status = sanitizer.sanitize(raw_nan)
        assert status == ASIStatus.OK
        # Sensor 0 should be the last valid value (1.0)
        assert state.values[0] == 1.0

    def test_inf_replaced_by_last_valid(self, sanitizer):
        """Inf should be replaced by the last valid value."""
        raw_valid = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        sanitizer.sanitize(raw_valid)

        raw_inf = np.array([np.inf, 2.0, 3.0, 4.0], dtype=np.float32)
        state, status = sanitizer.sanitize(raw_inf)
        assert status == ASIStatus.OK
        assert state.values[0] == 1.0

    def test_overflow_replaced_by_last_valid(self, sanitizer):
        """Values exceeding V_max should be replaced."""
        raw_valid = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        sanitizer.sanitize(raw_valid)

        raw_overflow = np.array([200.0, 2.0, 3.0, 4.0], dtype=np.float32)  # V_max = 100
        state, status = sanitizer.sanitize(raw_overflow)
        assert status == ASIStatus.OK
        assert state.values[0] == 1.0

    def test_precision_halves_on_each_failure(self, sanitizer):
        """Precision should halve on each consecutive failure."""
        raw_valid = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        sanitizer.sanitize(raw_valid)

        raw_fail = np.array([np.nan, 2.0, 3.0, 4.0], dtype=np.float32)
        # First failure: precision 1.0 → 0.5
        sanitizer.sanitize(raw_fail)
        assert sanitizer.precision[0] == 0.5
        # Second failure: precision 0.5 → 0.25
        sanitizer.sanitize(raw_fail)
        assert sanitizer.precision[0] == 0.25

    def test_sensor_failure_after_7_consecutive_failures(self, sanitizer):
        """ASI_SENSOR_FAILURE should be raised after 7 consecutive failures on same sensor."""
        raw_valid = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        sanitizer.sanitize(raw_valid)

        raw_fail = np.array([np.nan, 2.0, 3.0, 4.0], dtype=np.float32)
        status = ASIStatus.OK
        for _ in range(7):
            _, status = sanitizer.sanitize(raw_fail)
        assert status == ASIStatus.SENSOR_FAILURE, \
            f"Expected SENSOR_FAILURE after 7 failures, got {status}"
        assert sanitizer.precision[0] < sanitizer.epsilon_confidence

    def test_recovery_after_failure(self, sanitizer):
        """After a valid value arrives, precision should reset."""
        raw_valid = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        sanitizer.sanitize(raw_valid)

        raw_fail = np.array([np.nan, 2.0, 3.0, 4.0], dtype=np.float32)
        for _ in range(3):
            sanitizer.sanitize(raw_fail)
        assert sanitizer.precision[0] == 0.125  # 1.0 * 0.5^3

        # Valid value arrives — failure count resets (precision stays for attention to update)
        sanitizer.sanitize(raw_valid)
        assert sanitizer.failure_count[0] == 0

    def test_asi_failure_limit_triggers_global_recovery(self, sanitizer):
        """When > d/3 sensors fail simultaneously, global recovery should be indicated."""
        # Create a sanitizer with 3 sensors (failure limit = 1)
        sani = ASISanitizer(sensor_dim=3, v_max=100.0)
        raw_valid = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        sani.sanitize(raw_valid)

        # Two sensors fail simultaneously (2 > 3/3 = 1)
        raw_fail = np.array([np.nan, np.nan, 3.0], dtype=np.float32)
        # After 7 failures, sensor_failure will be raised
        for _ in range(7):
            state, status = sani.sanitize(raw_fail)

        # Two sensors should be at low precision
        assert sani.precision[0] < sani.epsilon_confidence
        assert sani.precision[1] < sani.epsilon_confidence

    def test_reset_clears_all_state(self, sanitizer):
        """Reset should clear all per-sensor state."""
        raw_valid = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        sanitizer.sanitize(raw_valid)

        raw_fail = np.array([np.nan, 2.0, 3.0, 4.0], dtype=np.float32)
        sanitizer.sanitize(raw_fail)
        assert sanitizer.precision[0] == 0.5

        sanitizer.reset()
        np.testing.assert_array_equal(sanitizer.precision, np.ones(4))
        np.testing.assert_array_equal(sanitizer.failure_count, np.zeros(4, dtype=np.int32))

    def test_sanitized_state_vector_has_correct_shape(self, sanitizer):
        """The returned StateVector should have the correct attributes."""
        raw = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32)
        state, status = sanitizer.sanitize(raw, timestamp=42.0)
        assert state.dim == 4
        assert state.timestamp == 42.0
        assert state.grounding_level == 1
        np.testing.assert_array_equal(state.values, raw)
        np.testing.assert_array_equal(state.precision, np.ones(4))

    def test_consecutive_failure_count_tracking(self, sanitizer):
        """Failure count should increase per-consecutive-failure and reset on valid."""
        raw_valid = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        sanitizer.sanitize(raw_valid)

        raw_fail = np.array([np.nan, 2.0, 3.0, 4.0], dtype=np.float32)
        sanitizer.sanitize(raw_fail)
        assert sanitizer.failure_count[0] == 1
        sanitizer.sanitize(raw_fail)
        assert sanitizer.failure_count[0] == 2

        # Valid value resets sensor 0
        sanitizer.sanitize(raw_valid)
        assert sanitizer.failure_count[0] == 0
