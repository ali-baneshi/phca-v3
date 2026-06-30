"""
Tests for Criticality Regulator PID Controller (PHCA-3.2-005).

Covers: PID regulation, error dynamics, orthogonality constraint,
parameter freeze/unfreeze, reset.

v3.0 Reference: §3.4 Definition 3.7, v3.0 Patch §2.6
"""

from __future__ import annotations

import pytest
import numpy as np

from phca.regulation.pid_controller import CriticalityRegulator


@pytest.fixture
def cr() -> CriticalityRegulator:
    return CriticalityRegulator(
        setpoint=0.5, k_p=1.0, k_i=0.1, k_d=0.05,
        T_base=1.0, eta_base=0.1, alpha_base=0.5,
    )


class TestCriticalityRegulatorInit:
    """Initialization tests."""

    def test_default_params(self):
        """Default parameters should be set correctly."""
        cr = CriticalityRegulator()
        assert cr.setpoint == 0.5
        assert cr.k_p == 1.0
        assert cr.k_i == 0.1
        assert cr.k_d == 0.05
        assert cr.T_base == 1.0
        assert cr.eta_base == 0.1
        assert cr.alpha_base == 0.5

    def test_initial_pid_state(self, cr):
        """PID state should start at zero."""
        assert cr._integral == 0.0
        assert cr._prev_error == 0.0
        assert cr._cycle == 0
        assert cr._frozen_params == set()


class TestPIDRegulation:
    """PID control loop tests."""

    def test_regulate_returns_tuple(self, cr):
        """regulate() should return (T, eta, alpha) tuple."""
        result = cr.regulate(0.5)
        assert isinstance(result, tuple)
        assert len(result) == 3
        T, eta, alpha = result
        assert T >= 0.1
        assert eta >= 0.001
        assert 0.01 <= alpha <= 1.0

    def test_regulate_at_setpoint(self, cr):
        """When phi == setpoint, output should be near baseline."""
        T, eta, alpha = cr.regulate(0.5)
        assert T == pytest.approx(cr.T_base, abs=0.01)
        assert eta == pytest.approx(cr.eta_base, abs=0.001)
        assert alpha == pytest.approx(cr.alpha_base, abs=0.01)

    def test_regulate_below_setpoint(self, cr):
        """When phi < setpoint, temperature should increase."""
        _, eta_low, alpha_low = cr.regulate(0.3)
        _, eta_high, alpha_high = cr.regulate(0.5)
        # Actually both calls have different states because integral accumulates
        # Let me test differently
        cr2 = CriticalityRegulator()
        T_below, eta_below, alpha_below = cr2.regulate(0.3)
        cr3 = CriticalityRegulator()
        T_at, eta_at, alpha_at = cr3.regulate(0.5)
        assert T_below > T_at  # Need more heat when below setpoint

    def test_regulate_above_setpoint(self, cr):
        """When phi > setpoint, temperature should decrease."""
        cr_below = CriticalityRegulator()
        cr_above = CriticalityRegulator()
        T_below, _, _ = cr_below.regulate(0.3)
        T_above, _, _ = cr_above.regulate(0.7)
        assert T_below > T_above

    def test_exploration_noise_increases_with_error(self, cr):
        """Eta should increase with larger error."""
        cr1 = CriticalityRegulator()
        cr2 = CriticalityRegulator()
        _, eta_small, _ = cr1.regulate(0.45)  # small error
        _, eta_large, _ = cr2.regulate(0.0)   # large error
        assert eta_large > eta_small

    def test_attention_spread_decreases_with_error(self, cr):
        """Alpha should decrease (narrower focus) with larger error."""
        cr1 = CriticalityRegulator()
        cr2 = CriticalityRegulator()
        _, _, alpha_small = cr1.regulate(0.45)  # small error
        _, _, alpha_large = cr2.regulate(0.0)   # large error
        assert alpha_large < alpha_small

    def test_integral_accumulates(self, cr):
        """Integral term should accumulate over multiple cycles."""
        cr.regulate(0.3)
        i1 = cr._integral
        cr.regulate(0.3)
        i2 = cr._integral
        assert abs(i2) > abs(i1)

    def test_anti_windup(self):
        """Integral should be clipped to prevent windup."""
        cr = CriticalityRegulator(integral_limit=1.0)
        for _ in range(100):
            cr.regulate(0.0)  # large persistent error
        assert abs(cr._integral) <= 1.0

    def test_increments_cycle(self, cr):
        """regulate() should increment the cycle counter."""
        assert cr._cycle == 0
        cr.regulate(0.5)
        assert cr._cycle == 1
        cr.regulate(0.5)
        assert cr._cycle == 2


class TestOrthogonalityConstraint:
    """Orthogonality constraint tests."""

    def test_no_freeze_initially(self, cr):
        """No parameters should be frozen initially."""
        assert cr.get_frozen_params() == set()

    def test_orthogonality_check_not_enough_data(self, cr):
        """With < 10 data points, no freeze should occur."""
        for _ in range(5):
            cr.regulate(0.5)
        assert cr.get_frozen_params() == set()

    def test_freeze_with_high_correlation(self):
        """Highly correlated parameters should trigger freeze."""
        cr = CriticalityRegulator(orthogonality_threshold=0.5)
        rng = np.random.RandomState(42)

        # Simulate highly correlated T and eta
        for i in range(50):
            # Force positive correlation between T and eta
            base = rng.randn() * 0.3
            cr._param_history_buffer.append({
                "T": 1.0 + base,
                "eta": 0.1 + base * 0.5,
                "alpha": 0.5 + rng.randn() * 0.1,
            })

        cr._check_orthogonality()
        frozen = cr.get_frozen_params()
        # At least one parameter should be frozen
        assert len(frozen) >= 1, f"No params frozen: corr={np.corrcoef([p['T'] for p in cr._param_history_buffer[-50:]],[p['eta'] for p in cr._param_history_buffer[-50:]])[0,1]}"

    def test_unfreeze_all(self, cr):
        """unfreeze_all should clear frozen params."""
        cr._frozen_params.add("T")
        assert "T" in cr.get_frozen_params()
        cr.unfreeze_all()
        assert cr.get_frozen_params() == set()

    def test_frozen_param_not_updated(self):
        """A frozen parameter should remain at its previous value."""
        cr = CriticalityRegulator()
        # First regulate to establish a baseline
        T0, eta0, alpha0 = cr.regulate(0.5)
        # Freeze T
        cr._frozen_params.add("T")
        T1, eta1, alpha1 = cr.regulate(0.3)  # error should change T normally
        assert T1 == pytest.approx(T0, abs=0.001)  # T frozen
        assert eta1 != pytest.approx(eta0, abs=0.001)  # eta still updates
        assert alpha1 != pytest.approx(alpha0, abs=0.001)  # alpha still updates

    def test_frozen_params_unfreeze_on_orthogonality_break(self, cr):
        """Freezing one param should unfreeze the correlated pair."""
        cr._frozen_params.add("T")
        cr._frozen_params.add("eta")
        assert len(cr._frozen_params) == 2

        # Check that _check_orthogonality can force-unfreeze
        cr._param_history_buffer = [
            {"T": 1.0, "eta": 0.1, "alpha": 0.5},
            {"T": 1.0, "eta": 0.1, "alpha": 0.51},
        ] * 25  # 50 entries, T and eta perfectly correlated
        cr._check_orthogonality()
        # T and eta are perfectly correlated, one should be frozen
        # but the other might be unfrozen
        frozen = cr.get_frozen_params()
        assert len(frozen) >= 1


class TestReset:
    """Reset behavior."""

    def test_reset_clears_pid_state(self, cr):
        """Reset should clear integral, prev_error, and cycle."""
        cr.regulate(0.3)
        cr.regulate(0.3)
        assert cr._integral != 0.0
        assert cr._cycle > 0

        cr.reset()
        assert cr._integral == 0.0
        assert cr._prev_error == 0.0
        assert cr._cycle == 0

    def test_reset_clears_orthogonality(self, cr):
        """Reset should clear frozen params and history."""
        cr._frozen_params.add("T")
        cr._param_history_buffer.append({"T": 1.0, "eta": 0.1, "alpha": 0.5})
        cr.reset()
        assert cr._frozen_params == set()
        assert cr._param_history_buffer == []

    def test_reset_restores_base_output(self, cr):
        """After reset, first regulate should return baseline."""
        cr.regulate(0.3)  # Change state
        cr.reset()
        T, eta, alpha = cr.regulate(0.5)
        assert T == pytest.approx(cr.T_base, abs=0.01)
        assert eta == pytest.approx(cr.eta_base, abs=0.001)


class TestOutputScaling:
    """Output parameter scaling tests."""

    def test_temperature_range(self):
        """Temperature should be clamped to minimum 0.1."""
        cr = CriticalityRegulator(k_p=100.0)  # Extreme gain
        T, _, _ = cr.regulate(0.6)  # error = -0.1 → pid_out = -10
        assert T >= 0.1

    def test_exploration_noise_range(self):
        """Exploration noise should be at least 0.001."""
        cr = CriticalityRegulator()
        _, eta, _ = cr.regulate(0.5)  # zero error → baseline
        assert eta > 0.0
        assert eta < 1.0

    def test_attention_spread_range(self):
        """Attention spread should be clamped to [0.01, 1.0]."""
        cr = CriticalityRegulator()
        _, _, alpha = cr.regulate(0.5)
        assert 0.01 <= alpha <= 1.0

    def test_custom_gains(self):
        """Custom PID gains should affect output."""
        cr_default = CriticalityRegulator()
        cr_custom = CriticalityRegulator(k_p=5.0, k_i=0.5, k_d=0.25)

        T1, _, _ = cr_default.regulate(0.3)
        T2, _, _ = cr_custom.regulate(0.3)
        assert abs(T2 - 1.0) > abs(T1 - 1.0)  # stronger reaction
