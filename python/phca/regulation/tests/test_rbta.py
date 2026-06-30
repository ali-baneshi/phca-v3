"""
Tests for the RBTA Constraint Enforcer (PHCA-3.1-006).

Cross-ref: v3.0 §2.1 Definition 2.2, Definition 2.3, Theorem 2.1, Theorem 3.1
"""

from __future__ import annotations

import pytest
import numpy as np

from phca.config import ConstraintViolation, ResourceBounds
from phca.regulation.rbta_enforcer import RBTAEnforcer, EnforcerAction


DEFAULT_BOUNDS: dict[str, ResourceBounds] = {
    "ASI": ResourceBounds(B_time=0.002, B_mem=100_000, B_energy=10.0),
    "WM": ResourceBounds(B_time=0.005, B_mem=50_000, B_energy=5.0),
    "G'": ResourceBounds(B_time=0.020, B_mem=500_000, B_energy=50.0, entropy_floor=0.01),
    "PE": ResourceBounds(B_time=0.025, B_mem=200_000, B_energy=20.0),
    "TSPL-P": ResourceBounds(B_time=0.020, B_mem=300_000, B_energy=30.0),
}


@pytest.fixture
def enforcer() -> RBTAEnforcer:
    return RBTAEnforcer(DEFAULT_BOUNDS)


class TestRBTAEnforcer:
    """Core RBTA enforcer tests."""

    def test_single_module_time_violation(self, enforcer: RBTAEnforcer):
        """Runtime > B_time → 1 violation, action = INTERRUPT."""
        violations, action = enforcer.check_cycle(
            runtime_log={"ASI": 0.010},  # B_time = 0.002, exceeded!
            memory_log={},
            energy_log={},
            belief_entropies={},
            sensor_failure_count=0,
        )
        assert len(violations) == 1
        assert violations[0].bound_type == "TIME"
        assert violations[0].module_id == "ASI"
        assert violations[0].measured == 0.010
        assert violations[0].allowed == 0.002
        assert action == EnforcerAction.INTERRUPT

    def test_all_bounds_satisfied(self, enforcer: RBTAEnforcer):
        """All modules within bounds → 0 violations, action = CONTINUE."""
        violations, action = enforcer.check_cycle(
            runtime_log={"ASI": 0.001, "WM": 0.003, "G'": 0.015, "PE": 0.020},
            memory_log={"ASI": 10_000, "WM": 5_000, "G'": 100_000, "PE": 50_000},
            energy_log={"ASI": 1.0, "WM": 0.5, "G'": 10.0, "PE": 5.0},
            belief_entropies={},
            sensor_failure_count=0,
        )
        assert len(violations) == 0
        assert action == EnforcerAction.CONTINUE

    def test_memory_violation(self, enforcer: RBTAEnforcer):
        """Memory > B_mem → 1 violation."""
        violations, action = enforcer.check_cycle(
            runtime_log={},
            memory_log={"WM": 60_000},  # B_mem = 50_000
            energy_log={},
            belief_entropies={},
            sensor_failure_count=0,
        )
        assert len(violations) == 1
        assert violations[0].bound_type == "MEM"
        assert violations[0].module_id == "WM"
        assert action == EnforcerAction.INTERRUPT

    def test_entropy_floor_violation(self, enforcer: RBTAEnforcer):
        """H(beliefs) < entropy_floor → 1 violation."""
        violations, _ = enforcer.check_cycle(
            runtime_log={},
            memory_log={},
            energy_log={},
            belief_entropies={"G'": 0.001},  # below floor of 0.01
            sensor_failure_count=0,
        )
        assert len(violations) == 1
        assert violations[0].bound_type == "ENTROPY"
        assert violations[0].module_id == "G'"
        assert violations[0].measured == 0.001
        assert violations[0].allowed == 0.01

    def test_sensor_failure_limit(self, enforcer: RBTAEnforcer):
        """sensor_failure > asi_failure_limit → SENSOR_FAILURE violation.

        Default limit is 5; 10 failures exceeds it.
        Sensor failure is 1 violation → action = INTERRUPT (1..=2 range).
        """
        violations, action = enforcer.check_cycle(
            runtime_log={},
            memory_log={},
            energy_log={},
            belief_entropies={},
            sensor_failure_count=10,
            asi_failure_limit=5,
        )
        sensor_violations = [v for v in violations if v.bound_type == "SENSOR"]
        assert len(sensor_violations) == 1
        assert sensor_violations[0].module_id == "ASI"
        assert sensor_violations[0].measured == 10.0
        assert sensor_violations[0].allowed == 5.0
        assert action == EnforcerAction.INTERRUPT

    def test_multiple_violations_triggers_terminate(self, enforcer: RBTAEnforcer):
        """3+ violations → action = TERMINATE."""
        # Make 3 modules exceed their time bounds
        violations, action = enforcer.check_cycle(
            runtime_log={
                "ASI": 0.010,   # exceeds 0.002
                "WM": 0.010,    # exceeds 0.005
                "G'": 0.100,    # exceeds 0.020
                "PE": 0.050,    # exceeds 0.025
            },
            memory_log={},
            energy_log={},
            belief_entropies={},
            sensor_failure_count=0,
        )
        assert len(violations) >= 3
        assert action == EnforcerAction.TERMINATE

    def test_missing_module_id_no_crash(self, enforcer: RBTAEnforcer):
        """Module not in logs → treated as 0.0, no crash."""
        # Logs reference a module that's not in bounds
        violations, action = enforcer.check_cycle(
            runtime_log={"UNKNOWN_MODULE": 999.0},
            memory_log={},
            energy_log={},
            belief_entropies={},
            sensor_failure_count=0,
        )
        # Unknown module is not in self._bounds, so it's ignored
        assert len(violations) == 0
        assert action == EnforcerAction.CONTINUE

    def test_energy_violation(self, enforcer: RBTAEnforcer):
        """Energy > B_energy → 1 violation."""
        violations, action = enforcer.check_cycle(
            runtime_log={},
            memory_log={},
            energy_log={"G'": 100.0},  # B_energy = 50.0
            belief_entropies={},
            sensor_failure_count=0,
        )
        assert len(violations) == 1
        assert violations[0].bound_type == "ENERGY"
        assert violations[0].module_id == "G'"
        assert violations[0].measured == 100.0
        assert violations[0].allowed == 50.0
        assert action == EnforcerAction.INTERRUPT

    def test_missing_log_entry_no_crash(self, enforcer: RBTAEnforcer):
        """Module in bounds but not in a specific log → treated as 0.0 for that resource."""
        # G' is in bounds but only appears in runtime_log, not memory_log
        violations, action = enforcer.check_cycle(
            runtime_log={"G'": 0.010},  # within B_time = 0.020
            memory_log={},               # G' missing → 0.0 (no violation)
            energy_log={},
            belief_entropies={},
            sensor_failure_count=0,
        )
        # G' has runtime=0.010 < 0.020, memory=0 < 500_000 → no violations
        assert len(violations) == 0
        assert action == EnforcerAction.CONTINUE

    def test_asi_failure_limit_setter(self):
        """asi_failure_limit setter validates positive values."""
        enforcer = RBTAEnforcer(DEFAULT_BOUNDS)
        assert enforcer.asi_failure_limit == 5  # default

        enforcer.asi_failure_limit = 10
        assert enforcer.asi_failure_limit == 10

        with pytest.raises(AssertionError):
            enforcer.asi_failure_limit = 0

        with pytest.raises(AssertionError):
            enforcer.asi_failure_limit = -1

    def test_update_bounds(self, enforcer: RBTAEnforcer):
        """update_bounds changes bounds for an existing or new module."""
        enforcer.update_bounds("ASI", ResourceBounds(B_time=0.100, B_mem=200_000, B_energy=20.0))
        enforcer.update_bounds("NEW_MOD", ResourceBounds(B_time=0.050, B_mem=100_000, B_energy=10.0))

        # ASI now has higher tolerance
        violations, action = enforcer.check_cycle(
            runtime_log={"ASI": 0.050},  # old B_time=0.002 would fail; new B_time=0.100 passes
            memory_log={},
            energy_log={},
            belief_entropies={},
            sensor_failure_count=0,
        )
        assert len(violations) == 0
        assert action == EnforcerAction.CONTINUE

        # New module enforces its bounds
        assert enforcer.get_bounds("NEW_MOD") is not None
        assert enforcer.get_bounds("NONEXISTENT") is None
