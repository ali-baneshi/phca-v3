"""
Tests for the RBTA Constraint Enforcer (PHCA-3.1-006).

Cross-ref: v3.0 §2.1 Definition 2.2, Definition 2.3, Theorem 2.1, Theorem 3.1
"""

from __future__ import annotations

import pytest

from phca.config import ResourceBounds
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
        """H(beliefs) < entropy_floor → 1 violation → INTERRUPT."""
        violations, action = enforcer.check_cycle(
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
        assert action == EnforcerAction.INTERRUPT, (
            f"ENTROPY floor violation should trigger INTERRUPT, got {action}"
        )

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
        assert "NEW_MOD" in enforcer._bounds
        assert "NONEXISTENT" not in enforcer._bounds


class TestCompositionTree:
    """Tests for HPM composition tree checking (v3.0 Theorem 2.1 / Theorem 3.1).

    Implements:
        SEQUENCE:  B_time = sum(child_time) + τ_comp (τ_comp = 1ms)
        PARALLEL:  B_time = max(child_time) + τ_sync (τ_sync = 2ms)
    """

    TAU_COMP = 0.001
    TAU_SYNC = 0.002

    def test_sequence_time_computation(self, enforcer: RBTAEnforcer):
        """SEQUENCE: total = sum(child_times) + τ_comp."""
        tree = {
            "type": "SEQUENCE", "id": "pipeline",
            "children": ["ASI", "G'", "PE"],
            "bounds": {"B_time": 0.100},
        }
        runtime_log = {"ASI": 0.002, "G'": 0.015, "PE": 0.025}
        expected = (0.002 + 0.015 + 0.025) + self.TAU_COMP
        actual = enforcer._compute_subtree_runtime(tree, runtime_log)
        assert abs(actual - expected) < 1e-6, f"Expected {expected}, got {actual}"

    def test_parallel_time_computation(self, enforcer: RBTAEnforcer):
        """PARALLEL: total = max(child_times) + τ_sync."""
        tree = {
            "type": "PARALLEL", "id": "branch",
            "children": ["ASI", "WM", "G'"],
            "bounds": {"B_time": 0.050},
        }
        runtime_log = {"ASI": 0.002, "WM": 0.004, "G'": 0.015}
        expected = max(0.002, 0.004, 0.015) + self.TAU_SYNC
        actual = enforcer._compute_subtree_runtime(tree, runtime_log)
        assert abs(actual - expected) < 1e-6, f"Expected {expected}, got {actual}"

    def test_nested_sequence_in_parallel(self, enforcer: RBTAEnforcer):
        """Nested tree: PARALLEL with one child being a SEQUENCE subtree."""
        tree = {
            "type": "PARALLEL", "id": "root",
            "children": [
                "ASI",
                {
                    "type": "SEQUENCE", "id": "subpipe",
                    "children": ["G'", "PE"],
                    "bounds": {"B_time": 0.100},
                },
            ],
            "bounds": {"B_time": 0.100},
        }
        runtime_log = {"ASI": 0.002, "G'": 0.015, "PE": 0.025}
        sub_time = (0.015 + 0.025) + self.TAU_COMP
        expected = max(0.002, sub_time) + self.TAU_SYNC
        actual = enforcer._compute_subtree_runtime(tree, runtime_log)
        assert abs(actual - expected) < 1e-6, f"Expected {expected}, got {actual}"

    def test_nested_parallel_in_sequence(self, enforcer: RBTAEnforcer):
        """Nested tree: SEQUENCE with one child being a PARALLEL subtree."""
        tree = {
            "type": "SEQUENCE", "id": "root",
            "children": [
                "ASI",
                {
                    "type": "PARALLEL", "id": "subbranch",
                    "children": ["G'", "WM"],
                    "bounds": {"B_time": 0.050},
                },
                "PE",
            ],
            "bounds": {"B_time": 0.200},
        }
        runtime_log = {"ASI": 0.002, "G'": 0.015, "WM": 0.003, "PE": 0.025}
        sub_time = max(0.015, 0.003) + self.TAU_SYNC
        expected = (0.002 + sub_time + 0.025) + self.TAU_COMP
        actual = enforcer._compute_subtree_runtime(tree, runtime_log)
        assert abs(actual - expected) < 1e-6, f"Expected {expected}, got {actual}"

    def test_sequence_violation_detected(self, enforcer: RBTAEnforcer):
        """SEQUENCE time > bound → composite violation → INTERRUPT."""
        tree = {
            "type": "SEQUENCE", "id": "tight_pipe",
            "children": ["ASI", "G'", "PE"],
            "bounds": {"B_time": 0.020},  # too tight
        }
        runtime_log = {"ASI": 0.010, "G'": 0.010, "PE": 0.010}
        violations, action = enforcer.check_cycle(
            runtime_log=runtime_log,
            memory_log={}, energy_log={},
            belief_entropies={}, sensor_failure_count=0,
            composition_tree=tree,
        )
        composite_violations = [v for v in violations if v.module_id.startswith("composite:")]
        assert len(composite_violations) == 1
        assert composite_violations[0].bound_type == "TIME"
        expected = (0.010 + 0.010 + 0.010) + self.TAU_COMP
        assert abs(composite_violations[0].measured - expected) < 1e-6
        assert action == EnforcerAction.INTERRUPT

    def test_parallel_within_bound(self, enforcer: RBTAEnforcer):
        """PARALLEL time ≤ bound → no composite violation → CONTINUE."""
        tree = {
            "type": "PARALLEL", "id": "fast_branch",
            "children": ["ASI", "WM"],
            "bounds": {"B_time": 0.010},
        }
        runtime_log = {"ASI": 0.002, "WM": 0.003}
        violations, action = enforcer.check_cycle(
            runtime_log=runtime_log,
            memory_log={}, energy_log={},
            belief_entropies={}, sensor_failure_count=0,
            composition_tree=tree,
        )
        composite_violations = [v for v in violations if v.module_id.startswith("composite:")]
        assert len(composite_violations) == 0
        assert action == EnforcerAction.CONTINUE

    def test_empty_children_no_crash(self, enforcer: RBTAEnforcer):
        """Empty children list → no crash, no violation → CONTINUE."""
        tree = {"type": "SEQUENCE", "children": [], "id": "empty"}
        runtime_log = {"ASI": 0.010}
        violations, action = enforcer.check_cycle(
            runtime_log=runtime_log,
            memory_log={}, energy_log={},
            belief_entropies={}, sensor_failure_count=0,
            composition_tree=tree,
        )
        composite_violations = [v for v in violations if v.module_id.startswith("composite:")]
        assert len(composite_violations) == 0
        # ASI runtime=0.010 > B_time=0.005 → 1 TIME violation → INTERRUPT
        assert action == EnforcerAction.INTERRUPT

    def test_unknown_operator_skipped(self, enforcer: RBTAEnforcer):
        """Unknown composition type → skipped, no composite violation."""
        tree = {
            "type": "CUSTOM_FORK", "id": "custom",
            "children": ["ASI", "G'"],
            "bounds": {"B_time": 0.005},
        }
        runtime_log = {"ASI": 0.010, "G'": 0.020}
        violations, action = enforcer.check_cycle(
            runtime_log=runtime_log,
            memory_log={}, energy_log={},
            belief_entropies={}, sensor_failure_count=0,
            composition_tree=tree,
        )
        composite_violations = [v for v in violations if v.module_id.startswith("composite:")]
        assert len(composite_violations) == 0
        # ASI=0.010 > 0.005, G'=0.020 == 0.020 → 1 TIME violation → INTERRUPT
        assert action == EnforcerAction.INTERRUPT

    def test_no_bounds_no_violation(self, enforcer: RBTAEnforcer):
        """No bounds field → no composite bound check (implicit ∞)."""
        tree = {
            "type": "SEQUENCE", "id": "nobounds",
            "children": ["ASI", "G'"],
        }
        runtime_log = {"ASI": 999.0, "G'": 999.0}
        violations, action = enforcer.check_cycle(
            runtime_log=runtime_log,
            memory_log={}, energy_log={},
            belief_entropies={}, sensor_failure_count=0,
            composition_tree=tree,
        )
        composite_violations = [v for v in violations if v.module_id.startswith("composite:")]
        assert len(composite_violations) == 0
        # ASI=999 >> 0.005, G'=999 >> 0.020 → 2 extreme violations → TERMINATE
        assert action == EnforcerAction.TERMINATE

    def test_three_level_deep_nesting(self, enforcer: RBTAEnforcer):
        """Three-level nesting: SEQUENCE → PARALLEL → SEQUENCE."""
        tree = {
            "type": "SEQUENCE", "id": "l1",
            "children": [
                "ASI",
                {
                    "type": "PARALLEL", "id": "l2",
                    "children": [
                        "WM",
                        {
                            "type": "SEQUENCE", "id": "l3",
                            "children": ["G'", "PE"],
                            "bounds": {"B_time": 0.100},
                        },
                    ],
                    "bounds": {"B_time": 0.080},
                },
            ],
            "bounds": {"B_time": 0.200},
        }
        runtime_log = {"ASI": 0.002, "WM": 0.003, "G'": 0.015, "PE": 0.025}
        l3_time = (0.015 + 0.025) + self.TAU_COMP
        l2_time = max(0.003, l3_time) + self.TAU_SYNC
        expected = (0.002 + l2_time) + self.TAU_COMP
        actual = enforcer._compute_subtree_runtime(tree, runtime_log)
        assert abs(actual - expected) < 1e-6, f"Expected {expected}, got {actual}"
