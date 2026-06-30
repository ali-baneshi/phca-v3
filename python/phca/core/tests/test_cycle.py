"""Tests for PHCA-3.1-011: Cognitive Cycle Orchestrator."""

from __future__ import annotations

import time

import numpy as np
import pytest

from phca.config import StateVector, EnforcerAction
from phca.core.cycle import CognitiveCycle, CycleMetrics
from phca.regulation.rbta_enforcer import RBTAEnforcer

# Reduce G' inference test load — use small graphs
pytestmark = pytest.mark.timeout(30)


class TestCognitiveCycleBuild:
    """Tests for CognitiveCycle.build_for_env()."""

    def test_build_5x5_defaults(self):
        """Building for 5×5 grid should succeed with all components wired."""
        cycle = CognitiveCycle.build_for_env(size=5, seed=42)
        assert cycle.env.size == 5
        assert cycle.state_dim == 5 * 5 * 3 + 9  # 84
        assert cycle.sanitizer is not None
        assert cycle.gprime is not None
        assert cycle.engine is not None
        assert cycle.peu is not None
        assert cycle.tspl is not None
        assert cycle.rbta is not None
        assert cycle.m2 is not None
        assert cycle.mdim is not None
        assert cycle.attention is not None

    def test_build_5x5_cycle_count(self):
        """Build should initialize cycle_count = 0."""
        cycle = CognitiveCycle.build_for_env(size=5, seed=42)
        assert cycle.cycle_count == 0


class TestCognitiveCycleStep:
    """Tests for single cycle execution."""

    def test_single_step_returns_metrics(self):
        """A single step should return CycleMetrics."""
        cycle = CognitiveCycle.build_for_env(size=5, seed=42)
        metrics = cycle.step()
        assert isinstance(metrics, CycleMetrics)
        assert metrics.cycle_id == 0
        assert metrics.latency_ms >= 0
        assert metrics.action_taken in range(5)

    def test_single_step_increments_counter(self):
        """Step should increment cycle_count."""
        cycle = CognitiveCycle.build_for_env(size=5, seed=42)
        cycle.step()
        assert cycle.cycle_count == 1

    def test_single_step_populates_metrics(self):
        """Step metrics should have all fields populated."""
        cycle = CognitiveCycle.build_for_env(size=5, seed=42)
        metrics = cycle.step()
        assert metrics.prediction_error >= 0
        assert 0.0 <= metrics.prediction_confidence <= 1.0
        assert metrics.rbta_action in ("CONTINUE", "INTERRUPT", "TERMINATE")
        assert metrics.action_name in ("MOVE_N", "MOVE_S", "MOVE_E", "MOVE_W", "STAY")
        assert len(metrics.module_timings) > 0

    def test_two_steps_increment_properly(self):
        """Two steps should result in cycle_count=2."""
        cycle = CognitiveCycle.build_for_env(size=5, seed=42)
        cycle.step()
        cycle.step()
        assert cycle.cycle_count == 2
        assert len(cycle.metrics_history) == 2


class TestCognitiveCycleRun:
    """Tests for multi-cycle run()."""

    def test_run_10_cycles(self):
        """Run 10 cycles should complete with valid summary."""
        cycle = CognitiveCycle.build_for_env(size=5, seed=42)
        summary = cycle.run(n_cycles=10)
        assert summary["total_cycles"] == 10
        assert summary["avg_latency_ms"] > 0
        assert 0 <= summary["total_violations"]
        assert isinstance(summary["skill_compiled"], bool)

    def test_run_stores_all_metrics(self):
        """Run should store one metrics entry per cycle."""
        cycle = CognitiveCycle.build_for_env(size=5, seed=42)
        summary = cycle.run(n_cycles=5)
        assert len(summary["early_errors"]) == 10  # 5 cycles × 2 halves
        assert len(summary["late_errors"]) == 10
        assert len(summary["actions_taken"]) == 5

    def test_run_empty_returns_safe_defaults(self):
        """run(n_cycles=0) should return safe defaults."""
        cycle = CognitiveCycle.build_for_env(size=5, seed=42)
        summary = cycle.run(n_cycles=0)
        assert summary["total_cycles"] == 0
        assert summary["avg_latency_ms"] == 0.0


class TestCognitiveCycleActionSelection:
    """Tests for action selection (Step 9)."""

    def test_action_in_valid_range(self):
        """_select_action should return 0-4."""
        cycle = CognitiveCycle.build_for_env(size=5, seed=42)
        cycle.current_state = StateVector(
            values=np.ones(84, dtype=np.float32),
            precision=np.ones(84, dtype=np.float32),
        )
        action = cycle._select_action()
        assert 0 <= action <= 4

    def test_no_state_returns_stay(self):
        """_select_action with no current_state should return STAY (4)."""
        cycle = CognitiveCycle.build_for_env(size=5, seed=42)
        cycle.current_state = None
        action = cycle._select_action()
        assert action == 4


class TestCognitiveCycleCollectLogs:
    """Tests for runtime log collection."""

    def test_collect_runtime_log_populates_all(self):
        """_collect_runtime_log should populate all logs."""
        cycle = CognitiveCycle.build_for_env(size=5, seed=42)
        cycle._collect_runtime_log()
        assert "ASI" in cycle.runtime_log
        assert "WM" in cycle.runtime_log
        assert "G'" in cycle.runtime_log
        assert "CYCLE" in cycle.runtime_log
        assert len(cycle.runtime_log) >= 6

    def test_belief_entropies_decrease_with_cycles(self):
        """Belief entropies should decrease as cycle_count increases."""
        cycle = CognitiveCycle.build_for_env(size=5, seed=42)
        cycle._collect_runtime_log()
        e1 = cycle.belief_entropies["G'"]
        cycle.cycle_count = 100
        cycle._collect_runtime_log()
        e2 = cycle.belief_entropies["G'"]
        assert e2 <= e1
