"""Integration tests: CognitiveCycle with MuJoCo physics environments.

Verifies that the full PHCA cognitive cycle can drive a MuJoCo
environment without errors, that dimensions are consistent, and
that the cycle produces meaningful metrics.

Cross-ref: docs/mujoco_integration_plan.md §5.2
"""

import numpy as np
import pytest

from phca.core.cycle import CognitiveCycle


def test_build_for_mujoco_cartpole():
    """build_for_mujoco creates a correctly-configured cycle for Cartpole."""
    cycle = CognitiveCycle.build_for_mujoco(
        "InvertedPendulum-v5", seed=42, use_mlp=True,
    )
    assert cycle.env.action_space_size == 3
    assert cycle.state_dim == 4
    assert cycle.env.get_state_dim() == 4
    assert cycle.env.env_name == "InvertedPendulum-v5"


def test_build_for_mujoco_pendulum():
    """build_for_mujoco creates a correctly-configured cycle for Pendulum."""
    cycle = CognitiveCycle.build_for_mujoco(
        "Pendulum-v1", seed=42, use_mlp=True,
    )
    assert cycle.env.action_space_size == 3
    assert cycle.state_dim == 3


def test_single_cycle_cartpole():
    """A single cognitive cycle completes without errors."""
    cycle = CognitiveCycle.build_for_mujoco(
        "InvertedPendulum-v5", seed=42, use_mlp=True,
    )
    metrics = cycle.step()
    assert metrics.cycle_id == 0
    assert metrics.latency_ms > 0
    assert 0 <= metrics.action_taken < 3
    assert metrics.prediction_error >= 0
    assert np.isfinite(metrics.prediction_error)
    assert metrics.action_name in ("PUSH_LEFT", "STAY", "PUSH_RIGHT")


def test_multi_cycle_cartpole():
    """Run 10 cycles and verify all produce valid metrics."""
    cycle = CognitiveCycle.build_for_mujoco(
        "InvertedPendulum-v5", seed=42, use_mlp=True,
    )
    for i in range(10):
        metrics = cycle.step()
        assert metrics.cycle_id == i
        assert metrics.latency_ms > 0, f"Cycle {i}: zero latency"
        assert metrics.latency_ms < 5000, f"Cycle {i}: latency too high"
        assert np.isfinite(metrics.prediction_error), f"Cycle {i}: non-finite error"
        assert 0 <= metrics.action_taken < 3, f"Cycle {i}: invalid action"


def test_multi_cycle_pendulum():
    """Run 10 Pendulum cycles and verify all produce valid metrics."""
    cycle = CognitiveCycle.build_for_mujoco(
        "Pendulum-v1", seed=42, use_mlp=True,
    )
    for i in range(10):
        metrics = cycle.step()
        assert metrics.cycle_id == i
        assert metrics.latency_ms > 0
        assert np.isfinite(metrics.prediction_error)


def test_run_method_cartpole():
    """cycle.run(n_cycles) completes and returns summary stats."""
    cycle = CognitiveCycle.build_for_mujoco(
        "InvertedPendulum-v5", seed=42, use_mlp=True,
    )
    summary = cycle.run(n_cycles=20)
    assert summary["total_cycles"] == 20
    assert summary["avg_latency_ms"] > 0
    assert summary["avg_prediction_error"] >= 0
    assert np.isfinite(summary["avg_prediction_error"])
    assert len(summary["actions_taken"]) == 20


def test_rbta_no_excessive_violations():
    """Cycle should not trigger excessive RBTA violations."""
    cycle = CognitiveCycle.build_for_mujoco(
        "InvertedPendulum-v5", seed=42, use_mlp=True,
    )
    total_violations = 0
    for _ in range(20):
        metrics = cycle.step()
        total_violations += metrics.violations_count
    # Allow some violations, but not excessive
    assert total_violations < 10, f"Too many RBTA violations: {total_violations}"


def test_goal_position_fallback():
    """_compute_distance_gain falls back to 0.5 for non-grid envs."""
    cycle = CognitiveCycle.build_for_mujoco(
        "InvertedPendulum-v5", seed=42, use_mlp=True,
    )
    # Trigger state initialisation via step
    cycle.step()
    # All actions should get neutral distance gain
    for action_idx in range(cycle.env.action_space_size):
        gain = cycle._compute_distance_gain(action_idx)
        assert gain == 0.5, f"Action {action_idx}: expected 0.5, got {gain}"


def test_single_cycle_continuous_gprime():
    """Cognitive cycle with continuous Gaussian G' completes."""
    cycle = CognitiveCycle.build_for_mujoco(
        "InvertedPendulum-v5", seed=42, use_mlp=False, use_continuous=True,
    )
    metrics = cycle.step()
    assert metrics.cycle_id == 0
    assert metrics.latency_ms > 0
    assert np.isfinite(metrics.prediction_error)


def test_episode_reset_handling():
    """Cycle handles terminal states and resets correctly."""
    cycle = CognitiveCycle.build_for_mujoco(
        "InvertedPendulum-v5", seed=42, use_mlp=True,
    )
    # Force a terminal state by running many steps
    # Cartpole typically falls within ~100 steps with random actions
    for _ in range(200):
        metrics = cycle.step()
        # The cycle should not crash even after terminal → reset
        assert np.isfinite(metrics.prediction_error)
