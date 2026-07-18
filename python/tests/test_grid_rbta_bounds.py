"""Tests for grid-scaled RBTA bounds on large GridWorld configs."""

from __future__ import annotations

import pytest

from phca.config import DEFAULT_MODULE_BOUNDS
from phca.core.cycle import CognitiveCycle
from phca.evaluation.metrics.phi_iq import generate_goal_pursuit_obstacles
from phca.world_model.mlp import (
    apply_grid_rbta_bounds,
    estimate_gaussian_gprime_memory_bytes,
    estimate_mlp_gprime_time_bound,
    grid_scale,
    grid_rbta_bounds,
)


def test_gaussian_10x10_gprime_mem_bound_covers_estimate():
  cycle = CognitiveCycle.build_for_env(
      size=10, seed=42, use_continuous=True, use_mlp=False,
  )
  expected_mem = estimate_gaussian_gprime_memory_bytes(cycle.state_dim)
  g_bounds = cycle.rbta._bounds["G'"]
  assert g_bounds.B_mem >= expected_mem
  assert grid_scale(cycle.state_dim) > 1.0


def test_mlp_10x10_action_bound_scaled():
  cycle = CognitiveCycle.build_for_env(size=10, seed=42, use_mlp=True)
  action_bounds = cycle.rbta._bounds["ACTION"]
  base = DEFAULT_MODULE_BOUNDS["ACTION"].B_time
  expected = estimate_mlp_gprime_time_bound(cycle.state_dim, base) * 20.0
  assert action_bounds.B_time == expected


def test_grid_rbta_reduces_violations_vs_unscaled_mem():
  """Scaled G' mem bound should produce fewer violations than fixed 500 KB cap."""
  unscaled = CognitiveCycle.build_for_env(
      size=10, seed=42, use_continuous=True, use_mlp=False,
  )
  from phca.config import ResourceBounds
  unscaled.rbta.update_bounds(
      "G'", ResourceBounds(B_time=0.020, B_mem=500_000, B_energy=50.0),
  )
  v_bad = sum(unscaled.step().violations_count for _ in range(15))

  scaled = CognitiveCycle.build_for_env(
      size=10, seed=42, use_continuous=True, use_mlp=False,
  )
  v_good = sum(scaled.step().violations_count for _ in range(15))
  assert v_good < v_bad


def test_5x5_gaussian_no_extra_bounds():
  cycle = CognitiveCycle.build_for_env(
      size=5, seed=42, use_continuous=True, use_mlp=False,
  )
  assert grid_rbta_bounds(cycle=cycle) == {}


@pytest.mark.slow
def test_l2_10x10_gaussian_violation_rate_under_15pct():
    """L2 Goal Pursuit on 10×10 Gaussian G': RBTA violation gate at 15% after D-152 recalibration.

    ACTION time has high stochastic variance (0.02–2.0s depending on goal-pursuit
    complexity); the 15% gate accommodates normal spikes while still catching
    systemic bound regression.
    """
    level = 2
    seed = 44
    obstacles = generate_goal_pursuit_obstacles(seed + level, 10)
    cycle = CognitiveCycle.build_for_env(
        size=10,
        seed=seed + level,
        use_continuous=True,
        use_mlp=False,
        obstacles=obstacles,
    )
    apply_grid_rbta_bounds(cycle, b_time=0.020)

    warmup = 10
    n_cycles = 80
    for _ in range(warmup):
        cycle.step()

    violations = sum(cycle.step().violations_count for _ in range(n_cycles))
    assert violations / n_cycles < 0.15


@pytest.mark.slow
def test_action_latency_spread_within_20x():
    """D-154: ACTION p99 latency must not exceed 20× p50 (variance regression gate).

    The Round 5 review flagged that ACTION latency has a 100× spread between
    typical (0.02s) and worst-case (2.0s) under goal pursuit. This test asserts
    the spread stays within a 20× bound, independent of the absolute RBTA bound.
    """
    level = 2
    seed = 44
    obstacles = generate_goal_pursuit_obstacles(seed + level, 10)
    cycle = CognitiveCycle.build_for_env(
        size=10,
        seed=seed + level,
        use_continuous=True,
        use_mlp=False,
        obstacles=obstacles,
    )
    apply_grid_rbta_bounds(cycle, b_time=0.020)

    warmup = 10
    n_cycles = 80
    for _ in range(warmup):
        cycle.step()

    action_times_ms = []
    for _ in range(n_cycles):
        m = cycle.step()
        t = m.module_timings.get("action_selection", 0.0)
        action_times_ms.append(t)

    action_times_s = sorted([t / 1000.0 for t in action_times_ms])
    n = len(action_times_s)
    p50 = action_times_s[n // 2]
    p99 = action_times_s[int(n * 0.99)]
    ratio = p99 / max(p50, 1e-9)
    assert ratio < 20.0, (
        f"ACTION latency p99/p50 ratio {ratio:.1f}× exceeds 20× "
        f"(p50={p50:.4f}s, p99={p99:.4f}s, max={max(action_times_s):.4f}s)"
    )
