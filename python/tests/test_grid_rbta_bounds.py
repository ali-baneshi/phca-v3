"""Tests for grid-scaled RBTA bounds on large GridWorld configs."""

from __future__ import annotations

from phca.config import DEFAULT_MODULE_BOUNDS
from phca.core.cycle import CognitiveCycle
from phca.world_model.mlp import (
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
  expected = estimate_mlp_gprime_time_bound(cycle.state_dim, base)
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
