"""Tests for intervention config and cycle integration."""

from __future__ import annotations

from phca.core.cycle import CognitiveCycle
from phca.evaluation.interventions import InterventionConfig
from phca.evaluation.trace import TraceCollector


def test_no_prediction_intervention():
    trace = TraceCollector()
    cycle = CognitiveCycle.build_for_env(
        size=5, seed=42, use_mlp=False, use_continuous=True,
        interventions=InterventionConfig.no_prediction(),
        trace_collector=trace,
    )
    for _ in range(20):
        cycle.step()
    assert len(trace) == 20


def test_minimal_cycle_intervention():
    cycle = CognitiveCycle.build_for_env(
        size=5, seed=42,
        interventions=InterventionConfig.minimal(),
    )
    m = cycle.step()
    assert m.cycle_id == 0
    assert cycle.current_goal.drive_id == 1


def test_causal_fairness_disables_heuristics():
    cfg = InterventionConfig.from_ablation_dict({"enable_prediction": False}, causal_fair=True)
    assert cfg.disable_task_lock is True
    assert cfg.disable_planning_grid is True
    cycle = CognitiveCycle.build_for_env(
        size=5, seed=42, use_mlp=False,
        interventions=cfg,
    )
    assert cycle._task_lock is False
    assert cycle._planning_grid is None


def test_bandit_runner_smoke():
    from phca.evaluation.runner import run_benchmark_level
    from phca.evaluation.result_schema import BenchmarkConfig

    config = BenchmarkConfig(n_cycles=20, use_mlp=False, environment="bandit")
    summary = run_benchmark_level(0, config, seed=42)
    assert "phi_iq" in summary.metrics
