"""Tests for forgetting-rate metrics and Level-4-lite integration."""

from __future__ import annotations

import pytest
import numpy as np

from phca.core.cycle import CognitiveCycle, CycleMetrics
from phca.evaluation.continual.gridworld_tasks import build_task_sequence
from phca.evaluation.metrics.forgetting import (
    delta_perf,
    delta_perf_valid_only,
    eval_window_accuracy,
    forgetting_rate,
    max_rolling_task_accuracy,
    passes_forgetting_gate,
    task_accuracy,
)


class TestForgettingMetrics:
    def test_delta_perf_negative_on_drop(self):
        baseline = {0: 0.8, 1: 0.6}
        current = {0: 0.7, 1: 0.5}
        delta = delta_perf(baseline, current)
        assert delta[0] == pytest.approx(-0.125)
        assert delta[1] == pytest.approx(-1.0 / 6.0)

    def test_forgetting_rate_max_drop_only(self):
        delta = {0: -0.03, 1: -0.08, 2: 0.5}
        assert forgetting_rate(delta) == pytest.approx(0.08)

    def test_forgetting_rate_ignores_improvement(self):
        delta = {0: -1.0, 1: 1.0}
        assert forgetting_rate(delta) == pytest.approx(1.0)

    def test_passes_gate_threshold(self):
        assert passes_forgetting_gate({0: -0.04, 1: -0.03})
        assert not passes_forgetting_gate({0: -0.04, 1: -0.06})

    def test_max_rolling_beats_last_window(self):
        hist = {
            0: [
                CycleMetrics(goal_reached=False),
                CycleMetrics(goal_reached=True),
                CycleMetrics(goal_reached=True),
                CycleMetrics(goal_reached=False),
                CycleMetrics(goal_reached=False),
            ]
        }
        last = task_accuracy(hist, task_id=0, window=2)
        best = max_rolling_task_accuracy(hist, task_id=0, window=2)
        assert best >= last

    def test_delta_perf_valid_only_excludes_low_baselines(self):
        baseline = {0: 0.8, 1: 0.0}
        current = {0: 0.7, 1: 0.5}
        delta, excluded = delta_perf_valid_only(baseline, current)
        assert 1 in excluded
        assert 0 in delta
        assert 1 not in delta

    def test_task_accuracy_goal_rate(self):
        hist = {
            0: [
                CycleMetrics(goal_reached=True),
                CycleMetrics(goal_reached=False),
                CycleMetrics(goal_reached=True),
            ]
        }
        assert task_accuracy(hist, task_id=0) == pytest.approx(2.0 / 3.0)

    def test_eval_window_accuracy(self):
        hist = {
            1: [CycleMetrics(goal_reached=i % 2 == 0) for i in range(10)]
        }
        acc = eval_window_accuracy(hist, task_id=1, eval_cycles=5)
        assert 0.0 <= acc <= 1.0


class TestGridWorldTasks:
    def test_build_task_sequence_size(self):
        tasks = build_task_sequence(n_tasks=10, grid_size=5, base_seed=42)
        assert len(tasks) == 10
        assert all(t.task_id == i for i, t in enumerate(tasks))


class TestMitigationHooks:
    def test_on_task_boundary_protects_tspl(self):
        cycle = CognitiveCycle.build_for_env(size=5, seed=42, use_mlp=True)
        cycle.on_task_boundary(3)
        assert cycle._current_task_id == 3
        assert cycle._forgetting_mitigation_active
        assert cycle.tspl.theta_protected.keys() == cycle.tspl.theta.keys()
        for key in cycle.tspl.theta:
            assert np.array_equal(cycle.tspl.theta_protected[key], cycle.tspl.theta[key])

    def test_on_forgetting_detected_halves_alpha(self):
        cycle = CognitiveCycle.build_for_env(size=5, seed=42, use_mlp=True)
        from phca.config import StreamID
        alpha_before = cycle.tspl.configs[StreamID.P_STREAM].alpha
        cycle.on_forgetting_detected()
        assert cycle.tspl.configs[StreamID.P_STREAM].alpha == pytest.approx(alpha_before * 0.5)
        assert cycle.gprime.replay_boost

    def test_m3_replay_prior_tasks_after_boundary(self):
        cycle = CognitiveCycle.build_for_env(size=5, seed=42, use_mlp=True)
        cycle.on_task_boundary(0)
        for _ in range(30):
            cycle.step()
        before = cycle._m3_replay_total
        cycle.on_task_boundary(1)
        for _ in range(30):
            cycle.step()
        assert cycle._m3_replay_total > before


@pytest.mark.slow
class TestLevel4LiteIntegration:
    def test_two_task_sequence_runs(self):
        from scripts import benchmark_level4

        report = benchmark_level4.run_level4_benchmark(
            n_tasks=2,
            task_cycles=20,
            eval_cycles=5,
            seeds=1,
            use_mlp=True,
            grid_size=5,
            mitigation=True,
        )
        assert "forgetting_rate" in report
        assert "passes_gate" in report
        assert len(report["per_task_accuracy"]) == 2
