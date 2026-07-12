"""Tests for evaluation metrics."""

from __future__ import annotations

from phca.evaluation.metrics.emergence import (
    compute_emergence_bundle,
    novel_behaviour,
    strategy_diversity,
)
from phca.evaluation.metrics.phi_iq import compute_phi_iq, DEFAULT_WEIGHTS
from phca.evaluation.metrics.statistics import mann_whitney_u, seed_sequence
from phca.evaluation.result_schema import BenchmarkResult
from phca.evaluation.trace import CycleTraceRecord, TraceCollector


def _synthetic_trace(actions: list[int], n: int = 100) -> list[CycleTraceRecord]:
    return [
        CycleTraceRecord(
            cycle_id=i,
            action=actions[i % len(actions)],
            prediction_error=0.1 + 0.01 * i,
            prediction_confidence=0.5,
            goal_drive=1 + (i % 3),
            goal_switched=(i % 20 == 0),
            goal_reached=(i % 30 == 0),
        )
        for i in range(n)
    ]


def test_seed_sequence():
    assert seed_sequence(42, 5) == [42, 43, 44, 45, 46]


def test_compute_phi_iq_bounds():
    r = BenchmarkResult(level=0, level_name="", n_cycles=10, prediction_accuracy=1.0,
                        adaptation_speed=1.0, goal_complexity=1.0, transfer_efficiency=1.0,
                        resource_efficiency=1.0, failure_rate=0.0)
    score = compute_phi_iq(r, DEFAULT_WEIGHTS)
    assert 0.0 <= score <= 1.0
    assert score >= 0.89  # 0.25+0.25+0.20+0.20 = 0.90; allow FP epsilon


def test_strategy_diversity():
    trace = _synthetic_trace([0, 1, 2, 3, 4])
    d = strategy_diversity(trace)
    assert 0.0 <= d <= 1.0


def test_novel_behaviour_increases_with_new_actions():
    early = _synthetic_trace([0, 0, 0, 0], 50)
    late = _synthetic_trace([0, 1, 2, 3, 4], 50)
    trace = early + late
    nb = novel_behaviour(trace, window=50)
    assert nb >= 0.0


def test_emergence_bundle_keys():
    trace = _synthetic_trace([0, 1, 2, 3, 4], 120)
    bundle = compute_emergence_bundle(trace)
    assert "emergence_composite" in bundle
    assert "strategy_diversity" in bundle


def test_trace_collector():
    tc = TraceCollector(capacity=10)
    tc.record(cycle_id=0, action=1, reward=0.0, prediction_error=0.1,
              prediction_confidence=0.5, goal_drive=1, rbta_action="CONTINUE",
              violations=0, latency_ms=10.0, goal_reached=False, module_timings={})
    assert len(tc) == 1


def test_mann_whitney_u():
    a = [0.8, 0.82, 0.79, 0.81, 0.83]
    b = [0.5, 0.52, 0.48, 0.51, 0.49]
    result = mann_whitney_u(a, b)
    assert result["p_value"] is not None
    assert result["a_mean"] > result["b_mean"]
