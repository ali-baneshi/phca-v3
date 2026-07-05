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
