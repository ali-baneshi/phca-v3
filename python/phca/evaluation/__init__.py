"""Evaluation layer: interventions, traces, statistics, baselines.

Benchmark harnesses that produced Φ-IQ reports were moved to ``attic/`` in
favour of the ``phca bench`` command.
"""

from phca.evaluation.interventions import InterventionConfig
from phca.evaluation.metrics.statistics import (
    aggregate_runs,
    bootstrap_ci,
    compare_groups,
    seed_sequence,
)
from phca.evaluation.trace import (
    CycleTraceRecord,
    TraceCollector,
    summarize_action_selection,
)

__all__ = [
    "CycleTraceRecord",
    "InterventionConfig",
    "TraceCollector",
    "aggregate_runs",
    "bootstrap_ci",
    "compare_groups",
    "seed_sequence",
    "summarize_action_selection",
]
