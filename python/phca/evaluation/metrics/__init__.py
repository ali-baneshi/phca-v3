"""Pure metric functions over cycle traces.

Φ-IQ, emergence, interaction, synergy, and forgetting metrics moved to
``attic/`` with the benchmark harnesses that consumed them.
"""

from phca.evaluation.metrics.statistics import (
    aggregate_runs,
    bootstrap_ci,
    compare_groups,
    mann_whitney_u,
    seed_sequence,
)

__all__ = [
    "aggregate_runs",
    "bootstrap_ci",
    "compare_groups",
    "mann_whitney_u",
    "seed_sequence",
]
