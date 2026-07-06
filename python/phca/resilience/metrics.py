"""Recovery-rate metrics for cognitive resilience benchmarks."""

from __future__ import annotations

from typing import Dict, Sequence


def recovery_rate(
    events: Sequence[str],
    outcomes: Dict[str, bool],
    window: int = 10,
) -> float:
    """Fraction of detectable failure scenarios mitigated within ``window`` cycles.

    Args:
        events: Scenario or mode identifiers that were injected/detected.
        outcomes: Map scenario_id → True if primary KPI recovered in time.
        window: Maximum cycles allowed for mitigation (informational; outcomes
            should already reflect the window constraint).

    Returns:
        Fraction in [0, 1] of events with a True outcome.
    """
    if not events:
        return 1.0
    mitigated = sum(1 for e in events if outcomes.get(e, False))
    return mitigated / len(events)
