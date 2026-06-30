"""PHCA v3.0 — Consolidation Scheduler.

Phase 3.2: Periodic E→S transfer (episodic to statistical consolidation)
using MVCC snapshot isolation. Runs as a sleep-cycle analogue.

Note: "Statistical" not "Semantic" — facts are extracted via frequency-based
pattern matching, not genuine semantic understanding. See G-007.

v3.0 Reference: v3.0 Patch §2.3 (Memory concurrency), §3.1 Table
"""

from phca.consolidation.scheduler import ConsolidationScheduler

__all__ = [
    "ConsolidationScheduler",
]
