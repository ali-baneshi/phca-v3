"""Predictive Learning — P-Stream only (E-Stream and S-Stream removed in Phase 3.3).

Phase 3.3 (current):
    - P-Stream active with unified learning rule (α=0.08, λ=0.01, η=0.1).
    - Skill compilation at ≥95% accuracy.
    - Consolidation runs on fixed 10-cycle timer (not TSPL-mediated).

v3.0 Reference: §3.1 Definitions 3.2, 3.3.3
"""

from phca.learning.tspl import TSPL, StreamConfig, DEFAULT_STREAM_CONFIGS

__all__ = [
    "TSPL",
    "StreamConfig",
    "DEFAULT_STREAM_CONFIGS",
]
