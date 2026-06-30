"""Three-Stream Predictive Learning — P-Stream, E-Stream, S-Stream with EWC/GEM/skill compilation.

Phase 3.1:
    - P-Stream active with unified learning rule (α=0.05, λ=0.01, η=0.1).
    - E-Stream and S-Stream are stubs (deferred to Phase 3.2+).
    - Skill compilation at ≥95% accuracy.

Phase 3.2+:
    - E-Stream with GEM projection.
    - S-Stream with EWC penalty.
    - Multi-stream consolidation scheduling via CONSOL module.

v3.0 Reference: §3.1 Definitions 3.2, 3.3.3
"""

from phca.learning.tspl import TSPL, StreamConfig, DEFAULT_STREAM_CONFIGS
from phca.learning.skill_compilation import SkillLibrary

__all__ = [
    "TSPL",
    "StreamConfig",
    "DEFAULT_STREAM_CONFIGS",
    "SkillLibrary",
]
