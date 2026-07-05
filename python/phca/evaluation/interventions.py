"""Config-driven intervention points for causal ablations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


CANONICAL_STAGE_ORDER = [
    "sanitize",
    "memory_write",
    "prediction",
    "regulation",
    "rbta_preflight",
    "action",
    "feedback",
    "rbta_post",
    "consolidation",
]

MINIMAL_STAGE_ORDER = [
    "sanitize",
    "action",
    "feedback",
    "consolidation",
]

DESYNC_STAGE_ORDER = [
    "sanitize",
    "memory_write",
    "feedback",
    "prediction",
    "regulation",
    "rbta_preflight",
    "action",
    "rbta_post",
    "consolidation",
]


@dataclass
class InterventionConfig:
    """Single place to configure module enables and cycle variants."""

    enable_prediction: bool = True
    enable_mdim: bool = True
    enable_m3_write: bool = True
    enable_consolidation: bool = True
    enable_attention: bool = True
    enable_tspl: bool = True
    enable_apc: bool = True
    enable_m2: bool = True
    enable_gprime_learn: bool = True
    stage_order: Optional[List[str]] = None
    minimal_cycle: bool = False
    resource_policy: str = "time"  # time | energy | memory
    prediction_mode: str = "normal"  # normal | zero | scramble

    def effective_stage_order(self) -> List[str]:
        if self.minimal_cycle:
            return list(MINIMAL_STAGE_ORDER)
        if self.stage_order is not None:
            return list(self.stage_order)
        return list(CANONICAL_STAGE_ORDER)

    @classmethod
    def no_prediction(cls) -> "InterventionConfig":
        return cls(enable_prediction=False)

    @classmethod
    def no_drives(cls) -> "InterventionConfig":
        return cls(enable_mdim=False)

    @classmethod
    def no_memory(cls) -> "InterventionConfig":
        return cls(enable_m3_write=False, enable_consolidation=False)

    @classmethod
    def desynchronized(cls) -> "InterventionConfig":
        return cls(stage_order=list(DESYNC_STAGE_ORDER))

    @classmethod
    def minimal(cls) -> "InterventionConfig":
        return cls(minimal_cycle=True)

    @classmethod
    def from_ablation_dict(cls, d: dict) -> "InterventionConfig":
        """Build from experiment manifest ablation block."""
        return cls(
            enable_prediction=d.get("enable_prediction", True),
            enable_mdim=d.get("enable_mdim", True),
            enable_m3_write=d.get("enable_m3_write", True),
            enable_consolidation=d.get("enable_consolidation", True),
            enable_attention=d.get("enable_attention", True),
            enable_tspl=d.get("enable_tspl", True),
            enable_apc=d.get("enable_apc", True),
            enable_m2=d.get("enable_m2", True),
            enable_gprime_learn=d.get("enable_gprime_learn", True),
            minimal_cycle=d.get("minimal_cycle", False),
            stage_order=d.get("stage_order"),
            resource_policy=d.get("resource_policy", "time"),
            prediction_mode=d.get("prediction_mode", "normal"),
        )

    def label(self) -> str:
        parts = []
        if self.minimal_cycle:
            return "minimal_cycle"
        if not self.enable_prediction:
            parts.append("no_pred")
        if not self.enable_mdim:
            parts.append("no_mdim")
        if not self.enable_m3_write and not self.enable_consolidation:
            parts.append("no_mem")
        elif not self.enable_consolidation:
            parts.append("no_consol")
        if self.stage_order == DESYNC_STAGE_ORDER:
            parts.append("desync")
        if self.prediction_mode != "normal":
            parts.append(f"pred_{self.prediction_mode}")
        if self.resource_policy != "time":
            parts.append(f"rbta_{self.resource_policy}")
        return "_".join(parts) if parts else "full"
