"""Config-driven intervention points for causal ablations."""

from __future__ import annotations

from dataclasses import dataclass
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
    "regulation",
    "prediction",
    "rbta_preflight",
    "action",
    "feedback",
    "rbta_post",
    "consolidation",
]
# Note: feedback (PEU/TSPL/learn) runs after action because it needs the env
# transition. Desync swaps regulation before prediction only (see cycle.py).


@dataclass
class InterventionConfig:
    """Single place to configure module enables and cycle variants.

    Default state: ``disable_blended_scorer=True`` — GridWorld action selection
    uses pure BFS/Manhattan geometry, not prediction-scored evaluation.  This
    is intentional (per D-156 / D-161): the blended scorer does not outperform
    geometry at either 5×5 or 10×10 L2 causal gate after the RBTA fix.
    Set ``disable_blended_scorer=False`` to enable prediction-scored selection
    with adaptive confidence gating and agreement-based fallback.
    """

    enable_prediction: bool = True
    enable_mdim: bool = True
    enable_m3_write: bool = True
    enable_consolidation: bool = True
    enable_attention: bool = True
    enable_tspl: bool = True
    enable_apc: bool = True
    enable_m2: bool = True
    enable_gprime_learn: bool = True
    disable_task_lock: bool = False
    disable_planning_grid: bool = False
    disable_blended_scorer: bool = True
    adaptive_confidence_gating: bool = True  # D-156: fall back to geometry when G' confidence < threshold
    agreement_gating: bool = True            # Round 7 (NEW-10): fall back to geometry when blended scorer persistently disagrees with geometry
    agreement_window: int = 30               # Round 7: number of recent cycles to track agreement rate
    agreement_threshold: float = 0.3         # Round 7: minimum agreement rate below which blended scorer is overridden
    before_blended_warmup_cycles: int = 50  # D-157: pure geometry for first N cycles before enabling blended scorer
    calibration_probe_interval: int = 100  # D-158: cycles between self-calibration probes
    calibration_probe_samples: int = 16    # D-158: number of (s,a,s') triples per probe
    calibration_min_threshold: float = 0.4 # D-158: floor for adaptive threshold
    calibration_max_threshold: float = 0.95# D-158: ceiling for adaptive threshold
    stage_order: Optional[List[str]] = None
    minimal_cycle: bool = False
    resource_policy: str = "time"  # time | energy | memory
    prediction_mode: str = "normal"  # normal | zero | scramble | ensemble
    rbta_energy_scale: float = 1.0  # <1 tightens energy bounds for RBTA variants

    def apply_causal_fairness(self) -> "InterventionConfig":
        """Bypass GridWorld task_lock and planning heuristics for causal tests."""
        self.disable_task_lock = True
        self.disable_planning_grid = True
        return self

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
    def from_ablation_dict(cls, d: dict, *, causal_fair: bool = False) -> "InterventionConfig":
        """Build from experiment manifest ablation block."""
        cfg = cls(
            enable_prediction=d.get("enable_prediction", True),
            enable_mdim=d.get("enable_mdim", True),
            enable_m3_write=d.get("enable_m3_write", True),
            enable_consolidation=d.get("enable_consolidation", True),
            enable_attention=d.get("enable_attention", True),
            enable_tspl=d.get("enable_tspl", True),
            enable_apc=d.get("enable_apc", True),
            enable_m2=d.get("enable_m2", True),
            enable_gprime_learn=d.get("enable_gprime_learn", True),
            disable_task_lock=d.get("disable_task_lock", False),
            disable_planning_grid=d.get("disable_planning_grid", False),
            disable_blended_scorer=d.get("disable_blended_scorer", True),
            adaptive_confidence_gating=d.get("adaptive_confidence_gating", True),
            agreement_gating=d.get("agreement_gating", True),
            agreement_window=int(d.get("agreement_window", 30)),
            agreement_threshold=float(d.get("agreement_threshold", 0.3)),
            before_blended_warmup_cycles=int(d.get("before_blended_warmup_cycles", 50)),
            calibration_probe_interval=int(d.get("calibration_probe_interval", 100)),
            calibration_probe_samples=int(d.get("calibration_probe_samples", 16)),
            calibration_min_threshold=float(d.get("calibration_min_threshold", 0.4)),
            calibration_max_threshold=float(d.get("calibration_max_threshold", 0.95)),
            minimal_cycle=d.get("minimal_cycle", False),
            stage_order=d.get("stage_order"),
            resource_policy=d.get("resource_policy", "time"),
            prediction_mode=d.get("prediction_mode", "normal"),  # normal | zero | scramble | ensemble
            rbta_energy_scale=float(d.get("rbta_energy_scale", 1.0)),
        )
        if causal_fair or not cfg.enable_prediction or cfg.minimal_cycle:
            cfg.apply_causal_fairness()
        return cfg

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
