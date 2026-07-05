"""Typed result schemas for benchmarks and experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

DEFAULT_WEIGHTS = {
    "prediction_accuracy": 0.20,
    "adaptation_speed": 0.20,
    "goal_complexity": 0.15,
    "transfer_efficiency": 0.15,
    "resource_efficiency": 0.20,
    "failure_rate": 0.10,
}


@dataclass
class BenchmarkConfig:
    """Configuration for a benchmark run."""
    n_cycles: int = 100
    warmup: int = 10
    seed: int = 42
    grid_size: int = 5
    use_continuous: bool = True
    use_mlp: bool = False
    diagnose_level: int = -1
    dynamic_goals: bool = False
    dynamic_goals_every: int = 100
    action_slip: float = 0.0
    environment: str = "gridworld"
    mujoco_env: str = "Pendulum-v1"
    weights: Dict[str, float] = field(default_factory=lambda: DEFAULT_WEIGHTS.copy())


@dataclass
class BenchmarkResult:
    """Results from a single benchmark level."""
    level: int
    level_name: str
    n_cycles: int
    prediction_accuracy: float = 0.0
    adaptation_speed: float = 0.0
    goal_complexity: float = 0.0
    transfer_efficiency: float = 0.0
    resource_efficiency: float = 0.0
    failure_rate: float = 0.0
    phi_iq: float = 0.0
    raw_metrics: Dict[str, Any] = field(default_factory=dict)
    emergence: Dict[str, float] = field(default_factory=dict)


@dataclass
class BenchmarkReport:
    """Full benchmark report across all levels."""
    config: BenchmarkConfig
    results: List[BenchmarkResult] = field(default_factory=list)
    overall_phi_iq: float = 0.0
    total_cycles: int = 0
    duration_s: float = 0.0
    pass_criteria: Dict[str, bool] = field(default_factory=dict)


@dataclass
class RunSummary:
    """Per-seed run summary for experiment manifests."""
    seed: int
    metrics: Dict[str, float] = field(default_factory=dict)
    emergence: Dict[str, float] = field(default_factory=dict)
    failures: Dict[str, Any] = field(default_factory=dict)
    duration_s: float = 0.0


@dataclass
class ExperimentResult:
    """Aggregate result from an experiment manifest."""
    name: str
    hypothesis: str
    predicted_failure: str
    config: Dict[str, Any] = field(default_factory=dict)
    runs: List[RunSummary] = field(default_factory=list)
    aggregate: Dict[str, Any] = field(default_factory=dict)
    comparisons: Dict[str, Any] = field(default_factory=dict)
    passed: Optional[bool] = None
