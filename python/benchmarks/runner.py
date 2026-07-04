"""
PHCA v3.0 — Φ-IQ Benchmark Suite Runner.

DEPRECATED (AD-3): Use ``scripts/benchmark.py`` as the canonical entry point.
This module remains for backward compatibility only and may drift from the
primary script.

CLI for running benchmark levels 0-3.
Implements the Φ-Intelligence composite metric and multi-level benchmark suite
from the Phase 3.3 specification, ported from scripts/benchmark.py.

Φ-IQ = w₁·PredictionAccuracy + w₂·AdaptationSpeed + w₃·GoalComplexity
     + w₄·TransferEfficiency + w₅·ResourceEfficiency - w₆·FailureRate

Benchmark levels:
  Level 0: Stationary prediction (no action required)
  Level 1: Reactive control (single feedback loop)
  Level 2: Goal pursuit (external goal → planning)
  Level 3: Self-motivated exploration (no external goals)

Usage:
    python -m phca.benchmarks.runner --level=0 --output=results/level_0.json
    python -m phca.benchmarks.runner --all --cycles=50 --output=results/full.json
"""

from __future__ import annotations

import json
import sys
import warnings
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from phca.core.cycle import CognitiveCycle, CycleMetrics
from phca.config import ResourceBounds, CYCLE_TARGET


# ── Φ-IQ Sub-Metric Weights ────────────────────────────────

# Default weights from Phase 2.4 specification
DEFAULT_WEIGHTS = {
    "prediction_accuracy": 0.20,
    "adaptation_speed": 0.20,
    "goal_complexity": 0.15,
    "transfer_efficiency": 0.15,
    "resource_efficiency": 0.20,
    "failure_rate": 0.10,  # subtracted
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


# ── Individual Level Runners ────────────────────────────────


def _generate_goal_pursuit_obstacles(seed: int) -> List[Tuple[int, int]]:
    """Generate wall positions for Level 2 (Goal Pursuit)."""
    rng = np.random.RandomState(seed)
    obstacles = []
    gap_row = rng.randint(0, 5)
    for r in range(5):
        if r != gap_row:
            obstacles.append((r, 2))
    for _ in range(2):
        r, c = rng.randint(0, 5, size=2)
        if (r, c) not in obstacles:
            obstacles.append((r, c))
    return obstacles


def _build_cycle(
    level: int,
    config: BenchmarkConfig,
) -> CognitiveCycle:
    """Build a CognitiveCycle for the given benchmark level."""
    obstacles: Optional[List[Tuple[int, int]]] = None
    if level == 2:
        obstacles = _generate_goal_pursuit_obstacles(config.seed + level)

    cycle = CognitiveCycle.build_for_env(
        size=config.grid_size,
        seed=config.seed + level,
        use_continuous=config.use_continuous,
        use_mlp=config.use_mlp,
        obstacles=obstacles,
    )

    if config.use_mlp:
        cycle.rbta.update_bounds(
            "G'", ResourceBounds(B_time=0.050, B_mem=500_000, B_energy=50.0),
        )

    return cycle


def _run_cycles(
    cycle: CognitiveCycle,
    n: int,
    warmup: int,
) -> List[CycleMetrics]:
    """Run warmup + benchmark cycles and return metric history."""
    for _ in range(warmup):
        cycle.step()
    return [cycle.step() for _ in range(n)]


def _compute_level_0(
    history: List[CycleMetrics],
    cycle: CognitiveCycle,
    result: BenchmarkResult,
) -> BenchmarkResult:
    """Level 0: Stationary prediction."""
    result.level_name = "Stationary Prediction"

    errors = [m.prediction_error for m in history]
    confidences = [m.prediction_confidence for m in history]
    latencies = [m.latency_ms for m in history]
    violations = sum(m.violations_count for m in history)

    mean_error = float(np.mean(errors)) if errors else 0.0
    result.prediction_accuracy = max(0.0, 1.0 - min(mean_error / 10.0, 1.0))

    if len(errors) >= 10:
        early = float(np.mean(errors[:len(errors)//2]))
        late = float(np.mean(errors[len(errors)//2:]))
        improvement = (early - late) / max(early, 0.001)
        maintenance = max(0.0, 1.0 - late / 10.0)
        result.adaptation_speed = float(np.clip(max(improvement, maintenance), 0.0, 1.0))

    if hasattr(cycle, 'mdim') and cycle.mdim is not None:
        drive_summary = {name: d.deficit for name, d in cycle.mdim.drives.items()}
        active_drives = sum(1 for v in drive_summary.values() if v > 0.01)
        result.goal_complexity = min(1.0, active_drives / 5.0)
    else:
        result.goal_complexity = 0.0

    mean_latency = float(np.mean(latencies)) if latencies else 500.0
    target_ms = CYCLE_TARGET * 1000
    result.resource_efficiency = max(0.0, 1.0 - mean_latency / target_ms)
    result.failure_rate = violations / max(len(history), 1)

    result.raw_metrics = {
        "mean_error": mean_error,
        "mean_confidence": float(np.mean(confidences)) if confidences else 0.0,
        "mean_latency_ms": mean_latency,
        "p95_latency_ms": float(np.percentile(latencies, 95)) if latencies else 0.0,
        "violations": violations,
        "skill_accuracy": cycle.tspl.skill_accuracy,
        "skill_compiled": cycle.tspl.skill_compiled,
        "active_drives": int(result.goal_complexity * 5),
    }
    return result


def _compute_level_1(
    history: List[CycleMetrics],
    cycle: CognitiveCycle,
    result: BenchmarkResult,
) -> BenchmarkResult:
    """Level 1: Reactive control."""
    result.level_name = "Reactive Control"

    errors = [m.prediction_error for m in history]
    latencies = [m.latency_ms for m in history]
    violations = sum(m.violations_count for m in history)

    mean_error = float(np.mean(errors)) if errors else 0.0
    result.prediction_accuracy = max(0.0, 1.0 - min(mean_error / 10.0, 1.0))

    if len(errors) >= 10:
        early = float(np.mean(errors[:max(1, len(errors)//4)]))
        late = float(np.mean(errors[-max(1, len(errors)//4):]))
        improvement = (early - late) / max(early, 0.001)
        maintenance = max(0.0, 1.0 - late / 10.0)
        result.adaptation_speed = float(np.clip(max(improvement, maintenance), 0.0, 1.0))

    actions = [m.action_taken for m in history if m.action_taken >= 0]
    unique_actions = len(set(actions)) if actions else 0
    result.goal_complexity = min(1.0, unique_actions / 5.0)

    mean_latency = float(np.mean(latencies)) if latencies else 500.0
    target_ms = CYCLE_TARGET * 1000
    result.resource_efficiency = max(0.0, 1.0 - mean_latency / target_ms)
    result.failure_rate = violations / max(len(history), 1)

    result.raw_metrics = {
        "mean_error": mean_error,
        "mean_latency_ms": mean_latency,
        "p95_latency_ms": float(np.percentile(latencies, 95)) if latencies else 0.0,
        "violations": violations,
        "unique_actions": unique_actions,
        "skill_accuracy": cycle.tspl.skill_accuracy,
        "skill_compiled": cycle.tspl.skill_compiled,
    }
    return result


def _compute_level_2(
    history: List[CycleMetrics],
    cycle: CognitiveCycle,
    result: BenchmarkResult,
) -> BenchmarkResult:
    """Level 2: Goal pursuit."""
    result.level_name = "Goal Pursuit"

    errors = [m.prediction_error for m in history]
    latencies = [m.latency_ms for m in history]
    goals = [m.goal_reached for m in history]
    violations = sum(m.violations_count for m in history)

    mean_error = float(np.mean(errors)) if errors else 0.0
    result.prediction_accuracy = max(0.0, 1.0 - min(mean_error / 10.0, 1.0))

    if len(goals) >= 10:
        early_goals = float(np.mean(goals[:len(goals)//2]))
        late_goals = float(np.mean(goals[len(goals)//2:]))
        result.adaptation_speed = float(np.clip(late_goals - early_goals, 0.0, 1.0))

    result.goal_complexity = float(np.mean(goals)) if goals else 0.0

    mean_latency = float(np.mean(latencies)) if latencies else 500.0
    target_ms = CYCLE_TARGET * 1000
    result.resource_efficiency = max(0.0, 1.0 - mean_latency / target_ms)
    result.failure_rate = violations / max(len(history), 1)

    result.raw_metrics = {
        "mean_error": mean_error,
        "mean_latency_ms": mean_latency,
        "p95_latency_ms": float(np.percentile(latencies, 95)) if latencies else 0.0,
        "violations": violations,
        "goals_reached": int(sum(goals)),
        "goal_rate": float(np.mean(goals)) if goals else 0.0,
        "skill_accuracy": cycle.tspl.skill_accuracy,
    }
    return result


def _compute_level_3(
    history: List[CycleMetrics],
    cycle: CognitiveCycle,
    result: BenchmarkResult,
) -> BenchmarkResult:
    """Level 3: Self-motivated exploration."""
    result.level_name = "Self-Motivated Exploration"

    errors = [m.prediction_error for m in history]
    latencies = [m.latency_ms for m in history]
    violations = sum(m.violations_count for m in history)
    actions = [m.action_taken for m in history if m.action_taken >= 0]

    mean_error = float(np.mean(errors)) if errors else 0.0
    result.prediction_accuracy = max(0.0, 1.0 - min(mean_error / 10.0, 1.0))

    unique_actions = len(set(actions)) if actions else 0
    result.adaptation_speed = min(1.0, unique_actions / 5.0)

    if hasattr(cycle, 'mdim') and cycle.mdim is not None:
        drive_summary = {name: d.deficit for name, d in cycle.mdim.drives.items()}
        active_drives = sum(1 for v in drive_summary.values() if v > 0.01)
        result.goal_complexity = min(1.0, active_drives / 5.0)
    else:
        result.goal_complexity = 0.2

    mean_latency = float(np.mean(latencies)) if latencies else 500.0
    target_ms = CYCLE_TARGET * 1000
    result.resource_efficiency = max(0.0, 1.0 - mean_latency / target_ms)
    result.failure_rate = violations / max(len(history), 1)

    consol_stats = cycle.consolidation.get_stats() if hasattr(cycle, 'consolidation') else {}
    m3_size = cycle.consolidation.m3.count() if hasattr(cycle, 'consolidation') else 0

    result.raw_metrics = {
        "mean_error": mean_error,
        "mean_latency_ms": mean_latency,
        "p95_latency_ms": float(np.percentile(latencies, 95)) if latencies else 0.0,
        "violations": violations,
        "unique_actions": unique_actions,
        "active_drives": int(result.goal_complexity * 5),
        "m3_episodes": m3_size,
        "consolidation_facts": consol_stats.get("total_facts_stored", 0),
        "skill_accuracy": cycle.tspl.skill_accuracy,
    }
    return result


def _compute_phi_iq(
    result: BenchmarkResult,
    weights: Dict[str, float],
) -> float:
    """Compute the Φ-IQ composite score from sub-metrics."""
    score = (
        weights["prediction_accuracy"] * result.prediction_accuracy
        + weights["adaptation_speed"] * result.adaptation_speed
        + weights["goal_complexity"] * result.goal_complexity
        + weights["transfer_efficiency"] * result.transfer_efficiency
        + weights["resource_efficiency"] * result.resource_efficiency
        - weights["failure_rate"] * result.failure_rate
    )
    return float(np.clip(score, 0.0, 1.0))


# ── Public API ───────────────────────────────────────────────


def run_level_0(
    output_path: Optional[str] = None,
    n_cycles: int = 100,
    warmup: int = 10,
    seed: int = 42,
    use_continuous: bool = True,
    use_mlp: bool = False,
) -> dict:
    """Level 0: Stationary prediction."""
    config = BenchmarkConfig(
        n_cycles=n_cycles, warmup=warmup, seed=seed,
        use_continuous=use_continuous, use_mlp=use_mlp,
    )
    cycle = _build_cycle(0, config)
    history = _run_cycles(cycle, n=n_cycles, warmup=warmup)
    result = BenchmarkResult(level=0, level_name="Stationary Prediction", n_cycles=n_cycles)
    result = _compute_level_0(history, cycle, result)
    result.transfer_efficiency = result.adaptation_speed * result.prediction_accuracy
    result.phi_iq = _compute_phi_iq(result, config.weights)

    output = {
        "level": 0,
        "status": "completed",
        "phi_iq": result.phi_iq,
        **asdict(result),
    }
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(json.dumps(output, indent=2, default=str))
    return output


def run_level_1(
    output_path: Optional[str] = None,
    n_cycles: int = 100,
    warmup: int = 10,
    seed: int = 42,
    use_continuous: bool = True,
    use_mlp: bool = False,
) -> dict:
    """Level 1: Reactive control."""
    config = BenchmarkConfig(
        n_cycles=n_cycles, warmup=warmup, seed=seed,
        use_continuous=use_continuous, use_mlp=use_mlp,
    )
    cycle = _build_cycle(1, config)
    history = _run_cycles(cycle, n=n_cycles, warmup=warmup)
    result = BenchmarkResult(level=1, level_name="Reactive Control", n_cycles=n_cycles)
    result = _compute_level_1(history, cycle, result)
    result.transfer_efficiency = result.adaptation_speed * result.prediction_accuracy
    result.phi_iq = _compute_phi_iq(result, config.weights)

    output = {
        "level": 1,
        "status": "completed",
        "phi_iq": result.phi_iq,
        **asdict(result),
    }
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(json.dumps(output, indent=2, default=str))
    return output


def run_level_2(
    output_path: Optional[str] = None,
    n_cycles: int = 100,
    warmup: int = 10,
    seed: int = 42,
    use_continuous: bool = True,
    use_mlp: bool = False,
) -> dict:
    """Level 2: Goal pursuit."""
    config = BenchmarkConfig(
        n_cycles=n_cycles, warmup=warmup, seed=seed,
        use_continuous=use_continuous, use_mlp=use_mlp,
    )
    cycle = _build_cycle(2, config)
    history = _run_cycles(cycle, n=n_cycles, warmup=warmup)
    result = BenchmarkResult(level=2, level_name="Goal Pursuit", n_cycles=n_cycles)
    result = _compute_level_2(history, cycle, result)
    result.transfer_efficiency = result.adaptation_speed * result.prediction_accuracy
    result.phi_iq = _compute_phi_iq(result, config.weights)

    output = {
        "level": 2,
        "status": "completed",
        "phi_iq": result.phi_iq,
        **asdict(result),
    }
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(json.dumps(output, indent=2, default=str))
    return output


def run_level_3(
    output_path: Optional[str] = None,
    n_cycles: int = 100,
    warmup: int = 10,
    seed: int = 42,
    use_continuous: bool = True,
    use_mlp: bool = False,
) -> dict:
    """Level 3: Self-motivated exploration."""
    config = BenchmarkConfig(
        n_cycles=n_cycles, warmup=warmup, seed=seed,
        use_continuous=use_continuous, use_mlp=use_mlp,
    )
    cycle = _build_cycle(3, config)
    history = _run_cycles(cycle, n=n_cycles, warmup=warmup)
    result = BenchmarkResult(level=3, level_name="Self-Motivated Exploration", n_cycles=n_cycles)
    result = _compute_level_3(history, cycle, result)
    result.transfer_efficiency = result.adaptation_speed * result.prediction_accuracy
    result.phi_iq = _compute_phi_iq(result, config.weights)

    output = {
        "level": 3,
        "status": "completed",
        "phi_iq": result.phi_iq,
        **asdict(result),
    }
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(json.dumps(output, indent=2, default=str))
    return output


def run_all_levels(
    output_path: Optional[str] = None,
    n_cycles: int = 50,
    warmup: int = 10,
    seed: int = 42,
    use_continuous: bool = True,
    use_mlp: bool = False,
) -> dict:
    """Run all benchmark levels 0-3 and produce a composite report."""
    t_start = time.perf_counter()
    level_map = {0: run_level_0, 1: run_level_1, 2: run_level_2, 3: run_level_3}
    results = []
    all_phi_iqs = []

    for level in sorted(level_map.keys()):
        result = level_map[level](
            output_path=None,
            n_cycles=n_cycles,
            warmup=warmup,
            seed=seed,
            use_continuous=use_continuous,
            use_mlp=use_mlp,
        )
        results.append(result)
        all_phi_iqs.append(result["phi_iq"])

    overall_phi_iq = float(np.mean(all_phi_iqs)) if all_phi_iqs else 0.0

    # Check pass criteria
    total_violations = sum(
        r.get("raw_metrics", {}).get("violations", 0) for r in results
    )
    total_cycles = sum(r.get("n_cycles", 0) for r in results)
    level3 = [r for r in results if r.get("level") == 3]

    pass_criteria = {
        "latency_under_500ms": all(
            r.get("raw_metrics", {}).get("mean_latency_ms", 0) < 500.0
            for r in results
        ),
        "failure_rate_under_10pct": (
            total_violations / max(total_cycles, 1) < 0.1
        ),
        "goal_autonomy_achieved": (
            bool(level3) and level3[0].get("goal_complexity", 0) > 0.1
        ),
        "phi_iq_above_0_5": overall_phi_iq > 0.5,
    }

    report = {
        "overall_phi_iq": overall_phi_iq,
        "total_cycles": total_cycles,
        "duration_s": time.perf_counter() - t_start,
        "pass_criteria": pass_criteria,
        "all_pass": all(pass_criteria.values()),
        "results": results,
    }

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(json.dumps(report, indent=2, default=str))
    return report


if __name__ == "__main__":
    warnings.warn(
        "python -m phca.benchmarks.runner is deprecated; use "
        "PYTHONPATH=python python scripts/benchmark.py instead (AD-3).",
        DeprecationWarning,
        stacklevel=1,
    )
    import argparse
    parser = argparse.ArgumentParser(description="PHCA Φ-IQ Benchmark Runner")
    parser.add_argument("--level", type=int, default=0, help="Benchmark level (0-3)")
    parser.add_argument("--all", dest="run_all", action="store_true",
                        help="Run all levels 0-3")
    parser.add_argument("--cycles", type=int, default=50, help="Cycles per level")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup cycles")
    parser.add_argument("--use-mlp", action="store_true", help="Use MLP world model")
    parser.add_argument("--output", type=str, default=None, help="Output file path")
    args = parser.parse_args()

    if args.run_all:
        report = run_all_levels(
            output_path=args.output,
            n_cycles=args.cycles,
            warmup=args.warmup,
            use_mlp=args.use_mlp,
        )
        print(json.dumps(report, indent=2, default=str))
        if not report["all_pass"]:
            sys.exit(1)
    else:
        level_fn = {0: run_level_0, 1: run_level_1, 2: run_level_2, 3: run_level_3}
        fn = level_fn.get(args.level, run_level_0)
        result = fn(
            output_path=args.output,
            n_cycles=args.cycles,
            warmup=args.warmup,
            use_mlp=args.use_mlp,
        )
        print(json.dumps(result, indent=2, default=str))
        sys.exit(0)
