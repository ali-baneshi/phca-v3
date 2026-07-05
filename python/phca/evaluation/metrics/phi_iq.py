"""Φ-IQ composite metric — pure functions from cycle history."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from phca.config import CYCLE_TARGET
from phca.core.cycle import CycleMetrics, CognitiveCycle
from phca.evaluation.result_schema import BenchmarkReport, BenchmarkResult, DEFAULT_WEIGHTS


def compute_phi_iq(result: BenchmarkResult, weights: Optional[Dict[str, float]] = None) -> float:
    """Compute the Φ-IQ composite score from sub-metrics."""
    w = weights or DEFAULT_WEIGHTS
    score = (
        w["prediction_accuracy"] * result.prediction_accuracy
        + w["adaptation_speed"] * result.adaptation_speed
        + w["goal_complexity"] * result.goal_complexity
        + w["transfer_efficiency"] * result.transfer_efficiency
        + w["resource_efficiency"] * result.resource_efficiency
        - w["failure_rate"] * result.failure_rate
    )
    return float(np.clip(score, 0.0, 1.0))


def _prediction_accuracy(errors: List[float]) -> float:
    mean_error = float(np.mean(errors)) if errors else 0.0
    return max(0.0, 1.0 - min(mean_error / 10.0, 1.0))


def _resource_efficiency(latencies: List[float]) -> float:
    mean_latency = float(np.mean(latencies)) if latencies else 500.0
    target_ms = CYCLE_TARGET * 1000
    return max(0.0, 1.0 - mean_latency / target_ms)


def _failure_rate(history: List[CycleMetrics]) -> float:
    violations = sum(m.violations_count for m in history)
    return violations / max(len(history), 1)


def compute_level_0(
    history: List[CycleMetrics], cycle: CognitiveCycle, result: BenchmarkResult,
) -> BenchmarkResult:
    """Level 0: Stationary prediction."""
    result.level_name = "Stationary Prediction"
    errors = [m.prediction_error for m in history]
    confidences = [m.prediction_confidence for m in history]
    latencies = [m.latency_ms for m in history]
    violations = sum(m.violations_count for m in history)

    mean_error = float(np.mean(errors)) if errors else 0.0
    result.prediction_accuracy = _prediction_accuracy(errors)

    if len(errors) >= 10:
        early = float(np.mean(errors[: len(errors) // 2]))
        late = float(np.mean(errors[len(errors) // 2 :]))
        improvement = (early - late) / max(early, 0.001)
        maintenance = max(0.0, 1.0 - late / 10.0)
        result.adaptation_speed = float(np.clip(max(improvement, maintenance), 0.0, 1.0))

    if hasattr(cycle, "mdim") and cycle.mdim is not None:
        drive_summary = {name: d.deficit for name, d in cycle.mdim.drives.items()}
        active_drives = sum(1 for v in drive_summary.values() if v > 0.01)
        result.goal_complexity = min(1.0, active_drives / 5.0)

    result.resource_efficiency = _resource_efficiency(latencies)
    result.failure_rate = violations / max(len(history), 1)
    result.raw_metrics = {
        "mean_error": mean_error,
        "mean_confidence": float(np.mean(confidences)) if confidences else 0.0,
        "mean_latency_ms": float(np.mean(latencies)) if latencies else 500.0,
        "p95_latency_ms": float(np.percentile(latencies, 95)) if latencies else 0.0,
        "violations": violations,
        "skill_accuracy": cycle.tspl.skill_accuracy,
        "skill_compiled": cycle.tspl.skill_compiled,
        "active_drives": int(result.goal_complexity * 5),
    }
    return result


def compute_level_1(
    history: List[CycleMetrics], cycle: CognitiveCycle, result: BenchmarkResult,
) -> BenchmarkResult:
    """Level 1: Reactive control."""
    result.level_name = "Reactive Control"
    errors = [m.prediction_error for m in history]
    latencies = [m.latency_ms for m in history]
    violations = sum(m.violations_count for m in history)

    mean_error = float(np.mean(errors)) if errors else 0.0
    result.prediction_accuracy = _prediction_accuracy(errors)

    if len(errors) >= 10:
        early = float(np.mean(errors[: max(1, len(errors) // 4)]))
        late = float(np.mean(errors[-max(1, len(errors) // 4) :]))
        improvement = (early - late) / max(early, 0.001)
        maintenance = max(0.0, 1.0 - late / 10.0)
        result.adaptation_speed = float(np.clip(max(improvement, maintenance), 0.0, 1.0))

    actions = [m.action_taken for m in history if m.action_taken >= 0]
    unique_actions = len(set(actions)) if actions else 0
    result.goal_complexity = min(1.0, unique_actions / 5.0)
    result.resource_efficiency = _resource_efficiency(latencies)
    result.failure_rate = violations / max(len(history), 1)
    result.raw_metrics = {
        "mean_error": mean_error,
        "mean_latency_ms": float(np.mean(latencies)) if latencies else 500.0,
        "p95_latency_ms": float(np.percentile(latencies, 95)) if latencies else 0.0,
        "violations": violations,
        "unique_actions": unique_actions,
        "skill_accuracy": cycle.tspl.skill_accuracy,
        "skill_compiled": cycle.tspl.skill_compiled,
    }
    return result


def compute_level_2(
    history: List[CycleMetrics], cycle: CognitiveCycle, result: BenchmarkResult,
) -> BenchmarkResult:
    """Level 2: Goal pursuit."""
    result.level_name = "Goal Pursuit"
    errors = [m.prediction_error for m in history]
    latencies = [m.latency_ms for m in history]
    goals = [m.goal_reached for m in history]
    violations = sum(m.violations_count for m in history)

    mean_error = float(np.mean(errors)) if errors else 0.0
    result.prediction_accuracy = _prediction_accuracy(errors)

    if len(goals) >= 10:
        early_goals = float(np.mean(goals[: len(goals) // 2]))
        late_goals = float(np.mean(goals[len(goals) // 2 :]))
        improvement = (late_goals - early_goals) / max(1.0 - early_goals, 0.001)
        maintenance = late_goals
        result.adaptation_speed = float(np.clip(max(improvement, maintenance), 0.0, 1.0))

    result.goal_complexity = float(np.mean(goals)) if goals else 0.0
    result.resource_efficiency = _resource_efficiency(latencies)
    result.failure_rate = violations / max(len(history), 1)
    result.raw_metrics = {
        "mean_error": mean_error,
        "mean_latency_ms": float(np.mean(latencies)) if latencies else 500.0,
        "p95_latency_ms": float(np.percentile(latencies, 95)) if latencies else 0.0,
        "violations": violations,
        "goals_reached": int(sum(goals)),
        "goal_rate": float(np.mean(goals)) if goals else 0.0,
        "skill_accuracy": cycle.tspl.skill_accuracy,
    }
    return result


def compute_level_3(
    history: List[CycleMetrics], cycle: CognitiveCycle, result: BenchmarkResult,
) -> BenchmarkResult:
    """Level 3: Self-motivated exploration."""
    result.level_name = "Self-Motivated Exploration"
    errors = [m.prediction_error for m in history]
    latencies = [m.latency_ms for m in history]
    violations = sum(m.violations_count for m in history)
    actions = [m.action_taken for m in history if m.action_taken >= 0]

    mean_error = float(np.mean(errors)) if errors else 0.0
    result.prediction_accuracy = _prediction_accuracy(errors)
    unique_actions = len(set(actions)) if actions else 0
    result.adaptation_speed = min(1.0, unique_actions / 5.0)

    if hasattr(cycle, "mdim") and cycle.mdim is not None:
        drive_summary = {name: d.deficit for name, d in cycle.mdim.drives.items()}
        active_drives = sum(1 for v in drive_summary.values() if v > 0.01)
        result.goal_complexity = min(1.0, active_drives / 5.0)
    else:
        result.goal_complexity = 0.2

    result.resource_efficiency = _resource_efficiency(latencies)
    result.failure_rate = violations / max(len(history), 1)
    consol_stats = cycle.consolidation.get_stats() if hasattr(cycle, "consolidation") else {}
    m3_size = cycle.consolidation.m3.count() if hasattr(cycle, "consolidation") else 0
    result.raw_metrics = {
        "mean_error": mean_error,
        "mean_latency_ms": float(np.mean(latencies)) if latencies else 500.0,
        "p95_latency_ms": float(np.percentile(latencies, 95)) if latencies else 0.0,
        "violations": violations,
        "unique_actions": unique_actions,
        "active_drives": int(result.goal_complexity * 5),
        "m3_episodes": m3_size,
        "consolidation_facts": consol_stats.get("total_facts_stored", 0),
        "skill_accuracy": cycle.tspl.skill_accuracy,
    }
    return result


LEVEL_COMPUTERS = {
    0: compute_level_0,
    1: compute_level_1,
    2: compute_level_2,
    3: compute_level_3,
}


def compute_level_metrics(
    level: int,
    history: List[CycleMetrics],
    cycle: CognitiveCycle,
    n_cycles: int,
    weights: Optional[Dict[str, float]] = None,
) -> BenchmarkResult:
    """Compute all sub-metrics and Φ-IQ for one benchmark level."""
    result = BenchmarkResult(level=level, level_name="", n_cycles=n_cycles)
    computer = LEVEL_COMPUTERS.get(level)
    if computer is None:
        return result
    result = computer(history, cycle, result)
    result.transfer_efficiency = result.adaptation_speed * result.prediction_accuracy
    result.phi_iq = compute_phi_iq(result, weights)
    return result


def check_pass_criteria(report: BenchmarkReport) -> Dict[str, bool]:
    """Check pass criteria from whitepaper §1.3."""
    criteria: Dict[str, bool] = {}
    all_results = report.results

    latencies = [
        r.raw_metrics["mean_latency_ms"]
        for r in all_results
        if "mean_latency_ms" in r.raw_metrics
    ]
    criteria["latency_under_500ms"] = bool(latencies) and max(latencies) < 500.0

    total_violations = sum(r.raw_metrics.get("violations", 0) for r in all_results)
    total_cycles = sum(r.n_cycles for r in all_results)
    criteria["failure_rate_under_10pct"] = total_violations / max(total_cycles, 1) < 0.1

    level3 = [r for r in all_results if r.level == 3]
    if level3:
        criteria["goal_autonomy_achieved"] = level3[0].goal_complexity > 0.1

    criteria["phi_iq_above_0_5"] = report.overall_phi_iq > 0.5
    return criteria


def generate_goal_pursuit_obstacles(seed: int, grid_size: int = 5) -> List[Tuple[int, int]]:
    """Generate wall positions for Level 2 goal pursuit."""
    rng = np.random.RandomState(seed)
    obstacles: List[Tuple[int, int]] = []
    gap_row = rng.randint(0, grid_size)
    mid_col = max(1, grid_size // 2)
    for r in range(grid_size):
        if r != gap_row:
            obstacles.append((r, mid_col))
    for _ in range(2):
        r, c = rng.randint(0, grid_size, size=2)
        if (r, c) not in obstacles:
            obstacles.append((r, c))
    return obstacles
