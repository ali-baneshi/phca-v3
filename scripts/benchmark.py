#!/usr/bin/env python3
"""
PHCA v3.0 — Φ-IQ Benchmark Suite (Phase 2.4 Evaluation).

Implements the Φ-Intelligence composite metric and multi-level benchmark suite.

Φ-IQ = w₁·PredictionAccuracy + w₂·AdaptationSpeed + w₃·GoalComplexity
     + w₄·TransferEfficiency + w₅·ResourceEfficiency - w₆·FailureRate

Benchmark levels:
  Level 0: Stationary prediction (no action required)
  Level 1: Reactive control (single feedback loop)
  Level 2: Goal pursuit (external goal → planning)
  Level 3: Self-motivated exploration (no external goals)

Usage:
    python scripts/benchmark.py --levels=0-3 --cycles=100 --output=benchmark_report.json
    python scripts/benchmark.py --quick  (Level 0 only, 20 cycles)
"""

from __future__ import annotations

import argparse
import json
import time
import statistics
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from phca.core.cycle import CognitiveCycle, CycleMetrics
from phca.logging import ensure_logging
from phca.config import GoalVector, ResourceBounds, StateVector, CYCLE_TARGET


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
    diagnose_level: int = -1  # if >=0, dump per-step history for this level to CSV
    dynamic_goals: bool = False  # Week 3: L2 curriculum (static 100 cyc, then relocate every N)
    dynamic_goals_every: int = 100  # Phase 5 / D-094: relocation cadence in cycles (every-100 default)
    weights: Dict[str, float] = field(default_factory=lambda: DEFAULT_WEIGHTS.copy())


@dataclass
class BenchmarkResult:
    """Results from a single benchmark level."""
    level: int
    level_name: str
    n_cycles: int
    prediction_accuracy: float = 0.0    # 1 - normalized prediction error
    adaptation_speed: float = 0.0       # error improvement over time
    goal_complexity: float = 0.0        # goal diversity / autonomy rate
    transfer_efficiency: float = 0.0    # cross-task retention (heuristic: adapt × pred)
    resource_efficiency: float = 0.0    # 1 - (cycle_time / target)
    failure_rate: float = 0.0           # violations per cycle
    phi_iq: float = 0.0                 # composite score
    raw_metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkReport:
    """Full benchmark report across all levels."""
    config: BenchmarkConfig
    results: List[BenchmarkResult] = field(default_factory=list)
    overall_phi_iq: float = 0.0
    total_cycles: int = 0
    duration_s: float = 0.0
    pass_criteria: Dict[str, bool] = field(default_factory=dict)


# ── Benchmark Runner ────────────────────────────────────────


class BenchmarkRunner:
    """Runs the Φ-IQ benchmark suite across multiple levels."""

    def __init__(self, config: Optional[BenchmarkConfig] = None):
        self.config = config or BenchmarkConfig()
        self.report = BenchmarkReport(config=self.config)

    @staticmethod
    def _generate_goal_pursuit_obstacles(seed: int) -> List[Tuple[int, int]]:
        """Generate wall positions for Level 2 (Goal Pursuit).

        Creates a 5×5 grid with walls forming a maze-like structure
        that requires the agent to navigate around obstacles to reach the goal.

        Returns:
            List of (row, col) wall positions.
        """
        rng = np.random.RandomState(seed)
        obstacles = []
        # Place a vertical wall barrier in column 2, with 1-2 gaps
        gap_row = rng.randint(0, 5)
        for r in range(5):
            if r != gap_row:
                obstacles.append((r, 2))
        # Place a few scattered walls for variety
        for _ in range(2):
            r, c = rng.randint(0, 5, size=2)
            if (r, c) not in obstacles:
                obstacles.append((r, c))
        return obstacles

    def run_all(self, levels: Optional[List[int]] = None) -> BenchmarkReport:
        """Run all specified benchmark levels."""
        if levels is None:
            levels = [0, 1, 2, 3]

        t_start = time.perf_counter()

        level_names = {
            0: "Stationary Prediction",
            1: "Reactive Control",
            2: "Goal Pursuit",
            3: "Self-Motivated Exploration",
        }

        for level in levels:
            if level not in level_names:
                print(f"  [SKIP] Unknown level {level}")
                continue
            print(f"\n{'='*60}")
            print(f"  Level {level}: {level_names[level]}")
            print(f"{'='*60}")
            result = self._run_level(level)
            self.report.results.append(result)
            print(f"  → Φ-IQ: {result.phi_iq:.4f}  "
                  f"(pred_acc={result.prediction_accuracy:.3f}, "
                  f"adapt={result.adaptation_speed:.3f}, "
                  f"goals={result.goal_complexity:.3f}, "
                  f"transfer={result.transfer_efficiency:.3f}, "
                  f"resource={result.resource_efficiency:.3f}, "
                  f"failures={result.failure_rate:.3f})")

        # Compute overall Φ-IQ (mean across levels)
        if self.report.results:
            self.report.overall_phi_iq = float(np.mean([r.phi_iq for r in self.report.results]))

        self.report.total_cycles = sum(r.n_cycles for r in self.report.results)
        self.report.duration_s = time.perf_counter() - t_start

        # Check pass criteria from whitepaper §1.3
        self._check_pass_criteria()

        return self.report

    def _run_level(self, level: int) -> BenchmarkResult:
        """Run a single benchmark level and compute metrics."""
        # MLP needs more cycles to learn (200 vs 50)
        n = self.config.n_cycles if not self.config.use_mlp else max(self.config.n_cycles, 200)
        warmup = self.config.warmup

        # Build a cycle configured for this level
        # Level 2 (Goal Pursuit) gets obstacles to make navigation interesting
        obstacles: Optional[List[Tuple[int, int]]] = None
        if level == 2:
            obstacles = self._generate_goal_pursuit_obstacles(self.config.seed + level)

        cycle = CognitiveCycle.build_for_env(
            size=self.config.grid_size,
            seed=self.config.seed + level,
            use_continuous=self.config.use_continuous,
            use_mlp=self.config.use_mlp,
            obstacles=obstacles,
        )

        # Override G' timing bound for MLP (learn() takes ~55ms with 8×64 batch)
        if self.config.use_mlp:
            cycle.rbta.update_bounds(
                "G'", ResourceBounds(B_time=0.080, B_mem=500_000, B_energy=50.0),
            )

        # Warmup
        for _ in range(warmup):
            cycle.step()

        # Benchmark cycles
        history: List[CycleMetrics] = []
        # Week 3 dynamic-goal curriculum (L2 only): static for the first 100
        # cycles so the agent stabilises, then relocate the goal every 100
        # cycles — gentler than the rejected Iteration B (every 50). Gives
        # adaptation_speed real improvement headroom without overwhelming.
        relocate_every = self.config.dynamic_goals_every if (level == 2 and self.config.dynamic_goals) else 0
        for i in range(n):
            if relocate_every and i > 0 and i % relocate_every == 0:
                cycle.env.relocate_goal()
            metrics = cycle.step()
            history.append(metrics)

        result = BenchmarkResult(level=level, level_name="", n_cycles=n)

        # Diagnostic dump: per-step history for the diagnosed level (read-only analysis)
        if level == self.config.diagnose_level:
            self._dump_history_csv(history, level)

        if level == 0:
            result = self._compute_level_0(history, cycle, result)
        elif level == 1:
            result = self._compute_level_1(history, cycle, result)
        elif level == 2:
            result = self._compute_level_2(history, cycle, result)
        elif level == 3:
            result = self._compute_level_3(history, cycle, result)

        # Compute transfer_efficiency heuristic (proxy for cross-task retention)
        # When not measured directly, use the product of adaptation and prediction
        # as a reasonable estimate: a system that predicts well AND adapts quickly
        # is likely to transfer well across tasks.
        result.transfer_efficiency = result.adaptation_speed * result.prediction_accuracy

        # Compute Φ-IQ composite
        result.phi_iq = self._compute_phi_iq(result)
        return result

    # ── Level 0: Stationary Prediction ───────────────────────

    def _dump_history_csv(self, history: List[CycleMetrics], level: int) -> None:
        """Dump per-step cycle metrics to logs/diagnose_level{N}.csv (read-only diagnostic)."""
        import csv
        path = f"logs/diagnose_level{level}.csv"
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["cycle", "pred_error", "pred_conf", "goal_reached",
                        "action", "latency_ms", "violations"])
            for m in history:
                w.writerow([m.cycle_id, f"{m.prediction_error:.4f}",
                            f"{m.prediction_confidence:.4f}", int(m.goal_reached),
                            m.action_taken, f"{m.latency_ms:.2f}", m.violations_count])
        print(f"  [diagnose] per-step history dumped to {path} ({len(history)} rows)")

    # ── Level 0: Stationary Prediction ───────────────────────

    def _compute_level_0(
        self, history: List[CycleMetrics], cycle: CognitiveCycle, result: BenchmarkResult,
    ) -> BenchmarkResult:
        """Level 0: Stationary prediction — measure prediction accuracy only."""
        result.level_name = "Stationary Prediction"

        errors = [m.prediction_error for m in history]
        confidences = [m.prediction_confidence for m in history]
        latencies = [m.latency_ms for m in history]
        violations = sum(m.violations_count for m in history)

        # Prediction accuracy: inverse of normalized RMSE
        mean_error = float(np.mean(errors)) if errors else 0.0
        result.prediction_accuracy = max(0.0, 1.0 - min(mean_error / 10.0, 1.0))

        # Adaptation speed: blend of improvement and maintenance
        # improvement = (early - late) / early  (relative error reduction)
        # maintenance = 1 - late / 10.0          (sustained accuracy, same /10 scaling as pred_acc)
        if len(errors) >= 10:
            early = float(np.mean(errors[:len(errors)//2]))
            late = float(np.mean(errors[len(errors)//2:]))
            improvement = (early - late) / max(early, 0.001)
            maintenance = max(0.0, 1.0 - late / 10.0)
            result.adaptation_speed = float(np.clip(max(improvement, maintenance), 0.0, 1.0))

        # Goal complexity: MDIM drive diversity in stationary env
        if hasattr(cycle, 'mdim') and cycle.mdim is not None:
            drive_summary = {name: d.deficit for name, d in cycle.mdim.drives.items()}
            active_drives = sum(1 for v in drive_summary.values() if v > 0.01)
            result.goal_complexity = min(1.0, active_drives / 5.0)
        else:
            result.goal_complexity = 0.0

        # Resource efficiency
        mean_latency = float(np.mean(latencies)) if latencies else 500.0
        target_ms = CYCLE_TARGET * 1000  # 500ms
        result.resource_efficiency = max(0.0, 1.0 - mean_latency / target_ms)

        # Failure rate
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

    # ── Level 1: Reactive Control ────────────────────────────

    def _compute_level_1(
        self, history: List[CycleMetrics], cycle: CognitiveCycle, result: BenchmarkResult,
    ) -> BenchmarkResult:
        """Level 1: Reactive control — single feedback loop accuracy."""
        result.level_name = "Reactive Control"

        errors = [m.prediction_error for m in history]
        latencies = [m.latency_ms for m in history]
        violations = sum(m.violations_count for m in history)

        mean_error = float(np.mean(errors)) if errors else 0.0
        result.prediction_accuracy = max(0.0, 1.0 - min(mean_error / 10.0, 1.0))

        # Adaptation: error reduction under active control
        if len(errors) >= 10:
            early = float(np.mean(errors[:max(1, len(errors)//4)]))
            late = float(np.mean(errors[-max(1, len(errors)//4):]))
            improvement = (early - late) / max(early, 0.001)
            maintenance = max(0.0, 1.0 - late / 10.0)
            result.adaptation_speed = float(np.clip(max(improvement, maintenance), 0.0, 1.0))

        # Goal complexity: action diversity
        actions = [m.action_taken for m in history if m.action_taken >= 0]
        unique_actions = len(set(actions)) if actions else 0
        result.goal_complexity = min(1.0, unique_actions / 5.0)

        # Resource efficiency
        mean_latency = float(np.mean(latencies)) if latencies else 500.0
        target_ms = CYCLE_TARGET * 1000
        result.resource_efficiency = max(0.0, 1.0 - mean_latency / target_ms)

        # Failure rate
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

    # ── Level 2: Goal Pursuit ────────────────────────────────

    def _compute_level_2(
        self, history: List[CycleMetrics], cycle: CognitiveCycle, result: BenchmarkResult,
    ) -> BenchmarkResult:
        """Level 2: Goal pursuit — measure goal reaching rate."""
        result.level_name = "Goal Pursuit"

        errors = [m.prediction_error for m in history]
        latencies = [m.latency_ms for m in history]
        goals = [m.goal_reached for m in history]
        violations = sum(m.violations_count for m in history)

        mean_error = float(np.mean(errors)) if errors else 0.0
        result.prediction_accuracy = max(0.0, 1.0 - min(mean_error / 10.0, 1.0))

        # Adaptation: goal-reaching improvement AND sustained excellence.
        # L0/L1 use max(improvement, maintenance); L2 now aligned (was late-early
        # only, which collides with the ceiling when goal_rate is high throughout).
        if len(goals) >= 10:
            early_goals = float(np.mean(goals[:len(goals)//2]))
            late_goals = float(np.mean(goals[len(goals)//2:]))
            improvement = (late_goals - early_goals) / max(1.0 - early_goals, 0.001)
            maintenance = late_goals  # sustained high goal rate IS adaptation
            result.adaptation_speed = float(np.clip(max(improvement, maintenance), 0.0, 1.0))

        # Goal complexity: actual goal reaching rate
        result.goal_complexity = float(np.mean(goals)) if goals else 0.0

        # Resource efficiency
        mean_latency = float(np.mean(latencies)) if latencies else 500.0
        target_ms = CYCLE_TARGET * 1000
        result.resource_efficiency = max(0.0, 1.0 - mean_latency / target_ms)

        # Failure rate
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

    # ── Level 3: Self-Motivated Exploration ──────────────────

    def _compute_level_3(
        self, history: List[CycleMetrics], cycle: CognitiveCycle, result: BenchmarkResult,
    ) -> BenchmarkResult:
        """Level 3: Self-motivated exploration — measure goal autonomy."""
        result.level_name = "Self-Motivated Exploration"

        errors = [m.prediction_error for m in history]
        latencies = [m.latency_ms for m in history]
        violations = sum(m.violations_count for m in history)
        actions = [m.action_taken for m in history if m.action_taken >= 0]

        mean_error = float(np.mean(errors)) if errors else 0.0
        result.prediction_accuracy = max(0.0, 1.0 - min(mean_error / 10.0, 1.0))

        # Adaptation: exploration diversity over time
        unique_actions = len(set(actions)) if actions else 0
        result.adaptation_speed = min(1.0, unique_actions / 5.0)

        # Goal complexity: drive diversity from MDIM
        if hasattr(cycle, 'mdim') and cycle.mdim is not None:
            drive_summary = {name: d.deficit for name, d in cycle.mdim.drives.items()}
            active_drives = sum(1 for v in drive_summary.values() if v > 0.01)
            result.goal_complexity = min(1.0, active_drives / 5.0)
        else:
            result.goal_complexity = 0.2  # default

        # Resource efficiency
        mean_latency = float(np.mean(latencies)) if latencies else 500.0
        target_ms = CYCLE_TARGET * 1000
        result.resource_efficiency = max(0.0, 1.0 - mean_latency / target_ms)

        # Failure rate
        result.failure_rate = violations / max(len(history), 1)

        # Check goal autonomy criterion: ≥1 novel goal per 100 cycles
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

    # ── Φ-IQ Composite ───────────────────────────────────────

    def _compute_phi_iq(self, result: BenchmarkResult) -> float:
        """Compute the Φ-IQ composite score from sub-metrics."""
        w = self.config.weights
        score = (
            w["prediction_accuracy"] * result.prediction_accuracy
            + w["adaptation_speed"] * result.adaptation_speed
            + w["goal_complexity"] * result.goal_complexity
            + w["transfer_efficiency"] * result.transfer_efficiency
            + w["resource_efficiency"] * result.resource_efficiency
            - w["failure_rate"] * result.failure_rate
        )
        return float(np.clip(score, 0.0, 1.0))

    def _check_pass_criteria(self) -> None:
        """Check pass criteria from whitepaper §1.3."""
        criteria = {}
        all_results = self.report.results

        # C1: Cycle latency < 500ms
        latencies = []
        for r in all_results:
            if "mean_latency_ms" in r.raw_metrics:
                latencies.append(r.raw_metrics["mean_latency_ms"])
        criteria["latency_under_500ms"] = (
            bool(latencies) and max(latencies) < 500.0
        )

        # C2: Failure recovery (violations < 10% of cycles)
        total_violations = sum(r.raw_metrics.get("violations", 0) for r in all_results)
        total_cycles = sum(r.n_cycles for r in all_results)
        criteria["failure_rate_under_10pct"] = (
            total_violations / max(total_cycles, 1) < 0.1
        )

        # C3: Goal autonomy (Level 3 only — N/A for --quick / L0-only runs)
        level3 = [r for r in all_results if r.level == 3]
        if level3:
            criteria["goal_autonomy_achieved"] = level3[0].goal_complexity > 0.1

        # C4: Overall Φ-IQ > 0.5
        criteria["phi_iq_above_0_5"] = self.report.overall_phi_iq > 0.5

        self.report.pass_criteria = criteria


def print_report(report: BenchmarkReport) -> None:
    """Print a formatted benchmark report."""
    print(f"\n{'='*60}")
    print(f"  PHCA v3.0 — Φ-IQ Benchmark Report")
    print(f"{'='*60}")
    model = "MLP" if report.config.use_mlp else ("Gaussian G'" if report.config.use_continuous else "Discrete G'")
    print(f"  Config: {report.config.n_cycles} cycles/level, "
          f"grid={report.config.grid_size}x{report.config.grid_size}, "
          f"{model}")
    print(f"  Duration: {report.duration_s:.1f}s")
    print(f"\n  {'Level':<8} {'Φ-IQ':<8} {'Pred':<8} {'Adapt':<8} {'Goals':<8} {'Transfer':<8} {'Resource':<8} {'Fail':<8}")
    print(f"  {'-'*64}")
    for r in report.results:
        print(f"  L{r.level:<7} {r.phi_iq:.4f}  {r.prediction_accuracy:.4f}  "
              f"{r.adaptation_speed:.4f}  {r.goal_complexity:.4f}  "
              f"{r.transfer_efficiency:.4f}  {r.resource_efficiency:.4f}  {r.failure_rate:.4f}")

    print(f"\n  {'─'*64}")
    print(f"  {'OVERALL Φ-IQ':<42} {report.overall_phi_iq:.4f}")
    print(f"  {'─'*64}")

    print(f"\n  Pass Criteria:")
    for criterion, passed in report.pass_criteria.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"    [{status}] {criterion}")

    all_pass = all(report.pass_criteria.values())
    print(f"\n  Overall: {'✓ PASS' if all_pass else '✗ FAIL'}")
    print(f"{'='*60}\n")


def save_report(report: BenchmarkReport, path: str) -> None:
    """Save benchmark report to JSON."""
    data = {
        "config": asdict(report.config),
        "results": [asdict(r) for r in report.results],
        "overall_phi_iq": report.overall_phi_iq,
        "total_cycles": report.total_cycles,
        "duration_s": report.duration_s,
        "pass_criteria": report.pass_criteria,
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, indent=2, default=str))
    print(f"Report saved to {path}")


def _run_mujoco(env_name: str, n_cycles: int, use_mlp: bool, output: str) -> dict:
    """Run a single MuJoCo environment benchmark (Week 2).

    Reports latency, prediction-error trend, and RBTA violations — the grid
    goal_reached metric does not apply to MuJoCo, so no Φ-IQ composite.
    """
    from phca.core.cycle import CognitiveCycle
    cycle = CognitiveCycle.build_for_mujoco(env_name, seed=42, use_mlp=use_mlp)
    for _ in range(10):  # warmup
        cycle.step()
    errors, latencies, violations = [], [], 0
    for _ in range(n_cycles):
        m = cycle.step()
        errors.append(m.prediction_error)
        latencies.append(m.latency_ms)
        violations += m.violations_count
    early = float(np.mean(errors[:max(1, len(errors)//4)]))
    late = float(np.mean(errors[-max(1, len(errors)//4):]))
    report = {
        "env": env_name, "n_cycles": n_cycles, "model": "MLP" if use_mlp else "Gaussian",
        "mean_latency_ms": float(np.mean(latencies)),
        "p95_latency_ms": float(np.percentile(latencies, 95)),
        "max_latency_ms": float(np.max(latencies)),
        "mean_error": float(np.mean(errors)),
        "early_error": early, "late_error": late,
        "error_improved": late < early,
        "violations": violations,
        "violation_rate": violations / max(n_cycles, 1),
        "no_errors": all(np.isfinite(e) for e in errors),
    }
    print(f"\n{'='*60}\n  PHCA v3.0 — MuJoCo Benchmark ({env_name})\n{'='*60}")
    print(f"  Cycles: {n_cycles}  Model: {report['model']}")
    print(f"  Latency mean/p95/max: {report['mean_latency_ms']:.1f}/"
          f"{report['p95_latency_ms']:.1f}/{report['max_latency_ms']:.1f} ms")
    print(f"  Error early→late: {early:.3f}→{late:.3f}  improved={report['error_improved']}")
    print(f"  RBTA violations: {violations} ({report['violation_rate']*100:.1f}%)")
    c1 = report["no_errors"]
    c3 = report["mean_latency_ms"] < 200.0
    c4 = report["error_improved"] or late < 1.0
    c6 = report["violation_rate"] < 0.1
    print(f"  Criteria: C1(no err)={'PASS' if c1 else 'FAIL'} "
          f"C3(<200ms)={'PASS' if c3 else 'FAIL'} "
          f"C4(err↓)={'PASS' if c4 else 'FAIL'} "
          f"C6(<10%viol)={'PASS' if c6 else 'FAIL'}\n{'='*60}\n")
    out = output or "logs/benchmark_mujoco_report.json"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(report, indent=2))
    print(f"Report saved to {out}")
    return report


def main() -> None:
    ensure_logging()  # enable file logging to logs/phca.log (A-002 fix)
    parser = argparse.ArgumentParser(description="PHCA Φ-IQ Benchmark Suite")
    parser.add_argument("--levels", type=str, default="0,1,2,3",
                        help="Comma-separated list of levels to run (default: 0,1,2,3)")
    parser.add_argument("--cycles", type=int, default=50,
                        help="Number of cognitive cycles per level (default: 50)")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: Level 0 only, 20 cycles")
    parser.add_argument("--use-mlp", action="store_true",
                        help="Use MLP world model instead of Gaussian G'")
    parser.add_argument("--env", type=str, default="gridworld",
                        choices=["gridworld", "cartpole", "pendulum", "reacher"],
                        help="Environment: gridworld (4-level Φ-IQ) or a MuJoCo env "
                             "(single-level latency/error report)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON report path")
    parser.add_argument("--diagnose-level", type=int, default=-1,
                        help="Dump per-step history CSV for this level (default: off)")
    parser.add_argument("--dynamic-goals", action="store_true",
                        help="L2 curriculum: static goal, then relocate every --dynamic-goals-every cycles")
    parser.add_argument("--dynamic-goals-every", type=int, default=100,
                        help="L2 goal-relocation cadence in cycles (default 100; Phase 5 / D-094)")
    args = parser.parse_args()

    if args.quick:
        levels = [0]
        n_cycles = 20
    else:
        levels = [int(l.strip()) for l in args.levels.split(",")]
        n_cycles = args.cycles

    # MuJoCo environments: single-level latency/error report (no grid goal_reached).
    if args.env in ("cartpole", "pendulum", "reacher"):
        env_name = {
            "cartpole": "InvertedPendulum-v5",
            "pendulum": "Pendulum-v1",
            "reacher": "Reacher-v5",
        }[args.env]
        report = _run_mujoco(env_name, n_cycles, args.use_mlp, args.output)
        sys.exit(0 if report["no_errors"] else 1)

    config = BenchmarkConfig(n_cycles=n_cycles, use_mlp=args.use_mlp,
                             diagnose_level=args.diagnose_level,
                             dynamic_goals=args.dynamic_goals,
                             dynamic_goals_every=args.dynamic_goals_every)
    runner = BenchmarkRunner(config)
    report = runner.run_all(levels)
    print_report(report)

    output = args.output or "logs/benchmark_report.json"
    save_report(report, output)

    # Return exit code based on pass/fail
    if not all(report.pass_criteria.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
