#!/usr/bin/env python3
"""
PHCA v3.0 — Φ-IQ Benchmark Suite (Phase 2.4 Evaluation).

Usage:
    python scripts/benchmark.py --levels=0-3 --cycles=100 --output=benchmark_report.json
    python scripts/benchmark.py --quick  (Level 0 only, 20 cycles)
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import _bootstrap  # noqa: F401

from phca.core.cycle import CognitiveCycle, CycleMetrics
from phca.world_model.mlp import apply_grid_rbta_bounds
from phca.evaluation.metrics.phi_iq import (
    check_pass_criteria,
    compute_level_metrics,
    compute_phi_iq,
    generate_goal_pursuit_obstacles,
)
from phca.evaluation.result_schema import DEFAULT_WEIGHTS
from phca.evaluation.metrics.statistics import seed_sequence
from phca.evaluation.result_schema import BenchmarkConfig, BenchmarkReport, BenchmarkResult
from phca.logging import ensure_logging


class BenchmarkRunner:
    """Runs the Φ-IQ benchmark suite across multiple levels."""

    def __init__(self, config: Optional[BenchmarkConfig] = None):
        self.config = config or BenchmarkConfig()
        self.report = BenchmarkReport(config=self.config)

    def run_all(self, levels: Optional[List[int]] = None) -> BenchmarkReport:
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

        if self.report.results:
            self.report.overall_phi_iq = float(np.mean([r.phi_iq for r in self.report.results]))

        self.report.total_cycles = sum(r.n_cycles for r in self.report.results)
        self.report.duration_s = time.perf_counter() - t_start
        self.report.pass_criteria = check_pass_criteria(self.report)
        return self.report

    def _run_level(self, level: int) -> BenchmarkResult:
        n = self.config.n_cycles if not self.config.use_mlp else max(self.config.n_cycles, 200)
        warmup = self.config.warmup
        obstacles = None
        if level == 2:
            obstacles = generate_goal_pursuit_obstacles(
                self.config.seed + level, self.config.grid_size,
            )

        cycle = CognitiveCycle.build_for_env(
            size=self.config.grid_size,
            seed=self.config.seed + level,
            use_continuous=self.config.use_continuous,
            use_mlp=self.config.use_mlp,
            obstacles=obstacles,
            action_slip=self.config.action_slip,
        )

        apply_grid_rbta_bounds(
            cycle,
            b_time=0.080 if self.config.use_mlp else 0.020,
        )

        for _ in range(warmup):
            cycle.step()

        history: List[CycleMetrics] = []
        relocate_every = (
            self.config.dynamic_goals_every if (level == 2 and self.config.dynamic_goals) else 0
        )
        for i in range(n):
            if relocate_every and i > 0 and i % relocate_every == 0:
                cycle.env.relocate_goal()
            metrics = cycle.step()
            history.append(metrics)

        if level == self.config.diagnose_level:
            self._dump_history_csv(history, level)

        result = compute_level_metrics(level, history, cycle, n, self.config.weights)
        result.transfer_efficiency = result.adaptation_speed * result.prediction_accuracy
        result.phi_iq = compute_phi_iq(result, self.config.weights)
        return result

    def _dump_history_csv(self, history: List[CycleMetrics], level: int) -> None:
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


def print_report(report: BenchmarkReport) -> None:
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
    if report.config.grid_size != 5 or not report.config.use_mlp:
        print(
            "  Note: failure_rate_under_10pct is calibrated for canonical "
            "5×5 MLP (200 cycles). Other grid sizes / Gaussian mode may still fail."
        )
    print(f"{'='*60}\n")


def save_report(report: BenchmarkReport, path: str, multi_seed: Optional[Dict[str, Any]] = None) -> None:
    data = {
        "config": asdict(report.config),
        "results": [asdict(r) for r in report.results],
        "overall_phi_iq": report.overall_phi_iq,
        "total_cycles": report.total_cycles,
        "duration_s": report.duration_s,
        "pass_criteria": report.pass_criteria,
    }
    if multi_seed is not None:
        data["multi_seed"] = multi_seed
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, indent=2, default=str))
    print(f"Report saved to {path}")


def run_multiseed(
    levels: List[int],
    config: BenchmarkConfig,
    n_seeds: int,
) -> Tuple[BenchmarkReport, Dict[str, Any]]:
    base_seed = config.seed
    overall_scores: List[float] = []
    per_level: Dict[int, List[float]] = {level: [] for level in levels}
    last_report: Optional[BenchmarkReport] = None

    for i, seed in enumerate(seed_sequence(base_seed, n_seeds)):
        run_config = BenchmarkConfig(
            n_cycles=config.n_cycles,
            warmup=config.warmup,
            seed=seed,
            grid_size=config.grid_size,
            use_continuous=config.use_continuous,
            use_mlp=config.use_mlp,
            diagnose_level=config.diagnose_level,
            dynamic_goals=config.dynamic_goals,
            dynamic_goals_every=config.dynamic_goals_every,
            action_slip=config.action_slip,
            weights=config.weights.copy(),
        )
        print(f"\n{'#'*60}\n  Seed {seed} ({i + 1}/{n_seeds})\n{'#'*60}")
        runner = BenchmarkRunner(run_config)
        report = runner.run_all(levels)
        last_report = report
        overall_scores.append(report.overall_phi_iq)
        for r in report.results:
            per_level[r.level].append(r.phi_iq)

    assert last_report is not None
    level_stats = {
        str(level): {
            "mean": float(np.mean(scores)),
            "std": float(np.std(scores)) if len(scores) > 1 else 0.0,
            "min": float(np.min(scores)),
            "max": float(np.max(scores)),
            "n": len(scores),
            "values": scores,
        }
        for level, scores in per_level.items()
        if scores
    }
    multi_seed = {
        "n_seeds": n_seeds,
        "base_seed": base_seed,
        "overall_phi_iq": {
            "mean": float(np.mean(overall_scores)),
            "std": float(np.std(overall_scores)) if len(overall_scores) > 1 else 0.0,
            "min": float(np.min(overall_scores)),
            "max": float(np.max(overall_scores)),
            "values": overall_scores,
        },
        "per_level_phi_iq": level_stats,
    }
    print(f"\n{'='*60}")
    print(f"  Multi-seed summary ({n_seeds} seeds, base={base_seed})")
    print(f"  Overall Φ-IQ: {multi_seed['overall_phi_iq']['mean']:.4f} "
          f"± {multi_seed['overall_phi_iq']['std']:.4f}")
    for level, stats in level_stats.items():
        print(f"  L{level}: {stats['mean']:.4f} ± {stats['std']:.4f}")
    print(f"{'='*60}\n")
    return last_report, multi_seed


def _run_mujoco(env_name: str, n_cycles: int, use_mlp: bool, output: str) -> dict:
    from phca.evaluation.runner import run_mujoco_smoke_report

    report = run_mujoco_smoke_report(env_name, n_cycles, use_mlp, seed=42)
    early = report["early_error"]
    late = report["late_error"]
    violations = report["violations"]
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
    ensure_logging()
    parser = argparse.ArgumentParser(description="PHCA Φ-IQ Benchmark Suite")
    parser.add_argument("--levels", type=str, default="0,1,2,3")
    parser.add_argument("--cycles", type=int, default=50)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--use-mlp", action="store_true")
    parser.add_argument("--env", type=str, default="gridworld",
                        choices=["gridworld", "cartpole", "pendulum", "reacher"])
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--diagnose-level", type=int, default=-1)
    parser.add_argument("--dynamic-goals", action="store_true")
    parser.add_argument("--dynamic-goals-every", type=int, default=100)
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--grid-size", type=int, default=5, choices=[5, 10, 20])
    parser.add_argument("--action-slip", type=float, default=0.0)
    parser.add_argument("--allow-fail", action="store_true",
                        help="Exit 0 even when pass criteria fail (validation scaling)")
    args = parser.parse_args()

    if args.quick:
        levels = [0]
        n_cycles = 20
    else:
        levels = [int(l.strip()) for l in args.levels.split(",")]
        n_cycles = args.cycles

    if args.env in ("cartpole", "pendulum", "reacher"):
        env_name = {
            "cartpole": "InvertedPendulum-v5",
            "pendulum": "Pendulum-v1",
            "reacher": "Reacher-v5",
        }[args.env]
        report = _run_mujoco(env_name, n_cycles, args.use_mlp, args.output)
        sys.exit(0 if report["no_errors"] else 1)

    config = BenchmarkConfig(
        n_cycles=n_cycles, use_mlp=args.use_mlp,
        diagnose_level=args.diagnose_level,
        dynamic_goals=args.dynamic_goals,
        dynamic_goals_every=args.dynamic_goals_every,
        grid_size=args.grid_size,
        action_slip=args.action_slip,
    )
    multi_seed_data: Optional[Dict[str, Any]] = None
    if args.seeds > 1:
        report, multi_seed_data = run_multiseed(levels, config, args.seeds)
        print_report(report)
    else:
        runner = BenchmarkRunner(config)
        report = runner.run_all(levels)
        print_report(report)

    output = args.output or ("logs/benchmark_multiseed.json" if args.seeds > 1
                             else "logs/benchmark_report.json")
    save_report(report, output, multi_seed=multi_seed_data)

    if not all(report.pass_criteria.values()) and not args.allow_fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
