#!/usr/bin/env python3
"""MuJoCo integration benchmark for the PHCA cognitive cycle.

Usage:
    # Run with MLP world model (recommended for continuous MuJoCo obs)
    python scripts/benchmark_mujoco.py --env=InvertedPendulum-v5 --cycles=100

    # Run with Gaussian continuous G'
    python scripts/benchmark_mujoco.py --env=Pendulum-v1 --cycles=100 --no-mlp

    # Quick sanity check
    python scripts/benchmark_mujoco.py --env=InvertedPendulum-v5 --cycles=10 --quick

Cross-ref: docs/mujoco_integration_plan.md §5.3
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

import _bootstrap  # noqa: F401

from phca.core.cycle import CognitiveCycle


def run_benchmark(
    env_name: str,
    n_cycles: int = 100,
    use_mlp: bool = True,
    seed: int = 42,
) -> dict:
    """Run the MuJoCo integration benchmark.

    Args:
        env_name: gymnasium MuJoCo environment ID.
        n_cycles: Number of cognitive cycles to run.
        use_mlp: If True, use MLP world model; else continuous Gaussian G'.
        seed: Random seed.

    Returns:
        Dict of benchmark metrics.
    """
    cycle = CognitiveCycle.build_for_mujoco(
        env_name=env_name, seed=seed, use_mlp=use_mlp,
    )

    # Warmup — let the world model stabilise
    for _ in range(10):
        cycle.step()

    # Benchmark phase
    errors: list[float] = []
    latencies: list[float] = []
    violations_count: int = 0
    terminal_count: int = 0
    start_time = time.perf_counter()

    for _ in range(n_cycles):
        metrics = cycle.step()
        latencies.append(metrics.latency_ms)
        errors.append(metrics.prediction_error)
        violations_count += metrics.violations_count

        # Count how many times the environment resets
        if metrics.goal_reached:
            terminal_count += 1

    elapsed = time.perf_counter() - start_time

    # Compute pass criteria
    all_finite = all(np.isfinite(e) for e in errors)
    mean_error = float(np.mean(errors))
    mean_latency = float(np.mean(latencies))
    p95_latency = float(np.percentile(latencies, 95))
    max_latency = float(np.max(latencies))

    # C4: Prediction improves? Check error trend
    if len(errors) >= 10:
        early = float(np.mean(errors[: len(errors) // 2]))
        late = float(np.mean(errors[len(errors) // 2 :]))
        error_improves = late < early
    else:
        early = 0.0
        late = 0.0
        error_improves = False

    pass_criteria = {
        "C1_no_errors": all_finite,
        "C2_latency_under_200ms": mean_latency < 200.0,
        "C3_latency_under_5000ms": max_latency < 5000.0,
        "C4_error_improves": error_improves or mean_error < 1.0,
        "C5_violations_under_10pct": violations_count / max(n_cycles, 1) < 0.1,
    }

    return {
        "env_name": env_name,
        "n_cycles": n_cycles,
        "model": "MLP" if use_mlp else "Gaussian G'",
        "state_dim": cycle.state_dim,
        "action_space_size": cycle.env.action_space_size,
        # Error metrics
        "mean_prediction_error": mean_error,
        "median_prediction_error": float(np.median(errors)),
        "min_prediction_error": float(np.min(errors)),
        "max_prediction_error": float(np.max(errors)),
        "early_error_mean": early,
        "late_error_mean": late,
        "prediction_improves": error_improves,
        # Latency metrics
        "mean_latency_ms": mean_latency,
        "p95_latency_ms": p95_latency,
        "max_latency_ms": max_latency,
        "cycles_per_second": n_cycles / elapsed if elapsed > 0 else 0.0,
        "total_time_s": elapsed,
        # RBTA / health
        "total_rbta_violations": violations_count,
        "terminal_episodes": terminal_count,
        # Pass/fail
        "all_cycles_finite": all_finite,
        "pass_criteria": pass_criteria,
        "all_pass": all(pass_criteria.values()),
    }


def print_benchmark_report(result: dict) -> None:
    """Print a human-readable benchmark report."""
    print(f"\n{'=' * 60}")
    print(f"  MuJoCo Benchmark — {result['env_name']}")
    print(f"  Model: {result['model']}  |  "
          f"State dim: {result['state_dim']}  |  "
          f"Actions: {result['action_space_size']}")
    print(f"{'=' * 60}")

    print(f"\n  Prediction Error:")
    print(f"    Mean:   {result['mean_prediction_error']:.4f}")
    print(f"    Median: {result['median_prediction_error']:.4f}")
    print(f"    Range:  [{result['min_prediction_error']:.4f}, "
          f"{result['max_prediction_error']:.4f}]")
    if result["early_error_mean"] and result["late_error_mean"]:
        direction = "↓ improving" if result["prediction_improves"] else "↑ worsening"
        print(f"    Trend:  Early={result['early_error_mean']:.4f} → "
              f"Late={result['late_error_mean']:.4f} ({direction})")

    print(f"\n  Latency:")
    print(f"    Mean:   {result['mean_latency_ms']:.1f} ms")
    print(f"    P95:    {result['p95_latency_ms']:.1f} ms")
    print(f"    Max:    {result['max_latency_ms']:.1f} ms")
    print(f"    Rate:   {result['cycles_per_second']:.1f} cycles/s")

    print(f"\n  Health:")
    print(f"    Cycles:       {result['n_cycles']}")
    print(f"    RBTA violations: {result['total_rbta_violations']}")
    print(f"    Terminal episodes: {result['terminal_episodes']}")
    print(f"    All finite:   {result['all_cycles_finite']}")

    print(f"\n  Pass Criteria:")
    for criterion, passed in result["pass_criteria"].items():
        status = "✓" if passed else "✗"
        print(f"    [{status}] {criterion}")

    print(f"\n  {'✓ ALL PASS' if result['all_pass'] else '✗ SOME FAILED'}")
    print(f"{'=' * 60}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MuJoCo PHCA Integration Benchmark",
    )
    parser.add_argument(
        "--env", default="InvertedPendulum-v5",
        choices=["InvertedPendulum-v5", "Pendulum-v1", "Reacher-v5"],
        help="gymnasium MuJoCo environment ID",
    )
    parser.add_argument(
        "--cycles", type=int, default=100,
        help="Number of cognitive cycles to benchmark",
    )
    parser.add_argument(
        "--no-mlp", dest="use_mlp", action="store_false",
        help="Use Gaussian G' instead of MLP world model",
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick mode: 10 cycles, no output file",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output JSON file path",
    )
    parser.set_defaults(use_mlp=True)

    args = parser.parse_args()

    if args.quick:
        args.cycles = 10
        args.output = None

    print(f"Benchmarking {args.env} with "
          f"{'MLP' if args.use_mlp else 'Gaussian G''} "
          f"({args.cycles} cycles)...")

    result = run_benchmark(
        env_name=args.env,
        n_cycles=args.cycles,
        use_mlp=args.use_mlp,
    )

    print_benchmark_report(result)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2, default=str))
        print(f"Report saved to {args.output}")

    # Exit code: 0 if all pass, 1 if any fail
    if not result["all_pass"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
