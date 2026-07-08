#!/usr/bin/env python3
"""
PHCA v3.0 — Noise Robustness Benchmark (Grounding Simulation).

Demonstrates cognitive cycle robustness under increasing sensor noise.
Runs the agent in GridWorld with configurable noise injection, measuring
prediction error, confidence, and goal rate across noise intensities.

Usage:
    python scripts/benchmark_noise_robustness.py --cycles=200 --output=noise_report.json
    python scripts/benchmark_noise_robustness.py --quick  (50 cycles)
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

import _bootstrap  # noqa: F401

from phca.core.cycle import CognitiveCycle
from phca.world_model.mlp import apply_grid_rbta_bounds
from phca.asi.noise_injector import NoiseInjector, NoiseProfile
from phca.environments.grid_world import GridWorld
from phca.logging import ensure_logging


NOISE_PROFILES = ["gaussian", "dropout", "drift", "salt_pepper"]
INTENSITIES = np.linspace(0.0, 0.9, 10).tolist()


def run_benchmark(
    cycles: int = 100,
    grid_size: int = 5,
    profile: str = "gaussian",
    intensity: float = 0.0,
    seed: int = 42,
) -> Dict[str, Any]:
    env = GridWorld(size=grid_size, seed=seed)
    cycle = CognitiveCycle.build(env, seed=seed, use_mlp=True, mlp_lr=0.2)
    cycle._noise_injector = NoiseInjector(
        sensor_dim=env.get_state_dim(),
        initial_profile=profile,
        initial_intensity=intensity,
        seed=seed + 1,
    )
    apply_grid_rbta_bounds(cycle)
    result = cycle.run(n_cycles=cycles)
    result["intensity"] = float(intensity)
    result["profile"] = profile
    result["grid_size"] = grid_size
    result["cycles"] = cycles
    result["emergencies"] = cycle._fallback_controller.total_emergencies
    return result


def run_sweep(
    cycles: int = 100,
    grid_size: int = 5,
    profile: str = "gaussian",
    seeds: int = 3,
) -> List[Dict[str, Any]]:
    results = []
    for intensity in INTENSITIES:
        for s in range(seeds):
            r = run_benchmark(
                cycles=cycles, grid_size=grid_size,
                profile=profile, intensity=intensity,
                seed=s * 100 + 42,
            )
            results.append(r)
            print(f"  {profile} @ intensity={intensity:.1f} seed={s}: "
                  f"err={r['avg_prediction_error']:.4f} "
                  f"conf={r.get('median_prediction_error', 0):.4f} "
                  f"goals={r['goals_reached']} "
                  f"emergencies={r['emergencies']}")
    return results


def summarize(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_intensity: Dict[float, List[Dict[str, Any]]] = {}
    for r in results:
        i = r["intensity"]
        by_intensity.setdefault(i, []).append(r)

    summary = {}
    for i, group in sorted(by_intensity.items()):
        errors = [g["avg_prediction_error"] for g in group]
        goals = [g["goals_reached"] for g in group]
        emergencies = [g["emergencies"] for g in group]
        summary[f"intensity_{i:.1f}"] = {
            "avg_prediction_error_mean": float(np.mean(errors)),
            "avg_prediction_error_std": float(np.std(errors)),
            "goals_reached_mean": float(np.mean(goals)),
            "goals_reached_std": float(np.std(goals)),
            "emergencies_mean": float(np.mean(emergencies)),
            "emergencies_total": int(sum(emergencies)),
        }
    return summary


def main():
    ensure_logging()
    parser = argparse.ArgumentParser(description="PHCA Noise Robustness Benchmark")
    parser.add_argument("--cycles", type=int, default=100)
    parser.add_argument("--grid-size", type=int, default=5)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    cycles = 50 if args.quick else args.cycles
    grid_size = args.grid_size

    t_start = time.perf_counter()
    all_results: List[Dict[str, Any]] = []

    for profile in NOISE_PROFILES:
        print(f"\n=== Noise Profile: {profile} ===")
        results = run_sweep(cycles=cycles, grid_size=grid_size, profile=profile, seeds=3)
        all_results.extend(results)

    duration = time.perf_counter() - t_start
    summary = summarize(all_results)

    report = {
        "config": {
            "cycles": cycles,
            "grid_size": grid_size,
            "profiles": NOISE_PROFILES,
            "intensities": INTENSITIES,
        },
        "duration_seconds": round(duration, 2),
        "summary": summary,
    }

    print(f"\n=== Summary ({duration:.1f}s) ===")
    for key, val in sorted(report["summary"].items()):
        print(f"  {key}: err={val['avg_prediction_error_mean']:.4f} "
              f"goals={val['goals_reached_mean']:.1f} "
              f"emergencies={val['emergencies_total']}")

    if args.output:
        path = Path(args.output)
        path.write_text(json.dumps(report, indent=2))
        print(f"\nReport saved to {path.resolve()}")
    else:
        print(f"\n=== Full Report ({duration:.1f}s) ===")
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
