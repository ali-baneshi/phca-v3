#!/usr/bin/env python3
"""
PHCA v3.0 — Closed-Loop Noise Robustness Benchmark.

Demonstrates the full resilience pipeline under a time-varying noise schedule:
  noise rises → ASI sanitizer catches failures → PEU error climbs
  → belief entropy increases → FallbackController triggers STOP
  → noise drops → system recovers

Shows this as per-cycle time-series data.

Usage:
    python scripts/benchmark_noise_closedloop.py --cycles=500 --output=closedloop.json
    python scripts/benchmark_noise_closedloop.py --quick  (200 cycles)
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


def make_noise_schedule(total_cycles: int) -> List[float]:
    """Generate a noise schedule that rises then falls.

    Pattern: 0→0.3→0.7→0.3→0 over 5 equal blocks.
    """
    block = total_cycles // 5
    schedule = []
    # Block 1: clean
    schedule.extend([0.0] * block)
    # Block 2: low noise
    schedule.extend([0.3] * block)
    # Block 3: high noise
    schedule.extend([0.7] * block)
    # Block 4: recovery (low)
    schedule.extend([0.3] * block)
    # Block 5: clean
    schedule.extend([0.0] * (total_cycles - len(schedule)))
    return schedule[:total_cycles]


def run_closedloop(
    cycles: int = 500,
    grid_size: int = 5,
    seed: int = 42,
    profile: str = "gaussian",
) -> Dict[str, Any]:
    env = GridWorld(size=grid_size, seed=seed)
    cycle = CognitiveCycle.build(env, seed=seed, use_mlp=True, mlp_lr=0.2)
    apply_grid_rbta_bounds(cycle)
    noise_schedule = make_noise_schedule(cycles)
    injector = NoiseInjector(
        sensor_dim=env.get_state_dim(),
        initial_profile=profile,
        initial_intensity=0.0,
        seed=seed + 1,
    )
    cycle._noise_injector = injector

    time_series: List[Dict[str, float]] = []

    for i in range(cycles):
        injector.set_intensity(noise_schedule[i])
        cycle.step()
        m = cycle.metrics_history[-1]
        time_series.append({
            "cycle": i,
            "noise_intensity": noise_schedule[i],
            "prediction_error": float(m.prediction_error),
            "prediction_confidence": float(m.prediction_confidence),
            "goal_reached": float(m.goal_reached),
            "emergency_active": float(m.emergency_active),
            "emergency_entropy": float(m.emergency_entropy),
            "failure_events": ",".join(m.failure_events),
            "rbta_action": m.rbta_action,
            "latency_ms": float(m.latency_ms),
        })

    total_goals = sum(t["goal_reached"] for t in time_series)
    emergencies = cycle._fallback_controller.total_emergencies
    trigger = cycle._fallback_controller.active_trigger()

    return {
        "config": {
            "cycles": cycles,
            "grid_size": grid_size,
            "profile": profile,
            "seed": seed,
            "noise_schedule": noise_schedule,
        },
        "summary": {
            "total_goals": int(total_goals),
            "emergencies": int(emergencies),
            "active_trigger": trigger,
            "mean_prediction_error": float(np.mean([t["prediction_error"] for t in time_series])),
            "mean_confidence": float(np.mean([t["prediction_confidence"] for t in time_series])),
        },
        "time_series": time_series,
        "duration_seconds": round(time.perf_counter() - 0, 2),
    }


def print_analysis(report: Dict[str, Any]) -> None:
    ts = report["time_series"]
    blocks = {
        "clean_1": [t for t in ts if t["noise_intensity"] == 0.0 and t["cycle"] < len(ts) // 2],
        "noisy": [t for t in ts if t["noise_intensity"] >= 0.5],
        "recovery": [t for t in ts if t["noise_intensity"] <= 0.3 and t["cycle"] > len(ts) * 3 // 5],
    }

    print(f"\n=== Closed-Loop Analysis ===")
    print(f"Profile: {report['config']['profile']}")
    print(f"Total emergencies: {report['summary']['emergencies']} (trigger: {report['summary']['active_trigger']})")
    print(f"Total goals: {report['summary']['total_goals']}")

    for label, block in blocks.items():
        if not block:
            continue
        err = np.mean([t["prediction_error"] for t in block])
        conf = np.mean([t["prediction_confidence"] for t in block])
        emer = sum(t["emergency_active"] for t in block)
        print(f"  {label:>10}: err={err:.4f} conf={conf:.4f} emergency_cycles={emer}")

    # Find emergency events
    emergencies_by_block = []
    in_emergency = False
    block_start = 0
    for t in ts:
        if t["emergency_active"] and not in_emergency:
            block_start = t["cycle"]
            in_emergency = True
        elif not t["emergency_active"] and in_emergency:
            emergencies_by_block.append((block_start, t["cycle"]))
            in_emergency = False
    if in_emergency:
        emergencies_by_block.append((block_start, ts[-1]["cycle"]))

    if emergencies_by_block:
        print(f"\nEmergency episodes ({len(emergencies_by_block)}):")
        for start, end in emergencies_by_block:
            noise_block = ts[start]["noise_intensity"]
            print(f"  cycles {start}–{end}  (noise={noise_block})")


def main():
    ensure_logging()
    parser = argparse.ArgumentParser(description="PHCA Closed-Loop Noise Robustness")
    parser.add_argument("--cycles", type=int, default=500)
    parser.add_argument("--grid-size", type=int, default=5)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--profile", type=str, default="gaussian",
                        choices=["gaussian", "dropout", "drift", "salt_pepper"])
    args = parser.parse_args()

    cycles = 200 if args.quick else args.cycles
    t_start = time.perf_counter()
    report = run_closedloop(
        cycles=cycles,
        grid_size=args.grid_size,
        profile=args.profile,
    )
    report["duration_seconds"] = round(time.perf_counter() - t_start, 2)

    print_analysis(report)

    if args.output:
        path = Path(args.output)
        path.write_text(json.dumps(report, indent=2))
        print(f"\nReport saved to {path.resolve()}")
    else:
        print(f"\n(duration: {report['duration_seconds']}s)")


if __name__ == "__main__":
    main()
