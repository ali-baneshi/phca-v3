#!/usr/bin/env python3
"""PHCA v3.0 Phase 5 — per-module-timings probe for the MLP path.

Measures `gprime_learn` (and the full cycle) on the MLP Level-2 config
(the most stressful path, mirroring `longrun_probe.py`) over N cycles,
discarding the warm-up cycles before the replay buffer fills at
~`batch_size` (=64) cycles. Reports steady-state mean/p95/max.

This is the Phase-5 perf target: `gprime_learn` is ~95% of cycle time
(D-091 profile). The success criterion is a >=20% reduction to <=25.4ms
mean with Φ-IQ >= 0.73 and the gate PASS.

Read-only diagnostic; does not modify any production code.

Usage:
    MUJOCO_GL=disabled PYTHONPATH=python python scripts/profile_mlp_learn.py --cycles=200
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from phca.core.cycle import CognitiveCycle
from phca.logging import ensure_logging
from phca.world_model.mlp import gprime_stress_bounds


def _build_l2_cycle(seed: int = 42, train_steps: int | None = None,
                    batch_size: int | None = None) -> CognitiveCycle:
    """Build the Level-2 MLP cycle (mirrors longrun_probe.py config)."""
    rng = np.random.RandomState(seed + 2)
    obstacles = []
    gap_row = rng.randint(0, 5)
    for r in range(5):
        if r != gap_row:
            obstacles.append((r, 2))
    for _ in range(2):
        r, c = rng.randint(0, 5, size=2)
        if (r, c) not in obstacles:
            obstacles.append((r, c))

    cycle = CognitiveCycle.build_for_env(
        size=5, seed=seed + 2, use_continuous=True, use_mlp=True,
        obstacles=obstacles,
    )
    cycle.rbta.update_bounds("G'", gprime_stress_bounds(cycle))
    # Allow overriding the perf levers for re-measurement after a change.
    if train_steps is not None and hasattr(cycle.gprime, "train_steps"):
        cycle.gprime.train_steps = train_steps
    if batch_size is not None and hasattr(cycle.gprime, "batch_size"):
        cycle.gprime.batch_size = batch_size
    return cycle


def main() -> None:
    ensure_logging()
    parser = argparse.ArgumentParser(description="Profile MLP gprime_learn (Phase 5 perf target)")
    parser.add_argument("--cycles", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup", type=int, default=64,
                        help="Cycles to discard (replay buffer fills at ~batch_size=64)")
    parser.add_argument("--train-steps", type=int, default=None,
                        help="Override MLP train_steps (perf lever)")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override MLP batch_size (perf lever)")
    parser.add_argument("--output", default="logs/profile_mlp_learn.json")
    args = parser.parse_args()

    cycle = _build_l2_cycle(args.seed, args.train_steps, args.batch_size)

    gprime_learn, full_lat = [], []
    for i in range(args.cycles):
        m = cycle.step()
        if i < args.warmup:
            continue
        gprime_learn.append(m.module_timings.get("gprime_learn", 0.0))
        full_lat.append(m.latency_ms)

    g = np.array(gprime_learn)
    f = np.array(full_lat)
    summary = {
        "cycles": args.cycles,
        "warmup_discarded": args.warmup,
        "measured_cycles": len(g),
        "train_steps": getattr(cycle.gprime, "train_steps", None),
        "batch_size": getattr(cycle.gprime, "batch_size", None),
        "gprime_learn_mean_ms": float(g.mean()),
        "gprime_learn_p95_ms": float(np.percentile(g, 95)),
        "gprime_learn_max_ms": float(g.max()),
        "cycle_mean_ms": float(f.mean()),
        "cycle_p95_ms": float(np.percentile(f, 95)),
        "a1_p95_under_500ms": bool(np.percentile(f, 95) < 500.0),
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(summary, indent=2))

    print(f"\n=== MLP gprime_learn profile ({args.cycles} cyc, discard {args.warmup}) ===")
    print(f"  train_steps={summary['train_steps']}  batch_size={summary['batch_size']}")
    print(f"  gprime_learn  mean={summary['gprime_learn_mean_ms']:.2f}ms  "
          f"p95={summary['gprime_learn_p95_ms']:.2f}ms  max={summary['gprime_learn_max_ms']:.2f}ms")
    print(f"  full cycle    mean={summary['cycle_mean_ms']:.2f}ms  "
          f"p95={summary['cycle_p95_ms']:.2f}ms")
    print(f"  A1 (p95<500ms): {'PASS' if summary['a1_p95_under_500ms'] else 'FAIL'}")
    print(f"  Target (Phase 5): gprime_learn mean <= 25.4ms (20% off baseline ~31.8ms)")
    print(f"  Saved to {args.output}")
    sys.exit(0)


if __name__ == "__main__":
    main()
