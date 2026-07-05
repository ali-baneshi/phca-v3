#!/usr/bin/env python3
"""Long-horizon emergence runs with checkpoint/resume support."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from phca.config import ResourceBounds
from phca.core.cycle import CognitiveCycle
from phca.evaluation.metrics.emergence import compute_emergence_bundle
from phca.evaluation.metrics.phi_iq import generate_goal_pursuit_obstacles
from phca.evaluation.metrics.synergy import synergy_score
from phca.evaluation.trace import TraceCollector
from phca.logging import ensure_logging


def _rolling_windows(trace, window: int = 1000) -> List[Dict[str, float]]:
    records = trace.snapshot()
    if len(records) < window:
        return [compute_emergence_bundle(records)]
    out = []
    for i in range(0, len(records) - window + 1, window):
        chunk = records[i : i + window]
        bundle = compute_emergence_bundle(chunk)
        bundle["synergy"] = synergy_score(chunk)
        bundle["window_start"] = i
        out.append(bundle)
    return out


def save_checkpoint(path: str, cycle: CognitiveCycle, trace: TraceCollector, cycle_count: int) -> None:
    ckpt = {
        "cycle_count": cycle_count,
        "rng_state": np.random.get_state(),
        "trace_records": trace.snapshot(),
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_bytes(pickle.dumps(ckpt))


def load_checkpoint(path: str) -> Dict[str, Any]:
    return pickle.loads(Path(path).read_bytes())


def main() -> None:
    ensure_logging()
    parser = argparse.ArgumentParser(description="PHCA long-horizon runner")
    parser.add_argument("--cycles", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--grid-size", type=int, default=5, choices=[5, 10, 20])
    parser.add_argument("--checkpoint-every", type=int, default=1000)
    parser.add_argument("--checkpoint", default=None, help="Resume from checkpoint path")
    parser.add_argument("--output", default="logs/horizon_run.json")
    parser.add_argument("--window", type=int, default=1000)
    args = parser.parse_args()

    trace = TraceCollector(capacity=max(args.cycles + 100, 5000))
    start_cycle = 0

    obstacles = generate_goal_pursuit_obstacles(args.seed + 2, args.grid_size)
    cycle = CognitiveCycle.build_for_env(
        size=args.grid_size, seed=args.seed + 2,
        use_continuous=True, use_mlp=True, obstacles=obstacles,
        trace_collector=trace,
    )
    cycle.rbta.update_bounds(
        "G'", ResourceBounds(B_time=0.080, B_mem=500_000, B_energy=50.0),
    )

    if args.checkpoint and Path(args.checkpoint).exists():
        ckpt = load_checkpoint(args.checkpoint)
        start_cycle = ckpt["cycle_count"]
        np.random.set_state(ckpt["rng_state"])
        for rec in ckpt.get("trace_records", []):
            trace.record(
                cycle_id=rec.cycle_id, action=rec.action, reward=rec.reward,
                prediction_error=rec.prediction_error,
                prediction_confidence=rec.prediction_confidence,
                goal_drive=rec.goal_drive, rbta_action=rec.rbta_action,
                violations=rec.violations, latency_ms=rec.latency_ms,
                goal_reached=rec.goal_reached, module_timings=rec.module_timings,
            )
        cycle.cycle_count = start_cycle
        print(f"Resumed from checkpoint at cycle {start_cycle}")

    t0 = time.perf_counter()
    ckpt_path = Path(args.output).with_suffix(".ckpt")
    for i in range(start_cycle, args.cycles):
        cycle.step()
        if args.checkpoint_every and (i + 1) % args.checkpoint_every == 0:
            save_checkpoint(str(ckpt_path), cycle, trace, i + 1)

    duration = time.perf_counter() - t0
    windows = _rolling_windows(trace, args.window)
    final_emergence = compute_emergence_bundle(trace.snapshot())
    final_emergence["synergy"] = synergy_score(trace.snapshot())

    report = {
        "cycles": args.cycles,
        "seed": args.seed,
        "grid_size": args.grid_size,
        "duration_s": duration,
        "final_emergence": final_emergence,
        "rolling_windows": windows,
        "checkpoint": str(ckpt_path) if ckpt_path.exists() else None,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2, default=str))
    print(f"Horizon run complete: {args.cycles} cycles in {duration:.1f}s → {args.output}")
    print(f"  emergence_composite={final_emergence.get('emergence_composite', 0):.4f}")


if __name__ == "__main__":
    main()
