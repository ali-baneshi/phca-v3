#!/usr/bin/env python3
"""Phase 4 W1 — long-duration stability probe.

Runs the MLP cognitive cycle (Level 2 config, the most stressful path) for
N cycles and reports per-block latency creep + RSS memory growth to detect
leaks. Read-only diagnostic; does not modify any existing file.

Usage:
    MUJOCO_GL=disabled PYTHONPATH=python python scripts/longrun_probe.py --cycles=1000
"""
from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np

from phca.core.cycle import CognitiveCycle
from phca.logging import ensure_logging
from phca.world_model.mlp import gprime_stress_bounds


def rss_kb() -> float:
    """Current process RSS in KiB (Linux ru_maxrss is KiB; ru_ixrss is ticks)."""
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def main() -> None:
    ensure_logging()
    parser = argparse.ArgumentParser(description="PHCA long-duration stability probe")
    parser.add_argument("--cycles", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--block", type=int, default=100, help="cycles per measurement block")
    parser.add_argument("--output", default="logs/longrun_probe.json")
    args = parser.parse_args()

    # Level-2 config: 5x5 grid with obstacles (mirrors benchmark L2).
    rng = np.random.RandomState(args.seed + 2)
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
        size=5, seed=args.seed + 2, use_continuous=True, use_mlp=True, obstacles=obstacles,
    )
    cycle.rbta.update_bounds("G'", gprime_stress_bounds(cycle))

    rss_start = rss_kb()
    blocks = []
    block_latencies: list[float] = []
    t0 = time.perf_counter()
    for i in range(args.cycles):
        m = cycle.step()
        block_latencies.append(m.latency_ms)
        if (i + 1) % args.block == 0:
            arr = np.array(block_latencies)
            blocks.append({
                "block": (i + 1) // args.block,
                "cycles": f"{i+1-args.block+1}-{i+1}",
                "mean_ms": float(arr.mean()),
                "p95_ms": float(np.percentile(arr, 95)),
                "max_ms": float(arr.max()),
                "violations": int(m.violations_count) if False else None,
            })
            block_latencies = []
    elapsed = time.perf_counter() - t0
    rss_end = rss_kb()

    means = [b["mean_ms"] for b in blocks]
    p95s = [b["p95_ms"] for b in blocks]
    creep_mean = (means[-1] - means[0]) / max(means[0], 1e-6)
    creep_p95 = (p95s[-1] - p95s[0]) / max(p95s[0], 1e-6)

    summary = {
        "cycles": args.cycles,
        "block_size": args.block,
        "elapsed_s": elapsed,
        "cycles_per_s": args.cycles / elapsed,
        "rss_start_kb": rss_start,
        "rss_end_kb": rss_end,
        "rss_growth_kb": rss_end - rss_start,
        "rss_growth_pct": 100.0 * (rss_end - rss_start) / max(rss_start, 1.0),
        "mean_latency_creep_pct": 100.0 * creep_mean,
        "p95_latency_creep_pct": 100.0 * creep_p95,
        "first_block_mean_ms": means[0],
        "last_block_mean_ms": means[-1],
        "first_block_p95_ms": p95s[0],
        "last_block_p95_ms": p95s[-1],
        "overall_p95_ms": float(np.percentile([b["p95_ms"] for b in blocks], 95)),
        "blocks": blocks,
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(summary, indent=2))
    print(f"\n=== Long-run probe ({args.cycles} cycles, block={args.block}) ===")
    print(f"Elapsed: {elapsed:.1f}s ({summary['cycles_per_s']:.1f} cyc/s)")
    print(f"RSS: {rss_start:.0f} -> {rss_end:.0f} KiB (+{summary['rss_growth_pct']:.2f}%)")
    print(f"Mean latency: {means[0]:.2f} -> {means[-1]:.2f} ms (creep {summary['mean_latency_creep_pct']:+.2f}%)")
    print(f"p95 latency:  {p95s[0]:.2f} -> {p95s[-1]:.2f} ms (creep {summary['p95_latency_creep_pct']:+.2f}%)")
    print(f"Per-block mean (ms): " + " ".join(f"{m:.1f}" for m in means))
    print(f"Saved to {args.output}")

    # Acceptance signal: p95 < 500 throughout, no large monotonic creep, modest RSS growth
    ok = (summary["last_block_p95_ms"] < 500.0
          and abs(summary["p95_latency_creep_pct"]) < 50.0
          and summary["rss_growth_pct"] < 50.0)
    print(f"ACCEPTANCE: {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
