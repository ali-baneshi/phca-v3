#!/usr/bin/env python3
"""
PHCA v3.0 — Cognitive Cycle Profiler.

Measures per-cycle latency for implemented Phase 3.1 components.
Full cycle profiling targets PHCA-3.1-011 (cognitive cycle orchestrator).

Usage:
    python scripts/profile_cycle.py --cycles=100 --max-ms=500
"""

import time
import json
import argparse
import statistics
from pathlib import Path


def profile_asi_sanitizer(n_cycles: int = 100) -> dict:
    """Profile the ASI sanitizer in isolation."""
    import numpy as np
    from phca.asi.sanitizer import ASISanitizer

    sanitizer = ASISanitizer(sensor_dim=64, v_max=100.0)
    timings = []
    for _ in range(n_cycles):
        raw = np.random.randn(64).astype(np.float32)
        t0 = time.perf_counter_ns()
        sanitizer.sanitize(raw)
        t1 = time.perf_counter_ns()
        timings.append((t1 - t0) / 1e6)  # ms

    return {
        "component": "ASI_Sanitizer",
        "n_cycles": n_cycles,
        "mean_ms": statistics.mean(timings),
        "median_ms": statistics.median(timings),
        "p95_ms": sorted(timings)[int(len(timings) * 0.95)],
        "max_ms": max(timings),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Profile PHCA cognitive cycle")
    parser.add_argument("--cycles", type=int, default=100, help="Number of cycles to profile")
    parser.add_argument("--max-ms", type=float, default=500, help="Maximum acceptable latency (ms)")
    parser.add_argument("--output", type=str, default=None, help="Output file")
    args = parser.parse_args()

    results = {"profiles": []}

    # Profile ASI sanitizer
    asi_profile = profile_asi_sanitizer(args.cycles)
    results["profiles"].append(asi_profile)

    thresholds = {
        "ASI_Sanitizer": {"mean_ms": 0.01, "p95_ms": 0.02},  # target < 10μs mean
    }

    print(f"=== PHCA Cycle Profiler ({args.cycles} cycles) ===")
    for p in results["profiles"]:
        threshold = thresholds.get(p["component"], {})
        status = "✅" if p["mean_ms"] <= threshold.get("mean_ms", float("inf")) else "⚠️"
        print(f"  {status} {p['component']}: median={p['median_ms']:.4f}ms, "
              f"p95={p['p95_ms']:.4f}ms, max={p['max_ms']:.4f}ms")

    print(f"\nThreshold: {args.max_ms}ms")
    overall_status = all(
        p["mean_ms"] <= thresholds.get(p["component"], {}).get("mean_ms", float("inf"))
        for p in results["profiles"]
    )
    print(f"Overall: {'✅ PASS' if overall_status else '⚠️ SOME COMPONENTS ABOVE THRESHOLD'}")

    results["overall_pass"] = overall_status
    results["threshold_ms"] = args.max_ms

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(results, indent=2))
        print(f"Results written to {args.output}")
