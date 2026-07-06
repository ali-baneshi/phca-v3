#!/usr/bin/env python3
"""
PHCA v3.0 - Cognitive Cycle Profiler.

Measures per-cycle latency for the full Phase 3.1 cognitive cycle.
Supports both per-component profiling and full end-to-end cycle profiling.

Usage:
    python scripts/profile_cycle.py --cycles=100 --max-ms=500
    python scripts/profile_cycle.py --cycles=100 --max-ms=500 --output=logs/profile.json
"""

import time
import json
import argparse
import statistics
from pathlib import Path

import _bootstrap  # noqa: F401

from phca.core.cycle import CognitiveCycle


def profile_full_cycle(n_cycles: int = 100) -> dict:
    """Profile the full cognitive cycle end-to-end.

    Args:
        n_cycles: Number of cycles to profile (must be >= 1).

    Returns:
        Dict with latency stats and summary metrics.
    """
    if n_cycles < 1:
        return {"component": "Full_Cognitive_Cycle", "n_cycles": 0,
                "mean_ms": 0.0, "median_ms": 0.0, "p95_ms": 0.0,
                "p99_ms": 0.0, "max_ms": 0.0, "min_ms": 0.0}

    cycle = CognitiveCycle.build_for_env(size=5, seed=42)

    # Warmup: 5 cycles (discard metrics, stabilize JIT/caching)
    for _ in range(5):
        cycle.step()

    timings = []
    for _ in range(n_cycles):
        t0 = time.perf_counter()
        metrics = cycle.step()
        t1 = time.perf_counter()
        timings.append((t1 - t0) * 1000)  # ms

    return {
        "component": "Full_Cognitive_Cycle",
        "n_cycles": n_cycles,
        "mean_ms": statistics.mean(timings),
        "median_ms": statistics.median(timings),
        "p95_ms": sorted(timings)[int(len(timings) * 0.95)],
        "p99_ms": sorted(timings)[int(len(timings) * 0.99)],
        "max_ms": max(timings),
        "min_ms": min(timings),
        "errors_mean": float(statistics.mean([m.prediction_error for m in cycle.metrics_history[-n_cycles:]])),
        "violations": sum(m.violations_count for m in cycle.metrics_history[-n_cycles:]),
        "actions_taken": [m.action_name for m in cycle.metrics_history[-n_cycles:]],
    }


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
    parser.add_argument("--output", type=str, default=None, help="Output file path")
    args = parser.parse_args()

    results = {"profiles": [], "overall_pass": True}

    # Profile ASI sanitizer
    asi_profile = profile_asi_sanitizer(args.cycles)
    results["profiles"].append(asi_profile)

    # Profile full cognitive cycle
    cycle_profile = profile_full_cycle(args.cycles)
    results["profiles"].append(cycle_profile)

    thresholds = {
        "ASI_Sanitizer": {"mean_ms": 0.01, "p95_ms": 0.02},
        "Full_Cognitive_Cycle": {"mean_ms": args.max_ms, "p95_ms": args.max_ms * 2},
    }

    print(f"\n=== PHCA Cycle Profiler ({args.cycles} cycles) ===")
    for p in results["profiles"]:
        threshold = thresholds.get(p["component"], {})
        mean_ok = p["mean_ms"] <= threshold.get("mean_ms", float("inf"))
        p95_ok = p["p95_ms"] <= threshold.get("p95_ms", float("inf"))
        status = "PASS" if (mean_ok and p95_ok) else "FAIL"
        if status == "FAIL":
            results["overall_pass"] = False

        print(f"  [{status}] {p['component']}")
        print(f"         mean={p['mean_ms']:.4f}ms  median={p['median_ms']:.4f}ms")
        print(f"         p95={p['p95_ms']:.4f}ms  max={p['max_ms']:.4f}ms")
        if "errors_mean" in p:
            print(f"         errors={p['errors_mean']:.6f}  violations={p['violations']}")

    print(f"\nThreshold: {args.max_ms}ms max median")
    print(f"Overall: {'PASS' if results['overall_pass'] else 'FAIL'}")
    print()

    results["threshold_ms"] = args.max_ms

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(results, indent=2))
        print(f"Results written to {args.output}")
