"""PHCA v3.0 — Sync vs Async benchmark (Phase B).

Runs 500 cycles each in sync and async mode, captures:
  - Average latency per cycle (ms)
  - Learning step frequency (G'.learn calls / total cycles)
  - RBTA violations per 1000 cycles
  - Task success rate (goal_reached ratio)

Output: benchmarks/async_vs_sync_validation.json
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from phca.core.cycle import CognitiveCycle

N_CYCLES = 500
OUT = Path(__file__).resolve().parents[1] / "benchmarks" / "async_vs_sync_validation.json"


def _capture(cycle: CognitiveCycle, label: str) -> dict:
    hist = cycle.metrics_history
    if not hist:
        return {"mode": label, "error": "no_metrics"}

    latencies = [m.latency_ms for m in hist]
    violations = sum(m.violations_count for m in hist)
    goals = sum(1 for m in hist if m.goal_reached)

    # Count G'.learn calls from module_timings
    learn_calls = sum(
        1 for m in hist if m.module_timings.get("gprime_learn", 0) > 0
    )

    return {
        "mode": label,
        "n_cycles": len(hist),
        "avg_latency_ms": round(float(sum(latencies) / len(latencies)), 3),
        "median_latency_ms": round(float(sorted(latencies)[len(latencies) // 2]), 3),
        "p95_latency_ms": round(float(sorted(latencies)[int(len(latencies) * 0.95)]), 3),
        "rbta_violations_per_1k": round(violations / max(len(hist), 1) * 1000, 3),
        "goal_success_rate": round(goals / max(len(hist), 1), 4),
        "learn_call_ratio": round(learn_calls / max(len(hist), 1), 4),
        "total_violations": violations,
        "goals_reached": goals,
    }


def benchmark_sync() -> dict:
    print("  Building sync cycle ...", end=" ", flush=True)
    cycle = CognitiveCycle.build_for_env(size=5, use_mlp=True)
    print("done.")
    print(f"  Running {N_CYCLES} sync cycles ...", end=" ", flush=True)
    t0 = time.perf_counter()
    cycle.run(N_CYCLES)
    elapsed = time.perf_counter() - t0
    print(f"done ({elapsed:.2f}s).")
    return _capture(cycle, "sync")


def benchmark_async() -> dict:
    print("  Building async cycle ...", end=" ", flush=True)
    cycle = CognitiveCycle.build_for_env(size=5, use_mlp=True)
    print("done.")
    print(f"  Running {N_CYCLES} async cycles ...", end=" ", flush=True)
    t0 = time.perf_counter()
    cycle._async_mode = True
    cycle.run(N_CYCLES)
    elapsed = time.perf_counter() - t0
    print(f"done ({elapsed:.2f}s).")
    return _capture(cycle, "async")


if __name__ == "__main__":
    OUT.parent.mkdir(parents=True, exist_ok=True)
    print("=== PHCA Sync vs Async Benchmark ===")
    print()

    sync_result = benchmark_sync()
    print(f"  Sync:  avg_latency={sync_result['avg_latency_ms']}ms, "
          f"goals={sync_result['goal_success_rate']:.2%}, "
          f"violations/1k={sync_result['rbta_violations_per_1k']:.2f}")

    async_result = benchmark_async()
    print(f"  Async: avg_latency={async_result['avg_latency_ms']}ms, "
          f"goals={async_result['goal_success_rate']:.2%}, "
          f"violations/1k={async_result['rbta_violations_per_1k']:.2f}")

    summary = {
        "benchmark": "async_vs_sync",
        "n_cycles": N_CYCLES,
        "date": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "sync": sync_result,
        "async": async_result,
    }
    OUT.write_text(json.dumps(summary, indent=2))
    print(f"\nResults written to {OUT}")
    print("Done.")
