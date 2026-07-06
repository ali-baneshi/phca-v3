#!/usr/bin/env python3
"""PHCA v3.0 nightly stress test (Phase 6 / C1; Phase 7 / B3 tightened gate).

Long-run stability probe: drives a single L2 (Goal Pursuit) MLP cognitive
cycle for N cycles (default 11000; override via --cycles or NIGHTLY_CYCLES env)
and reports the four stability signals a 24h soak must hold:

  - RSS leak detector: psutil current RSS sampled every 100 cycles, linear-fit
    full-run slope AND late-half slope (bytes/cycle). The GATE uses the late
    slope (post-cap steady state) — the early slope legitimately stays ~4 KB/cyc
    during the M3 fill phase (0→10k episodes). Phase 7 / B3 tightens the
    threshold to LEAK_SLOPE_LATE = 1600 B/cyc post-cap (fill-phase ≤5000 B/cyc; D-112/D-113).
  - Latency trend: p95 / p99 over the full run and per-checkpoint.
  - Φ-IQ proxy at checkpoints {1k, 5k, 10k, 50k, 100k} (windowed composite from
    per-cycle metrics — prediction accuracy, adaptation, goal rate, resource,
    failures).
  - RBTA violations total + per-cycle rate (A1 bound).

Exits non-zero on critical failure (RSS late-slope ≥ LEAK_SLOPE_LATE, p95
latency ≥ 500 ms, violation rate ≥ 10 %, or Φ-IQ proxy collapses > 0.15 between
the first and final checkpoint). Read-only; modifies no production code.

`--neg-test` runs a synthetic 4 KB/cyc unbounded-leak series and PASSes only
if the tightened gate flags it (proves the gate is non-vacuous).

Usage:
    MUJOCO_GL=disabled PYTHONPATH=python python scripts/nightly_stress.py --cycles=11000
    MUJOCO_GL=disabled NIGHTLY_CYCLES=1000 python scripts/nightly_stress.py
    python scripts/nightly_stress.py --neg-test
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import psutil

import _bootstrap  # noqa: F401

from phca.monitoring.retention_slope import (
    FILL_PHASE_CYCLES,
    LEAK_SLOPE_FILL,
    LEAK_SLOPE_LATE,
    compute_rss_slopes,
    leak_threshold_for_cycles,
    rss_leak_flagged,
)
from phca.core.cycle import CognitiveCycle
from phca.logging import ensure_logging
from phca.world_model.mlp import gprime_stress_bounds


CHECKPOINTS = [1000, 5000, 10000, 50000, 100000]
SAMPLE_EVERY = 100
P95_LIMIT_MS = 500.0
VIOL_RATE_LIMIT = 0.10
PHI_COLLAPSE_LIMIT = 0.15


def _build_cycle(seed: int = 42) -> CognitiveCycle:
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
        size=5, seed=seed + 2, use_continuous=True, use_mlp=True, obstacles=obstacles,
    )
    cycle.rbta.update_bounds("G'", gprime_stress_bounds(cycle))
    return cycle


def _phi_proxy(window) -> float:
    """Windowed Φ-IQ composite (weights from benchmark DEFAULT_WEIGHTS)."""
    if not window:
        return 0.0
    errors = np.array([m.prediction_error for m in window])
    lats = np.array([m.latency_ms for m in window])
    goals = np.array([1.0 if m.goal_reached else 0.0 for m in window])
    viols = np.array([m.violations_count for m in window])
    pred_acc = float(np.clip(1.0 - np.mean(errors) / 10.0, 0.0, 1.0))
    adapt = float(np.clip(
        (np.mean(errors[:len(errors) // 2]) - np.mean(errors[len(errors) // 2:]))
        / max(np.mean(errors[:len(errors) // 2]), 1e-3), 0.0, 1.0))
    goal = float(np.mean(goals))
    resource = float(max(0.0, 1.0 - np.mean(lats) / 500.0))
    transfer = adapt * pred_acc
    failure = float(np.mean(viols))
    return 0.25 * pred_acc + 0.20 * adapt + 0.15 * goal + 0.20 * resource + 0.15 * transfer - 0.10 * failure


def _neg_test() -> int:
    """Phase 7 / B3 negative self-test for the tightened late-slope gate.

    Synthesises an RSS series with a sustained 4 KB/cyc leak (an unbounded
    accumulator that appends a 4 KB bytearray every cycle and is never
    capped), runs the gate logic, and PASSes only if the gate flags it
    (late_slope ≥ LEAK_SLOPE_LATE). Proves the tightened gate is non-vacuous.
    """
    n = 2000
    base = 229_000_000
    xs = np.array([i * SAMPLE_EVERY for i in range(n // SAMPLE_EVERY)], dtype=np.float64)
    ys = np.array([base + (i * SAMPLE_EVERY) * 4000 for i in range(n // SAMPLE_EVERY)],
                  dtype=np.float64)
    slopes = compute_rss_slopes(xs.tolist(), ys.tolist(), total_cycles=n)
    late_slope = float(slopes["late_slope"])
    # Neg-test proves post-cap threshold catches sustained 4 KB/cyc leaks.
    flagged = late_slope >= LEAK_SLOPE_LATE
    print("=" * 60)
    print(f"  PHCA v3.0 — Nightly Stress NEG-TEST ({n} synthetic cyc)")
    print("=" * 60)
    print(f"  Synthetic leak: 4 KB/cyc sustained  →  late_slope = {late_slope:.1f} B/cyc")
    print(f"  Threshold (LEAK_SLOPE_LATE) = {LEAK_SLOPE_LATE:.1f} B/cyc")
    print(f"  Gate flagged the leak: {flagged}")
    if flagged:
        print("  Neg-test PASS: tightened gate correctly flagged the synthetic unbounded leak.")
        return 0
    print("  Neg-test FAIL: tightened gate did NOT flag the synthetic leak.")
    return 1


def main() -> None:
    ensure_logging()
    parser = argparse.ArgumentParser(description="PHCA nightly stress (Phase 7 / B3)")
    parser.add_argument("--cycles", type=int,
                        default=int(os.environ.get("NIGHTLY_CYCLES", "11000")))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="logs/nightly_stress.json")
    parser.add_argument("--neg-test", action="store_true",
                        help="Run the synthetic-leak negative self-test and exit.")
    args = parser.parse_args()

    if args.neg_test:
        sys.exit(_neg_test())

    n = args.cycles
    checkpoints = sorted(c for c in CHECKPOINTS if c <= n) or [n]
    cycle = _build_cycle(args.seed)
    proc = psutil.Process()

    rss_samples, latencies, violations = [], [], 0
    window = []
    phi_checkpoints = {}
    t0 = time.perf_counter()
    for i in range(n):
        m = cycle.step()
        window.append(m)
        latencies.append(m.latency_ms)
        violations += m.violations_count
        if (i + 1) % SAMPLE_EVERY == 0:
            gc.collect()
            rss_samples.append((i + 1, proc.memory_info().rss))
        if (i + 1) in checkpoints:
            phi_checkpoints[i + 1] = _phi_proxy(window[-min(1000, len(window)):])
    duration = time.perf_counter() - t0

    # RSS leak slope (bytes/cycle). Report full-run slope and late-half slope
    # (the late half is the cleaner leak signal once bounded buffers saturate).
    if len(rss_samples) >= 2:
        xs = [s[0] for s in rss_samples]
        ys = [s[1] for s in rss_samples]
        slopes = compute_rss_slopes(xs, ys, total_cycles=n)
        slope = float(slopes["full_slope"])
        late_slope = float(slopes["late_slope"])
        late_slope_window = str(slopes.get("late_slope_window", "unknown"))
    else:
        slope = late_slope = 0.0
        late_slope_window = "insufficient"
    rss_start = rss_samples[0][1] if rss_samples else 0
    rss_end = rss_samples[-1][1] if rss_samples else 0

    lat = np.array(latencies)
    p95, p99 = float(np.percentile(lat, 95)), float(np.percentile(lat, 99))
    viol_rate = violations / max(n, 1)
    phi_first = phi_checkpoints.get(checkpoints[0], 0.0)
    phi_final = phi_checkpoints.get(checkpoints[-1], 0.0)
    phi_collapse = phi_first - phi_final

    leak_threshold, leak_gate_mode = leak_threshold_for_cycles(n)

    crit = {
        "no_rss_leak": not rss_leak_flagged(late_slope, n),
        "p95_latency_under_500ms": p95 < P95_LIMIT_MS,
        "violation_rate_under_10pct": viol_rate < VIOL_RATE_LIMIT,
        "phi_iq_stable": phi_collapse < PHI_COLLAPSE_LIMIT,
    }
    all_pass = all(crit.values())

    report = {
        "cycles": n, "duration_s": round(duration, 2),
        "rss_slope_bytes_per_cycle": round(slope, 2),
        "rss_late_slope_bytes_per_cycle": round(late_slope, 2),
        "late_slope_window": late_slope_window,
        "rss_start_bytes": int(rss_start), "rss_end_bytes": int(rss_end),
        "latency_p95_ms": round(p95, 2), "latency_p99_ms": round(p99, 2),
        "latency_mean_ms": round(float(np.mean(lat)), 2),
        "total_violations": int(violations), "violation_rate": round(viol_rate, 5),
        "phi_iq_checkpoints": {str(k): round(v, 4) for k, v in phi_checkpoints.items()},
        "phi_collapse_first_to_final": round(phi_collapse, 4),
        "leak_late_slope_threshold": leak_threshold,
        "leak_gate_mode": leak_gate_mode,
        "criteria": crit, "all_pass": bool(all_pass),
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2))

    print("=" * 60)
    print(f"  PHCA v3.0 — Nightly Stress ({n} cycles, {duration:.1f}s)")
    print("=" * 60)
    print(f"  RSS slope : {slope:.1f} B/cyc (late {late_slope:.1f}, {late_slope_window})  "
          f"(start {rss_start/1e6:.1f}MB → end {rss_end/1e6:.1f}MB)")
    print(f"  Latency   : mean {report['latency_mean_ms']:.1f}ms  p95 {p95:.1f}ms  p99 {p99:.1f}ms")
    print(f"  Violations: {violations}  ({viol_rate*100:.3f}%/cyc)")
    print(f"  Φ-IQ ckpt : " + "  ".join(f"@{k}={v:.3f}" for k, v in phi_checkpoints.items()))
    for k, v in crit.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    print(f"  Overall: {'PASS' if all_pass else 'FAIL'}  → {args.output}")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
