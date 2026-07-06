#!/usr/bin/env python3
"""Cognitive recovery benchmark — B1, C1, F5 scenarios."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest import mock

import numpy as np

import _bootstrap  # noqa: F401

from phca.core.cycle import CognitiveCycle
from phca.memory.m3_episodic import M3EpisodicMemory
from phca.resilience.metrics import recovery_rate


def _baseline_kpi(cycle: CognitiveCycle, n: int = 30) -> Dict[str, float]:
    for _ in range(n):
        cycle.step()
    recent = cycle.metrics_history[-10:]
    return {
        "prediction_error": float(np.mean([m.prediction_error for m in recent])),
        "violations": float(np.mean([m.violations_count for m in recent])),
        "fact_count": float(cycle.consolidation.get_stats().get("total_facts_stored", 0)),
    }


def _within_10pct(current: float, baseline: float) -> bool:
    if baseline < 1e-8:
        return current < 0.1
    return abs(current - baseline) / baseline <= 0.10


def run_b1_scenario(cycle: CognitiveCycle, *, warmup: int = 50, fault_cycles: int = 5) -> Dict[str, Any]:
    """Distribution shift via action_slip spike."""
    baseline = _baseline_kpi(cycle, warmup)
    cycle.env.action_slip = 0.8
    for _ in range(fault_cycles):
        cycle.step()
    cycle.env.action_slip = 0.0
    recovered = False
    recovery_cycle: Optional[int] = None
    for i in range(10):
        cycle.step()
        err = cycle.metrics_history[-1].prediction_error
        if _within_10pct(err, baseline["prediction_error"]):
            recovered = True
            recovery_cycle = i + 1
            break
    return {
        "scenario": "b1",
        "recovered": recovered,
        "recovery_cycle": recovery_cycle,
        "baseline_error": baseline["prediction_error"],
        "final_error": cycle.metrics_history[-1].prediction_error,
    }


def run_c1_scenario(cycle: CognitiveCycle, *, warmup: int = 30) -> Dict[str, Any]:
    """Feedback instability via artificial slow module → RBTA TERMINATE streak."""
    baseline = _baseline_kpi(cycle, warmup)
    original_collect = cycle._collect_runtime_log

    def slow_collect(metrics=None):
        original_collect(metrics)
        cycle.runtime_log["G'"] = 10.0

    cycle._collect_runtime_log = slow_collect  # type: ignore[method-assign]
    for _ in range(5):
        cycle.step()
    cycle._collect_runtime_log = original_collect  # type: ignore[method-assign]

    recovered = False
    recovery_cycle: Optional[int] = None
    for i in range(10):
        cycle.step()
        if cycle.metrics_history[-1].rbta_action != "TERMINATE":
            recovered = True
            recovery_cycle = i + 1
            break
    return {
        "scenario": "c1",
        "recovered": recovered,
        "recovery_cycle": recovery_cycle,
        "baseline_violations": baseline["violations"],
        "final_rbta": cycle.metrics_history[-1].rbta_action,
    }


def run_f5_scenario(cycle: CognitiveCycle, *, warmup: int = 20) -> Dict[str, Any]:
    """Consolidation failure — fill M3 near cap with synthetic episodes."""
    from phca.config import StateVector

    _baseline_kpi(cycle, warmup)
    m3 = cycle.consolidation.m3
    state_dim = cycle.state_dim
    action_dim = cycle.env.action_space_size
    target_fill = int(0.92 * m3._max_episodes)
    sv = StateVector(
        values=np.zeros(state_dim, dtype=np.float32),
        precision=np.ones(state_dim, dtype=np.float32),
    )
    action = np.zeros(action_dim, dtype=np.float32)
    for i in range(target_fill - m3.count()):
        m3.store_episode(sv, action, sv, 0.1, timestamp=i)

    cycle.consolidation._consolidation_interval = 10_000
    for _ in range(55):
        cycle.step()

    facts_before = cycle.consolidation.get_stats().get("total_facts_stored", 0)
    cycle.consolidation.force_step(cycle.cycle_count)
    facts_after = cycle.consolidation.get_stats().get("total_facts_stored", 0)

    recovered = facts_after > facts_before or cycle.metrics_history[-1].failure_events
    return {
        "scenario": "f5",
        "recovered": bool(recovered),
        "facts_before": facts_before,
        "facts_after": facts_after,
        "m3_fill": m3.count() / m3._max_episodes,
    }


def run_recovery_benchmark(
    scenarios: List[str],
    *,
    cycles: int = 200,
    seed: int = 42,
) -> Dict[str, Any]:
    """Run injectable failure scenarios and measure recovery rate."""
    results: Dict[str, Any] = {}
    outcomes: Dict[str, bool] = {}

    if "b1" in scenarios:
        cycle = CognitiveCycle.build_for_env(size=5, seed=seed, use_mlp=True)
        results["b1"] = run_b1_scenario(cycle)
        outcomes["b1"] = results["b1"]["recovered"]

    if "c1" in scenarios:
        cycle = CognitiveCycle.build_for_env(size=5, seed=seed + 1, use_mlp=True)
        results["c1"] = run_c1_scenario(cycle)
        outcomes["c1"] = results["c1"]["recovered"]

    if "f5" in scenarios:
        cycle = CognitiveCycle.build_for_env(size=5, seed=seed + 2, use_mlp=True)
        results["f5"] = run_f5_scenario(cycle)
        outcomes["f5"] = results["f5"]["recovered"]

    rate = recovery_rate(scenarios, outcomes, window=10)
    return {
        "config": {"scenarios": scenarios, "cycles": cycles, "seed": seed},
        "scenarios": results,
        "recovery_rate": rate,
        "outcomes": outcomes,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Cognitive recovery benchmark")
    parser.add_argument("--scenarios", type=str, default="b1,c1,f5")
    parser.add_argument("--cycles", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="logs/benchmark_recovery.json")
    args = parser.parse_args()

    scenarios = [s.strip().lower() for s in args.scenarios.split(",") if s.strip()]
    t0 = time.perf_counter()
    report = run_recovery_benchmark(scenarios, cycles=args.cycles, seed=args.seed)
    report["duration_s"] = time.perf_counter() - t0

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str))

    print(f"recovery_rate={report['recovery_rate']:.2f}")
    for sid, detail in report["scenarios"].items():
        print(f"  {sid}: recovered={detail.get('recovered')}")
    print(f"Report saved to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
