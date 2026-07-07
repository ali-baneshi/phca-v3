#!/usr/bin/env python3
"""Level-4-lite continual learning benchmark (forgetting rate / AT-2-lite gate)."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Literal

import numpy as np

import _bootstrap  # noqa: F401

from phca.core.cycle import CognitiveCycle, CycleMetrics
from phca.evaluation.continual.gridworld_tasks import build_task_sequence
from phca.evaluation.metrics.forgetting import (
    DEFAULT_BASELINE_MIN_VALID,
    delta_perf_valid_only,
    eval_window_accuracy,
    forgetting_rate,
    max_rolling_task_accuracy,
    passes_forgetting_gate,
    task_accuracy,
)
from phca.world_model.mlp import apply_grid_rbta_bounds

BaselineMethod = Literal["last_window", "max_rolling"]


def _train_goal_rate_curve(
    train_hist: List[CycleMetrics],
    *,
    sample_every: int = 10,
) -> List[Dict[str, float]]:
    if not train_hist:
        return []
    curve: List[Dict[str, float]] = []
    for end in range(sample_every, len(train_hist) + 1, sample_every):
        window = train_hist[max(0, end - 20) : end]
        rate = float(np.mean([float(m.goal_reached) for m in window])) if window else 0.0
        curve.append({"cycle": end, "goal_rate": rate})
    if len(train_hist) % sample_every != 0:
        window = train_hist[-20:]
        rate = float(np.mean([float(m.goal_reached) for m in window])) if window else 0.0
        curve.append({"cycle": len(train_hist), "goal_rate": rate})
    return curve


def _baseline_for_task(
    train_hist: List[CycleMetrics],
    *,
    task_id: int,
    method: BaselineMethod,
    window: int,
) -> float:
    if method == "max_rolling":
        return max_rolling_task_accuracy(
            {task_id: train_hist},
            task_id=task_id,
            window=window,
        )
    return task_accuracy(
        {task_id: train_hist},
        task_id=task_id,
        metric="goal_rate",
        window=min(window, len(train_hist)),
    )


def _m3_counts_by_task(cycle: CognitiveCycle, n_tasks: int) -> Dict[int, int]:
    m3 = cycle.consolidation.m3
    return {tid: m3.count_for_task(tid) for tid in range(n_tasks)}


def run_level4_benchmark(
    *,
    n_tasks: int = 10,
    task_cycles: int = 80,
    eval_cycles: int = 20,
    eval_warmup_cycles: int = 0,
    seeds: int = 3,
    use_mlp: bool = True,
    grid_size: int = 5,
    base_seed: int = 42,
    mitigation: bool = True,
    baseline_method: BaselineMethod = "max_rolling",
    baseline_window: int = 20,
    baseline_min_valid: float = DEFAULT_BASELINE_MIN_VALID,
    diagnostic: bool = False,
) -> Dict[str, Any]:
    """Run continual task sequence and measure forgetting rate."""
    tasks = build_task_sequence(n_tasks=n_tasks, grid_size=grid_size, base_seed=base_seed)
    seed_results: List[Dict[str, Any]] = []

    for seed_idx in range(seeds):
        seed = base_seed + seed_idx * 100
        cycle = CognitiveCycle.build_for_env(
            size=grid_size,
            seed=seed,
            use_mlp=use_mlp,
            obstacles=tasks[0].obstacles,
        )
        if grid_size > 5:
            apply_grid_rbta_bounds(cycle, b_time=0.080 if use_mlp else 0.020)

        per_task_train: Dict[int, List[CycleMetrics]] = {}
        baselines: Dict[int, float] = {}
        train_curves: Dict[int, List[Dict[str, float]]] = {}
        b4_events = 0
        m3_replay_at_train_end = 0

        for task in tasks:
            cycle.env.apply_task_layout(task.goal_pos, task.obstacles)
            if mitigation:
                cycle.on_task_boundary(task.task_id)
            else:
                cycle._current_task_id = task.task_id

            train_hist: List[CycleMetrics] = []
            for _ in range(task_cycles):
                m = cycle.step()
                m.task_id = task.task_id
                train_hist.append(m)
                if diagnostic and m.failure_events and "B4" in m.failure_events:
                    b4_events += 1

            per_task_train[task.task_id] = train_hist
            baseline = _baseline_for_task(
                train_hist,
                task_id=task.task_id,
                method=baseline_method,
                window=baseline_window,
            )
            baselines[task.task_id] = baseline
            cycle.record_task_baseline(task.task_id, baseline)
            if diagnostic:
                train_curves[task.task_id] = _train_goal_rate_curve(train_hist)
                m3_replay_at_train_end = cycle._m3_replay_total

        current: Dict[int, float] = {}
        per_task_history: Dict[int, List[CycleMetrics]] = {}
        for task in tasks:
            cycle.env.apply_task_layout(task.goal_pos, task.obstacles)
            if mitigation:
                cycle.on_task_boundary(task.task_id)
            else:
                cycle._current_task_id = task.task_id

            for _ in range(eval_warmup_cycles):
                cycle.step()

            eval_hist: List[CycleMetrics] = []
            for _ in range(eval_cycles):
                m = cycle.step()
                m.task_id = task.task_id
                eval_hist.append(m)
                cycle.record_task_eval(task.task_id, m.goal_reached)
                if diagnostic and m.failure_events and "B4" in m.failure_events:
                    b4_events += 1

            per_task_history[task.task_id] = eval_hist
            current[task.task_id] = eval_window_accuracy(
                {task.task_id: eval_hist},
                task_id=task.task_id,
                eval_cycles=eval_cycles,
            )

        delta, excluded = delta_perf_valid_only(
            baselines,
            current,
            min_valid=baseline_min_valid,
        )
        seed_entry: Dict[str, Any] = {
            "seed": seed,
            "baselines": baselines,
            "current": current,
            "delta_perf": delta,
            "excluded_tasks": excluded,
            "forgetting_rate": forgetting_rate(delta),
            "passes_gate": passes_forgetting_gate(delta),
            "per_task_accuracy": current,
        }
        if diagnostic:
            seed_entry["diagnostic"] = {
                "train_curves": train_curves,
                "m3_counts_by_task": _m3_counts_by_task(cycle, n_tasks),
                "m3_replay_total": cycle._m3_replay_total,
                "m3_replay_at_train_end": m3_replay_at_train_end,
                "b4_event_cycles": b4_events,
                "mitigation_active": mitigation,
                "baseline_method": baseline_method,
                "baseline_window": baseline_window,
                "baseline_min_valid": baseline_min_valid,
            }
        seed_results.append(seed_entry)

    agg_delta = {
        tid: float(np.mean([r["delta_perf"].get(tid, 0.0) for r in seed_results]))
        for tid in range(n_tasks)
        if any(tid in r["delta_perf"] for r in seed_results)
    }
    agg_forgetting = forgetting_rate(agg_delta)
    return {
        "config": {
            "tasks": n_tasks,
            "task_cycles": task_cycles,
            "eval_cycles": eval_cycles,
            "eval_warmup_cycles": eval_warmup_cycles,
            "seeds": seeds,
            "use_mlp": use_mlp,
            "grid_size": grid_size,
            "mitigation": mitigation,
            "baseline_method": baseline_method,
            "baseline_window": baseline_window,
            "baseline_min_valid": baseline_min_valid,
            "diagnostic": diagnostic,
        },
        "tasks": [
            {
                "task_id": t.task_id,
                "goal_pos": t.goal_pos,
                "n_obstacles": len(t.obstacles),
            }
            for t in tasks
        ],
        "seed_results": seed_results,
        "delta_perf": agg_delta,
        "forgetting_rate": agg_forgetting,
        "passes_gate": passes_forgetting_gate(agg_delta),
        "per_task_accuracy": {
            tid: float(np.mean([r["per_task_accuracy"].get(tid, 0.0) for r in seed_results]))
            for tid in range(n_tasks)
        },
        "duration_s": 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Level-4-lite forgetting benchmark")
    parser.add_argument("--tasks", type=int, default=10)
    parser.add_argument("--task-cycles", type=int, default=80)
    parser.add_argument("--eval-cycles", type=int, default=20)
    parser.add_argument("--eval-warmup", type=int, default=0, dest="eval_warmup_cycles")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--use-mlp", action="store_true", default=True)
    parser.add_argument("--no-mlp", action="store_false", dest="use_mlp")
    parser.add_argument("--grid-size", type=int, default=5, choices=[5, 10, 20])
    parser.add_argument("--no-mitigation", action="store_true")
    parser.add_argument(
        "--baseline-method",
        choices=["last_window", "max_rolling"],
        default="max_rolling",
    )
    parser.add_argument("--baseline-window", type=int, default=20)
    parser.add_argument("--baseline-min-valid", type=float, default=DEFAULT_BASELINE_MIN_VALID)
    parser.add_argument("--diagnostic", action="store_true")
    parser.add_argument("--output", type=str, default="logs/benchmark_level4.json")
    args = parser.parse_args()

    t0 = time.perf_counter()
    report = run_level4_benchmark(
        n_tasks=args.tasks,
        task_cycles=args.task_cycles,
        eval_cycles=args.eval_cycles,
        eval_warmup_cycles=args.eval_warmup_cycles,
        seeds=args.seeds,
        use_mlp=args.use_mlp,
        grid_size=args.grid_size,
        mitigation=not args.no_mitigation,
        baseline_method=args.baseline_method,
        baseline_window=args.baseline_window,
        baseline_min_valid=args.baseline_min_valid,
        diagnostic=args.diagnostic,
    )
    report["duration_s"] = time.perf_counter() - t0

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str))

    print(f"forgetting_rate={report['forgetting_rate']:.4f}")
    print(f"passes_gate={report['passes_gate']}")
    print(f"per_task_accuracy={report['per_task_accuracy']}")
    if args.diagnostic and report["seed_results"]:
        diag = report["seed_results"][0].get("diagnostic", {})
        print(f"m3_replay_total={diag.get('m3_replay_total')}")
        print(f"excluded_tasks={report['seed_results'][0].get('excluded_tasks')}")
    print(f"Report saved to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
