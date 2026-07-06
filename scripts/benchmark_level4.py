#!/usr/bin/env python3
"""Level-4-lite continual learning benchmark (forgetting rate / AT-2-lite gate)."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

import _bootstrap  # noqa: F401

from phca.core.cycle import CognitiveCycle, CycleMetrics
from phca.evaluation.continual.gridworld_tasks import build_task_sequence
from phca.evaluation.metrics.forgetting import (
    delta_perf,
    eval_window_accuracy,
    forgetting_rate,
    passes_forgetting_gate,
    task_accuracy,
)
from phca.world_model.mlp import apply_grid_rbta_bounds


def run_level4_benchmark(
    *,
    n_tasks: int = 10,
    task_cycles: int = 80,
    eval_cycles: int = 20,
    seeds: int = 3,
    use_mlp: bool = True,
    grid_size: int = 5,
    base_seed: int = 42,
    mitigation: bool = True,
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

        per_task_history: Dict[int, List[CycleMetrics]] = {}
        baselines: Dict[int, float] = {}

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

            baseline = task_accuracy(
                {task.task_id: train_hist},
                task_id=task.task_id,
                metric="goal_rate",
                window=min(20, len(train_hist)),
            )
            baselines[task.task_id] = baseline
            cycle.record_task_baseline(task.task_id, baseline)

        # Final eval pass over all tasks (no task-boundary signal to agent)
        current: Dict[int, float] = {}
        for task in tasks:
            cycle.env.apply_task_layout(task.goal_pos, task.obstacles)
            cycle._current_task_id = task.task_id
            eval_hist: List[CycleMetrics] = []
            for _ in range(eval_cycles):
                m = cycle.step()
                m.task_id = task.task_id
                eval_hist.append(m)
                cycle.record_task_eval(task.task_id, m.goal_reached)
            per_task_history[task.task_id] = eval_hist
            current[task.task_id] = eval_window_accuracy(
                {task.task_id: eval_hist},
                task_id=task.task_id,
                eval_cycles=eval_cycles,
            )

        delta = delta_perf(baselines, current)
        seed_results.append({
            "seed": seed,
            "baselines": baselines,
            "current": current,
            "delta_perf": delta,
            "forgetting_rate": forgetting_rate(delta),
            "passes_gate": passes_forgetting_gate(delta),
            "per_task_accuracy": current,
        })

    agg_delta = {
        tid: float(np.mean([r["delta_perf"].get(tid, 0.0) for r in seed_results]))
        for tid in range(n_tasks)
    }
    agg_forgetting = forgetting_rate(agg_delta)
    return {
        "config": {
            "tasks": n_tasks,
            "task_cycles": task_cycles,
            "eval_cycles": eval_cycles,
            "seeds": seeds,
            "use_mlp": use_mlp,
            "grid_size": grid_size,
            "mitigation": mitigation,
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
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--use-mlp", action="store_true", default=True)
    parser.add_argument("--no-mlp", action="store_false", dest="use_mlp")
    parser.add_argument("--grid-size", type=int, default=5, choices=[5, 10, 20])
    parser.add_argument("--no-mitigation", action="store_true")
    parser.add_argument("--output", type=str, default="logs/benchmark_level4.json")
    args = parser.parse_args()

    t0 = time.perf_counter()
    report = run_level4_benchmark(
        n_tasks=args.tasks,
        task_cycles=args.task_cycles,
        eval_cycles=args.eval_cycles,
        seeds=args.seeds,
        use_mlp=args.use_mlp,
        grid_size=args.grid_size,
        mitigation=not args.no_mitigation,
    )
    report["duration_s"] = time.perf_counter() - t0

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str))

    print(f"forgetting_rate={report['forgetting_rate']:.4f}")
    print(f"passes_gate={report['passes_gate']}")
    print(f"per_task_accuracy={report['per_task_accuracy']}")
    print(f"Report saved to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
