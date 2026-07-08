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
from phca.evaluation.continual.gridworld_tasks import build_task_sequence, GridWorldTask
from phca.evaluation.continual.pendulum_tasks import (
    build_pendulum_task_sequence,
    pendulum_task_to_params,
    PendulumTask,
)
from phca.evaluation.metrics.forgetting import (
    DEFAULT_BASELINE_MIN_VALID,
    delta_perf_valid_only,
    eval_window_accuracy,
    forgetting_rate,
    forward_transfer,
    max_rolling_task_accuracy,
    passes_forgetting_gate,
    task_accuracy,
)
from phca.world_model.mlp import apply_grid_rbta_bounds
EnvTask = GridWorldTask | PendulumTask

BaselineMethod = Literal["last_window", "max_rolling"]


def _training_perf_curve(
    train_hist: List[CycleMetrics],
    *,
    metric: str = "goal_rate",
    sample_every: int = 10,
) -> List[Dict[str, float]]:
    if not train_hist:
        return []
    curve: List[Dict[str, float]] = []
    for end in range(sample_every, len(train_hist) + 1, sample_every):
        window = train_hist[max(0, end - 20) : end]
        if window:
            if metric == "goal_rate":
                val = float(np.mean([float(m.goal_reached) for m in window]))
            else:
                errs = [m.prediction_error for m in window]
                mean_err = float(np.mean(errs)) if errs else 0.0
                val = max(0.0, 1.0 - min(mean_err / 10.0, 1.0))
        else:
            val = 0.0
        curve.append({"cycle": end, "perf": val})
    if len(train_hist) % sample_every != 0:
        window = train_hist[-20:]
        if window:
            if metric == "goal_rate":
                val = float(np.mean([float(m.goal_reached) for m in window]))
            else:
                errs = [m.prediction_error for m in window]
                mean_err = float(np.mean(errs)) if errs else 0.0
                val = max(0.0, 1.0 - min(mean_err / 10.0, 1.0))
        else:
            val = 0.0
        curve.append({"cycle": len(train_hist), "perf": val})
    return curve


def _baseline_for_task(
    train_hist: List[CycleMetrics],
    *,
    task_id: int,
    method: BaselineMethod,
    window: int,
    metric: str = "goal_rate",
) -> float:
    if method == "max_rolling":
        return max_rolling_task_accuracy(
            {task_id: train_hist},
            task_id=task_id,
            window=window,
            metric=metric,
        )
    return task_accuracy(
        {task_id: train_hist},
        task_id=task_id,
        metric=metric,
        window=min(window, len(train_hist)),
    )


def _m3_counts_by_task(cycle: CognitiveCycle, n_tasks: int) -> Dict[int, int]:
    m3 = cycle.consolidation.m3
    return {tid: m3.count_for_task(tid) for tid in range(n_tasks)}


def _run_eval_on_task(
    cycle: CognitiveCycle,
    task: EnvTask,
    *,
    mitigation: bool,
    warmup_cycles: int,
    eval_cycles: int,
    diagnostic: bool,
    b4_events: int,
    env_type: str = "gridworld",
    metric: str = "goal_rate",
    train_start_pos: tuple | None = None,
) -> tuple[List[CycleMetrics], int]:
    if env_type == "pendulum":
        assert isinstance(task, PendulumTask)
        cycle.env.apply_task_layout(pendulum_task_to_params(task))
    else:
        assert isinstance(task, GridWorldTask)
        cycle.env.apply_task_layout(task.goal_pos, task.obstacles)
        if train_start_pos is not None and hasattr(cycle.env, "agent_pos"):
            cycle.env.agent_pos = train_start_pos
            cycle.env.start_pos = train_start_pos
    if mitigation:
        cycle.on_task_boundary(task.task_id)
    else:
        cycle._current_task_id = task.task_id

    for _ in range(warmup_cycles):
        cycle.step()

    eval_hist: List[CycleMetrics] = []
    saved_learn = cycle.interventions.enable_gprime_learn
    cycle.interventions.enable_gprime_learn = False
    for _ in range(eval_cycles):
        m = cycle.step()
        m.task_id = task.task_id
        eval_hist.append(m)
        if metric == "prediction_accuracy":
            err = max(m.prediction_error, 0.0)
            acc_val = max(0.0, 1.0 - min(err / 10.0, 1.0))
            cycle.record_task_eval(task.task_id, bool(acc_val > 0.5))
        else:
            cycle.record_task_eval(task.task_id, m.goal_reached)
        if diagnostic and m.failure_events and "B4" in m.failure_events:
            b4_events += 1
    cycle.interventions.enable_gprime_learn = saved_learn
    return eval_hist, b4_events


def _interleaved_eval_prior_tasks(
    cycle: CognitiveCycle,
    tasks: List[EnvTask],
    *,
    through_task_id: int,
    eval_cycles: int,
    warmup_cycles: int,
    mitigation: bool,
    diagnostic: bool,
    b4_events: int,
    env_type: str = "gridworld",
    metric: str = "goal_rate",
) -> int:
    """Brief eval on tasks 0..through_task_id after training each new task."""
    for t in tasks:
        if t.task_id > through_task_id:
            break
        _, b4_events = _run_eval_on_task(
            cycle,
            t,
            mitigation=mitigation,
            warmup_cycles=warmup_cycles,
            eval_cycles=eval_cycles,
            diagnostic=diagnostic,
            b4_events=b4_events,
            env_type=env_type,
            metric=metric,
        )
    return b4_events


def run_level4_benchmark(
    *,
    n_tasks: int = 10,
    task_cycles: int = 80,
    eval_cycles: int = 20,
    eval_warmup_cycles: int = 0,
    seeds: int = 3,
    use_mlp: bool = True,
    grid_size: int = 5,
    env_type: str = "gridworld",
    metric: str = "goal_rate",
    base_seed: int = 42,
    mitigation: bool = True,
    baseline_method: BaselineMethod = "max_rolling",
    baseline_window: int = 20,
    baseline_min_valid: float = DEFAULT_BASELINE_MIN_VALID,
    diagnostic: bool = False,
    m3_replay_budget: int = 16,
    interleaved_eval_cycles: int = 0,
    interleaved_warmup_cycles: int = 0,
    noise_profile: Optional[str] = None,
    noise_intensity: float = 0.1,
) -> Dict[str, Any]:
    """Run continual task sequence and measure forgetting rate."""
    if env_type == "pendulum":
        tasks: List[EnvTask] = build_pendulum_task_sequence(n_tasks=n_tasks, base_seed=base_seed)
    else:
        tasks = build_task_sequence(n_tasks=n_tasks, grid_size=grid_size, base_seed=base_seed)
    seed_results: List[Dict[str, Any]] = []

    for seed_idx in range(seeds):
        seed = base_seed + seed_idx * 100
        if env_type == "pendulum":
            cycle = CognitiveCycle.build_for_mujoco(
                env_name="Pendulum-v1",
                seed=seed,
                noise_profile=noise_profile,
                noise_intensity=noise_intensity,
            )
            cycle.set_m3_replay_budget(m3_replay_budget)
        else:
            cycle = CognitiveCycle.build_for_env(
                size=grid_size,
                seed=seed,
                use_mlp=use_mlp,
                obstacles=tasks[0].obstacles,
                noise_profile=noise_profile,
                noise_intensity=noise_intensity,
            )
            if grid_size > 5:
                apply_grid_rbta_bounds(cycle, b_time=0.080 if use_mlp else 0.020)
            cycle.set_m3_replay_budget(m3_replay_budget)

        per_task_train: Dict[int, List[CycleMetrics]] = {}
        baselines: Dict[int, float] = {}
        train_curves: Dict[int, List[Dict[str, float]]] = {}
        b4_events = 0
        m3_replay_at_train_end = 0
        train_end_positions: Dict[int, tuple] = {}

        for task in tasks:
            if env_type == "pendulum":
                assert isinstance(task, PendulumTask)
                cycle.env.apply_task_layout(pendulum_task_to_params(task))
            else:
                assert isinstance(task, GridWorldTask)
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
            if hasattr(cycle.env, "agent_pos"):
                train_end_positions[task.task_id] = cycle.env.agent_pos
            baseline = _baseline_for_task(
                train_hist,
                task_id=task.task_id,
                method=baseline_method,
                window=baseline_window,
                metric=metric,
            )
            baselines[task.task_id] = baseline
            cycle.record_task_baseline(task.task_id, baseline)
            train_curves[task.task_id] = _training_perf_curve(train_hist, metric=metric)
            if diagnostic:
                m3_replay_at_train_end = cycle._m3_replay_total

            if interleaved_eval_cycles > 0:
                b4_events = _interleaved_eval_prior_tasks(
                    cycle,
                    tasks,
                    through_task_id=task.task_id,
                    eval_cycles=interleaved_eval_cycles,
                    warmup_cycles=interleaved_warmup_cycles,
                    mitigation=mitigation,
                    diagnostic=diagnostic,
                    b4_events=b4_events,
                    env_type=env_type,
                    metric=metric,
                )

        current: Dict[int, float] = {}
        per_task_history: Dict[int, List[CycleMetrics]] = {}
        for task in tasks:
            eval_hist, b4_events = _run_eval_on_task(
                cycle,
                task,
                mitigation=mitigation,
                warmup_cycles=eval_warmup_cycles,
                eval_cycles=eval_cycles,
                diagnostic=diagnostic,
                b4_events=b4_events,
                env_type=env_type,
                metric=metric,
                train_start_pos=train_end_positions.get(task.task_id),
            )

            per_task_history[task.task_id] = eval_hist
            current[task.task_id] = eval_window_accuracy(
                {task.task_id: eval_hist},
                task_id=task.task_id,
                eval_cycles=eval_cycles,
                metric=metric,
            )

        delta, excluded = delta_perf_valid_only(
            baselines,
            current,
            min_valid=baseline_min_valid,
        )
        _forward_transfer: Dict[int, float] = {}
        if train_curves:
            _perf_only = {
                tid: [p["perf"] for p in pts]
                for tid, pts in train_curves.items()
            }
            try:
                _forward_transfer = forward_transfer(_perf_only)
            except Exception:
                pass
        seed_entry: Dict[str, Any] = {
            "seed": seed,
            "baselines": baselines,
            "current": current,
            "delta_perf": delta,
            "excluded_tasks": excluded,
            "forgetting_rate": forgetting_rate(delta),
            "passes_gate": passes_forgetting_gate(delta),
            "per_task_accuracy": current,
            "forward_transfer": _forward_transfer,
            "m3_replay_total": cycle._m3_replay_total,
            "novel_goal_rate": cycle.mdim.snapshot().get("novel_goal_rate", 0.0) if hasattr(cycle, "mdim") and cycle.mdim is not None else 0.0,
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
                "m3_replay_budget": m3_replay_budget,
                "interleaved_eval_cycles": interleaved_eval_cycles,
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
            "m3_replay_budget": m3_replay_budget,
            "interleaved_eval_cycles": interleaved_eval_cycles,
            "interleaved_warmup_cycles": interleaved_warmup_cycles,
            "noise_profile": noise_profile,
            "noise_intensity": noise_intensity,
        },
        "tasks": [
            {
                "task_id": t.task_id,
                "goal_pos": list(t.goal_pos) if isinstance(t, GridWorldTask) else None,
                "target_angle": t.target_angle if isinstance(t, PendulumTask) else None,
                "n_obstacles": len(t.obstacles) if isinstance(t, GridWorldTask) else 0,
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
        "forward_transfer": {
            tid: float(np.mean([r["forward_transfer"].get(tid, 0.0) for r in seed_results]))
            for tid in range(n_tasks)
        },
        "novel_goal_rate": float(np.mean([r.get("novel_goal_rate", 0.0) for r in seed_results])),
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
    parser.add_argument("--env-type", choices=["gridworld", "pendulum"], default="gridworld")
    parser.add_argument("--metric", choices=["goal_rate", "prediction_accuracy"], default=None)
    parser.add_argument("--no-mitigation", action="store_true")
    parser.add_argument(
        "--baseline-method",
        choices=["last_window", "max_rolling"],
        default="max_rolling",
    )
    parser.add_argument("--baseline-window", type=int, default=20)
    parser.add_argument("--baseline-min-valid", type=float, default=DEFAULT_BASELINE_MIN_VALID)
    parser.add_argument("--diagnostic", action="store_true")
    parser.add_argument("--m3-replay-budget", type=int, default=16)
    parser.add_argument("--interleaved-eval", type=int, default=0, dest="interleaved_eval_cycles")
    parser.add_argument("--interleaved-warmup", type=int, default=0, dest="interleaved_warmup_cycles")
    parser.add_argument("--noise-profile", type=str, default=None, choices=["gaussian", "dropout", "drift", "salt_pepper"])
    parser.add_argument("--noise-intensity", type=float, default=0.1)
    parser.add_argument("--output", type=str, default="logs/benchmark_level4.json")
    args = parser.parse_args()
    if args.metric is None:
        args.metric = "prediction_accuracy" if args.env_type == "pendulum" else "goal_rate"

    t0 = time.perf_counter()
    report = run_level4_benchmark(
        n_tasks=args.tasks,
        task_cycles=args.task_cycles,
        eval_cycles=args.eval_cycles,
        eval_warmup_cycles=args.eval_warmup_cycles,
        seeds=args.seeds,
        use_mlp=args.use_mlp,
        grid_size=args.grid_size,
        env_type=args.env_type,
        metric=args.metric,
        mitigation=not args.no_mitigation,
        baseline_method=args.baseline_method,
        baseline_window=args.baseline_window,
        baseline_min_valid=args.baseline_min_valid,
        diagnostic=args.diagnostic,
        m3_replay_budget=args.m3_replay_budget,
        interleaved_eval_cycles=args.interleaved_eval_cycles,
        interleaved_warmup_cycles=args.interleaved_warmup_cycles,
        noise_profile=args.noise_profile,
        noise_intensity=args.noise_intensity,
    )


    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str))

    print(f"forgetting_rate={report['forgetting_rate']:.4f}")
    print(f"passes_gate={report['passes_gate']}")
    print(f"per_task_accuracy={report['per_task_accuracy']}")
    ft_agg = report.get("forward_transfer", {})
    if ft_agg:
        print(f"forward_transfer={ft_agg}")
    if report["seed_results"]:
        print(f"m3_replay_total={report['seed_results'][0].get('m3_replay_total')}")
    if args.diagnostic and report["seed_results"]:
        diag = report["seed_results"][0].get("diagnostic", {})
        print(f"excluded_tasks={report['seed_results'][0].get('excluded_tasks')}")
    print(f"Report saved to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
