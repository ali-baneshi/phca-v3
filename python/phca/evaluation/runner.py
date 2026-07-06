"""Experiment and benchmark runner."""

from __future__ import annotations

import json
import time

import numpy as np
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from phca.config import ResourceBounds
from phca.core.cycle import CognitiveCycle
from phca.world_model.mlp import gprime_stress_bounds
from phca.evaluation.interventions import InterventionConfig
from phca.evaluation.metrics.emergence import compute_emergence_bundle
from phca.evaluation.metrics.phi_iq import (
    compute_level_metrics,
    generate_goal_pursuit_obstacles,
)
from phca.evaluation.metrics.statistics import aggregate_runs, compare_groups, seed_sequence
from phca.evaluation.metrics.synergy import synergy_score
from phca.evaluation.result_schema import BenchmarkConfig, RunSummary
from phca.evaluation.trace import TraceCollector
from phca.environments.bandit_env import BanditEnv
from phca.environments.grid_world import GridWorld


def _apply_intervention_rbta(cycle: CognitiveCycle, interventions: Optional[InterventionConfig]) -> None:
    if interventions is None:
        return
    if interventions.resource_policy == "energy" or interventions.rbta_energy_scale < 1.0:
        scale = interventions.rbta_energy_scale
        bounds_map = getattr(cycle.rbta, "_bounds", None) or getattr(cycle.rbta, "module_bounds", {})
        for mod in ("G'", "ACTION", "PE", "PEU"):
            b = bounds_map.get(mod)
            if b is not None:
                cycle.rbta.update_bounds(
                    mod,
                    ResourceBounds(
                        B_time=b.B_time,
                        B_mem=b.B_mem,
                        B_energy=max(0.5, b.B_energy * scale),
                    ),
                )


def build_cycle(
    *,
    seed: int,
    grid_size: int = 5,
    use_mlp: bool = False,
    use_continuous: bool = True,
    level: int = 2,
    action_slip: float = 0.0,
    environment: str = "gridworld",
    mujoco_env: str = "Pendulum-v1",
    interventions: Optional[InterventionConfig] = None,
    trace_collector: Optional[TraceCollector] = None,
) -> CognitiveCycle:
    if environment == "bandit":
        env = BanditEnv(seed=seed)
        cycle = CognitiveCycle.build(
            env=env, seed=seed, use_mlp=use_mlp, use_continuous=use_continuous,
            interventions=interventions, trace_collector=trace_collector,
        )
        _apply_intervention_rbta(cycle, interventions)
        return cycle
    if environment == "mujoco":
        cycle = CognitiveCycle.build_for_mujoco(
            env_name=mujoco_env, seed=seed, use_mlp=use_mlp,
            interventions=interventions, trace_collector=trace_collector,
        )
        _apply_intervention_rbta(cycle, interventions)
        return cycle

    obstacles = None
    if level == 2:
        obstacles = generate_goal_pursuit_obstacles(seed + level, grid_size)
    env = GridWorld(
        size=grid_size,
        obstacles=obstacles if obstacles is not None else [],
        seed=seed,
        action_slip=action_slip,
    )
    cycle = CognitiveCycle.build(
        env=env,
        seed=seed,
        use_mlp=use_mlp,
        use_continuous=use_continuous,
        interventions=interventions,
        trace_collector=trace_collector,
    )
    if use_mlp:
        cycle.rbta.update_bounds("G'", gprime_stress_bounds(cycle))
    _apply_intervention_rbta(cycle, interventions)
    return cycle


def run_benchmark_level(
    level: int,
    config: BenchmarkConfig,
    seed: int,
    interventions: Optional[InterventionConfig] = None,
) -> RunSummary:
    """Run one benchmark level and return summary."""
    n = config.n_cycles if not config.use_mlp else max(config.n_cycles, 200)
    trace = TraceCollector(verbose=False)
    cycle = build_cycle(
        seed=seed + level,
        grid_size=config.grid_size,
        use_mlp=config.use_mlp,
        use_continuous=config.use_continuous,
        level=level,
        action_slip=config.action_slip,
        environment=getattr(config, "environment", "gridworld"),
        mujoco_env=getattr(config, "mujoco_env", "Pendulum-v1"),
        interventions=interventions,
        trace_collector=trace,
    )
    for _ in range(config.warmup):
        cycle.step()
    relocate_every = (
        config.dynamic_goals_every if (level == 2 and config.dynamic_goals) else 0
    )
    for i in range(n):
        if relocate_every and i > 0 and i % relocate_every == 0 and hasattr(cycle.env, "relocate_goal"):
            cycle.env.relocate_goal()
            cycle._env_goal_relocated = True
        cycle.step()
    history = list(cycle.metrics_history)
    result = compute_level_metrics(level, history, cycle, n, config.weights)
    emergence = compute_emergence_bundle(trace.snapshot())
    synergy = synergy_score(trace.snapshot())
    failures = _collect_failures(history, cycle)
    return RunSummary(
        seed=seed,
        metrics={
            "phi_iq": result.phi_iq,
            "prediction_accuracy": result.prediction_accuracy,
            "adaptation_speed": result.adaptation_speed,
            "goal_complexity": result.goal_complexity,
            "transfer_efficiency": result.transfer_efficiency,
            "resource_efficiency": result.resource_efficiency,
            "failure_rate": result.failure_rate,
            "synergy": synergy,
            **{f"level_{level}_phi_iq": result.phi_iq},
        },
        emergence={**emergence, "synergy": synergy},
        failures=failures,
    )


def run_full_benchmark(
    config: BenchmarkConfig,
    seed: int,
    levels: Optional[List[int]] = None,
    interventions: Optional[InterventionConfig] = None,
) -> RunSummary:
    levels = levels or [0, 1, 2, 3]
    summaries = [
        run_benchmark_level(level, config, seed, interventions) for level in levels
    ]
    phi_scores = [s.metrics["phi_iq"] for s in summaries]
    merged_metrics: Dict[str, float] = {"phi_iq": float(np.mean(phi_scores))}
    merged_emergence: Dict[str, float] = {}
    merged_failures: Dict[str, Any] = {}
    for s in summaries:
        merged_metrics.update(s.metrics)
        for k, v in s.emergence.items():
            merged_emergence[k] = max(merged_emergence.get(k, 0.0), v)
        for k, v in s.failures.items():
            merged_failures[k] = merged_failures.get(k, 0) + v
    return RunSummary(
        seed=seed,
        metrics=merged_metrics,
        emergence=merged_emergence,
        failures=merged_failures,
    )


def _collect_failures(history, cycle) -> Dict[str, Any]:
    """Log competence boundary failures."""
    goal_thrash = 0
    rbta_terminate_streak = 0
    max_terminate_streak = 0
    prev_drive: Optional[int] = None
    for m in history:
        if m.rbta_action == "TERMINATE":
            rbta_terminate_streak += 1
            max_terminate_streak = max(max_terminate_streak, rbta_terminate_streak)
        else:
            rbta_terminate_streak = 0
        if prev_drive is not None and m.drive_id != prev_drive:
            goal_thrash += 1
        prev_drive = m.drive_id
    return {
        "rbta_terminate_max_streak": max_terminate_streak,
        "total_violations": sum(m.violations_count for m in history),
        "goal_thrash_events": goal_thrash,
    }


def run_multiseed_experiment(
    *,
    name: str,
    config: BenchmarkConfig,
    n_seeds: int,
    base_seed: int = 42,
    levels: Optional[List[int]] = None,
    interventions: Optional[InterventionConfig] = None,
    baseline_fn=None,
) -> Dict[str, Any]:
    """Run experiment across seeds and aggregate."""
    seeds = seed_sequence(base_seed, n_seeds)
    runs = []
    t0 = time.perf_counter()
    for seed in seeds:
        summary = run_full_benchmark(config, seed, levels, interventions)
        summary.seed = seed
        runs.append(summary)
    duration = time.perf_counter() - t0
    metric_runs = [r.metrics for r in runs]
    aggregate = aggregate_runs(
        metric_runs,
        ["phi_iq", "synergy", "transfer_efficiency", "prediction_accuracy", "adaptation_speed"],
    )
    emergence_runs = [r.emergence for r in runs]
    emergence_agg = aggregate_runs(emergence_runs)
    result = {
        "name": name,
        "intervention": interventions.label() if interventions else "full",
        "n_seeds": n_seeds,
        "base_seed": base_seed,
        "seeds": seeds,
        "duration_s": duration,
        "aggregate": {"metrics": aggregate, "emergence": emergence_agg},
        "runs": [asdict(r) for r in runs],
    }
    if baseline_fn is not None:
        baseline_scores = [baseline_fn(s) for s in seeds]
        phca_scores = [r.metrics.get("phi_iq", 0.0) for r in runs]
        result["comparison"] = compare_groups(phca_scores, baseline_scores)
    return result


def _violation_to_dict(v: Any, cycle_idx: int) -> Dict[str, Any]:
    return {
        "cycle": cycle_idx,
        "module_id": v.module_id,
        "bound_type": v.bound_type,
        "measured": float(v.measured),
        "allowed": float(v.allowed),
    }


def run_mujoco_smoke_report(
    env_name: str,
    n_cycles: int,
    use_mlp: bool,
    seed: int = 42,
) -> Dict[str, Any]:
    """MuJoCo latency/error report (CI-compatible) plus evaluation metrics."""
    trace = TraceCollector(verbose=False)
    cycle = build_cycle(
        seed=seed, use_mlp=use_mlp, environment="mujoco", mujoco_env=env_name,
        trace_collector=trace,
    )
    for _ in range(10):
        cycle.step()
    errors, latencies, violations = [], [], 0
    action_times_ms: List[float] = []
    env_times_ms: List[float] = []
    violation_details: List[Dict[str, Any]] = []
    for i in range(n_cycles):
        m = cycle.step()
        errors.append(m.prediction_error)
        latencies.append(m.latency_ms)
        action_times_ms.append(m.module_timings.get("action_selection", 0.0))
        env_times_ms.append(m.module_timings.get("env_step", 0.0))
        violations += m.violations_count
        if m.violations_count > 0:
            for v in cycle.last_violations:
                violation_details.append(_violation_to_dict(v, i))
    early = float(np.mean(errors[:max(1, len(errors) // 4)]))
    late = float(np.mean(errors[-max(1, len(errors) // 4):]))
    emergence = compute_emergence_bundle(trace.snapshot())
    synergy = synergy_score(trace.snapshot())
    failures = _collect_failures(cycle.metrics_history, cycle)
    result: Dict[str, Any] = {
        "env": env_name, "n_cycles": n_cycles, "model": "MLP" if use_mlp else "Gaussian",
        "mean_latency_ms": float(np.mean(latencies)),
        "p95_latency_ms": float(np.percentile(latencies, 95)),
        "max_latency_ms": float(np.max(latencies)),
        "action_selection_p99_ms": float(np.percentile(action_times_ms, 99)),
        "env_step_p99_ms": float(np.percentile(env_times_ms, 99)),
        "mean_error": float(np.mean(errors)),
        "early_error": early, "late_error": late,
        "error_improved": late < early,
        "violations": violations,
        "violation_rate": violations / max(n_cycles, 1),
        "no_errors": all(np.isfinite(e) for e in errors),
        "synergy": synergy,
        "emergence": emergence,
        "failures": failures,
    }
    if violation_details:
        result["violation_details"] = violation_details
    return result


def save_result(result: Dict[str, Any], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(result, indent=2, default=str))
