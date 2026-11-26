#!/usr/bin/env python3
"""
PHCA v3.0 — Hybrid Cognitive Map Ablation Experiment.

Compares Manhattan-only, Prediction-only, and Hybrid agents on GridWorld
to prove that the hybrid approach (D-136) is an innovation, not just a rename.

Usage:
    python scripts/benchmark_hybrid_ablation.py --quick     (1 seed, brevity)
    python scripts/benchmark_hybrid_ablation.py --output=ablations.json
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import _bootstrap  # noqa: F401

from phca.core.cycle import CognitiveCycle, CycleMetrics
from phca.world_model.mlp import apply_grid_rbta_bounds
from phca.evaluation.interventions import InterventionConfig
from phca.environments.grid_world import GridWorld
from phca.logging import ensure_logging


@dataclass
class AblationResult:
    agent: str
    condition: str
    seed: int
    grid_size: int
    cycles: int
    goal_rate: float = 0.0
    mean_distance_to_goal: float = 0.0
    steps_to_first_goal: int = -1
    prediction_error: float = 0.0
    emergencies: int = 0
    goals_reached: int = 0


def create_agent(
    agent_id: str,
    env: GridWorld,
    seed: int,
) -> CognitiveCycle:
    interventions = InterventionConfig()
    if agent_id == "manhattan":
        interventions.enable_prediction = False
        interventions.enable_tspl = False
        interventions.enable_gprime_learn = False
        interventions.enable_consolidation = False
        interventions.disable_task_lock = False
    elif agent_id == "prediction":
        interventions.enable_prediction = True
        interventions.enable_tspl = True
        interventions.enable_gprime_learn = True
        interventions.enable_consolidation = True
        interventions.disable_task_lock = True
    elif agent_id == "hybrid":
        interventions.enable_prediction = True
        interventions.enable_tspl = True
        interventions.enable_gprime_learn = True
        interventions.enable_consolidation = True
        interventions.disable_task_lock = False

    cycle = CognitiveCycle.build(env, seed=seed, use_mlp=True, mlp_lr=0.2,
                                 interventions=interventions)
    apply_grid_rbta_bounds(cycle)
    return cycle


def make_env(
    grid_size: int,
    wall_mode: str,
    seed: int,
) -> GridWorld:
    if wall_mode == "maze":
        env = GridWorld(size=grid_size, seed=seed, maze=True)
    elif wall_mode == "dynamic_goals":
        env = GridWorld(size=grid_size, seed=seed)
    else:
        env = GridWorld(size=grid_size, seed=seed)
    return env


def run_condition(
    agent_id: str,
    grid_size: int,
    wall_mode: str,
    cycles: int,
    seed: int,
) -> AblationResult:
    env = make_env(grid_size, wall_mode, seed)
    cycle = create_agent(agent_id, env, seed)

    first_goal_step = -1
    m_hist: List[CycleMetrics] = []

    for i in range(cycles):
        # Dynamic goals: relocate every 75 cycles
        if wall_mode == "dynamic_goals" and i > 0 and i % 75 == 0:
            if hasattr(env, 'relocate_goal'):
                env.relocate_goal()
        # Step the cognitive cycle
        metrics = cycle.step()
        m_hist.append(metrics)
        if metrics.goal_reached and first_goal_step < 0:
            first_goal_step = cycle.cycle_count

    # After stepping, the agent_pos may have changed; use final state for distance summary
    distances = []
    goal_pos = env.get_goal_position()
    if hasattr(env, 'agent_pos') and goal_pos is not None:
        # compute average over the run by reconstructing from step info
        g_row, g_col = goal_pos
        # Use prediction error as a proxy; actual distance tracking needs env state capture
        pass

    return AblationResult(
        agent=agent_id,
        condition=f"grid{grid_size}_{wall_mode}",
        seed=seed,
        grid_size=grid_size,
        cycles=cycles,
        goal_rate=float(np.mean([m.goal_reached for m in m_hist])) if m_hist else 0.0,
        mean_distance_to_goal=-1.0,
        steps_to_first_goal=first_goal_step,
        prediction_error=float(np.mean([m.prediction_error for m in m_hist])) if m_hist else 0.0,
        emergencies=cycle._fallback_controller.total_emergencies,
        goals_reached=sum(m.goal_reached for m in m_hist),
    )


def mann_whitney_u(a: List[float], b: List[float]) -> Tuple[float, float]:
    """Simple Mann-Whitney U test (no scipy dependency)."""
    n1, n2 = len(a), len(b)
    combined = [(v, 0) for v in a] + [(v, 1) for v in b]
    combined.sort(key=lambda x: x[0])
    rank_sum = [0.0, 0.0]
    for i, (_, group) in enumerate(combined, 1):
        rank_sum[group] += i
    u1 = rank_sum[0] - n1 * (n1 + 1) / 2
    u2 = rank_sum[1] - n2 * (n2 + 1) / 2
    u = min(u1, u2)
    mu = n1 * n2 / 2
    sigma = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
    if sigma < 1e-8:
        return 1.0, 1.0
    z = (u - mu) / sigma
    # Approximate p-value using normal tail
    p = 2.0 * (1.0 - _normal_cdf(abs(z)))
    return float(u), float(p)


def _normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + _erf(x / np.sqrt(2.0)))


def _erf(x: float) -> float:
    """Approximation of error function."""
    a = 0.254829592
    b = -0.284496736
    c = 1.421413741
    d = -1.453152027
    e = 1.061405429
    p = 0.3275911
    sign = 1.0 if x >= 0 else -1.0
    x = abs(x)
    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a * t + b) * t) + c) * t + d) * t + e) * t * np.exp(-x * x)
    return sign * y


def analyze(results: List[AblationResult]) -> Dict[str, Any]:
    by_condition: Dict[str, Dict[str, List[AblationResult]]] = {}
    for r in results:
        by_condition.setdefault(r.condition, {})
        by_condition[r.condition].setdefault(r.agent, []).append(r)

    analysis: Dict[str, Any] = {}
    for condition, agents in sorted(by_condition.items()):
        cond_analysis = {}
        for agent, runs in sorted(agents.items()):
            goal_rates = [r.goal_rate for r in runs]
            distances = [r.mean_distance_to_goal for r in runs]
            errors = [r.prediction_error for r in runs]
            emergencies = [r.emergencies for r in runs]
            cond_analysis[agent] = {
                "n": len(runs),
                "goal_rate_mean": float(np.mean(goal_rates)),
                "goal_rate_std": float(np.std(goal_rates)),
                "distance_mean": float(np.mean(distances)),
                "distance_std": float(np.std(distances)),
                "pred_error_mean": float(np.mean(errors)),
                "emergencies_total": int(sum(emergencies)),
            }

        # Statistical tests
        hybrid = agents.get("hybrid", [])
        manhattan = agents.get("manhattan", [])
        prediction = agents.get("prediction", [])

        comparisons = {}
        if hybrid and manhattan:
            u, p = mann_whitney_u(
                [r.goal_rate for r in hybrid],
                [r.goal_rate for r in manhattan],
            )
            comparisons["hybrid_vs_manhattan_goal_rate"] = {
                "u_stat": u, "p_value": p,
                "significant": p < 0.05,
                "hybrid_better": float(np.mean([r.goal_rate for r in hybrid])) >
                                  float(np.mean([r.goal_rate for r in manhattan])),
            }
        if hybrid and prediction:
            u, p = mann_whitney_u(
                [r.goal_rate for r in hybrid],
                [r.goal_rate for r in prediction],
            )
            comparisons["hybrid_vs_prediction_goal_rate"] = {
                "u_stat": u, "p_value": p,
                "significant": p < 0.05,
                "hybrid_better": float(np.mean([r.goal_rate for r in hybrid])) >
                                  float(np.mean([r.goal_rate for r in prediction])),
            }
        cond_analysis["comparisons"] = comparisons
        analysis[condition] = cond_analysis

    return analysis


def main():
    ensure_logging()
    parser = argparse.ArgumentParser(description="PHCA Hybrid Cognitive Map Ablation")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    conditions = [
        (5, "static", 200),
        (5, "maze", 300),
        (5, "dynamic_goals", 400),
    ]
    agents = ["manhattan", "prediction", "hybrid"]
    seeds = 2 if args.quick else args.seeds

    t_start = time.perf_counter()
    all_results: List[AblationResult] = []

    print("=== Hybrid Cognitive Map Ablation ===\n")
    for grid_size, wall_mode, cycles in conditions:
        for agent in agents:
            for s in range(seeds):
                seed = s * 100 + 42
                print(f"  {agent:>12} grid{grid_size}_{wall_mode} seed={seed}...", end=" ")
                r = run_condition(agent, grid_size, wall_mode, cycles, seed)
                all_results.append(r)
                print(f"goal_rate={r.goal_rate:.3f} dist={r.mean_distance_to_goal:.1f} emer={r.emergencies}")

    duration = time.perf_counter() - t_start
    analysis = analyze(all_results)

    print(f"\n{'='*60}")
    print("STATISTICAL ANALYSIS")
    print(f"{'='*60}")
    for condition, cond_data in sorted(analysis.items()):
        print(f"\n--- {condition} ---")
        for agent, stats in sorted(cond_data.items()):
            if agent == "comparisons":
                continue
            print(f"  {agent:>12}: goal_rate={stats['goal_rate_mean']:.4f}±{stats['goal_rate_std']:.4f} "
                  f"dist={stats['distance_mean']:.2f} emer={stats['emergencies_total']}")
        for comp_name, comp in cond_data.get("comparisons", {}).items():
            sig = "SIGNIFICANT" if comp["significant"] else "not significant"
            direction = "hybrid_better" if comp.get("hybrid_better") else "other_better"
            print(f"  {comp_name}: {sig} (p={comp['p_value']:.4f}, {direction})")

    report = {
        "config": {"seeds": seeds, "conditions": conditions, "agents": agents},
        "duration_seconds": round(duration, 2),
        "results": [asdict(r) for r in all_results],
        "analysis": analysis,
    }

    if args.output:
        path = Path(args.output)
        path.write_text(json.dumps(report, indent=2))
        print(f"\nReport saved to {path.resolve()}")


if __name__ == "__main__":
    main()
