#!/usr/bin/env python3
"""PHCA causal evidence gate.

Compares PHCA against non-PHCA GridWorld controls on extrinsic task metrics.
This answers whether PHCA improves agent behavior, not whether runtime works.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any, Deque, Dict, Iterable, List, Optional, Tuple

import numpy as np

import _bootstrap  # noqa: F401

from phca.core.cycle import CognitiveCycle
from phca.config import DiscreteSpace
from phca.environments.grid_world import ACTION_DELTAS, ACTION_NAMES, GridWorld
from phca.evaluation.baselines.greedy import greedy_action
from phca.evaluation.baselines.random_agent import random_action
from phca.evaluation.baselines.search import bfs_action
from phca.evaluation.interventions import (
    InterventionConfig,
    action_selection_interpretation,
)
from phca.evaluation.metrics.statistics import seed_sequence
from phca.evaluation.metrics.statistics import bootstrap_ci
from phca.config import ResourceBounds
from phca.world_model.mlp import gprime_stress_bounds

BASE_METRICS = (
    "goal_rate",
    "first_goal_cycle",
    "mean_distance_to_goal",
    "cumulative_reward",
)
LONG_HORIZON_METRICS = BASE_METRICS + ("coverage_rate", "switch_recovery_cycle")
HIGHER_IS_BETTER = {
    "goal_rate": True,
    "first_goal_cycle": False,
    "mean_distance_to_goal": False,
    "cumulative_reward": True,
    "coverage_rate": True,
    "switch_recovery_cycle": False,
}


@dataclass(frozen=True)
class ScenarioSpec:
    """Causal-evidence scenario configuration."""

    name: str
    description: str
    metrics: Tuple[str, ...] = BASE_METRICS
    sensor_noise: float = 0.0
    goal_delay: int = 0
    partial_map: bool = False
    dynamic_goals_every: int = 0
    dynamic_obstacles_every: int = 0
    sensor_dropout_every: int = 0
    sensor_dropout_len: int = 0
    sensor_dropout_fraction: float = 0.0
    gate_controls: Tuple[str, ...] = ("random",)
    partial_obs_radius: int = 0


SCENARIOS: Dict[str, ScenarioSpec] = {
    "level1": ScenarioSpec(
        name="level1",
        description="Simple goal navigation; greedy_full_info is expected to be the ceiling.",
        gate_controls=("random",),
    ),
    "level2": ScenarioSpec(
        name="level2",
        description="Constrained GridWorld: noisy/delayed observation, partial map, dynamic obstacles.",
        sensor_noise=0.05,
        goal_delay=3,
        partial_map=True,
        dynamic_obstacles_every=25,
        gate_controls=("random", "greedy_observed"),
    ),
    "level3": ScenarioSpec(
        name="level3",
        description="Long-horizon GridWorld: goal switching, interruption windows, partial map.",
        metrics=LONG_HORIZON_METRICS,
        sensor_noise=0.04,
        goal_delay=3,
        partial_map=True,
        dynamic_goals_every=50,
        dynamic_obstacles_every=35,
        sensor_dropout_every=45,
        sensor_dropout_len=5,
        sensor_dropout_fraction=0.2,
        gate_controls=("random", "greedy_observed"),
    ),
    "viewport1": ScenarioSpec(
        name="viewport1",
        description="GridWorld with partial_obs_radius=1 (3x3 viewport); tests gate robustness under tight FoV.",
        partial_obs_radius=1,
        gate_controls=("random", "greedy_observed"),
    ),
    "viewport2": ScenarioSpec(
        name="viewport2",
        description="GridWorld with partial_obs_radius=2 (5x5 viewport); moderate partial view.",
        partial_obs_radius=2,
        gate_controls=("random", "greedy_observed"),
    ),
    "viewport3": ScenarioSpec(
        name="viewport3",
        description="GridWorld with partial_obs_radius=3 (7x7 viewport); mild partial view.",
        partial_obs_radius=3,
        gate_controls=("random", "greedy_observed"),
    ),
}


class ScenarioGridWorld:
    """Harness-only GridWorld wrapper for constrained causal scenarios."""

    EMPTY = GridWorld.EMPTY
    WALL = GridWorld.WALL
    GOAL = GridWorld.GOAL
    HAZARD = GridWorld.HAZARD

    def __init__(
        self,
        *,
        size: int,
        obstacles: List[Tuple[int, int]],
        seed: int,
        spec: ScenarioSpec,
        action_slip: float = 0.0,
    ) -> None:
        self.base = GridWorld(size=size, obstacles=obstacles, seed=seed, action_slip=action_slip,
                              partial_obs_radius=spec.partial_obs_radius or None)
        self.size = self.base.size
        self.rng = np.random.RandomState(seed + 10_000)
        self.spec = spec
        self.step_count = 0
        self.max_steps = self.base.max_steps
        self.action_space_size = self.base.action_space_size
        self._managed_wall: Optional[Tuple[int, int]] = None
        self._goal_history: Deque[Tuple[int, int]] = deque(maxlen=max(spec.goal_delay + 1, 1))
        self._agent_history: Deque[Tuple[int, int]] = deque(maxlen=max(spec.goal_delay + 1, 1))
        self._known_walls = np.zeros((self.size, self.size), dtype=bool)
        self.switch_cycles: List[int] = []
        self.goal_reached_cycles: List[int] = []
        self.visited: set[Tuple[int, int]] = set()
        self.grid = self.base.grid.copy()
        self.agent_pos = self.base.agent_pos
        self.goal_pos = self.base.goal_pos
        self._sync_observed(force=True)

    @property
    def true_agent_pos(self) -> Tuple[int, int]:
        return tuple(self.base.agent_pos)

    @property
    def true_goal_pos(self) -> Tuple[int, int]:
        return tuple(self.base.goal_pos)

    @property
    def stay_action(self) -> int:
        return self.base.stay_action

    def get_state_dim(self) -> int:
        return self.base.get_state_dim()

    def get_possible_actions(self) -> List[str]:
        return list(ACTION_NAMES)

    def get_action_names(self) -> List[str]:
        return list(ACTION_NAMES)

    def get_action_deltas(self):
        return self.base.get_action_deltas()

    def get_action_space(self):
        return DiscreteSpace(n=self.action_space_size)

    def get_goal_position(self) -> Tuple[int, int] | None:
        return self.goal_pos

    def reset(self, seed: int | None = None) -> np.ndarray:
        obs = self.base.reset(seed=seed)
        self.step_count = 0
        self._goal_history.clear()
        self._agent_history.clear()
        self._known_walls.fill(False)
        self.switch_cycles.clear()
        self.goal_reached_cycles.clear()
        self.visited.clear()
        self._sync_observed(force=True)
        return self._transform_observation(obs)

    def relocate_goal(self) -> Tuple[int, int]:
        pos = self.base.relocate_goal()
        self.switch_cycles.append(self.step_count)
        self._sync_observed(force=True)
        return pos

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        if self.spec.dynamic_goals_every and self.step_count > 0:
            if self.step_count % self.spec.dynamic_goals_every == 0:
                self.relocate_goal()
        if self.spec.dynamic_obstacles_every and self.step_count > 0:
            if self.step_count % self.spec.dynamic_obstacles_every == 0:
                self._move_dynamic_wall()

        obs, reward, _terminal, info = self.base.step(action)
        self.step_count = self.base.step_count
        self.visited.add(self.true_agent_pos)
        if info.get("goal_reached", False):
            self.goal_reached_cycles.append(self.step_count - 1)
        self._sync_observed()
        wrapped_info = dict(info)
        wrapped_info["agent_pos"] = self.true_agent_pos
        wrapped_info["goal_pos"] = self.true_goal_pos
        return self._transform_observation(obs), reward, False, wrapped_info

    def get_observation(self) -> np.ndarray:
        """Public observation accessor (EnvironmentProtocol)."""
        return self._get_observation()

    def _get_observation(self) -> np.ndarray:
        return self._transform_observation(self.base._get_observation())

    def _sync_observed(self, *, force: bool = False) -> None:
        if self.spec.partial_obs_radius > 0:
            self.goal_pos = self.base.get_goal_position()
            self.agent_pos = tuple(self.base.agent_pos)
            self.grid = self.base.observed_grid
            return
        self._goal_history.append(self.true_goal_pos)
        self._agent_history.append(self.true_agent_pos)
        if force:
            while len(self._goal_history) < self._goal_history.maxlen:
                self._goal_history.append(self.true_goal_pos)
            while len(self._agent_history) < self._agent_history.maxlen:
                self._agent_history.append(self.true_agent_pos)
        delay = min(self.spec.goal_delay, len(self._goal_history) - 1)
        self.goal_pos = list(self._goal_history)[-1 - delay]
        self.agent_pos = list(self._agent_history)[-1 - delay]
        self._reveal_local_walls(self.true_agent_pos)
        self.grid = self._observed_grid()

    def _reveal_local_walls(self, pos: Tuple[int, int]) -> None:
        row, col = pos
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr, cc = row + dr, col + dc
                if 0 <= rr < self.size and 0 <= cc < self.size:
                    if self.base.grid[rr, cc] == self.WALL:
                        self._known_walls[rr, cc] = True

    def _observed_grid(self) -> np.ndarray:
        if not self.spec.partial_map:
            return self.base.grid.copy()
        observed = np.zeros_like(self.base.grid)
        observed[self._known_walls] = self.WALL
        observed[self.goal_pos] = self.GOAL
        return observed

    def _transform_observation(self, obs: np.ndarray) -> np.ndarray:
        out = np.asarray(obs, dtype=np.float32).copy()
        n = self.size * self.size

        if self.spec.partial_obs_radius == 0:
            out[n:2 * n] = 0.0
            gr, gc = self.goal_pos
            out[n + gr * self.size + gc] = 1.0

        if self.spec.partial_map and self.spec.partial_obs_radius == 0:
            out[2 * n:3 * n] = self._known_walls.astype(np.float32).ravel()

        if self.spec.sensor_noise > 0.0:
            out += self.rng.normal(0.0, self.spec.sensor_noise, size=out.shape).astype(np.float32)
            out = np.clip(out, 0.0, 1.0)

        if self._in_dropout_window():
            count = max(1, int(len(out) * self.spec.sensor_dropout_fraction))
            idx = self.rng.choice(len(out), size=count, replace=False)
            out[idx] = 0.0

        return out.astype(np.float32)

    def _in_dropout_window(self) -> bool:
        if not self.spec.sensor_dropout_every or not self.spec.sensor_dropout_len:
            return False
        return (self.step_count % self.spec.sensor_dropout_every) < self.spec.sensor_dropout_len

    def _move_dynamic_wall(self) -> None:
        if self._managed_wall is not None:
            row, col = self._managed_wall
            if self.base.grid[row, col] == self.WALL:
                self.base.grid[row, col] = self.EMPTY
            self._known_walls[row, col] = False

        blocked = {self.true_agent_pos, self.true_goal_pos, self.base.start_pos}
        candidates = [
            (r, c)
            for r in range(self.size)
            for c in range(self.size)
            if (r, c) not in blocked and self.base.grid[r, c] == self.EMPTY
        ]
        if not candidates:
            self._managed_wall = None
            return
        self._managed_wall = tuple(candidates[int(self.rng.randint(len(candidates)))])
        self.base.grid[self._managed_wall] = self.WALL


def generate_goal_pursuit_obstacles(seed: int, size: int = 5) -> List[Tuple[int, int]]:
    """Match benchmark.py Level-2 obstacle generation without importing the runner."""
    rng = np.random.RandomState(seed)
    obstacles: List[Tuple[int, int]] = []
    gap_row = int(rng.randint(0, size))
    for row in range(size):
        if row != gap_row:
            obstacles.append((row, 2))
    for _ in range(2):
        row, col = [int(x) for x in rng.randint(0, size, size=2)]
        if (row, col) not in obstacles:
            obstacles.append((row, col))
    return obstacles


def build_scenario_env(
    seed: int, size: int, spec: ScenarioSpec, action_slip: float = 0.0,
) -> ScenarioGridWorld:
    obstacles = generate_goal_pursuit_obstacles(seed + 2, size=size)
    return ScenarioGridWorld(
        size=size, obstacles=obstacles, seed=seed + 2, spec=spec, action_slip=action_slip,
    )


def distance_to_goal(env: ScenarioGridWorld) -> int:
    ar, ac = env.true_agent_pos
    gr, gc = env.true_goal_pos
    return abs(ar - gr) + abs(ac - gc)


def reward_from_env(env: ScenarioGridWorld) -> float:
    if env.true_agent_pos == env.true_goal_pos:
        return 1.0
    if env.base.grid[env.true_agent_pos] == env.HAZARD:
        return -0.5
    return -0.01


def greedy_distance_action(env: ScenarioGridWorld, *, full_info: bool) -> int:
    """One-step Manhattan-distance controller with no PHCA state or prediction."""
    agent_pos = env.true_agent_pos if full_info else env.agent_pos
    goal_pos = env.true_goal_pos if full_info else env.get_goal_position()
    grid = env.base.grid if full_info else env.grid
    best_action = env.stay_action
    if goal_pos is None:
        return best_action
    best_distance = abs(agent_pos[0] - goal_pos[0]) + abs(agent_pos[1] - goal_pos[1])
    for action, (dr, dc) in enumerate(ACTION_DELTAS):
        row = agent_pos[0] + dr
        col = agent_pos[1] + dc
        if not (0 <= row < env.size and 0 <= col < env.size):
            continue
        if grid[row, col] == env.WALL:
            continue
        dist = abs(row - goal_pos[0]) + abs(col - goal_pos[1])
        if dist < best_distance:
            best_distance = dist
            best_action = action
    return best_action


def switch_recovery_cycle(env: ScenarioGridWorld, cycles: int) -> Optional[float]:
    if not env.switch_cycles:
        return None
    recoveries: List[int] = []
    for switch in env.switch_cycles:
        after = [cycle for cycle in env.goal_reached_cycles if cycle >= switch]
        recoveries.append((after[0] - switch) if after else cycles + 1)
    return float(mean(recoveries)) if recoveries else None


def summarize_trace(
    *,
    agent: str,
    seed: int,
    cycles: int,
    env: ScenarioGridWorld,
    distances: List[float],
    rewards: List[float],
    goals: List[bool],
    rbta_violations: int = 0,
    rbta_violations_by_type: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    first_goal: Optional[int] = None
    for idx, reached in enumerate(goals):
        if reached:
            first_goal = idx
            break
    return {
        "agent": agent,
        "seed": int(seed),
        "cycles": int(cycles),
        "goal_rate": float(mean([1.0 if g else 0.0 for g in goals])) if goals else 0.0,
        "first_goal_cycle": first_goal,
        "mean_distance_to_goal": float(mean(distances)) if distances else 0.0,
        "cumulative_reward": float(sum(rewards)),
        "coverage_rate": float(len(env.visited) / float(env.size * env.size)),
        "switch_recovery_cycle": switch_recovery_cycle(env, cycles),
        "rbta_violation_rate": float(rbta_violations / max(cycles, 1)),
        "rbta_violations_by_type": dict(rbta_violations_by_type) if rbta_violations_by_type else {},
    }


def run_control_agent(
    agent: str,
    seed: int,
    cycles: int,
    size: int,
    spec: ScenarioSpec,
    action_slip: float = 0.0,
) -> Dict[str, Any]:
    env = build_scenario_env(seed, size, spec, action_slip=action_slip)
    rng = np.random.RandomState(seed)
    distances: List[float] = []
    rewards: List[float] = []
    goals: List[bool] = []
    for _ in range(cycles):
        if agent == "random":
            action = random_action(env, rng)
        elif agent == "greedy_observed":
            action = greedy_action(env, full_info=False)
        elif agent == "greedy_full_info":
            action = greedy_action(env, full_info=True)
        elif agent == "bfs_search":
            action = bfs_action(env, full_info=True)
        else:
            raise ValueError(f"unknown control agent {agent!r}")
        _, reward, _, info = env.step(action)
        distances.append(float(distance_to_goal(env)))
        rewards.append(float(reward))
        goals.append(bool(info.get("goal_reached", False)))
    return summarize_trace(
        agent=agent, seed=seed, cycles=cycles, env=env,
        distances=distances, rewards=rewards, goals=goals,
    )


def run_phca_agent(
    seed: int,
    cycles: int,
    size: int,
    spec: ScenarioSpec,
    *,
    use_mlp: bool,
    action_slip: float = 0.0,
    interventions: Optional[InterventionConfig] = None,
) -> Dict[str, Any]:
    env = build_scenario_env(seed, size, spec, action_slip=action_slip)
    cycle = CognitiveCycle.build(
        env=env,
        seed=seed + 2,
        use_mlp=use_mlp,
        gprime_b_time=0.080 if use_mlp else 0.020,
        interventions=interventions,
    )
    if use_mlp:
        cycle.rbta.update_bounds("G'", gprime_stress_bounds(cycle))
    po_radius = getattr(env, "partial_obs_radius", None)
    if po_radius is None:
        base = getattr(env, "base", None)
        po_radius = getattr(base, "partial_obs_radius", None) if base is not None else None
    if po_radius is not None:
        _scale = 1.0 + (10.0 - float(po_radius)) * 0.1
        for mid in list(cycle.rbta._bounds.keys()):
            b = cycle.rbta._bounds[mid]
            cycle.rbta.update_bounds(mid, ResourceBounds(
                B_time=b.B_time * _scale,
                B_mem=b.B_mem * _scale,
                B_energy=b.B_energy * _scale,
                entropy_floor=b.entropy_floor / _scale if b.entropy_floor > 0 else b.entropy_floor,
            ))
    distances: List[float] = []
    rewards: List[float] = []
    goals: List[bool] = []
    violations = 0
    violations_by_type: Dict[str, int] = {}
    selector_mode_counts: Dict[str, int] = {}
    for _ in range(cycles):
        metrics = cycle.step()
        distances.append(float(distance_to_goal(env)))
        rewards.append(reward_from_env(env))
        goals.append(bool(metrics.goal_reached))
        violations += int(metrics.violations_count)
        for bt, cnt in metrics.violations_by_type.items():
            violations_by_type[bt] = violations_by_type.get(bt, 0) + cnt
        mode = str(cycle.last_action_rationale.get("selector_mode") or "unknown")
        selector_mode_counts[mode] = selector_mode_counts.get(mode, 0) + 1
    row = summarize_trace(
        agent="phca", seed=seed, cycles=cycles, env=env,
        distances=distances, rewards=rewards, goals=goals,
        rbta_violations=violations,
        rbta_violations_by_type=violations_by_type,
    )
    row["model"] = "MLP" if use_mlp else "Gaussian"
    row["prediction_error_mean"] = float(mean(
        [m.prediction_error for m in cycle.metrics_history]
    )) if cycle.metrics_history else 0.0
    row["episode_count"] = int(
        cycle.consolidation.m3.count() if hasattr(cycle, "consolidation") else 0
    )
    row["fact_count"] = int(cycle.consolidation.get_stats().get("total_facts_stored", 0))
    row["selector_mode_counts"] = selector_mode_counts
    total_modes = sum(selector_mode_counts.values()) or 1
    row["selector_mode_pct"] = {
        k: round(100.0 * v / total_modes, 2) for k, v in selector_mode_counts.items()
    }
    geo_pct = (
        float(row["selector_mode_pct"].get("pure_geometry_ablation", 0.0))
        + float(row["selector_mode_pct"].get("adaptive_geometry_fallback", 0.0))
        + float(row["selector_mode_pct"].get("task_lock_planner", 0.0))
    )
    row["geometry_dominated"] = geo_pct >= 50.0
    return row


def _metric_value(row: Dict[str, Any], metric: str) -> float:
    value = row.get(metric)
    if value is None:
        if metric in ("first_goal_cycle", "switch_recovery_cycle"):
            return float(int(row.get("cycles", 0)) + 1)
        return 0.0
    return float(value)


def aggregate_runs(
    runs: List[Dict[str, Any]],
    metrics: Iterable[str] = BASE_METRICS,
) -> Dict[str, Dict[str, Any]]:
    by_agent: Dict[str, List[Dict[str, Any]]] = {}
    for row in runs:
        by_agent.setdefault(str(row["agent"]), []).append(row)
    summary: Dict[str, Dict[str, Any]] = {}
    for agent, rows in sorted(by_agent.items()):
        data: Dict[str, Any] = {"n": len(rows)}
        for metric in metrics:
            vals = [_metric_value(row, metric) for row in rows]
            data[f"{metric}_mean"] = float(mean(vals))
            data[f"{metric}_median"] = float(median(vals))
            data[f"{metric}_std"] = float(np.std(vals)) if len(vals) > 1 else 0.0
            ci_lo, ci_hi = bootstrap_ci(vals, seed=42)
            data[f"{metric}_ci95_lo"] = float(ci_lo)
            data[f"{metric}_ci95_hi"] = float(ci_hi)
        data["success_rate"] = float(mean([
            1.0 if row.get("first_goal_cycle") is not None else 0.0
            for row in rows
        ]))
        data["rbta_violation_rate_mean"] = float(mean([
            float(row.get("rbta_violation_rate", 0.0)) for row in rows
        ]))
        if any("prediction_error_mean" in row for row in rows):
            data["prediction_error_mean"] = float(mean([
                float(row.get("prediction_error_mean", 0.0)) for row in rows
            ]))
        if any("episode_count" in row for row in rows):
            data["episode_count_mean"] = float(mean([
                float(row.get("episode_count", 0.0)) for row in rows
            ]))
            data["fact_count_mean"] = float(mean([
                float(row.get("fact_count", 0.0)) for row in rows
            ]))
        if agent == "phca" and any("selector_mode_counts" in row for row in rows):
            merged: Dict[str, int] = {}
            for row in rows:
                for mode, cnt in (row.get("selector_mode_counts") or {}).items():
                    merged[str(mode)] = merged.get(str(mode), 0) + int(cnt)
            total = sum(merged.values()) or 1
            data["selector_mode_counts"] = merged
            data["selector_mode_pct"] = {
                k: round(100.0 * v / total, 2) for k, v in merged.items()
            }
            data["geometry_dominated_frac"] = float(mean([
                1.0 if row.get("geometry_dominated") else 0.0 for row in rows
            ]))
        summary[agent] = data
    return summary


def compare_agents(
    summary: Dict[str, Dict[str, Any]],
    *,
    metrics: Iterable[str] = BASE_METRICS,
    gate_controls: Iterable[str] = ("random",),
) -> Dict[str, Any]:
    phca = summary.get("phca")
    metric_list = list(metrics)
    gate_control_list = list(gate_controls)
    if phca is None:
        return {"gate": {"passed": False, "reason": "missing phca rows"}}
    comparisons: Dict[str, Any] = {}
    gate_details: Dict[str, Any] = {}
    gate_passed = True
    for other in ("random", "greedy_observed", "greedy_full_info"):
        if other not in summary:
            continue
        per_metric: Dict[str, Dict[str, Any]] = {}
        better_count = 0
        worse_count = 0
        for metric in metric_list:
            key = f"{metric}_mean"
            phca_value = float(phca[key])
            other_value = float(summary[other][key])
            delta = phca_value - other_value
            higher = HIGHER_IS_BETTER[metric]
            better = delta > 0 if higher else delta < 0
            worse = delta < 0 if higher else delta > 0
            better_count += int(better)
            worse_count += int(worse)
            per_metric[metric] = {
                "phca": phca_value,
                other: other_value,
                "delta_phca_minus_other": delta,
                "phca_better": bool(better),
                "phca_worse": bool(worse),
            }
        comparisons[other] = {
            "metrics": per_metric,
            "phca_better_count": better_count,
            "phca_worse_count": worse_count,
        }
        if other in gate_control_list:
            needed = max(1, int(np.ceil(len(metric_list) * 0.75)))
            passed = better_count >= needed
            gate_details[other] = {
                "passed": bool(passed),
                "phca_better_count": int(better_count),
                "required": int(needed),
            }
            gate_passed = gate_passed and passed
    phca_summary = summary.get("phca") or {}
    sample_sizes = {
        agent: int(data.get("n", 0)) for agent, data in summary.items()
    }
    minimum_causal_seeds = 30
    minimum_diagnostic_seeds = 10
    phca_n = int(phca_summary.get("n", 0))
    control_ns = {
        control: int((summary.get(control) or {}).get("n", 0))
        for control in gate_control_list
    }
    all_adequate = phca_n >= minimum_causal_seeds and all(
        n >= minimum_causal_seeds for n in control_ns.values()
    )
    all_diagnostic = phca_n >= minimum_diagnostic_seeds and all(
        n >= minimum_diagnostic_seeds for n in control_ns.values()
    )
    evidence_quality = (
        "causal_power" if all_adequate
        else "diagnostic_power" if all_diagnostic
        else "smoke_power"
    )
    geo_dom = float(phca_summary.get("geometry_dominated_frac", 0.0))
    pe_raw = phca_summary.get("prediction_error_mean")
    pe_mean = float(pe_raw) if pe_raw is not None else None
    secondary_prediction = {
        "gated": False,
        "prediction_error_mean": pe_mean,
        "note": (
            "PHCA-only model-fit dual-report (D-195 / L4 dual PE pattern). "
            "Scenario PASS under geometry-dominated selection ≠ prediction competence; "
            "PE does not affect gate.passed."
        ),
    }
    base_note = (
        "PASS/FAIL under geometry-dominated action selection measures planner "
        "competence vs baselines, not prediction-primary control (D-156/D-161). "
        "RBTA violation rate is reported but not gated."
        if geo_dom >= 0.5
        else (
            "PASS/FAIL under non-geometry-dominated selection; still compare "
            "selector_mode_pct and RBTA rates before claiming prediction-primary."
        )
    )
    comparisons["gate"] = {
        "passed": bool(gate_passed),
        "controls": gate_details,
        "rule": "PHCA must beat each gated control on >=75% of scenario metrics",
        "note": "greedy_full_info is reported as a ceiling unless explicitly gated",
        "sample_sizes": sample_sizes,
        "evidence_quality": evidence_quality,
        "statistical_power": {
            "minimum_diagnostic_seeds": minimum_diagnostic_seeds,
            "minimum_causal_seeds": minimum_causal_seeds,
            "phca_seeds": phca_n,
            "gated_control_seeds": control_ns,
            "promotion_ready": bool(all_adequate),
            "note": (
                "Scenario PASS/FAIL remains descriptive below 30 shared seeds; "
                "use the 30-seed profile for causal promotion decisions."
            ),
        },
        "rbta_violation_rate_mean": phca_summary.get("rbta_violation_rate_mean"),
        "geometry_dominated_frac": geo_dom,
        "selector_mode_pct": phca_summary.get("selector_mode_pct"),
        "prediction_error_mean": pe_mean,
        "secondary_prediction": secondary_prediction,
        "interpretation_note": (
            f"{base_note} See secondary_prediction for PE dual-report (not gated)."
        ),
    }
    return comparisons


def run_level(
    *,
    level: str,
    cycles: int,
    seeds: int,
    size: int,
    agents: Iterable[str],
    use_mlp: bool,
    base_seed: int = 42,
    action_slip: float = 0.0,
    interventions: Optional[InterventionConfig] = None,
) -> Dict[str, Any]:
    spec = SCENARIOS[level]
    selected = list(agents)
    runs: List[Dict[str, Any]] = []
    for seed in seed_sequence(base_seed, seeds):
        if "phca" in selected:
            runs.append(run_phca_agent(
                seed, cycles, size, spec, use_mlp=use_mlp, action_slip=action_slip,
                interventions=interventions,
            ))
        for agent in ("random", "greedy_observed", "greedy_full_info", "bfs_search"):
            if agent in selected:
                runs.append(run_control_agent(
                    agent, seed, cycles, size, spec, action_slip=action_slip,
                ))
    summary = aggregate_runs(runs, spec.metrics)
    iv = interventions or InterventionConfig()
    interp = action_selection_interpretation(iv, environment="gridworld")
    return {
        "config": {
            "level": level,
            "description": spec.description,
            "cycles": int(cycles),
            "seeds": int(seeds),
            "base_seed": int(base_seed),
            "grid_size": int(size),
            "action_slip": float(action_slip),
            "agents": selected,
            "metrics": list(spec.metrics),
            "gate_controls": list(spec.gate_controls),
            "phca_model": "MLP" if use_mlp else "Gaussian",
            "action_selection_mode": interp["action_selection_mode"],
            "disable_blended_scorer": bool(iv.disable_blended_scorer),
            "interpretation_caveat": interp["interpretation_caveat"],
        },
        "runs": runs,
        "summary": summary,
        "comparisons": compare_agents(
            summary, metrics=spec.metrics, gate_controls=spec.gate_controls,
        ),
    }


def parse_level_seeds(raw: str, default: int) -> Dict[str, int]:
    """Parse ``level3=10,level2=5`` overrides; unknown levels are ignored."""
    out: Dict[str, int] = {}
    if not raw.strip():
        return out
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise SystemExit(f"invalid --level-seeds entry {part!r} (want level=N)")
        level, count = part.split("=", 1)
        level = level.strip()
        if level not in SCENARIOS:
            raise SystemExit(f"unknown level in --level-seeds: {level!r}")
        out[level] = int(count.strip())
    return out


def run_evaluation(
    *,
    cycles: int,
    seeds: int,
    size: int,
    agents: Iterable[str],
    use_mlp: bool,
    levels: Iterable[str] = ("level1",),
    base_seed: int = 42,
    action_slip: float = 0.0,
    level_seeds: Optional[Dict[str, int]] = None,
    interventions: Optional[InterventionConfig] = None,
) -> Dict[str, Any]:
    selected_levels = list(levels)
    overrides = level_seeds or {}
    reports = {
        level: run_level(
            level=level, cycles=cycles,
            seeds=int(overrides.get(level, seeds)),
            size=size,
            agents=agents, use_mlp=use_mlp, base_seed=base_seed, action_slip=action_slip,
            interventions=interventions,
        )
        for level in selected_levels
    }
    if len(reports) == 1:
        return next(iter(reports.values()))
    return {
        "config": {
            "cycles": int(cycles),
            "seeds": int(seeds),
            "level_seeds": {k: int(v) for k, v in overrides.items()},
            "base_seed": int(base_seed),
            "grid_size": int(size),
            "action_slip": float(action_slip),
            "levels": selected_levels,
            "agents": list(agents),
            "phca_model": "MLP" if use_mlp else "Gaussian",
        },
        "levels": reports,
        "gate": {
            "passed": all(
                report["comparisons"]["gate"].get("passed", False)
                for report in reports.values()
            )
        },
    }


def parse_levels(raw: str) -> List[str]:
    if raw == "all":
        return ["level1", "level2", "level3"]
    levels = [part.strip() for part in raw.split(",") if part.strip()]
    unknown = [level for level in levels if level not in SCENARIOS]
    if unknown:
        raise SystemExit(f"unknown levels: {', '.join(unknown)}")
    return levels


def _print_gate_failures(comparisons: Dict[str, Any], *, level: str = "") -> None:
    """Print per-control gate details and metrics where PHCA did not win."""
    gate = comparisons.get("gate", {})
    prefix = f"{level}: " if level else ""
    for control, detail in gate.get("controls", {}).items():
        if detail.get("passed"):
            continue
        better = detail.get("phca_better_count", 0)
        required = detail.get("required", 0)
        print(f"  {prefix}{control}: {better}/{required} metrics (PHCA better)")
        comp = comparisons.get(control, {}).get("metrics", {})
        for metric, mdata in comp.items():
            if not mdata.get("phca_better") and mdata.get("phca_worse"):
                phca_v = mdata.get("phca")
                other_v = mdata.get(control)
                print(f"    {metric}: phca={phca_v} vs {control}={other_v}")


def _print_secondary_prediction(gate: Dict[str, Any]) -> None:
    """Print PE dual-report (not part of scenario PASS/FAIL)."""
    pe = gate.get("prediction_error_mean")
    if pe is None:
        sec = gate.get("secondary_prediction") or {}
        pe = sec.get("prediction_error_mean")
    if pe is not None:
        print(f"  prediction_error_mean={pe} (secondary; not gated)")


def main() -> None:
    parser = argparse.ArgumentParser(description="PHCA causal evidence evaluation")
    parser.add_argument("--cycles", type=int, default=200)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument(
        "--level-seeds", default="",
        help="Per-level seed overrides, e.g. level3=10 (default: --seeds for all)",
    )
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--grid-size", type=int, default=5, choices=[5, 10, 20])
    parser.add_argument("--action-slip", type=float, default=0.0)
    parser.add_argument("--levels", default="level1")
    parser.add_argument("--agents", default="phca,random,greedy_observed,greedy_full_info")
    parser.add_argument("--use-mlp", action="store_true")
    parser.add_argument("--enable-blended-scorer", action="store_true",
                        help="D-156: enable G' prediction blend (default now pure geometry)")
    parser.add_argument("--confidence-gated", action="store_true",
                        help="Enable opt-in confidence-gated discrete selection")
    parser.add_argument("--confidence-gated-warmup", type=int, default=50,
                        help="Geometry-only cycles before confidence-gated selection")
    parser.add_argument("--confidence-gated-threshold", type=float, default=0.9,
                        help="Minimum G' confidence for prediction-scored selection")
    parser.add_argument("--confidence-gated-window", type=int, default=30,
                        help="Rolling outcome window for confidence-gated fallback")
    parser.add_argument("--output", default="logs/phca_causal_eval.json")
    parser.add_argument("--gate", action="store_true")
    args = parser.parse_args()

    agents = [a.strip() for a in args.agents.split(",") if a.strip()]
    iv = InterventionConfig(
        disable_blended_scorer=not (args.enable_blended_scorer or args.confidence_gated),
        confidence_gated_selector=bool(args.confidence_gated),
        confidence_gated_warmup_cycles=args.confidence_gated_warmup,
        confidence_gated_threshold=args.confidence_gated_threshold,
        confidence_gated_window=args.confidence_gated_window,
    )
    report = run_evaluation(
        cycles=args.cycles,
        seeds=args.seeds,
        size=args.grid_size,
        agents=agents,
        use_mlp=args.use_mlp,
        levels=parse_levels(args.levels),
        base_seed=args.base_seed,
        action_slip=args.action_slip,
        level_seeds=parse_level_seeds(args.level_seeds, args.seeds),
        interventions=iv,
    )
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    if "levels" in report:
        print(f"Wrote {out}")
        print(
            f"Action selection: "
            f"{'confidence_gated' if args.confidence_gated else 'blended' if args.enable_blended_scorer else 'pure_geometry_default'}"
        )
        critical_failure = False
        for level, level_report in report["levels"].items():
            gate = level_report["comparisons"].get("gate", {})
            level_passed = bool(gate.get("passed", False))
            print(f"{level}: {'PASS' if level_passed else 'FAIL'} — {gate.get('rule')}")
            if gate.get("geometry_dominated_frac") is not None:
                print(
                    f"  geometry_dominated_frac={gate.get('geometry_dominated_frac')} "
                    f"rbta_violation_rate_mean={gate.get('rbta_violation_rate_mean')}"
                )
            _print_secondary_prediction(gate)
            if gate.get("interpretation_note"):
                print(f"  note: {gate['interpretation_note']}")
            if args.gate and not level_passed:
                _print_gate_failures(level_report["comparisons"], level=level)
                # L2 failure is expected per D-161; only L3 is critical
                if level in ("level3", "level4"):
                    critical_failure = True
        if args.gate and critical_failure:
            sys.exit(1)
    else:
        gate = report["comparisons"].get("gate", {})
        overall_passed = bool(gate.get("passed", False))
        print(f"Wrote {out}")
        print(
            f"Action selection: "
            f"{report.get('config', {}).get('action_selection_mode', 'unknown')}"
        )
        print(f"Gate: {'PASS' if overall_passed else 'FAIL'} — {gate.get('rule')}")
        if gate.get("geometry_dominated_frac") is not None:
            print(
                f"  geometry_dominated_frac={gate.get('geometry_dominated_frac')} "
                f"rbta_violation_rate_mean={gate.get('rbta_violation_rate_mean')}"
            )
        _print_secondary_prediction(gate)
        if gate.get("interpretation_note"):
            print(f"  note: {gate['interpretation_note']}")
        if args.gate and not overall_passed:
            _print_gate_failures(report["comparisons"])
            sys.exit(1)


if __name__ == "__main__":
    main()
