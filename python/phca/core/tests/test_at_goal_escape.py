"""At-goal parking escape + B5 suppression for Observatory GridWorld."""
from __future__ import annotations

import numpy as np

from phca.config import StateVector
from phca.core.cycle import CognitiveCycle
from phca.resilience.detector import FailureDetector
from phca.resilience.types import CycleSnapshot


def _ready_cycle(size: int = 10, seed: int = 44) -> CognitiveCycle:
    cycle = CognitiveCycle.build_for_env(
        size=size, seed=seed, use_mlp=True, use_continuous=True,
    )
    # Warm one step so MDIM/current_goal exist.
    cycle.step()
    return cycle


def test_at_goal_neighbor_probe_without_visited():
    cycle = _ready_cycle(seed=44)
    env = cycle.env
    env.agent_pos = env.goal_pos
    cycle.current_state = StateVector(
        values=env.get_observation().astype(np.float32),
        precision=np.ones(env.get_state_dim(), dtype=np.float32),
    )
    cycle.cycle_count = 50  # probe cadence
    action = cycle._select_action()
    assert int(action) != int(env.stay_action)
    assert cycle.last_action_rationale.get("at_goal_explore") is True
    assert len(cycle.last_candidate_scores) == env.action_space_size


def test_geometry_scores_populated_when_navigating():
    cycle = _ready_cycle(seed=45)
    env = cycle.env
    if tuple(env.agent_pos) == tuple(env.goal_pos):
        env.agent_pos = env.start_pos
    cycle.current_state = StateVector(
        values=env.get_observation().astype(np.float32),
        precision=np.ones(env.get_state_dim(), dtype=np.float32),
    )
    action = cycle._select_action()
    assert cycle.last_action_rationale.get("selector_mode") == "pure_geometry_ablation"
    assert len(cycle.last_candidate_scores) == env.action_space_size
    assert cycle.last_action_rationale.get("mechanism") == "greedy_fallback"
    assert isinstance(action, (int, np.integer))


def test_b5_suppressed_when_at_goal_geometry():
    det = FailureDetector()
    snap = CycleSnapshot(
        cycle_id=100,
        recent_confidences=[0.995] * 20,
        recent_actions=[4] * 20,
        extra={
            "goal_reached": True,
            "selector_mode": "pure_geometry_ablation",
        },
    )
    assert det._detect_b5(snap) == []


def test_b5_still_fires_off_goal():
    det = FailureDetector()
    snap = CycleSnapshot(
        cycle_id=100,
        recent_confidences=[0.995] * 20,
        recent_actions=[4] * 20,
        extra={
            "goal_reached": False,
            "selector_mode": "pure_geometry_ablation",
        },
    )
    events = det._detect_b5(snap)
    assert len(events) == 1
    assert events[0].mode_id == "B5"
