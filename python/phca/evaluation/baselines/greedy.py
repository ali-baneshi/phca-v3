"""Greedy Manhattan-distance baseline."""

from __future__ import annotations

from typing import Any

from phca.environments.grid_world import ACTION_DELTAS


def greedy_action(
    env: Any,
    *,
    full_info: bool = False,
) -> int:
    """One-step Manhattan controller."""
    agent_pos = (
        env.true_agent_pos if full_info and hasattr(env, "true_agent_pos") else env.agent_pos
    )
    goal_pos = env.true_goal_pos if full_info and hasattr(env, "true_goal_pos") else env.get_goal_position()
    grid = env.base.grid if full_info and hasattr(env, "base") else env.grid
    stay = getattr(env, "stay_action", 4)
    best_action = stay
    if goal_pos is None:
        return best_action
    best_distance = abs(agent_pos[0] - goal_pos[0]) + abs(agent_pos[1] - goal_pos[1])
    size = env.size
    for action, (dr, dc) in enumerate(ACTION_DELTAS):
        row = agent_pos[0] + dr
        col = agent_pos[1] + dc
        if not (0 <= row < size and 0 <= col < size):
            continue
        if grid[row, col] == 1:  # WALL
            continue
        dist = abs(row - goal_pos[0]) + abs(col - goal_pos[1])
        if dist < best_distance:
            best_distance = dist
            best_action = action
    return best_action
