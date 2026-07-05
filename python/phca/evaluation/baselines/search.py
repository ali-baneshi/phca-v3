"""BFS/DFS search baselines for GridWorld navigation."""

from __future__ import annotations

from collections import deque
from typing import Any, List, Optional, Set, Tuple


def _neighbors(pos: Tuple[int, int], size: int, grid) -> List[Tuple[int, int, int]]:
    """Return (row, col, action) reachable from pos."""
    from phca.environments.grid_world import ACTION_DELTAS

    result = []
    for action, (dr, dc) in enumerate(ACTION_DELTAS):
        if action == 4:
            continue
        r, c = pos[0] + dr, pos[1] + dc
        if 0 <= r < size and 0 <= c < size and grid[r, c] != 1:
            result.append((r, c, action))
    return result


def bfs_action(env: Any, *, full_info: bool = True) -> int:
    """First step of BFS path to goal."""
    agent_pos = tuple(env.true_agent_pos if hasattr(env, "true_agent_pos") else env.agent_pos)
    goal_pos = tuple(
        env.true_goal_pos if hasattr(env, "true_goal_pos") else env.get_goal_position() or env.goal_pos
    )
    grid = env.base.grid if hasattr(env, "base") else env.grid
    size = env.size
    if agent_pos == goal_pos:
        return getattr(env, "stay_action", 4)
    queue: deque = deque([(agent_pos, [])])
    visited: Set[Tuple[int, int]] = {agent_pos}
    while queue:
        pos, path = queue.popleft()
        if pos == goal_pos:
            return path[0] if path else getattr(env, "stay_action", 4)
        for r, c, action in _neighbors(pos, size, grid):
            if (r, c) not in visited:
                visited.add((r, c))
                queue.append(((r, c), path + [action]))
    return getattr(env, "stay_action", 4)


def dfs_action(env: Any, *, full_info: bool = True, max_depth: int = 50) -> int:
    """First step of DFS path to goal."""
    agent_pos = tuple(env.true_agent_pos if hasattr(env, "true_agent_pos") else env.agent_pos)
    goal_pos = tuple(
        env.true_goal_pos if hasattr(env, "true_goal_pos") else env.get_goal_position() or env.goal_pos
    )
    grid = env.base.grid if hasattr(env, "base") else env.grid
    size = env.size
    if agent_pos == goal_pos:
        return getattr(env, "stay_action", 4)

    def _dfs(pos: Tuple[int, int], path: List[int], visited: Set[Tuple[int, int]]) -> Optional[int]:
        if pos == goal_pos:
            return path[0] if path else getattr(env, "stay_action", 4)
        if len(path) >= max_depth:
            return None
        for r, c, action in _neighbors(pos, size, grid):
            if (r, c) not in visited:
                visited.add((r, c))
                result = _dfs((r, c), path + [action], visited)
                if result is not None:
                    return result
        return None

    result = _dfs(agent_pos, [], {agent_pos})
    return result if result is not None else getattr(env, "stay_action", 4)
