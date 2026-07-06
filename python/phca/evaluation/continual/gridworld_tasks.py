"""Level-4-lite GridWorld continual task sequence builder."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from phca.evaluation.metrics.phi_iq import generate_goal_pursuit_obstacles


@dataclass(frozen=True)
class GridWorldTask:
    """Single continual-learning task: goal position + obstacle layout."""

    task_id: int
    goal_pos: Tuple[int, int]
    obstacles: List[Tuple[int, int]]
    seed_offset: int


def _sample_goal_pos(
    rng: np.random.RandomState,
    grid_size: int,
    obstacles: List[Tuple[int, int]],
) -> Tuple[int, int]:
    blocked = set(obstacles)
    candidates = [
        (r, c)
        for r in range(grid_size)
        for c in range(grid_size)
        if (r, c) not in blocked
    ]
    if not candidates:
        return (grid_size - 1, grid_size - 1)
    idx = int(rng.randint(0, len(candidates)))
    return candidates[idx]


def build_task_sequence(
    n_tasks: int = 10,
    grid_size: int = 5,
    base_seed: int = 42,
) -> List[GridWorldTask]:
    """Build a rotated goal/obstacle sequence for Level-4-lite benchmarks.

    Each task uses a distinct obstacle layout (via ``generate_goal_pursuit_obstacles``)
    and goal position. No task-boundary signal is sent to the agent — the runner
    applies layouts externally via ``GridWorld.apply_task_layout``.
    """
    tasks: List[GridWorldTask] = []
    for task_id in range(n_tasks):
        seed_offset = base_seed + task_id * 17
        rng = np.random.RandomState(seed_offset)
        obstacles = generate_goal_pursuit_obstacles(seed_offset, grid_size)
        goal_pos = _sample_goal_pos(rng, grid_size, obstacles)
        tasks.append(
            GridWorldTask(
                task_id=task_id,
                goal_pos=goal_pos,
                obstacles=obstacles,
                seed_offset=seed_offset,
            )
        )
    return tasks
