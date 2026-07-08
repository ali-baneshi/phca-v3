"""Pendulum-v1 continual task sequence for forgetting benchmarks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass(frozen=True)
class PendulumTask:
    """Single continual-learning task: target angle + goal reference."""

    task_id: int
    target_angle: float
    goal_reference: np.ndarray
    goal_threshold: float


def build_pendulum_task_sequence(
    n_tasks: int = 2,
    base_seed: int = 42,
) -> List[PendulumTask]:
    """Build a sequence of Pendulum tasks with different target angles.

    Angles are spaced evenly from 0 (upright) to pi (hanging down).
    Each task uses the target angle as its goal reference, so the
    MPC policy drives the pendulum to different equilibrium points
    — creating distribution shift across tasks for the forgetting test.
    """
    if n_tasks <= 1:
        angles = [0.0]
    else:
        angles = np.linspace(0.0, np.pi, n_tasks)
    tasks: List[PendulumTask] = []
    for tid in range(n_tasks):
        angle = float(angles[tid])
        ref = np.array([np.cos(angle), np.sin(angle), 0.0], dtype=np.float32)
        tasks.append(PendulumTask(
            task_id=tid,
            target_angle=angle,
            goal_reference=ref,
            goal_threshold=0.90,
        ))
    return tasks


def pendulum_task_to_params(task: PendulumTask) -> dict:
    """Convert PendulumTask to param dict for env.apply_task_layout."""
    return {
        "goal_reference": task.goal_reference,
        "goal_threshold": task.goal_threshold,
    }
