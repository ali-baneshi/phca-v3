"""
Grid-World Environment — Phase 3.1 Benchmark Environment.

A configurable 2D grid-world for testing the PHCA cognitive cycle.
Agent navigates from start to goal using 5 discrete actions (N/S/E/W/STAY).

Supports 3 grid sizes: 5×5, 10×10, 20×20.

Cross-ref: Playbook §D.1, PHCA-3.1-002
"""

from __future__ import annotations

import numpy as np
from typing import Literal

from phca.logging import logger, _log

# Action mappings
Action = Literal[0, 1, 2, 3, 4]
ACTION_NAMES = ["MOVE_N", "MOVE_S", "MOVE_E", "MOVE_W", "STAY"]
ACTION_DELTAS = [
    (-1, 0),  # MOVE_N
    (1, 0),   # MOVE_S
    (0, 1),   # MOVE_E
    (0, -1),  # MOVE_W
    (0, 0),   # STAY
]


class GridWorld:
    """
    A 2D grid-world environment.

    The agent perceives its current cell type (one-hot encoded) plus
    a local 3×3 neighborhood view. The goal is to reach the target cell.

    Attributes:
        size: Grid dimensions (size × size).
        grid: size × size integer array (0=empty, 1=wall, 2=goal, 3=hazard).
        agent_pos: Current (row, col) position of the agent.
        goal_pos: (row, col) position of the goal.
        step_count: Number of steps taken in current episode.
        max_steps: Maximum steps before episode termination.
    """

    # Cell types
    EMPTY = 0
    WALL = 1
    GOAL = 2
    HAZARD = 3

    def __init__(self, size: int = 10, obstacles: list[tuple[int, int]] | None = None, seed: int = 42):
        """
        Initialize the grid-world.

        Args:
            size: Grid dimensions (size × size). Must be 5, 10, or 20.
            obstacles: List of (row, col) wall positions. If None, random walls generated.
            seed: Random seed for reproducibility.
        """
        assert size in (5, 10, 20), f"size must be 5, 10, or 20, got {size}"
        self.size = size
        self.rng = np.random.RandomState(seed)
        self.max_steps = size * size * 4

        # Initialize empty grid
        self.grid = np.zeros((size, size), dtype=np.int32)

        # Place walls
        if obstacles is not None:
            for r, c in obstacles:
                self.grid[r, c] = self.WALL
        else:
            self._generate_random_walls()

        # Place goal at a random empty cell
        empty_cells = self._get_empty_cells()
        self.goal_pos = tuple(empty_cells[self.rng.randint(len(empty_cells))])
        self.grid[self.goal_pos] = self.GOAL

        # Agent starts at a random empty cell (not goal)
        start_cells = [c for c in empty_cells if c != self.goal_pos]
        self.start_pos = tuple(start_cells[self.rng.randint(len(start_cells))])
        self.agent_pos = self.start_pos
        self.step_count = 0

        _log(logger, "info", "grid_world.init", size=size, start=self.start_pos, goal=self.goal_pos)

    def _generate_random_walls(self, wall_density: float = 0.15):
        """Place random walls with the given density."""
        for r in range(self.size):
            for c in range(self.size):
                if self.rng.random() < wall_density:
                    self.grid[r, c] = self.WALL

    def _get_empty_cells(self) -> list[tuple[int, int]]:
        """Return list of (row, col) for cells that are not walls."""
        return [(r, c) for r in range(self.size) for c in range(self.size)
                if self.grid[r, c] != self.WALL]

    def reset(self, seed: int | None = None) -> np.ndarray:
        """
        Reset the environment to the start state.

        Args:
            seed: Optional new seed for reproducibility.

        Returns:
            Initial state vector (flattened grid + agent position).
        """
        if seed is not None:
            self.rng = np.random.RandomState(seed)
        self.agent_pos = self.start_pos
        self.step_count = 0
        return self._get_observation()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        """
        Execute an action in the environment.

        Args:
            action: 0=NORTH, 1=SOUTH, 2=EAST, 3=WEST, 4=STAY

        Returns:
            (next_state, reward, terminal, info)
        """
        assert 0 <= action <= 4, f"Invalid action {action}"

        dr, dc = ACTION_DELTAS[action]
        new_r = self.agent_pos[0] + dr
        new_c = self.agent_pos[1] + dc

        # Check bounds and walls
        if (0 <= new_r < self.size and 0 <= new_c < self.size
                and self.grid[new_r, new_c] != self.WALL):
            self.agent_pos = (new_r, new_c)

        self.step_count += 1

        # Compute reward
        at_goal = self.agent_pos == self.goal_pos
        at_hazard = self.grid[self.agent_pos] == self.HAZARD
        reward = 1.0 if at_goal else -0.5 if at_hazard else -0.01

        # Terminal condition (not on goal — cognitive architecture doesn't reset on achievement)
        terminal = self.step_count >= self.max_steps

        info = {
            "agent_pos": self.agent_pos,
            "steps": self.step_count,
            "goal_reached": at_goal,
            "hazard_hit": at_hazard,
        }

        return self._get_observation(), reward, terminal, info

    def _get_observation(self) -> np.ndarray:
        """
        Compute the state vector for the current timestep.

        Returns a flattened representation:
          - One-hot encoded agent position (size × size values: 1 for agent, 0 elsewhere)
          - One-hot encoded goal position
          - One-hot encoded walls
          - Local 3×3 neighborhood centered on agent

        Total dimensionality: size² + size² + size² + 9
        """
        r, c = self.agent_pos

        # Agent position map
        agent_map = np.zeros((self.size, self.size), dtype=np.float32)
        agent_map[r, c] = 1.0

        # Goal position map
        goal_map = np.zeros((self.size, self.size), dtype=np.float32)
        gr, gc = self.goal_pos
        goal_map[gr, gc] = 1.0

        # Wall map
        wall_map = (self.grid == self.WALL).astype(np.float32)

        # Local 3×3 neighborhood
        local_view = np.zeros((3, 3), dtype=np.float32)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.size and 0 <= nc < self.size:
                    local_view[dr + 1, dc + 1] = float(self.grid[nr, nc])

        observation = np.concatenate([
            agent_map.ravel(),
            goal_map.ravel(),
            wall_map.ravel(),
            local_view.ravel(),
        ]).astype(np.float32)

        return observation

    def render(self) -> str:
        """Return an ASCII visualization of the current grid."""
        lines = []
        lines.append(f"Step {self.step_count} / {self.max_steps}")
        lines.append("+" + "---+" * self.size)
        for r in range(self.size):
            row = "|"
            for c in range(self.size):
                if (r, c) == self.agent_pos:
                    row += " A "
                elif (r, c) == self.goal_pos:
                    row += " G "
                elif self.grid[r, c] == self.WALL:
                    row += "███"
                elif self.grid[r, c] == self.HAZARD:
                    row += " H "
                else:
                    row += "   "
                row += "|"
            lines.append(row)
            lines.append("+" + "---+" * self.size)
        return "\n".join(lines)

    def get_state_dim(self) -> int:
        """Get the dimensionality of the state vector."""
        return self.size * self.size * 3 + 9

    @property
    def action_space_size(self) -> int:
        return 5

    @property
    def stay_action(self) -> int:
        """Return the index of the STAY action (last action)."""
        return 4

    def get_possible_actions(self) -> list[str]:
        return list(ACTION_NAMES)

    def get_action_names(self) -> list[str]:
        return list(ACTION_NAMES)

    def get_goal_position(self) -> tuple[int, int] | None:
        return self.goal_pos

    def relocate_goal(self) -> tuple[int, int]:
        """Move the goal to a new random empty cell (not a wall, not the agent cell).

        Used by the Level-2 dynamic-goal curriculum (Week 3) to force re-navigation
        so adaptation has real headroom. The cognitive cycle re-reads `goal_pos`
        every step (mdim_context + _compute_distance_gain), so no other wiring is
        needed. Returns the new goal position.
        """
        self.grid[self.goal_pos] = self.EMPTY
        empty = [c for c in self._get_empty_cells()
                 if c != tuple(self.agent_pos)]
        new_pos = tuple(empty[self.rng.randint(len(empty))])
        self.grid[new_pos] = self.GOAL
        self.goal_pos = new_pos
        return new_pos
