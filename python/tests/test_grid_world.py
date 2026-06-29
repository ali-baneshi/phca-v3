"""
Tests for the GridWorld environment (PHCA-3.1-002).

Verifies: all 5 actions valid, state vector dimension, terminal conditions, obstacles.
"""

import numpy as np
import pytest

from environments.grid_world import GridWorld, ACTION_NAMES


class TestGridWorld:
    """Test suite for GridWorld."""

    def test_all_actions_valid(self, small_grid_world):
        """All 5 actions should execute without errors."""
        for action in range(5):
            obs, reward, terminal, info = small_grid_world.step(action)
            assert obs is not None
            assert isinstance(terminal, bool)
            assert 0 <= info["agent_pos"][0] < small_grid_world.size
            assert 0 <= info["agent_pos"][1] < small_grid_world.size

    def test_state_vector_dimension(self, small_grid_world):
        """State vector dimension should match get_state_dim()."""
        expected_dim = small_grid_world.get_state_dim()
        obs = small_grid_world.reset()
        assert obs.shape == (expected_dim,), f"Expected ({expected_dim},), got {obs.shape}"

    def test_state_dim_for_all_sizes(self):
        """State dimension should be correct for all grid sizes."""
        for size in (5, 10, 20):
            gw = GridWorld(size=size, seed=42)
            obs = gw.reset()
            expected = size * size * 3 + 9
            assert obs.shape == (expected,), f"Size {size}: expected ({expected},), got {obs.shape}"
            assert gw.get_state_dim() == expected

    def test_goal_reached_triggers_terminal(self, small_grid_world):
        """Reaching the goal should set terminal=True and reward=1.0."""
        # Manually move agent to goal
        goal = small_grid_world.goal_pos
        small_grid_world.agent_pos = goal
        obs, reward, terminal, info = small_grid_world.step(4)  # STAY
        assert terminal, "Agent at goal should trigger terminal"
        assert reward == 1.0, f"Expected reward 1.0, got {reward}"
        assert info["goal_reached"]

    def test_obstacles_block_movement(self):
        """Walls should block agent movement."""
        gw = GridWorld(size=5, obstacles=[(0, 1)], seed=42)
        gw.agent_pos = (0, 0)
        # Try to move into wall at (0, 1)
        gw.step(2)  # MOVE_E
        assert gw.agent_pos == (0, 0), "Agent should not move into wall"

    def test_move_north(self):
        """MOVE_N should decrease row by 1."""
        gw = GridWorld(size=5, obstacles=[], seed=42)
        gw.agent_pos = (2, 2)
        gw.step(0)  # MOVE_N
        assert gw.agent_pos == (1, 2)

    def test_move_south(self):
        """MOVE_S should increase row by 1."""
        gw = GridWorld(size=5, obstacles=[], seed=42)
        gw.agent_pos = (2, 2)
        gw.step(1)  # MOVE_S
        assert gw.agent_pos == (3, 2)

    def test_move_east(self):
        """MOVE_E should increase column by 1."""
        gw = GridWorld(size=5, obstacles=[], seed=42)
        gw.agent_pos = (2, 2)
        gw.step(2)  # MOVE_E
        assert gw.agent_pos == (2, 3)

    def test_move_west(self):
        """MOVE_W should decrease column by 1."""
        gw = GridWorld(size=5, obstacles=[], seed=42)
        gw.agent_pos = (2, 2)
        gw.step(3)  # MOVE_W
        assert gw.agent_pos == (2, 1)

    def test_stay_does_not_move(self):
        """STAY should keep the agent in place."""
        gw = GridWorld(size=5, obstacles=[], seed=42)
        gw.agent_pos = (2, 2)
        gw.step(4)  # STAY
        assert gw.agent_pos == (2, 2)

    def test_max_steps_triggers_terminal(self):
        """Episode should terminate after max_steps without reaching goal."""
        gw = GridWorld(size=5, seed=42)
        gw.max_steps = 10
        gw.agent_pos = (0, 0)
        gw.goal_pos = (4, 4)  # far away
        for _ in range(9):
            obs, reward, terminal, info = gw.step(4)  # STAY
            assert not terminal
        obs, reward, terminal, info = gw.step(4)  # 10th step
        assert terminal, "Episode should terminate after max_steps"

    def test_render_returns_string(self, small_grid_world):
        """render() should return a non-empty string."""
        rendered = small_grid_world.render()
        assert isinstance(rendered, str)
        assert len(rendered) > 0
        assert "A" in rendered  # Agent visible
        assert "G" in rendered  # Goal visible

    def test_reset_returns_to_start(self, small_grid_world):
        """reset() should return the agent to the start position."""
        start = small_grid_world.start_pos
        small_grid_world.agent_pos = (0, 0)  # move away
        small_grid_world.reset()
        assert small_grid_world.agent_pos == start
        assert small_grid_world.step_count == 0

    def test_deterministic_with_seed(self):
        """Same seed should produce same environment."""
        gw1 = GridWorld(size=10, seed=42)
        gw2 = GridWorld(size=10, seed=42)
        assert gw1.start_pos == gw2.start_pos
        assert gw1.goal_pos == gw2.goal_pos
        np.testing.assert_array_equal(gw1.grid, gw2.grid)
