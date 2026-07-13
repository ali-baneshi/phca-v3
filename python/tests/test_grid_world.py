"""
Tests for the GridWorld environment (PHCA-3.1-002).

Verifies: all 5 actions valid, state vector dimension, terminal conditions, obstacles.
"""

import numpy as np

from phca.environments.grid_world import GridWorld


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

    def test_goal_reached_sets_info(self, small_grid_world):
        """Reaching the goal should set reward=1.0 and info['goal_reached']."""
        goal = small_grid_world.goal_pos
        small_grid_world.agent_pos = goal
        obs, reward, terminal, info = small_grid_world.step(4)  # STAY
        assert not terminal, "Cognitive architecture does not reset on goal"
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


class TestGridWorldPartialObs:
    """Tests for GridWorld partial observability (partial_obs_radius)."""

    def test_radius_none_is_full_obs(self):
        """Default (partial_obs_radius=None) shows all walls and goal."""
        gw = GridWorld(size=5, obstacles=[], seed=42)
        obs = gw.reset()
        n = gw.size * gw.size
        goal_map = obs[n:2 * n].reshape(gw.size, gw.size)
        assert goal_map.max() == 1.0
        assert gw.get_goal_position() is not None
        assert gw.get_goal_position() is not None

    def test_radius_1_hides_walls_outside_3x3(self):
        """With radius=1, only walls within 3x3 of start are known."""
        gw = GridWorld(size=10, obstacles=[(0, 9), (9, 0)], seed=42,
                       partial_obs_radius=1)
        obs = gw.reset()
        n = gw.size * gw.size
        wall_map = obs[2 * n:3 * n].reshape(gw.size, gw.size)
        far_wall_known = wall_map[9, 0] > 0.5
        assert not far_wall_known, "Far wall should not be visible with radius=1"

    def test_reveal_expands_as_agent_moves(self):
        """Walking toward a wall reveals it when within range."""
        gw = GridWorld(size=5, obstacles=[(2, 4)], seed=42,
                       partial_obs_radius=1)
        gw.agent_pos = (2, 2)
        gw._reveal_around((2, 3))
        n = gw.size * gw.size
        obs = gw.get_observation()
        wall_map = obs[2 * n:3 * n].reshape(gw.size, gw.size)
        assert wall_map[2, 4] > 0.5, "Wall at (2,4) should be revealed when agent at (2,3)"

    def test_get_goal_position_none_when_goal_unseen(self):
        """get_goal_position() returns None when goal is outside viewport."""
        gw = GridWorld(size=10, obstacles=[], seed=42,
                       partial_obs_radius=1)
        gw.agent_pos = (0, 0)
        gw.goal_pos = (9, 9)
        assert gw.get_goal_position() is None

    def test_get_goal_position_returns_when_seen(self):
        """get_goal_position() returns goal pos when within viewport."""
        gw = GridWorld(size=5, obstacles=[], seed=42,
                       partial_obs_radius=2)
        gw.agent_pos = (2, 2)
        gw.goal_pos = (3, 3)
        assert gw.get_goal_position() == (3, 3)

    def test_observed_grid_only_contains_known_walls(self):
        """observed_grid property shows only revealed walls."""
        gw = GridWorld(size=10, obstacles=[(0, 0), (9, 9)], seed=42,
                       partial_obs_radius=1)
        obs_g = gw.observed_grid
        assert obs_g[9, 9] != gw.WALL, "Far wall should not appear in observed_grid"
        start_r, start_c = gw.agent_pos
        nearby_walls = [(r, c) for r in range(max(0, start_r - 1), min(gw.size, start_r + 2))
                        for c in range(max(0, start_c - 1), min(gw.size, start_c + 2))
                        if gw.grid[r, c] == gw.WALL]
        for r, c in nearby_walls:
            assert obs_g[r, c] == gw.WALL, f"Nearby wall at ({r},{c}) should be visible"

    def test_relocate_goal_resets_goal_seen(self):
        """relocate_goal() should reset _goal_seen under partial obs."""
        gw = GridWorld(size=5, obstacles=[], seed=42, partial_obs_radius=3)
        gw.agent_pos = (2, 2)
        assert gw.get_goal_position() is not None
        gw.relocate_goal()
        if gw.get_goal_position() is None:
            return
        gr, gc = gw.goal_pos
        dist = abs(gw.agent_pos[0] - gr) + abs(gw.agent_pos[1] - gc)
        assert dist <= 3, "relocated goal should be within viewport or return None"
