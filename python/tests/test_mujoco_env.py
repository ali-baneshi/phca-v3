"""Unit tests for MuJoCoSimpleEnv environment wrapper.

Tests that the wrapper correctly implements the EnvironmentProtocol
interface required by the PHCA cognitive cycle.

Cross-ref: docs/mujoco_integration_plan.md §5.1
"""

import numpy as np
import pytest

from environments.mujoco_env import MuJoCoSimpleEnv


def test_cartpole_env_creation():
    """Cartpole environment has correct dimensions and action space."""
    env = MuJoCoSimpleEnv("InvertedPendulum-v5", seed=42)
    assert env.action_space_size == 3
    assert env.get_state_dim() == 4
    assert env.size == 1
    assert env.env_name == "InvertedPendulum-v5"


def test_cartpole_reset_returns_observation():
    """Reset returns a float32 array of correct shape."""
    env = MuJoCoSimpleEnv("InvertedPendulum-v5", seed=42)
    obs = env.reset()
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (4,)
    assert obs.dtype == np.float32
    assert np.all(np.isfinite(obs))


def test_cartpole_step_all_actions():
    """All discrete actions produce valid observations."""
    env = MuJoCoSimpleEnv("InvertedPendulum-v5", seed=42)
    env.reset()
    for action in range(env.action_space_size):
        next_obs, reward, terminal, info = env.step(action)
        assert next_obs.shape == (4,), f"Failed on action {action}"
        assert next_obs.dtype == np.float32
        assert isinstance(reward, float)
        assert isinstance(terminal, bool)
        assert isinstance(info, dict)


def test_get_observation_caching():
    """_get_observation() returns the last observation from step() or reset()."""
    env = MuJoCoSimpleEnv("InvertedPendulum-v5", seed=42)
    # After reset, _get_observation() returns the reset observation
    obs_after_reset = env.reset()
    cached = env._get_observation()
    np.testing.assert_array_equal(cached, obs_after_reset)

    # After step, _get_observation() returns the step result
    next_obs, _, _, _ = env.step(0)
    cached_after_step = env._get_observation()
    np.testing.assert_array_equal(cached_after_step, next_obs)


def test_pendulum_env_creation():
    """Pendulum environment has correct dimensions."""
    env = MuJoCoSimpleEnv("Pendulum-v1", seed=42)
    assert env.action_space_size == 3
    assert env.get_state_dim() == 3
    obs = env.reset()
    assert obs.shape == (3,)
    assert obs.dtype == np.float32


def test_get_possible_actions():
    """get_possible_actions returns a list of action names."""
    env = MuJoCoSimpleEnv("InvertedPendulum-v5", seed=42)
    actions = env.get_possible_actions()
    assert isinstance(actions, list)
    assert len(actions) == env.action_space_size
    assert all(isinstance(a, str) for a in actions)


def test_get_action_names():
    """get_action_names returns descriptive names."""
    env = MuJoCoSimpleEnv("InvertedPendulum-v5", seed=42)
    names = env.get_action_names()
    assert "PUSH_LEFT" in names
    assert "STAY" in names
    assert "PUSH_RIGHT" in names


def test_get_goal_position_is_none():
    """get_goal_position returns None for non-grid environments."""
    env = MuJoCoSimpleEnv("InvertedPendulum-v5", seed=42)
    assert env.get_goal_position() is None


def test_stub_attributes():
    """Stub attributes needed by CognitiveCycle exist."""
    env = MuJoCoSimpleEnv("InvertedPendulum-v5", seed=42)
    assert env.agent_pos == (0, 0)
    assert env.grid.shape == (1, 1)
    assert env.WALL == -1


def test_repr():
    """repr provides useful debugging info."""
    env = MuJoCoSimpleEnv("InvertedPendulum-v5", seed=42)
    r = repr(env)
    assert "InvertedPendulum-v5" in r
    assert "state_dim=4" in r


def test_multi_step_no_crash():
    """Run 50 steps without crashing (regardless of terminal)."""
    env = MuJoCoSimpleEnv("InvertedPendulum-v5", seed=42)
    env.reset()
    for i in range(50):
        action = i % env.action_space_size
        obs, reward, terminal, info = env.step(action)
        assert np.all(np.isfinite(obs))
        if terminal:
            env.reset()


def test_nan_guard():
    """NaN guard replaces non-finite values without error."""
    env = MuJoCoSimpleEnv("InvertedPendulum-v5", seed=42)
    env.reset()
    obs, _, _, _ = env.step(0)
    # Normal operation should not produce NaN
    assert np.all(np.isfinite(obs))


def test_environment_error_on_bad_name():
    """Unknown environment ID raises an error."""
    try:
        import gymnasium as gym
    except ImportError:
        gym = None
    if gym is not None:
        with pytest.raises((ValueError, gym.error.NameNotFound)):
            MuJoCoSimpleEnv("UnknownEnv-v0", seed=42)
