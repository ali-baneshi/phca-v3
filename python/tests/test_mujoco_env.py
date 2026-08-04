"""Unit tests for MuJoCoSimpleEnv environment wrapper.

Tests that the wrapper correctly implements the EnvironmentProtocol
interface required by the PHCA cognitive cycle.

Cross-ref: docs/mujoco_integration_plan.md §5.1
"""

import numpy as np
import pytest

pytest.importorskip("gymnasium")

from phca.environments.mujoco_env import MuJoCoSimpleEnv


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
    """Pendulum environment has a 1-D continuous action space (Phase 6 / A3)."""
    env = MuJoCoSimpleEnv("Pendulum-v1", seed=42)
    # A3: Pendulum is now continuous (torque ∈ [-2,2], dim 1).
    assert env.action_space_size == 1
    assert env.get_state_dim() == 3
    from phca.config import ContinuousSpace
    assert isinstance(env.get_action_space(), ContinuousSpace)
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


# ── Reacher-v5 (Phase 5 / D-093) ───────────────────────────


def test_reacher_env_creation():
    """Reacher-v5 wrapper has a 2D continuous action space and a 10-dim observation (Phase 7 / A1)."""
    from phca.config import ContinuousSpace
    env = MuJoCoSimpleEnv("Reacher-v5", seed=42)
    assert env.action_space_size == 2
    assert env.get_state_dim() == 10
    assert env.size == 1
    assert env.env_name == "Reacher-v5"
    sp = env.get_action_space()
    assert isinstance(sp, ContinuousSpace)
    assert sp.dim == 2
    np.testing.assert_allclose(sp.low, [-1.0, -1.0])
    np.testing.assert_allclose(sp.high, [1.0, 1.0])
    env.close()


def test_reacher_step_all_actions():
    """Continuous Reacher actions (sampled in [-1,1]^2) produce valid finite observations."""
    env = MuJoCoSimpleEnv("Reacher-v5", seed=42)
    env.reset()
    rng = np.random.RandomState(42)
    for _ in range(5):
        action = rng.uniform(-1.0, 1.0, size=2).astype(np.float32)
        next_obs, reward, terminal, info = env.step(action)
        assert next_obs.shape == (10,)
        assert next_obs.dtype == np.float32
        assert np.all(np.isfinite(next_obs)), "Non-finite obs on continuous Reacher action"
        assert isinstance(reward, float)
        assert isinstance(terminal, bool)
        assert isinstance(info, dict)
    env.close()


def test_reacher_goal_position_is_none():
    """Reacher has no grid goal position (continuous target, not grid-based)."""
    env = MuJoCoSimpleEnv("Reacher-v5", seed=42)
    assert env.get_goal_position() is None


def test_reacher_continuous_action_names():
    """Continuous Reacher exposes τ dim names, not discrete MOVE_* labels."""
    env = MuJoCoSimpleEnv("Reacher-v5", seed=42)
    names = env.get_action_names()
    assert names == ["τ₀", "τ₁"]
    assert not any(n.startswith("MOVE_") for n in names)
    env.close()


def test_pendulum_continuous_action_names():
    env = MuJoCoSimpleEnv("Pendulum-v1", seed=42)
    assert env.get_action_names() == ["τ"]
    env.close()


def test_reacher_goal_reached_not_always_true():
    """Default Reacher tip-error gate must not report success every step."""
    env = MuJoCoSimpleEnv("Reacher-v5", seed=42)
    env.reset()
    hits = 0
    n = 40
    rng = np.random.RandomState(0)
    for _ in range(n):
        action = rng.uniform(-1.0, 1.0, size=2).astype(np.float32)
        _, _, terminal, info = env.step(action)
        if info.get("goal_reached"):
            hits += 1
        if terminal:
            env.reset()
    assert hits < n, f"goal_reached always true ({hits}/{n})"
    assert "goal_reached" in info
    env.close()

