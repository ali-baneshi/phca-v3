"""Phase 6 / A4 + Phase 7 / A2 — unit tests for the continuous-action path.

Covers:
  - Pendulum-v1 exposes a true ContinuousSpace (dim=1, [-2,2]) + upright goal ref.
  - The cognitive cycle drives Pendulum continuously: finite obs, action_taken=-1,
    action_name="continuous", 0 RBTA violations, action within bounds.
  - Reacher-v5 exposes a true ContinuousSpace (dim=2, [-1,1]) + fingertip-on-target
    goal ref (Phase 7 / A1 — extends continuous control to Reacher's 2D action space).
  - Cartpole stays discrete (regression guard for the discrete path).
  - GridWorld stays discrete (regression guard for the canonical path).
"""

import numpy as np
import pytest

pytest.importorskip("gymnasium")

from phca.config import ContinuousSpace, DiscreteSpace
from phca.core.cycle import CognitiveCycle
from phca.environments.mujoco_env import MuJoCoSimpleEnv


def test_pendulum_action_space_continuous():
    """Pendulum-v1 action space is ContinuousSpace([-2,2], dim=1)."""
    env = MuJoCoSimpleEnv("Pendulum-v1", seed=42)
    sp = env.get_action_space()
    assert isinstance(sp, ContinuousSpace)
    assert sp.dim == 1
    assert env.action_space_size == 1
    np.testing.assert_allclose(sp.low, [-2.0])
    np.testing.assert_allclose(sp.high, [2.0])


def test_pendulum_get_goal_reference():
    """Pendulum goal reference is the upright state [cos=1, sin=0, ang_vel=0]."""
    env = MuJoCoSimpleEnv("Pendulum-v1", seed=42)
    ref = env.get_goal_reference()
    assert ref is not None
    ref = np.asarray(ref, dtype=np.float32)
    assert ref.shape == (3,)
    np.testing.assert_allclose(ref, [1.0, 0.0, 0.0])
    assert np.all(np.isfinite(ref))


def test_cycle_pendulum_continuous_step_no_nan():
    """A Pendulum continuous cycle steps without NaN and reports continuous action."""
    cycle = CognitiveCycle.build_for_mujoco("Pendulum-v1", seed=42, use_mlp=True)
    assert cycle._is_continuous is True
    for _ in range(20):
        m = cycle.step()
        assert np.isfinite(m.prediction_error)
        assert m.action_taken == -1
        assert m.action_name == "continuous"
        assert m.violations_count == 0, "RBTA violation on continuous cycle"
    cycle.env.close()


def test_continuous_action_within_bounds():
    """The continuous action selector returns an action within [low, high]."""
    cycle = CognitiveCycle.build_for_mujoco("Pendulum-v1", seed=42, use_mlp=True)
    cycle.step()  # initialise current_state
    a = cycle._select_action()
    assert isinstance(a, np.ndarray)
    assert a.shape == (1,)
    sp = cycle.action_space
    assert np.all(a >= sp.low - 1e-6)
    assert np.all(a <= sp.high + 1e-6)
    assert np.all(np.isfinite(a))
    cycle.env.close()


def test_reacher_action_space_continuous():
    """Reacher-v5 action space is ContinuousSpace([-1,1], dim=2) (Phase 7 / A1)."""
    env = MuJoCoSimpleEnv("Reacher-v5", seed=42)
    sp = env.get_action_space()
    assert isinstance(sp, ContinuousSpace)
    assert sp.dim == 2
    assert env.action_space_size == 2
    np.testing.assert_allclose(sp.low, [-1.0, -1.0])
    np.testing.assert_allclose(sp.high, [1.0, 1.0])
    env.close()


def test_reacher_get_goal_reference():
    """Reacher goal reference is the current obs with fingertip→target (last 2 dims) zeroed."""
    env = MuJoCoSimpleEnv("Reacher-v5", seed=42)
    ref = env.get_goal_reference()
    assert ref is not None
    ref = np.asarray(ref, dtype=np.float32)
    assert ref.shape == (10,)
    np.testing.assert_allclose(ref[-2:], [0.0, 0.0])
    assert np.all(np.isfinite(ref))
    env.close()


def test_cycle_reacher_continuous_step_no_nan():
    """A Reacher continuous cycle steps without NaN and reports continuous action."""
    cycle = CognitiveCycle.build_for_mujoco("Reacher-v5", seed=42, use_mlp=True)
    assert cycle._is_continuous is True
    for _ in range(20):
        m = cycle.step()
        assert np.isfinite(m.prediction_error)
        assert m.action_taken == -1
        assert m.action_name == "continuous"
        assert m.violations_count == 0, "RBTA violation on Reacher continuous cycle"
    cycle.env.close()


def test_reacher_continuous_action_within_bounds():
    """The Reacher continuous selector returns a 2D action within [-1, 1]^2."""
    cycle = CognitiveCycle.build_for_mujoco("Reacher-v5", seed=42, use_mlp=True)
    cycle.step()  # initialise current_state
    a = cycle._select_action()
    assert isinstance(a, np.ndarray)
    assert a.shape == (2,)
    sp = cycle.action_space
    assert np.all(a >= sp.low - 1e-6)
    assert np.all(a <= sp.high + 1e-6)
    assert np.all(np.isfinite(a))
    cycle.env.close()


def test_cartpole_still_discrete():
    """Cartpole stays on the 3-bin discrete path."""
    env = MuJoCoSimpleEnv("InvertedPendulum-v5", seed=42)
    sp = env.get_action_space()
    assert isinstance(sp, DiscreteSpace)
    assert sp.n == 3
    assert env.action_space_size == 3


def test_discrete_gridworld_unchanged():
    """GridWorld still uses the discrete int-action path (regression guard)."""
    cycle = CognitiveCycle.build_for_env(size=5, seed=42, use_continuous=True, use_mlp=True)
    assert cycle._is_continuous is False
    assert isinstance(cycle.action_space, DiscreteSpace)
    m = cycle.step()
    assert m.action_taken >= 0  # int action index, not -1
    assert m.action_name in ("MOVE_N", "MOVE_S", "MOVE_E", "MOVE_W", "STAY")
    assert np.isfinite(m.prediction_error)
