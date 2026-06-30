"""
MuJoCo Simple Environment Wrapper — PHCA v3.0

Wraps a gymnasium MuJoCo environment (Cartpole, Pendulum, Reacher)
into the PHCA cognitive cycle's environment interface.

Discretises continuous action spaces into ≤5 discrete bins.
State vector is the raw observation array from the MuJoCo environment.

Cross-ref: docs/mujoco_integration_plan.md
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np


class MuJoCoSimpleEnv:
    """Minimal MuJoCo environment wrapper for the PHCA cognitive cycle.

    Implements the EnvironmentProtocol interface (and extras needed by
    CognitiveCycle) so that MuJoCo physics environments can drive the
    cycle with no changes to the cycle orchestrator.

    Attributes:
        env_name: gymnasium environment ID (e.g. 'InvertedPendulum-v5').
        action_space_size: Number of discrete actions (≤5).
        size: Stub attribute for CognitiveCycle (set to 1, non-grid).
    """

    # ── Discretisation maps ──────────────────────────────────
    # Maps environment ID → {discrete_idx: continuous_action_vector}

    _ACTION_MAPS: Dict[str, Dict[int, np.ndarray]] = {
        "InvertedPendulum-v5": {
            0: np.array([-3.0], dtype=np.float32),   # push left
            1: np.array([0.0], dtype=np.float32),     # do nothing
            2: np.array([3.0], dtype=np.float32),     # push right
        },
        "Pendulum-v1": {
            0: np.array([-2.0], dtype=np.float32),    # torque left
            1: np.array([0.0], dtype=np.float32),     # no torque
            2: np.array([2.0], dtype=np.float32),     # torque right
        },
    }

    # Reacher uses 2D action space; discretise via grid of 5 points
    _REACHER_ACTIONS: Dict[int, np.ndarray] = {
        0: np.array([-1.0, -1.0], dtype=np.float32),
        1: np.array([-1.0,  1.0], dtype=np.float32),
        2: np.array([ 0.0,  0.0], dtype=np.float32),
        3: np.array([ 1.0, -1.0], dtype=np.float32),
        4: np.array([ 1.0,  1.0], dtype=np.float32),
    }

    # ── Action names (for PHCA logging and get_action_names) ─

    _ACTION_NAMES: Dict[str, List[str]] = {
        "InvertedPendulum-v5": ["PUSH_LEFT", "STAY", "PUSH_RIGHT"],
        "Pendulum-v1": ["TORQUE_LEFT", "STAY", "TORQUE_RIGHT"],
        "Reacher-v5": ["MOVE_SW", "MOVE_NW", "STAY", "MOVE_NE", "MOVE_SE"],
    }

    def __init__(
        self,
        env_name: str = "InvertedPendulum-v5",
        seed: int = 42,
        render_mode: Optional[str] = None,
    ):
        """Initialise the MuJoCo environment wrapper.

        Args:
            env_name: gymnasium MuJoCo environment ID.
            seed: Random seed for reproducibility.
            render_mode: Gymnasium render mode (None for headless,
                "human" for visualisation, "rgb_array" for recording).
        """
        self.env_name = env_name
        self._env = gym.make(env_name, render_mode=render_mode)
        self._rng = np.random.RandomState(seed)
        self._seed = seed

        # Build discrete-to-continuous action mapping
        self._action_map: Dict[int, np.ndarray] = self._build_action_map()
        self.action_space_size: int = len(self._action_map)

        # Observation cache — _get_observation() returns this
        self._last_obs: Optional[np.ndarray] = None

        # State normalisation statistics (populated lazily in Phase 2+)
        self._obs_min: Optional[np.ndarray] = None
        self._obs_max: Optional[np.ndarray] = None

        # Index of the "do nothing" action for CognitiveCycle._select_action()
        # Determined from action names: find the one labelled "STAY"
        self.stay_action: int = self._find_stay_action()

        # Stub for CognitiveCycle compatibility.
        # CognitiveCycle.__init__ logs grid_size=env.size.
        # _compute_distance_gain() checks hasattr(env, 'grid') first,
        # so this stub is never used for distance calculations.
        self.size = 1  # non-grid environment

        # Initial reset populates the observation cache
        obs, _ = self._env.reset(seed=seed)
        self._last_obs = obs.astype(np.float32)

    # ── Public interface (EnvironmentProtocol + extras) ──────

    def get_possible_actions(self) -> List[str]:
        """Return list of action names."""
        return list(self.get_action_names())

    def get_goal_position(self) -> Optional[Tuple[int, int]]:
        """Return None — MuJoCo environments don't have grid positions.

        The cognitive cycle's _compute_distance_gain() falls back to
        0.5 (neutral distance gain) when goal_position is None.
        """
        return None

    def get_action_names(self) -> List[str]:
        """Return human-readable action names for logging."""
        return list(
            self._ACTION_NAMES.get(
                self.env_name,
                [f"ACT_{i}" for i in range(self.action_space_size)],
            )
        )

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """Execute a discrete action in the MuJoCo environment.

        Args:
            action: Discrete action index (0 .. action_space_size - 1).

        Returns:
            (observation, reward, terminal, info) tuple compatible
            with EnvironmentProtocol. Observation is cached for
            subsequent _get_observation() calls.
        """
        continuous_action = self._action_map[action]
        obs, reward, terminated, truncated, info = self._env.step(continuous_action)
        terminal = terminated or truncated

        # MuJoCo uses np.float64 internally; cast to float32 for PHCA
        obs = obs.astype(np.float32)

        # NaN/Inf guard — replace any non-finite values with 0
        if not np.all(np.isfinite(obs)):
            obs = np.nan_to_num(obs, nan=0.0, posinf=10.0, neginf=-10.0)

        # Cache for _get_observation()
        self._last_obs = obs

        return obs, float(reward), bool(terminal), info

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        """Reset the MuJoCo environment.

        Args:
            seed: Optional seed for reproducibility.

        Returns:
            Initial observation vector (cached for _get_observation()).
        """
        obs, info = self._env.reset(seed=seed or self._seed)
        obs = obs.astype(np.float32)
        self._last_obs = obs
        return obs

    def get_state_dim(self) -> int:
        """Return the dimensionality of the MuJoCo observation vector."""
        return int(self._env.observation_space.shape[0])

    def close(self) -> None:
        """Release MuJoCo simulation resources."""
        self._env.close()

    # ── Internal methods called by CognitiveCycle ────────────

    def _get_observation(self) -> np.ndarray:
        """Get the last observation without stepping the simulation.

        Called by CognitiveCycle.step() Step 0 (ASI sanitisation)
        *before* action selection and env.step(). Returns the
        observation cached from the previous step() or reset().

        This implements a standard temporal-difference loop:
          predict(s_t) → select action → observe s_{t+1} → learn
        where s_t is the cached observation from the prior cycle.
        """
        # After reset() or the previous step(), _last_obs holds the
        # most recent observation. This is the "current state" for
        # prediction before we act.
        return self._last_obs

    # ── Private helpers ──────────────────────────────────────

    def _build_action_map(self) -> Dict[int, np.ndarray]:
        """Build the discrete-to-continuous action mapping.

        Returns:
            Dict mapping discrete action index → continuous action vector.

        Raises:
            ValueError: If the environment ID is not recognised and
                no catch-all handler applies.
        """
        if self.env_name in self._ACTION_MAPS:
            return dict(self._ACTION_MAPS[self.env_name])
        elif "Reacher" in self.env_name:
            return dict(self._REACHER_ACTIONS)
        else:
            raise ValueError(
                f"Unknown environment '{self.env_name}'. "
                f"Add an action map to _ACTION_MAPS or _REACHER_ACTIONS."
            )

    def _find_stay_action(self) -> int:
        """Find the index of the 'do nothing' (STAY) action.

        Searches action names for a label containing 'STAY'. If not
        found, falls back to the middle action index (intended as
        the neutral action in symmetric action sets).

        Returns:
            Integer index of the stay action.
        """
        names = self.get_action_names()
        for i, name in enumerate(names):
            if "STAY" in name.upper():
                return i
        # Fallback: middle action (symmetric sets have neutral in the middle)
        return len(names) // 2

    # ── Stub properties for grid-world compatibility ─────────

    @property
    def agent_pos(self) -> Tuple[int, int]:
        """Stub — MuJoCo environments don't have grid positions.

        The cognitive cycle's _compute_distance_gain() checks for
        hasattr(env, 'grid') and hasattr(env, 'WALL') before accessing
        agent_pos, so this stub is never used for distance calculations.
        Required only for attribute existence.
        """
        return (0, 0)

    @property
    def grid(self) -> np.ndarray:
        """Stub — triggers the non-grid fallback in _compute_distance_gain()."""
        return np.zeros((1, 1), dtype=np.int32)

    @property
    def WALL(self) -> int:
        """Stub — pairs with grid stub for attribute existence check."""
        return -1

    def __repr__(self) -> str:
        return (
            f"MuJoCoSimpleEnv(env_name={self.env_name!r}, "
            f"state_dim={self.get_state_dim()}, "
            f"actions={self.action_space_size})"
        )
