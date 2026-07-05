"""
MuJoCo Simple Environment Wrapper — PHCA v3.0

Wraps a gymnasium MuJoCo environment (Cartpole, Pendulum, Reacher)
into the PHCA cognitive cycle's environment interface.

Discretises continuous action spaces into ≤5 discrete bins.
State vector is the raw observation array from the MuJoCo environment.

Cross-ref: docs/mujoco_integration_plan.md
"""

from __future__ import annotations

import threading
from typing import Any, Dict, List, Optional, Tuple

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

    # Phase 6 / A3 + Phase 7 / A1: envs with a true continuous action space.
    # Pendulum-v1 torque ∈ [-2, 2], dim 1. Reacher-v5 actuator ∈ [-1, 1]^2, dim 2.
    _CONTINUOUS_ENVS: Dict[str, Tuple[np.ndarray, np.ndarray, int]] = {
        "Pendulum-v1": (np.array([-2.0], dtype=np.float32),
                        np.array([2.0], dtype=np.float32), 1),
        "Reacher-v5": (np.array([-1.0, -1.0], dtype=np.float32),
                       np.array([1.0, 1.0], dtype=np.float32), 2),
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
        enable_camera: bool = False,
    ):
        """Initialise the MuJoCo environment wrapper.

        Args:
            env_name: gymnasium MuJoCo environment ID.
            seed: Random seed for reproducibility.
            render_mode: Gymnasium render mode (None for headless,
                "human" for visualisation). ``rgb_array`` is accepted for
                backward compatibility but the live camera uses
                ``mujoco.Renderer`` instead of gymnasium's OffScreenViewer.
            enable_camera: When True, capture RGB via ``mujoco.Renderer``
                (observatory dashboard). Avoids gymnasium EGL/OSMesa conflicts
                with Qt on the main thread.
        """
        self.env_name = env_name
        self._enable_camera = bool(enable_camera or render_mode == "rgb_array")
        gym_mode = render_mode
        if self._enable_camera:
            gym_mode = None
        self._env = gym.make(env_name, render_mode=gym_mode)
        self._offscreen_renderer: Any = None
        self._renderer_tid: Optional[int] = None
        self._sim_lock = threading.RLock()
        self._rng = np.random.RandomState(seed)
        self._seed = seed

        # Build discrete-to-continuous action mapping
        self._action_map: Dict[int, np.ndarray] = self._build_action_map()
        # Phase 6 / A3: continuous envs expose a true continuous space;
        # action_space_size = continuous dim so the MLP learns continuous
        # dynamics. Discrete envs keep action_space_size = len(action_map).
        self._continuous_cfg = self._CONTINUOUS_ENVS.get(env_name)
        if self._continuous_cfg is not None:
            self.action_space_size: int = self._continuous_cfg[2]
        else:
            self.action_space_size = len(self._action_map)

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

    def get_action_space(self):
        """Return this env's action space (Phase 6 / A1, A3).

        Pendulum-v1 → ContinuousSpace([-2,2], dim=1) (true continuous torque).
        Cartpole / Reacher → DiscreteSpace over the configured action map
        (Reacher-continuous is a Phase 7 target).
        """
        from phca.config import DiscreteSpace, ContinuousSpace
        if self._continuous_cfg is not None:
            low, high, dim = self._continuous_cfg
            return ContinuousSpace(low=low, high=high, dim=dim)
        return DiscreteSpace(n=len(self._action_map))

    def neutral_action(self):
        """RBTA/safe-mode action: zero vector (continuous) or stay index (discrete).

        Continuous envs must not use ``stay_action`` (int) with ``env.step`` —
        gymnasium expects shape ``(dim,)``.
        """
        if self._continuous_cfg is not None:
            dim = int(self._continuous_cfg[2])
            return np.zeros(dim, dtype=np.float32)
        return int(self.stay_action)

    def get_goal_reference(self):
        """Homeostatic reference state for goal-directed continuous control.

        Pendulum-v1 obs = [cos(theta), sin(theta), angular_velocity]; upright
        balanced = theta=0 → [1, 0, 0]. Returns None for discrete envs.

        Reacher-v5 obs (10-dim) encodes the fingertip→target vector in its
        last 2 dims (verified against gymnasium: obs[-2:] == fingertip_xpos -
        target_com). The reference is "current posture with fingertip on
        target": a copy of the current observation with obs[-2:] = 0. Holding
        the joint/target context (dims 0–7) at the current value keeps the
        MPC scorer's full-state distance well-posed, so the alignment signal
        is dominated by whether the predicted next state drives the
        fingertip→target vector to 0. State-dependent (Reacher's target is
        re-randomised each reset), unlike Pendulum's fixed upright reference.
        """
        if self.env_name == "Pendulum-v1":
            return np.array([1.0, 0.0, 0.0], dtype=np.float32)
        if self.env_name == "Reacher-v5" and self._last_obs is not None:
            ref = self._last_obs.copy()
            ref[-2:] = 0.0
            return ref
        return None

    def step(self, action) -> Tuple[np.ndarray, float, bool, dict]:
        """Execute an action in the MuJoCo environment.

        Args:
            action: Discrete int index (discrete envs) OR a continuous
                np.ndarray (continuous envs — Phase 6 / A3, Pendulum-v1).
                The continuous vector is passed straight to gymnasium; the
                discrete index is mapped through `_action_map`.

        Returns:
            (observation, reward, terminal, info) tuple compatible
            with EnvironmentProtocol. Observation is cached for
            subsequent _get_observation() calls.
        """
        if self._continuous_cfg is not None:
            continuous_action = np.asarray(action, dtype=np.float32)
        else:
            continuous_action = self._action_map[action]
        with self._sim_lock:
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
        with self._sim_lock:
            obs, info = self._env.reset(seed=seed or self._seed)
        obs = obs.astype(np.float32)
        self._last_obs = obs
        return obs

    def get_state_dim(self) -> int:
        """Return the dimensionality of the MuJoCo observation vector."""
        return int(self._env.observation_space.shape[0])

    def get_dim_names(self) -> List[str]:
        """Human-readable obs dim labels for Reacher (10-d default)."""
        n = self.get_state_dim()
        defaults = ["cθ₀", "sθ₀", "cθ₁", "sθ₁", "ẋ₀", "ẋ₁", "ẋ₂", "ẋ₃", "tip_x", "tip_y"]
        if n <= len(defaults):
            return defaults[:n]
        return defaults + [f"d{i}" for i in range(len(defaults), n)]

    def close(self) -> None:
        """Release MuJoCo simulation resources."""
        with self._sim_lock:
            if self._offscreen_renderer is not None:
                try:
                    self._offscreen_renderer.close()
                except Exception:
                    pass
                self._offscreen_renderer = None
                self._renderer_tid = None
            self._env.close()

    def _reset_renderer_if_wrong_thread(self) -> None:
        tid = threading.get_ident()
        if self._offscreen_renderer is not None and self._renderer_tid != tid:
            try:
                self._offscreen_renderer.close()
            except Exception:
                pass
            self._offscreen_renderer = None
            self._renderer_tid = None

    def _render_offscreen(self) -> Optional[np.ndarray]:
        """Capture RGB via mujoco.Renderer (main-thread safe; sim lock held)."""
        try:
            import mujoco
            unwrapped = getattr(self._env, "unwrapped", self._env)
            model = getattr(unwrapped, "model", None)
            data = getattr(unwrapped, "data", None)
            if model is None or data is None:
                return None
            self._reset_renderer_if_wrong_thread()
            if self._offscreen_renderer is None:
                self._offscreen_renderer = mujoco.Renderer(model, height=480, width=480)
                self._renderer_tid = threading.get_ident()
            mujoco.mj_forward(model, data)
            self._offscreen_renderer.update_scene(data)
            img = self._offscreen_renderer.render()
            return np.asarray(img, dtype=np.uint8).copy()
        except Exception:
            return None

    def render_rgb(self) -> Optional[np.ndarray]:
        """Return the current MuJoCo camera frame as an (H,W,3) uint8 array.

        Uses ``mujoco.Renderer`` when ``enable_camera`` is set (observatory).
        Gymnasium's ``rgb_array`` OffScreenViewer is intentionally avoided —
        it often returns solid green/red on EGL+Qt setups.

        Thread-safe: holds ``_sim_lock`` so render never races ``step()``.
        """
        if not self._enable_camera:
            return None

        def _ok(arr: Optional[np.ndarray]) -> bool:
            if arr is None or arr.size == 0:
                return False
            from phca.monitoring.camera_render import is_glitchy_rgb_frame
            return not is_glitchy_rgb_frame(arr)

        with self._sim_lock:
            for attempt in range(2):
                arr = self._render_offscreen()
                if _ok(arr):
                    return arr
                if attempt == 0:
                    try:
                        import mujoco
                        unwrapped = getattr(self._env, "unwrapped", self._env)
                        model = getattr(unwrapped, "model", None)
                        data = getattr(unwrapped, "data", None)
                        if model is not None and data is not None:
                            mujoco.mj_forward(model, data)
                    except Exception:
                        pass
        return None

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
