"""Environment protocol for cognitive cycle decoupling."""

from __future__ import annotations

from typing import Dict, List, Optional, Protocol, Tuple, Union

import numpy as np

from phca.config import ActionSpace


class EnvironmentProtocol(Protocol):
    """Minimal environment interface for the cognitive cycle.

    Any environment implementing all members can drive the cycle.
    GridWorld conforms; custom environments must provide at minimum
    ``action_space_size``, ``step()``, and ``get_observation()``.
    """

    action_space_size: int

    @property
    def stay_action(self) -> Optional[int]:
        """Index of the 'do nothing' / energy-saving action, or None if
        the environment has no meaningful stay action (e.g. bandit)."""
        ...

    def get_possible_actions(self) -> List[str]:
        ...

    def get_goal_position(self) -> Optional[Tuple[int, int]]:
        ...

    def get_action_names(self) -> List[str]:
        ...

    def get_action_space(self) -> ActionSpace:
        """Return this env's action space (Discrete or Continuous).

        Default-implementing environments may omit this; the cycle falls
        back to ``DiscreteSpace(env.action_space_size)`` via getattr.
        """
        ...

    def get_observation(self) -> np.ndarray:
        """Return the current observation without stepping the simulation.

        Called by the cognitive cycle *before* action selection to obtain
        the current state for prediction. Implementations typically return
        a cached observation from the previous ``step()`` or ``reset()``.
        """
        ...

    def get_state_dim(self) -> int:
        """Return the dimensionality of the observation vector."""
        ...

    def neutral_action(self) -> Union[int, np.ndarray]:
        """Return a safe / zero-effect action for RBTA TERMINATE mode.

        Discrete environments return ``stay_action``; continuous environments
        return a zero vector of the correct dimension.
        """
        ...

    def get_goal_reference(self) -> Optional[np.ndarray]:
        """Return a goal reference state for continuous control, or None.

        Grid environments return None (goal expressed as grid position);
        continuous-control environments like Pendulum return a homeostatic
        target (e.g. ``[1, 0, 0]`` for upright balance).
        """
        ...

    def get_action_deltas(self) -> Optional[Dict[str, Tuple[int, int]]]:
        """Return mapping of action name → (row_delta, col_delta) for grid
        environments, or None for non-grid environments.

        Used by ``_compute_distance_gain`` and ``_select_greedy_grid_action``
        to predict the resulting grid position of each candidate action.
        """
        ...

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        ...

    def step(
        self, action: Union[int, np.ndarray]
    ) -> Tuple[np.ndarray, float, bool, dict]:
        ...
