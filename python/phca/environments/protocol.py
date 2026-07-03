"""Environment protocol for cognitive cycle decoupling."""

from __future__ import annotations

from typing import List, Optional, Protocol, Tuple

import numpy as np

from phca.config import ActionSpace


class EnvironmentProtocol(Protocol):
    """Minimal environment interface for the cognitive cycle.

    Any environment with these attributes can drive the cycle.
    GridWorld already conforms; custom environments need only
    provide `action_space_size` and a generic `step()`.
    """

    action_space_size: int

    def get_possible_actions(self) -> List[str]:
        ...

    def get_goal_position(self) -> Optional[Tuple[int, int]]:
        ...

    def get_action_names(self) -> List[str]:
        ...

    def get_action_space(self) -> ActionSpace:
        """Return this env's action space (Discrete or Continuous).

        Default-implementing environments may omit this; the cycle falls
        back to `DiscreteSpace(env.action_space_size)` via getattr.
        """
        ...

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, dict]:
        ...
