"""Environment protocol for cognitive cycle decoupling."""

from __future__ import annotations

from typing import List, Optional, Protocol, Tuple

import numpy as np


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

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, dict]:
        ...
