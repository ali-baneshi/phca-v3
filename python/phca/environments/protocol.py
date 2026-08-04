"""Environment protocols and runtime validation for cognitive-cycle decoupling."""

from __future__ import annotations

from typing import Dict, List, Optional, Protocol, Tuple, Union, runtime_checkable

import numpy as np

from phca.config import ActionSpace, ContinuousSpace, DiscreteSpace


@runtime_checkable
class EnvironmentProtocol(Protocol):
    """Required environment interface for the cognitive cycle."""

    action_space_size: int

    def get_action_names(self) -> List[str]:
        ...

    def get_action_space(self) -> ActionSpace:
        ...

    def get_observation(self) -> np.ndarray:
        ...

    def get_state_dim(self) -> int:
        ...

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        ...

    def step(
        self, action: Union[int, np.ndarray]
    ) -> Tuple[np.ndarray, float, bool, dict]:
        ...


@runtime_checkable
class NeutralActionCapability(Protocol):
    def neutral_action(self) -> Union[int, np.ndarray]:
        ...


@runtime_checkable
class GridPlanningCapability(Protocol):
    @property
    def stay_action(self) -> Optional[int]:
        ...

    def get_goal_position(self) -> Optional[Tuple[int, int]]:
        ...

    def get_action_deltas(self) -> Optional[Dict[str, Tuple[int, int]]]:
        ...


@runtime_checkable
class GoalReferenceCapability(Protocol):
    def get_goal_reference(self) -> Optional[np.ndarray]:
        ...


def validate_environment(env: object) -> ActionSpace:
    """Validate the required runtime contract and return its action space."""
    missing = [
        name
        for name in (
            "get_action_names",
            "get_action_space",
            "get_observation",
            "get_state_dim",
            "reset",
            "step",
        )
        if not callable(getattr(env, name, None))
    ]
    if not hasattr(env, "action_space_size"):
        missing.append("action_space_size")
    if missing:
        raise TypeError(
            "environment is missing required capabilities: "
            + ", ".join(sorted(missing))
        )

    state_dim = int(env.get_state_dim())
    action_space_size = int(env.action_space_size)
    if state_dim <= 0:
        raise ValueError("environment state_dim must be positive")
    if action_space_size <= 0:
        raise ValueError("environment action_space_size must be positive")

    action_space = env.get_action_space()
    if isinstance(action_space, DiscreteSpace):
        if action_space.n != action_space_size:
            raise ValueError(
                "environment action-space mismatch: "
                f"DiscreteSpace.n={action_space.n}, action_space_size={action_space_size}"
            )
    elif isinstance(action_space, ContinuousSpace):
        if action_space.dim != action_space_size:
            raise ValueError(
                "environment action-space mismatch: "
                f"ContinuousSpace.dim={action_space.dim}, "
                f"action_space_size={action_space_size}"
            )
    else:
        raise TypeError("environment get_action_space() must return an ActionSpace")
    return action_space
