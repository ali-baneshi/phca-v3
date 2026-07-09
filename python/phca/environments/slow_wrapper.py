"""Environment wrapper that injects artificial delays to simulate slowdowns."""

from __future__ import annotations

import random
import time
from typing import Dict, List, Optional, Tuple, Union

import numpy as np


class SlowWrapper:
    """Wraps an EnvironmentProtocol and injects delays into step().

    Each call to step() has a configurable probability of being delayed
    by a random duration within a configurable range.  All other methods
    delegate directly to the underlying environment.

    Attributes:
        delay_min_ms: Minimum delay in milliseconds (inclusive).
        delay_max_ms: Maximum delay in milliseconds (inclusive).
        probability: Probability (0.0–1.0) of injecting a delay per step().
        _rng: Deterministic random state for reproducibility.
    """

    def __init__(
        self,
        env: object,
        delay_min_ms: float = 50.0,
        delay_max_ms: float = 200.0,
        probability: float = 0.3,
        seed: Optional[int] = None,
    ):
        self._env = env
        self.delay_min_ms = delay_min_ms
        self.delay_max_ms = delay_max_ms
        self.probability = probability
        self._rng = random.Random(seed)
        self._slowdown_count: int = 0
        self._total_injected_ms: float = 0.0

    # ── Delegated attributes ─────────────────────────────────────

    @property
    def action_space_size(self) -> int:
        return self._env.action_space_size

    @property
    def stay_action(self) -> Optional[int]:
        return getattr(self._env, "stay_action", None)

    @property
    def size(self) -> Optional[int]:
        return getattr(self._env, "size", None)

    # ── Delegated methods ─────────────────────────────────────────

    def get_possible_actions(self) -> List[str]:
        return self._env.get_possible_actions()

    def get_goal_position(self) -> Optional[Tuple[int, int]]:
        return getattr(self._env, "get_goal_position", lambda: None)()

    def get_action_names(self) -> List[str]:
        return self._env.get_action_names()

    def get_observation(self) -> np.ndarray:
        return self._env.get_observation()

    def get_state_dim(self) -> int:
        return self._env.get_state_dim()

    def neutral_action(self) -> Union[int, np.ndarray]:
        return self._env.neutral_action()

    def get_goal_reference(self) -> Optional[np.ndarray]:
        fn = getattr(self._env, "get_goal_reference", None)
        return fn() if fn is not None else None

    def get_action_deltas(self) -> Optional[Dict[str, Tuple[int, int]]]:
        fn = getattr(self._env, "get_action_deltas", None)
        return fn() if fn is not None else None

    def relocate_goal(self) -> None:
        fn = getattr(self._env, "relocate_goal", None)
        if fn is not None:
            fn()

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        return self._env.reset(seed=seed)

    def step(self, action: Union[int, np.ndarray]) -> Tuple[np.ndarray, float, bool, dict]:
        if self._rng.random() < self.probability:
            delay_ms = self._rng.uniform(self.delay_min_ms, self.delay_max_ms)
            time.sleep(delay_ms / 1000.0)
            self._slowdown_count += 1
            self._total_injected_ms += delay_ms
        return self._env.step(action)

    @property
    def stats(self) -> dict:
        return {
            "slowdown_count": self._slowdown_count,
            "total_injected_ms": round(self._total_injected_ms, 1),
        }
