"""Synthetic sequential decision environment for OOD evaluation."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from phca.config import DiscreteSpace


class BanditEnv:
    """Non-stationary 3-arm bandit — tests sequential decision without grid dynamics."""

    domain_id = "sequential_decision"
    action_space_size = 3

    def __init__(self, seed: int = 42, n_arms: int = 3, switch_every: int = 50):
        self.rng = np.random.RandomState(seed)
        self.n_arms = n_arms
        self.switch_every = switch_every
        self.step_count = 0
        self.best_arm = 0
        self._arm_probs = self._new_probs()
        self.size = 1
        self.complexity_params: Dict[str, int] = {"n_arms": n_arms, "switch_every": switch_every}

    def _new_probs(self) -> np.ndarray:
        probs = self.rng.dirichlet(np.ones(self.n_arms) * 0.5)
        self.best_arm = int(np.argmax(probs))
        return probs.astype(np.float32)

    def get_state_dim(self) -> int:
        return self.n_arms + 3

    def get_possible_actions(self):
        return [f"ARM_{i}" for i in range(self.n_arms)]

    def get_action_names(self):
        return self.get_possible_actions()

    def get_action_space(self):
        return DiscreteSpace(n=self.n_arms)

    def get_goal_position(self) -> Optional[Tuple[int, int]]:
        return None

    def normalize_obs(self, raw: np.ndarray) -> np.ndarray:
        return raw.astype(np.float32)

    def reset(self, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            self.rng = np.random.RandomState(seed)
        self.step_count = 0
        self._arm_probs = self._new_probs()
        self._last_reward = 0.0
        self._last_action = 0
        return self._get_observation()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        if self.step_count > 0 and self.step_count % self.switch_every == 0:
            self._arm_probs = self._new_probs()
        action = int(action) % self.n_arms
        reward = 1.0 if self.rng.random() < self._arm_probs[action] else 0.0
        self._last_reward = reward
        self._last_action = action
        self.step_count += 1
        info = {"goal_reached": reward > 0.5, "best_arm": self.best_arm}
        return self._get_observation(), reward, False, info

    def _get_observation(self) -> np.ndarray:
        step_norm = min(1.0, self.step_count / max(self.switch_every * 4, 1))
        obs = np.concatenate([
            self._arm_probs,
            [step_norm, self._last_reward, self._last_action / max(self.n_arms - 1, 1)],
        ]).astype(np.float32)
        return obs
