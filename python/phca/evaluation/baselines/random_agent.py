"""Uniform random action baseline."""

from __future__ import annotations

import numpy as np


def random_action(env, rng: np.random.RandomState) -> int:
    return int(rng.randint(0, env.action_space_size))
