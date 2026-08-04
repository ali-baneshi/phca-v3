"""Runtime contract tests for bundled PHCA environments."""

import numpy as np
import pytest

from phca.config import DiscreteSpace
from phca.core.cycle import CognitiveCycle
from phca.environments.bandit_env import BanditEnv
from phca.environments.grid_world import GridWorld
from phca.environments.protocol import validate_environment
from phca.environments.slow_wrapper import SlowWrapper


@pytest.mark.parametrize(
    "env",
    [
        GridWorld(size=5, seed=42),
        BanditEnv(seed=42),
        SlowWrapper(GridWorld(size=5, seed=42), probability=0.0),
    ],
)
def test_bundled_environment_contracts(env):
    action_space = validate_environment(env)
    assert action_space is not None
    assert env.get_state_dim() == len(env.get_observation())
    assert env.neutral_action() is not None


def test_environment_contract_reports_missing_capabilities():
    class IncompleteEnvironment:
        action_space_size = 2

    with pytest.raises(TypeError, match="get_action_names"):
        validate_environment(IncompleteEnvironment())


def test_environment_contract_rejects_action_dimension_mismatch():
    class MismatchedEnvironment:
        action_space_size = 2

        def get_action_names(self):
            return ["A", "B"]

        def get_action_space(self):
            return DiscreteSpace(n=3)

        def get_observation(self):
            return np.zeros(2, dtype=np.float32)

        def get_state_dim(self):
            return 2

        def reset(self, seed=None):
            return self.get_observation()

        def step(self, action):
            return self.get_observation(), 0.0, False, {}

    with pytest.raises(ValueError, match="action-space mismatch"):
        validate_environment(MismatchedEnvironment())


def test_build_for_env_rejects_state_dimension_override():
    with pytest.warns(DeprecationWarning):
        with pytest.raises(ValueError, match="does not match"):
            CognitiveCycle.build_for_env(size=5, state_dim=1)


def test_builder_rejects_invalid_noise_intensity():
    env = GridWorld(size=5, seed=42)
    with pytest.raises(ValueError, match="noise_intensity"):
        CognitiveCycle.build(env=env, noise_profile="gaussian", noise_intensity=1.5)
