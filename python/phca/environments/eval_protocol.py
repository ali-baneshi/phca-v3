"""Extended environment protocol for evaluation adapters."""

from __future__ import annotations

from typing import Any, Dict, Protocol, Tuple

import numpy as np

from phca.environments.protocol import EnvironmentProtocol


class EvalEnvironmentProtocol(EnvironmentProtocol, Protocol):
    """Thin evaluation adapter interface."""

    domain_id: str
    complexity_params: Dict[str, Any]

    def get_state_dim(self) -> int:
        ...

    def normalize_obs(self, raw: np.ndarray) -> np.ndarray:
        ...
