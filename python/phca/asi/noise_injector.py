"""Grounding simulation — noise injection for robustness testing.

Simulates sensor degradation at the ASI level by injecting configurable
noise profiles into raw sensor observations before sanitization.

Demonstrates robustness without requiring real hardware or actual L0/L2
grounding implementations. Profiles: gaussian, dropout, drift, salt_pepper.
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, Optional

import numpy as np


class NoiseProfile(str, Enum):
    GAUSSIAN = "gaussian"
    DROPOUT = "dropout"
    DRIFT = "drift"
    SALT_PEPPER = "salt_pepper"


class NoiseInjector:
    """Injects configurable noise into sensor observations.

    Place before ASISanitizer.sanitize() to simulate grounding degradation.
    """

    def __init__(
        self,
        sensor_dim: int,
        initial_profile: str = "gaussian",
        initial_intensity: float = 0.1,
        seed: int = 42,
    ):
        self.sensor_dim = sensor_dim
        self.profile = NoiseProfile(initial_profile)
        self.intensity = float(np.clip(initial_intensity, 0.0, 1.0))
        self.rng = np.random.RandomState(seed)
        self._drift_state: np.ndarray = np.zeros(sensor_dim, dtype=np.float32)
        self._total_noise_cycles: int = 0

    def set_profile(self, name: str) -> None:
        self.profile = NoiseProfile(name)

    def set_intensity(self, level: float) -> None:
        self.intensity = float(np.clip(level, 0.0, 1.0))

    def inject(self, raw: np.ndarray, cycle: int = 0) -> np.ndarray:
        assert raw.shape == (self.sensor_dim,), \
            f"Expected ({self.sensor_dim},), got {raw.shape}"
        noisy = raw.copy()
        if self.intensity <= 0.0:
            return noisy

        if self.profile == NoiseProfile.GAUSSIAN:
            noise = self.rng.normal(0, self.intensity, size=raw.shape).astype(np.float32)
            noisy = raw + noise
            noisy = np.clip(noisy, 0.0, 2.0)

        elif self.profile == NoiseProfile.DROPOUT:
            mask = self.rng.random(size=raw.shape) < self.intensity
            noisy[mask] = 0.0

        elif self.profile == NoiseProfile.DRIFT:
            self._drift_state += self.rng.normal(
                0, self.intensity * 0.1, size=raw.shape
            ).astype(np.float32)
            noisy = raw + self._drift_state
            noisy = np.clip(noisy, 0.0, 2.0)

        elif self.profile == NoiseProfile.SALT_PEPPER:
            mask = self.rng.random(size=raw.shape) < self.intensity
            n_fail = int(mask.sum())
            if n_fail > 0:
                noisy[mask] = self.rng.choice(
                    [0.0, 1.0, 2.0], size=n_fail
                ).astype(np.float32)

        self._total_noise_cycles += 1
        return noisy.astype(np.float32)

    def reset(self) -> None:
        self._drift_state.fill(0.0)
        self._total_noise_cycles = 0

    def get_stats(self) -> Dict[str, float]:
        return {
            "profile": self.profile.value,
            "intensity": self.intensity,
            "total_noise_cycles": self._total_noise_cycles,
        }
