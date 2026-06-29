"""
M1 Sensory Buffer — v3.0 §3.1 Memory Hierarchy.

A circular buffer that holds the last 10 × d_sensor time steps of sensory data.
Overwritten on each cycle; no locking required (transient, single-writer).

Cross-ref: v3.0 §3.1 Table, Blueprint §B
"""

from __future__ import annotations

import numpy as np

from phca.logging import logger, _log
from ..config import StateVector


class M1SensoryBuffer:
    """
    M1 Sensory Buffer — circular buffer.

    Capacity: 10 × sensor_dim samples (transient, ~100ms window).
    Concurrency: Overwrite — no locking (single-writer).
    """

    def __init__(self, sensor_dim: int, capacity: int | None = None):
        """
        Initialize the sensory buffer.

        Args:
            sensor_dim: Dimensionality of the sensor vector.
            capacity: Number of samples to buffer (default: 10 × sensor_dim).
        """
        self.sensor_dim = sensor_dim
        self.capacity = capacity or (10 * sensor_dim)
        self.buffer = np.zeros((self.capacity, sensor_dim), dtype=np.float32)
        self.precision_buffer = np.zeros((self.capacity, sensor_dim), dtype=np.float32)
        self.timestamps = np.zeros(self.capacity, dtype=np.float64)
        self.write_pos = 0  # current write position
        self.count = 0  # total samples written

        _log(logger, "info", "m1.init", sensor_dim=sensor_dim, capacity=self.capacity)

    def write(self, state: StateVector) -> None:
        """
        Write a sensor state vector to the buffer.

        Args:
            state: Current sanitized state vector.
        """
        idx = self.write_pos % self.capacity
        self.buffer[idx] = state.values
        self.precision_buffer[idx] = state.precision
        self.timestamps[idx] = state.timestamp
        self.write_pos += 1
        self.count += 1

    def read_latest(self, n: int = 1) -> list[StateVector]:
        """
        Read the last n samples from the buffer.

        Args:
            n: Number of most recent samples to read.

        Returns:
            List of StateVector objects, most recent first.
        """
        if self.count == 0:
            return []
        n = min(n, min(self.count, self.capacity))
        result = []
        for i in range(n):
            idx = (self.write_pos - 1 - i) % self.capacity
            result.append(StateVector(
                values=self.buffer[idx].copy(),
                precision=self.precision_buffer[idx].copy(),
                timestamp=self.timestamps[idx],
            ))
        return result

    @property
    def is_full(self) -> bool:
        """Check if the buffer has been filled at least once."""
        return self.count >= self.capacity

    def reset(self) -> None:
        """Clear the buffer and reset write position."""
        self.buffer.fill(0.0)
        self.precision_buffer.fill(1.0)
        self.timestamps.fill(0.0)
        self.write_pos = 0
        self.count = 0
        _log(logger, "info", "m1.reset")
