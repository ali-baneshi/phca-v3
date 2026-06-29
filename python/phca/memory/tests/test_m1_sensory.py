"""
Tests for M1 Sensory Buffer (PHCA-3.1-005).

Covers: capacity, circular buffer write, read_latest ordering, reset.
"""

import numpy as np
import pytest

from phca.memory.m1_sensory import M1SensoryBuffer
from phca.config import StateVector


@pytest.fixture
def m1():
    return M1SensoryBuffer(sensor_dim=4, capacity=10)


class TestM1SensoryBuffer:
    """Test suite for M1 Sensory Buffer."""

    def test_init_has_zero_samples(self, m1):
        """Newly created buffer should have count=0."""
        assert m1.count == 0
        assert not m1.is_full

    def test_write_and_read_latest(self, m1):
        """After writing, read_latest(1) should return the written state."""
        sv = StateVector(values=np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
                         precision=np.ones(4, dtype=np.float32))
        m1.write(sv)
        results = m1.read_latest(1)
        assert len(results) == 1
        np.testing.assert_array_almost_equal(results[0].values, sv.values)

    def test_read_latest_ordering(self, m1):
        """read_latest should return most recent first."""
        for i in range(5):
            sv = StateVector(values=np.full(4, float(i), dtype=np.float32),
                             precision=np.ones(4, dtype=np.float32), timestamp=float(i))
            m1.write(sv)
        results = m1.read_latest(3)
        assert len(results) == 3
        assert results[0].timestamp == 4.0  # most recent
        assert results[1].timestamp == 3.0
        assert results[2].timestamp == 2.0

    def test_circular_buffer_wraparound(self, m1):
        """After writing more than capacity, the buffer should wrap around."""
        for i in range(15):  # capacity is 10
            sv = StateVector(values=np.full(4, float(i), dtype=np.float32),
                             precision=np.ones(4, dtype=np.float32), timestamp=float(i))
            m1.write(sv)
        assert m1.count == 15
        assert m1.is_full
        # Latest should be timestamp 14
        results = m1.read_latest(1)
        assert results[0].timestamp == 14.0

    def test_read_more_than_available(self, m1):
        """Reading more than available should return what's available."""
        for i in range(3):
            sv = StateVector(values=np.full(4, float(i), dtype=np.float32),
                             precision=np.ones(4, dtype=np.float32), timestamp=float(i))
            m1.write(sv)
        results = m1.read_latest(10)
        assert len(results) == 3

    def test_reset_clears_buffer(self, m1):
        """Reset should clear all data."""
        sv = StateVector(values=np.ones(4, dtype=np.float32), precision=np.ones(4, dtype=np.float32))
        m1.write(sv)
        m1.reset()
        assert m1.count == 0
        assert len(m1.read_latest(1)) == 0
