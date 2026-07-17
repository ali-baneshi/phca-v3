"""
Tests for M2 Working Memory (PHCA-3.1-005).

Covers: capacity 7±2 bounds, salience-based eviction, write.
"""

import numpy as np
import pytest

from phca.memory.m2_working import M2WorkingMemory
from phca.config import StateVector


@pytest.fixture
def wm():
    return M2WorkingMemory(capacity=7)


class TestM2WorkingMemory:
    """Test suite for M2 Working Memory."""

    def test_init_empty(self, wm):
        """New WM should have zero chunks."""
        assert len(wm.chunks) == 0
        assert wm.capacity == 7

    def test_write_increases_count(self, wm):
        """Writing should increase chunk count."""
        sv = StateVector(values=np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
                         precision=np.ones(4, dtype=np.float32))
        wm.write(sv)
        assert len(wm.chunks) == 1

    def test_write_up_to_capacity(self, wm):
        """Writing up to capacity should work without eviction."""
        sv = StateVector(values=np.ones(4, dtype=np.float32), precision=np.ones(4, dtype=np.float32))
        for _ in range(7):
            wm.write(sv)
        assert len(wm.chunks) == 7

    def test_eviction_on_overflow(self, wm):
        """Writing past capacity should evict the lowest-salience chunk."""
        sv = StateVector(values=np.ones(4, dtype=np.float32), precision=np.ones(4, dtype=np.float32))
        # Write 7 chunks with salience 0 to 6
        for i in range(7):
            wm.write(sv, salience=float(i))
        assert len(wm.chunks) == 7

        # Write one more — should evict chunk with lowest salience (0)
        wm.write(sv, salience=7.0)
        assert len(wm.chunks) == 7

        # All remaining chunks should have salience >= 1
        for chunk in wm.chunks:
            assert chunk.salience >= 1.0

    def test_capacity_bounds_valid(self):
        """Capacity must be 5-9."""
        for cap in (5, 7, 9):
            M2WorkingMemory(capacity=cap)  # should not raise
        with pytest.raises(ValueError):
            M2WorkingMemory(capacity=4)
        with pytest.raises(ValueError):
            M2WorkingMemory(capacity=10)


