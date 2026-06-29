"""
Tests for M2 Working Memory (PHCA-3.1-005).

Covers: capacity 7±2 bounds, salience-based eviction, read/write, clear, resize.
"""

import numpy as np
import pytest

from phca.memory.m2_working import M2WorkingMemory, Chunk
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
        assert wm.is_full

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

    def test_read_all_chunks(self, wm):
        """Read without chunk_id should return all chunks."""
        sv = StateVector(values=np.ones(4, dtype=np.float32), precision=np.ones(4, dtype=np.float32))
        for _ in range(3):
            wm.write(sv)
        assert len(wm.read()) == 3

    def test_read_by_chunk_id(self, wm):
        """Read with chunk_id should return specific chunk."""
        sv = StateVector(values=np.ones(4, dtype=np.float32), precision=np.ones(4, dtype=np.float32))
        chunk = wm.write(sv)
        found = wm.read(chunk.chunk_id)
        assert len(found) == 1
        assert found[0].chunk_id == chunk.chunk_id

    def test_update_salience(self, wm):
        """update_salience should change a chunk's salience."""
        sv = StateVector(values=np.ones(4, dtype=np.float32), precision=np.ones(4, dtype=np.float32))
        chunk = wm.write(sv, salience=0.5)
        wm.update_salience(chunk.chunk_id, 0.9)
        updated = wm.read(chunk.chunk_id)[0]
        assert updated.salience == 0.9

    def test_clear(self, wm):
        """Clear should remove all chunks."""
        sv = StateVector(values=np.ones(4, dtype=np.float32), precision=np.ones(4, dtype=np.float32))
        for _ in range(5):
            wm.write(sv)
        wm.clear()
        assert len(wm.chunks) == 0

    def test_capacity_bounds_valid(self):
        """Capacity must be 5-9."""
        for cap in (5, 7, 9):
            M2WorkingMemory(capacity=cap)  # should not raise
        with pytest.raises(AssertionError):
            M2WorkingMemory(capacity=4)
        with pytest.raises(AssertionError):
            M2WorkingMemory(capacity=10)

    def test_resize_larger(self, wm):
        """Resize to larger should work."""
        wm.resize(9)
        assert wm.capacity == 9

    def test_resize_smaller_evicts(self, wm):
        """Resize to smaller should evict lowest-salience chunks."""
        sv = StateVector(values=np.ones(4, dtype=np.float32), precision=np.ones(4, dtype=np.float32))
        for i in range(7):
            wm.write(sv, salience=float(i))
        wm.resize(5)  # Minimum valid capacity (7±2)
        assert wm.capacity == 5
        # Only the 5 highest-salience chunks remain (salience 2..6)
        for chunk in wm.chunks:
            assert chunk.salience >= 2.0
