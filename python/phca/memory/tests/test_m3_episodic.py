"""
Tests for M3 Episodic Memory (PHCA-3.2-002/009).

Covers: SQLite storage, store_episode, count, MVCC snapshots,
consolidation marking, eviction, version increment.

v3.0 Reference: v3.0 Patch §2.3 (Memory concurrency)
"""

from __future__ import annotations

import numpy as np
import pytest

from phca.memory.m3_episodic import M3EpisodicMemory
from phca.config import StateVector


@pytest.fixture
def m3() -> M3EpisodicMemory:
    """In-memory M3 with small capacity for testing."""
    return M3EpisodicMemory(
        db_path=":memory:", max_episodes=100,
        state_dim=4, action_dim=2,
    )


@pytest.fixture
def sample_episode_data():
    """Create sample episode data."""
    state_before = StateVector(
        values=np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32),
        precision=np.ones(4, dtype=np.float32),
        timestamp=0.0,
    )
    action = np.array([1.0, 0.0], dtype=np.float32)
    state_after = StateVector(
        values=np.array([0.0, 2.0, 0.0, 0.0], dtype=np.float32),
        precision=np.ones(4, dtype=np.float32),
        timestamp=1.0,
    )
    return state_before, action, state_after


class TestM3EpisodicMemory:
    """Core M3 tests."""

    def test_init_creates_empty_store(self, m3):
        """New M3 should have zero episodes."""
        assert m3.count() == 0
        assert m3.count(consolidated=0) == 0

    def test_store_increments_count(self, m3, sample_episode_data):
        """Storing episodes increments the count."""
        state_before, action, state_after = sample_episode_data
        m3.store_episode(state_before, action, state_after, 0.1)
        assert m3.count() == 1

        m3.store_episode(state_after, action, state_before, 0.2)
        assert m3.count() == 2

    def test_create_snapshot(self, m3, sample_episode_data):
        """Create an MVCC snapshot for consolidation."""
        state_before, action, state_after = sample_episode_data
        for i in range(5):
            m3.store_episode(state_before, action, state_after, 0.1)

        snapshot = m3.create_snapshot()
        assert snapshot.snapshot_version > 0
        assert len(snapshot.episodes) == 5
        assert "snap" in snapshot.snapshot_id

    def test_version_increment(self, m3):
        """Increment the MVCC version counter."""
        v1 = m3.increment_version()
        v2 = m3.increment_version()
        assert v2 == v1 + 1

    def test_eviction(self, m3, sample_episode_data):
        """Evict oldest episodes when over capacity."""
        state_before, action, state_after = sample_episode_data
        # Store more than max_episodes (100)
        for i in range(105):
            m3.store_episode(state_before, action, state_after, 0.1, timestamp=i)

        assert m3.count() <= 100  # should evict to max_episodes


# ── SQLite Hardening (G-010 / D-082) ──────────────────────────


class TestM3Hardening:
    """SQLite hardening: integrity check, VACUUM, WAL checkpoint (G-010)."""

    def test_integrity_check_on_file_db(self, tmp_path):
        """A fresh file-backed M3 must pass integrity_check on init."""
        db = tmp_path / "m3_integrity.db"
        m3 = M3EpisodicMemory(db_path=str(db), max_episodes=50,
                              state_dim=4, action_dim=2)
        # If integrity_check failed, db_path would have been switched to
        # ":memory:". A healthy init keeps the file path.
        assert m3.db_path == str(db)
        m3.close()

    def test_integrity_failure_falls_back_to_memory(self, tmp_path):
        """A corrupt file DB must fall back to in-memory (G-010)."""
        db = tmp_path / "m3_corrupt.db"
        # Write garbage into the file so SQLite integrity_check fails.
        db.write_bytes(b"not a sqlite database file")
        m3 = M3EpisodicMemory(db_path=str(db), max_episodes=50,
                              state_dim=4, action_dim=2)
        # The fallback rewrites db_path to ":memory:".
        assert m3.db_path == ":memory:"
        # The fallback store must be usable.
        state = StateVector(values=np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32),
                            precision=np.ones(4, dtype=np.float32), timestamp=0.0)
        action = np.array([1.0, 0.0], dtype=np.float32)
        m3.store_episode(state, action, state, 0.1, timestamp=0)
        assert m3.count() == 1
        m3.close()

    def test_vacuum_runs_after_threshold(self, tmp_path):
        """VACUUM must trigger after _vacuum_interval evictions on a file DB."""
        db = tmp_path / "m3_vacuum.db"
        m3 = M3EpisodicMemory(db_path=str(db), max_episodes=10,
                              state_dim=4, action_dim=2)
        # Force a low vacuum interval to trigger without 1000 writes.
        m3._vacuum_interval = 5
        state = StateVector(values=np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32),
                            precision=np.ones(4, dtype=np.float32), timestamp=0.0)
        action = np.array([1.0, 0.0], dtype=np.float32)
        # Store enough to trigger eviction > vacuum_interval.
        for i in range(30):
            m3.store_episode(state, action, state, 0.1, timestamp=i)
        # After evictions >= vacuum_interval, _episodes_since_vacuum resets.
        assert m3._episodes_since_vacuum == 0
        m3.close()

    def test_close_runs_wal_checkpoint_on_file_db(self, tmp_path):
        """close() must run wal_checkpoint(TRUNCATE) for file-backed DBs."""
        db = tmp_path / "m3_wal.db"
        m3 = M3EpisodicMemory(db_path=str(db), max_episodes=50,
                              state_dim=4, action_dim=2)
        state = StateVector(values=np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32),
                            precision=np.ones(4, dtype=np.float32), timestamp=0.0)
        action = np.array([1.0, 0.0], dtype=np.float32)
        m3.store_episode(state, action, state, 0.1, timestamp=0)
        # close() should not raise and should checkpoint the WAL.
        m3.close()
        assert m3._conn is None

    def test_close_no_checkpoint_for_in_memory(self):
        """close() must skip wal_checkpoint for in-memory DBs (no WAL)."""
        m3 = M3EpisodicMemory(db_path=":memory:", max_episodes=50,
                              state_dim=4, action_dim=2)
        # Should be a no-op (no exception) for in-memory DBs.
        m3.close()
        assert m3._conn is None

