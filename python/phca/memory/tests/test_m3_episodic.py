"""
Tests for M3 Episodic Memory (PHCA-3.2-002/009).

Covers: SQLite storage, read/write, query, batch, MVCC snapshots,
consolidation marking, eviction, reset.

v3.0 Reference: v3.0 Patch §2.3 (Memory concurrency)
"""

from __future__ import annotations

import numpy as np
import pytest

from phca.memory.m3_episodic import M3EpisodicMemory, EpisodeRecord
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

    def test_store_and_retrieve(self, m3, sample_episode_data):
        """Store an episode and retrieve it by ID."""
        state_before, action, state_after = sample_episode_data
        ep_id = m3.store_episode(state_before, action, state_after, 0.5, confidence=0.8)
        assert ep_id > 0

        retrieved = m3.get_episode(ep_id)
        assert retrieved is not None
        assert retrieved.prediction_error == pytest.approx(0.5)
        assert retrieved.confidence == pytest.approx(0.8)
        assert retrieved.state_before is not None
        np.testing.assert_array_almost_equal(
            retrieved.state_before.values, state_before.values
        )

    def test_store_increments_count(self, m3, sample_episode_data):
        """Storing episodes increments the count."""
        state_before, action, state_after = sample_episode_data
        m3.store_episode(state_before, action, state_after, 0.1)
        assert m3.count() == 1

        m3.store_episode(state_after, action, state_before, 0.2)
        assert m3.count() == 2

    def test_query_episodes_with_filters(self, m3, sample_episode_data):
        """Query episodes with consolidated filter."""
        state_before, action, state_after = sample_episode_data
        ids = []
        for i in range(5):
            eid = m3.store_episode(state_before, action, state_after, float(i) * 0.1)
            ids.append(eid)

        # Query unconsolidated
        results = m3.query_episodes(limit=10, consolidated=0)
        assert len(results) == 5

        # Mark one as consolidated
        m3.mark_consolidated([ids[0]])
        results = m3.query_episodes(limit=10, consolidated=0)
        assert len(results) == 4
        results_cons = m3.query_episodes(limit=10, consolidated=1)
        assert len(results_cons) == 1

    def test_batch_store(self, m3, sample_episode_data):
        """Store multiple episodes in a batch."""
        state_before, action, state_after = sample_episode_data
        episodes = [
            EpisodeRecord(
                state_before=state_before, action_taken=action,
                state_after=state_after, prediction_error=0.1,
                confidence=0.9, timestamp=i,
            )
            for i in range(5)
        ]
        ids = m3.store_batch(episodes)
        assert len(ids) == 5
        assert m3.count() == 5

    def test_sample_batch(self, m3, sample_episode_data):
        """Sample episodes for experience replay."""
        state_before, action, state_after = sample_episode_data
        for i in range(10):
            m3.store_episode(state_before, action, state_after, float(i) * 0.1)

        samples = m3.sample_batch(batch_size=3, seed=42)
        assert len(samples) == 3
        # All sampled episodes should be valid
        for s in samples:
            assert s.state_before is not None

    def test_sample_batch_empty(self, m3):
        """Sampling from empty store returns empty."""
        samples = m3.sample_batch(batch_size=5)
        assert samples == []

    def test_create_snapshot(self, m3, sample_episode_data):
        """Create an MVCC snapshot for consolidation."""
        state_before, action, state_after = sample_episode_data
        for i in range(5):
            m3.store_episode(state_before, action, state_after, 0.1)

        snapshot = m3.create_snapshot()
        assert snapshot.snapshot_version > 0
        assert len(snapshot.episodes) == 5
        assert "snap" in snapshot.snapshot_id

    def test_mark_consolidated(self, m3, sample_episode_data):
        """Mark episodes as consolidated."""
        state_before, action, state_after = sample_episode_data
        ids = [m3.store_episode(state_before, action, state_after, 0.1) for _ in range(3)]

        marked = m3.mark_consolidated(ids[:2])
        assert marked == 2

        uncons = m3.query_episodes(limit=10, consolidated=0)
        assert len(uncons) == 1

    def test_version_increment(self, m3):
        """Increment the MVCC version counter."""
        v1 = m3.increment_version()
        v2 = m3.increment_version()
        assert v2 == v1 + 1

    def test_reset_clears_all(self, m3, sample_episode_data):
        """Reset clears all episodes."""
        state_before, action, state_after = sample_episode_data
        m3.store_episode(state_before, action, state_after, 0.1)
        assert m3.count() > 0

        m3.reset()
        assert m3.count() == 0

    def test_eviction(self, m3, sample_episode_data):
        """Evict oldest episodes when over capacity."""
        state_before, action, state_after = sample_episode_data
        # Store more than max_episodes (100)
        for i in range(105):
            m3.store_episode(state_before, action, state_after, 0.1, timestamp=i)

        assert m3.count() <= 100  # should evict to max_episodes

    def test_store_with_drive_id(self, m3, sample_episode_data):
        """Store episode with drive_id metadata."""
        state_before, action, state_after = sample_episode_data
        ep_id = m3.store_episode(
            state_before, action, state_after, 0.5,
            drive_id=3,  # D3 competence
        )
        retrieved = m3.get_episode(ep_id)
        assert retrieved is not None
        assert retrieved.drive_id == 3

    def test_query_by_timestamp_range(self, m3, sample_episode_data):
        """Query episodes within a timestamp range."""
        state_before, action, state_after = sample_episode_data
        for i in range(10):
            sv = StateVector(
                values=np.array([float(i), 0.0, 0.0, 0.0], dtype=np.float32),
                precision=np.ones(4, dtype=np.float32),
                timestamp=float(i),
            )
            m3.store_episode(sv, action, state_after, 0.1, timestamp=i)

        results = m3.query_episodes(limit=100, min_timestamp=3, max_timestamp=7)
        assert len(results) == 5
        for r in results:
            assert 3 <= r.timestamp <= 7


class TestEpisodeRecord:
    """EpisodeRecord creation and serialization."""

    def test_record_from_storage(self, m3, sample_episode_data):
        """Episode record fields are preserved through store/retrieve."""
        state_before, action, state_after = sample_episode_data
        ep_id = m3.store_episode(
            state_before, action, state_after,
            prediction_error=0.42, confidence=0.85,
            drive_id=1,
        )
        ep = m3.get_episode(ep_id)
        assert ep is not None
        assert ep.episode_id == ep_id
        assert ep.version == 1
        assert ep.prediction_error == pytest.approx(0.42)
        assert ep.confidence == pytest.approx(0.85)
        assert ep.drive_id == 1
        assert ep.consolidated == 0

    def test_none_for_missing_id(self, m3):
        """Getting a non-existent episode returns None."""
        assert m3.get_episode(99999) is None
