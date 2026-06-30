"""Tests for PHCA-3.2-008: Consolidation Scheduler (E→S transfer) and full cycle integration."""

from __future__ import annotations

import numpy as np
import pytest

from phca.config import StateVector
from phca.memory.m3_episodic import M3EpisodicMemory, EpisodeRecord
from phca.consolidation.scheduler import ConsolidationScheduler, ConsolidationReport, SemanticFact


class TestConsolidationSchedulerInit:
    """Tests for ConsolidationScheduler initialization."""

    def test_init_defaults(self):
        """Default init should create a valid scheduler."""
        m3 = M3EpisodicMemory(max_episodes=100, state_dim=4, action_dim=2)
        cs = ConsolidationScheduler(m3=m3, state_dim=4)
        assert cs._consolidation_interval == 10
        assert cs._max_facts_per_cycle == 50
        assert cs._last_consolidation_cycle == 0
        assert cs.get_stats()["total_episodes_processed"] == 0

    def test_init_custom_params(self):
        """Custom parameters should be reflected."""
        m3 = M3EpisodicMemory(max_episodes=100, state_dim=4, action_dim=2)
        cs = ConsolidationScheduler(m3=m3, state_dim=4, consolidation_interval=25, max_facts_per_cycle=10)
        assert cs._consolidation_interval == 25
        assert cs._max_facts_per_cycle == 10


class TestConsolidationStep:
    """Tests for the consolidation step()."""

    def test_step_returns_empty_when_not_due(self):
        """step() before interval returns empty report."""
        m3 = M3EpisodicMemory(max_episodes=100, state_dim=4, action_dim=2)
        cs = ConsolidationScheduler(m3=m3, state_dim=4, consolidation_interval=10)
        report = cs.step(cycle_count=1)  # only 1 cycle since last
        assert report.episodes_processed == 0
        assert report.success is True

    def test_step_returns_empty_when_no_episodes(self):
        """step() at interval but with no episodes returns empty success."""
        m3 = M3EpisodicMemory(max_episodes=100, state_dim=4, action_dim=2)
        cs = ConsolidationScheduler(m3=m3, state_dim=4, consolidation_interval=1)
        report = cs.step(cycle_count=1, force=True)
        assert report.episodes_processed == 0
        assert report.success is True

    def test_step_processes_episodes(self):
        """step() should process unconsolidated episodes."""
        m3 = M3EpisodicMemory(max_episodes=100, state_dim=4, action_dim=2)
        cs = ConsolidationScheduler(m3=m3, state_dim=4, consolidation_interval=1, max_facts_per_cycle=10)

        # Store some episodes
        for i in range(5):
            state = StateVector(
                values=np.full(4, i * 0.1, dtype=np.float32),
                precision=np.ones(4, dtype=np.float32),
                timestamp=float(i),
            )
            m3.store_episode(
                state_before=state,
                action_taken=np.array([1.0, 0.0], dtype=np.float32),
                state_after=StateVector(
                    values=np.full(4, i * 0.1 + 0.05, dtype=np.float32),
                    precision=np.ones(4, dtype=np.float32),
                ),
                prediction_error=float(i) * 0.1,
                confidence=0.8 - i * 0.1,
                timestamp=i,
            )

        assert m3.count() == 5
        report = cs.step(cycle_count=10, force=True)
        assert report.episodes_processed > 0
        assert report.facts_generated > 0
        assert report.success is True

    def test_consolidated_episodes_marked(self):
        """Episodes should be marked consolidated after processing."""
        m3 = M3EpisodicMemory(max_episodes=100, state_dim=4, action_dim=2)
        cs = ConsolidationScheduler(m3=m3, state_dim=4, consolidation_interval=1, max_facts_per_cycle=10)

        state = StateVector(
            values=np.zeros(4, dtype=np.float32),
            precision=np.ones(4, dtype=np.float32),
        )
        m3.store_episode(
            state_before=state,
            action_taken=np.array([1.0, 0.0], dtype=np.float32),
            state_after=state,
            prediction_error=0.1,
            timestamp=0,
        )

        cs.step(cycle_count=10, force=True)
        unconsolidated = m3.count(consolidated=0)
        assert unconsolidated == 0

    def test_forced_consolidation(self):
        """force=True should consolidate regardless of interval."""
        m3 = M3EpisodicMemory(max_episodes=100, state_dim=4, action_dim=2)
        cs = ConsolidationScheduler(m3=m3, state_dim=4, consolidation_interval=100)

        state = StateVector(
            values=np.zeros(4, dtype=np.float32),
            precision=np.ones(4, dtype=np.float32),
        )
        m3.store_episode(
            state_before=state,
            action_taken=np.array([1.0, 0.0], dtype=np.float32),
            state_after=state,
            prediction_error=0.1,
            timestamp=0,
        )

        report = cs.step(cycle_count=1, force=True)
        assert report.episodes_processed > 0


class TestConsolidationExtractFacts:
    """Tests for fact extraction from episodes."""

    def test_extract_facts_from_episodes(self):
        """_extract_facts should produce SemanticFact objects."""
        m3 = M3EpisodicMemory(max_episodes=100, state_dim=4, action_dim=2)
        cs = ConsolidationScheduler(m3=m3, state_dim=4, max_facts_per_cycle=10)

        episodes = []
        for i in range(3):
            ep = EpisodeRecord(
                episode_id=i,
                state_before=StateVector(
                    values=np.full(4, i * 0.1, dtype=np.float32),
                    precision=np.ones(4, dtype=np.float32),
                ),
                action_taken=np.array([1.0, 0.0], dtype=np.float32),
                state_after=StateVector(
                    values=np.full(4, i * 0.1 + 0.05, dtype=np.float32),
                    precision=np.ones(4, dtype=np.float32),
                ),
                prediction_error=0.1,
                timestamp=i,
            )
            episodes.append(ep)

        facts = cs._extract_facts(episodes)
        assert len(facts) > 0
        assert all(isinstance(f, SemanticFact) for f in facts)
        assert all(f.fact_type in ("transition", "novelty", "well_known") for f in facts)

    def test_similar_episodes_merge(self):
        """Similar episodes should merge into fewer facts."""
        m3 = M3EpisodicMemory(max_episodes=100, state_dim=4, action_dim=2)
        cs = ConsolidationScheduler(m3=m3, state_dim=4, similarity_threshold=0.5, max_facts_per_cycle=10)

        episodes = []
        for i in range(5):
            ep = EpisodeRecord(
                episode_id=i,
                state_before=StateVector(
                    values=np.full(4, 0.1, dtype=np.float32),
                    precision=np.ones(4, dtype=np.float32),
                ),
                action_taken=np.array([1.0, 0.0], dtype=np.float32),
                state_after=StateVector(
                    values=np.full(4, 0.15, dtype=np.float32),
                    precision=np.ones(4, dtype=np.float32),
                ),
                prediction_error=0.1,
                timestamp=i,
            )
            episodes.append(ep)

        facts = cs._extract_facts(episodes)
        # All episodes are nearly identical, should merge into ~1-2 facts
        assert len(facts) <= 3
        # Frequency should reflect merging
        if len(facts) >= 1:
            assert facts[0].frequency >= 2

    def test_empty_episodes_returns_empty(self):
        """Empty episode list should return empty facts."""
        m3 = M3EpisodicMemory(max_episodes=100, state_dim=4, action_dim=2)
        cs = ConsolidationScheduler(m3=m3, state_dim=4)
        facts = cs._extract_facts([])
        assert facts == []


class TestConsolidationReports:
    """Tests for consolidation reporting."""

    def test_report_stored_in_history(self):
        """Successful consolidations should be stored in history."""
        m3 = M3EpisodicMemory(max_episodes=100, state_dim=4, action_dim=2)
        cs = ConsolidationScheduler(m3=m3, state_dim=4, consolidation_interval=1, max_facts_per_cycle=5)

        state = StateVector(
            values=np.zeros(4, dtype=np.float32),
            precision=np.ones(4, dtype=np.float32),
        )
        m3.store_episode(
            state_before=state, action_taken=np.array([1.0, 0.0], dtype=np.float32),
            state_after=state, prediction_error=0.1, timestamp=0,
        )

        cs.step(cycle_count=10, force=True)
        assert len(cs._history) >= 1
        assert cs._history[-1].episodes_processed > 0

    def test_stats_summary(self):
        """get_stats() should return correct totals."""
        m3 = M3EpisodicMemory(max_episodes=100, state_dim=4, action_dim=2)
        cs = ConsolidationScheduler(m3=m3, state_dim=4, consolidation_interval=1, max_facts_per_cycle=5)

        state = StateVector(
            values=np.zeros(4, dtype=np.float32),
            precision=np.ones(4, dtype=np.float32),
        )
        m3.store_episode(
            state_before=state, action_taken=np.array([1.0, 0.0], dtype=np.float32),
            state_after=state, prediction_error=0.1, timestamp=0,
        )

        cs.step(cycle_count=10, force=True)
        stats = cs.get_stats()
        assert stats["total_episodes_processed"] > 0
        assert stats["total_facts_stored"] > 0
        assert stats["last_report"] is not None


class TestSemanticFactRetrieval:
    """Tests for semantic fact queries."""

    def test_get_facts_by_type(self):
        """get_semantic_facts should filter by fact_type."""
        m3 = M3EpisodicMemory(max_episodes=100, state_dim=4, action_dim=2)
        cs = ConsolidationScheduler(m3=m3, state_dim=4, max_facts_per_cycle=10)

        # Store episodes with different error magnitudes to get different fact types
        for i in range(3):
            state = StateVector(
                values=np.full(4, i * 0.2, dtype=np.float32),
                precision=np.ones(4, dtype=np.float32),
            )
            m3.store_episode(
                state_before=state,
                action_taken=np.array([1.0, 0.0], dtype=np.float32),
                state_after=StateVector(
                    values=np.full(4, i * 0.2 + 0.1, dtype=np.float32),
                    precision=np.ones(4, dtype=np.float32),
                ),
                prediction_error=i * 0.3,
                timestamp=i,
            )

        cs.step(cycle_count=10, force=True)

        # At least some facts should exist
        all_facts = cs.get_semantic_facts()
        assert len(all_facts) > 0

    def test_fact_confidence_filter(self):
        """get_semantic_facts should filter by min_confidence."""
        m3 = M3EpisodicMemory(max_episodes=100, state_dim=4, action_dim=2)
        cs = ConsolidationScheduler(m3=m3, state_dim=4, max_facts_per_cycle=5)

        state = StateVector(
            values=np.zeros(4, dtype=np.float32),
            precision=np.ones(4, dtype=np.float32),
        )
        m3.store_episode(
            state_before=state, action_taken=np.array([1.0, 0.0], dtype=np.float32),
            state_after=state, prediction_error=0.1, timestamp=0,
        )

        cs.step(cycle_count=10, force=True)

        high_conf = cs.get_semantic_facts(min_confidence=0.9)
        low_conf = cs.get_semantic_facts(min_confidence=0.0)
        # high confidence filter should be subset (or same if all facts are high conf)
        assert len(high_conf) <= len(low_conf)


class TestCosineSimilarity:
    """Direct tests for _cosine_similarity static method."""

    def test_identical_vectors(self):
        """Cosine similarity of a vector with itself is 1.0."""
        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        sim = ConsolidationScheduler._cosine_similarity(a, a)
        assert abs(sim - 1.0) < 1e-6

    def test_orthogonal_vectors(self):
        """Cosine similarity of orthogonal vectors is 0.0."""
        a = np.array([1.0, 0.0], dtype=np.float32)
        b = np.array([0.0, 1.0], dtype=np.float32)
        sim = ConsolidationScheduler._cosine_similarity(a, b)
        assert abs(sim) < 1e-6

    def test_zero_vector(self):
        """Cosine similarity with zero vector is 0.0."""
        a = np.array([1.0, 2.0], dtype=np.float32)
        zero = np.zeros(2, dtype=np.float32)
        sim = ConsolidationScheduler._cosine_similarity(a, zero)
        assert sim == 0.0

    def test_empty_dim(self):
        """Zero-norm on both sides returns 0.0."""
        a = np.zeros(3, dtype=np.float32)
        b = np.zeros(3, dtype=np.float32)
        sim = ConsolidationScheduler._cosine_similarity(a, b)
        assert sim == 0.0

    def test_parallel_vectors(self):
        """Parallel vectors have similarity 1.0."""
        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        b = np.array([2.0, 4.0, 6.0], dtype=np.float32)
        sim = ConsolidationScheduler._cosine_similarity(a, b)
        assert abs(sim - 1.0) < 1e-6


class TestFactTypeClassification:
    """Tests for fact type classification based on prediction_error/confidence."""

    def test_novelty_fact_high_error(self):
        """prediction_error > 0.5 → fact_type == 'novelty'."""
        m3 = M3EpisodicMemory(max_episodes=100, state_dim=4, action_dim=2)
        cs = ConsolidationScheduler(m3=m3, state_dim=4, max_facts_per_cycle=10)
        ep = EpisodeRecord(
            episode_id=1,
            state_before=StateVector(values=np.zeros(4, dtype=np.float32), precision=np.ones(4)),
            action_taken=np.array([1.0, 0.0], dtype=np.float32),
            state_after=StateVector(values=np.ones(4, dtype=np.float32), precision=np.ones(4)),
            prediction_error=0.8,
            confidence=0.3,
            timestamp=0,
        )
        facts = cs._extract_facts([ep])
        assert len(facts) == 1
        assert facts[0].fact_type == "novelty"

    def test_well_known_fact_high_confidence(self):
        """confidence > 0.8 and low error → fact_type == 'well_known'."""
        m3 = M3EpisodicMemory(max_episodes=100, state_dim=4, action_dim=2)
        cs = ConsolidationScheduler(m3=m3, state_dim=4, max_facts_per_cycle=10)
        ep = EpisodeRecord(
            episode_id=1,
            state_before=StateVector(values=np.zeros(4, dtype=np.float32), precision=np.ones(4)),
            action_taken=np.array([1.0, 0.0], dtype=np.float32),
            state_after=StateVector(values=np.ones(4, dtype=np.float32), precision=np.ones(4)),
            prediction_error=0.1,
            confidence=0.9,
            timestamp=0,
        )
        facts = cs._extract_facts([ep])
        assert len(facts) == 1
        assert facts[0].fact_type == "well_known"

    def test_transition_fact_default(self):
        """Low error and low confidence → fact_type == 'transition'."""
        m3 = M3EpisodicMemory(max_episodes=100, state_dim=4, action_dim=2)
        cs = ConsolidationScheduler(m3=m3, state_dim=4, max_facts_per_cycle=10)
        ep = EpisodeRecord(
            episode_id=1,
            state_before=StateVector(values=np.zeros(4, dtype=np.float32), precision=np.ones(4)),
            action_taken=np.array([1.0, 0.0], dtype=np.float32),
            state_after=StateVector(values=np.ones(4, dtype=np.float32), precision=np.ones(4)),
            prediction_error=0.1,
            confidence=0.5,
            timestamp=0,
        )
        facts = cs._extract_facts([ep])
        assert len(facts) == 1
        assert facts[0].fact_type == "transition"


class TestStoreFactsPruning:
    """Tests for _store_facts pruning logic."""

    def test_prune_at_10000(self):
        """_store_facts prunes lowest-confidence facts when exceeding 10_000."""
        m3 = M3EpisodicMemory(max_episodes=10000, state_dim=2, action_dim=1)
        cs = ConsolidationScheduler(m3=m3, state_dim=2, max_facts_per_cycle=100)

        # Store just under limit
        facts = [
            SemanticFact(
                fact_id=f"f{i}", source_episode_id=i, fact_type="transition",
                state_pattern=StateVector(values=np.zeros(2, dtype=np.float32), precision=np.ones(2)),
                confidence=0.5, frequency=1,
            )
            for i in range(10_000)
        ]
        cs._store_facts(facts)
        assert len(cs._committed_facts) == 10_000

    def test_prune_removes_lowest(self):
        """After exceeding 10_000, only 5_000 highest-confidence facts remain."""
        m3 = M3EpisodicMemory(max_episodes=10000, state_dim=2, action_dim=1)
        cs = ConsolidationScheduler(m3=m3, state_dim=2, max_facts_per_cycle=100)

        facts = [
            SemanticFact(
                fact_id=f"f{i}", source_episode_id=i, fact_type="transition",
                state_pattern=StateVector(values=np.zeros(2, dtype=np.float32), precision=np.ones(2)),
                confidence=float(i) / 10_000, frequency=1,
            )
            for i in range(10_001)
        ]
        cs._store_facts(facts)
        assert len(cs._committed_facts) == 5_000
        # All remaining facts should be high-confidence (top half)
        min_conf = min(f.confidence for f in cs._committed_facts)
        assert min_conf > 0.5


class TestConsolidationErrorHandling:
    """Tests for consolidation error paths."""

    def test_step_handles_snapshot_failure(self):
        """step() should not crash if M3 snapshot creation fails."""
        m3 = M3EpisodicMemory(max_episodes=100, state_dim=4, action_dim=2)
        cs = ConsolidationScheduler(m3=m3, state_dim=4, consolidation_interval=1)

        # Corrupt the store so create_snapshot raises
        m3._conn.execute("DROP TABLE episodes")

        report = cs.step(cycle_count=10, force=True)
        assert report.success is False
        assert report.episodes_processed == 0

    def test_step_report_has_all_fields(self):
        """ConsolidationReport from a successful step has all fields populated."""
        m3 = M3EpisodicMemory(max_episodes=100, state_dim=4, action_dim=2)
        cs = ConsolidationScheduler(m3=m3, state_dim=4, consolidation_interval=1, max_facts_per_cycle=5)

        state = StateVector(values=np.zeros(4, dtype=np.float32), precision=np.ones(4))
        m3.store_episode(
            state_before=state, action_taken=np.array([1.0, 0.0], dtype=np.float32),
            state_after=state, prediction_error=0.1, timestamp=0,
        )

        report = cs.step(cycle_count=10, force=True)
        assert report.cycles_since_last >= 0
        assert report.episodes_processed > 0
        assert report.facts_generated > 0
        assert report.snapshot_version > 0
        assert report.duration_ms >= 0
        assert report.success is True
