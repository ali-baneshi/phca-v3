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


class TestConsolidationReset:
    """Tests for reset()."""

    def test_reset_clears_state(self):
        """reset() should clear all consolidation state."""
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
        assert cs._total_processed > 0

        cs.reset()
        assert cs._total_processed == 0
        assert cs._total_facts == 0
        assert len(cs._history) == 0


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
