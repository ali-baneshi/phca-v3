"""
PHCA v3.0 — Consolidation Scheduler (Sleep-Cycle Analogue).

Phase 3.2: Periodic E→S transfer consolidating unprocessed M3 episodes
into the S-Stream (semantic memory) via MVCC snapshot isolation.

Cycle steps 16-18 (blueprint xa77.B):
  Step 16: Create M3 snapshot for consolidation
  Step 17: Process snapshot → extract semantic facts
  Step 18: Write facts to S-Stream (atomic transaction)

Step 20 (sleep cycle):
  Periodic full consolidation cycle every N cognitive cycles.

v3.0 Reference: v3.0 Patch §2.3, §3.1 Table, Theorem 3.3
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from phca.config import StateVector, StreamID
from phca.memory.m3_episodic import M3EpisodicMemory, EpisodeRecord
from phca.logging import logger, _log


@dataclass
class ConsolidationReport:
    """Result of a consolidation cycle.

    Attributes:
        cycles_since_last: Cognitive cycles since last consolidation.
        episodes_processed: Number of episodes consolidated this cycle.
        facts_generated: Number of semantic facts extracted.
        snapshot_version: MVCC version of the snapshot used.
        duration_ms: Wall-clock time for consolidation.
        success: Whether consolidation completed without errors.
    """
    cycles_since_last: int = 0
    episodes_processed: int = 0
    facts_generated: int = 0
    snapshot_version: int = 0
    duration_ms: float = 0.0
    success: bool = True


@dataclass
class SemanticFact:
    """A single semantic fact extracted from an episode during consolidation.

    Attributes:
        fact_id: Unique identifier for this fact.
        source_episode_id: Episode this fact was extracted from.
        fact_type: Type of semantic knowledge (transition, reward, etc.).
        state_pattern: Prototypical state vector for this fact.
        confidence: How well-supported this fact is (0.0-1.0).
        frequency: How many episodes support this fact.
    """
    fact_id: str = ""
    source_episode_id: int = 0
    fact_type: str = ""
    state_pattern: Optional[StateVector] = None
    confidence: float = 0.0
    frequency: int = 1


class ConsolidationScheduler:
    """Periodic E→S consolidation with MVCC snapshot isolation.

    Phase 3.2:
        - Every N cognitive cycles, create an MVCC snapshot of M3
        - Process unconsolidated episodes into semantic facts
        - Update S-Stream (via TSPL S-Stream parameters)
        - Log consolidation progress

    Phase 3.3+:
        - Distributed consolidation across sleep cycles
        - Hierarchical fact extraction (episode → event → narrative)
    """

    def __init__(
        self,
        m3: M3EpisodicMemory,
        state_dim: int = 84,
        consolidation_interval: int = 10,
        max_facts_per_cycle: int = 50,
        similarity_threshold: float = 0.85,
    ):
        """Initialize the consolidation scheduler.

        Args:
            m3: Reference to M3 episodic memory.
            state_dim: Dimensionality of state vectors.
            consolidation_interval: Cognitive cycles between consolidations.
            max_facts_per_cycle: Max semantic facts to generate per cycle.
            similarity_threshold: Cosine similarity threshold for fact merging.
        """
        self.m3 = m3
        self._state_dim = state_dim
        self._consolidation_interval = consolidation_interval
        self._max_facts_per_cycle = max_facts_per_cycle
        self._similarity_threshold = similarity_threshold

        # Consolidation state
        self._last_consolidation_cycle: int = 0
        self._current_snapshot: Optional[Any] = None
        self._semantic_facts: List[SemanticFact] = []
        self._total_processed: int = 0
        self._total_facts: int = 0
        self._history: List[ConsolidationReport] = []

        _log(logger, "info", "consolidation.init",
             interval=consolidation_interval, max_facts=max_facts_per_cycle)

    # ── Main Entry Point ─────────────────────────────────────

    def step(
        self,
        cycle_count: int,
        force: bool = False,
    ) -> ConsolidationReport:
        """Run consolidation if it's time (or force=True).

        Args:
            cycle_count: Current cognitive cycle number.
            force: If True, run consolidation regardless of interval.

        Returns:
            ConsolidationReport with results.
        """
        cycles_since = cycle_count - self._last_consolidation_cycle

        if not force and cycles_since < self._consolidation_interval:
            # Not time yet — return empty report
            return ConsolidationReport(cycles_since_last=cycles_since)

        t0 = time.perf_counter()

        try:
            # Step 16: Create MVCC snapshot
            snapshot = self.m3.create_snapshot()
            self._current_snapshot = snapshot

            if not snapshot.episodes:
                report = ConsolidationReport(
                    cycles_since_last=cycles_since,
                    snapshot_version=snapshot.snapshot_version,
                    duration_ms=(time.perf_counter() - t0) * 1000,
                    success=True,
                )
                self._last_consolidation_cycle = cycle_count
                self._history.append(report)
                return report

            # Step 17: Process snapshot → extract semantic facts
            episode_ids = [ep.episode_id for ep in snapshot.episodes]
            facts = self._extract_facts(snapshot.episodes)

            # Step 18: Write facts to S-Stream (log and mark consolidated)
            n_facts = self._store_facts(facts)
            marked = self.m3.mark_consolidated(episode_ids)

            # Increment MVCC version for next writes
            self.m3.increment_version()

            t1 = time.perf_counter()
            report = ConsolidationReport(
                cycles_since_last=cycles_since,
                episodes_processed=marked,
                facts_generated=n_facts,
                snapshot_version=snapshot.snapshot_version,
                duration_ms=(t1 - t0) * 1000,
                success=True,
            )

            self._last_consolidation_cycle = cycle_count
            self._total_processed += marked
            self._total_facts += n_facts

            _log(logger, "info", "consolidation.complete",
                 episodes=marked, facts=n_facts,
                 version=snapshot.snapshot_version,
                 duration_ms=f"{report.duration_ms:.1f}",
                 total_processed=self._total_processed,
                 total_facts=self._total_facts)

        except Exception as e:
            t1 = time.perf_counter()
            _log(logger, "error", "consolidation.failed",
                 error=str(e), cycle=cycle_count)
            report = ConsolidationReport(
                cycles_since_last=cycles_since,
                duration_ms=(t1 - t0) * 1000,
                success=False,
            )

        self._history.append(report)
        return report

    # ── Fact Extraction ──────────────────────────────────────

    def _extract_facts(
        self, episodes: List[EpisodeRecord],
    ) -> List[SemanticFact]:
        """Extract semantic facts from a batch of episodes.

        Groups similar episodes by state similarity and extracts:
        - Transition facts: (state_before, action) → state_after patterns
        - Reward facts: high-prediction-error episodes (outliers)
        - Frequency facts: frequently-visited state regions

        Args:
            episodes: Batch of episodes to process.

        Returns:
            List of extracted SemanticFact objects.
        """
        if not episodes:
            return []

        facts: List[SemanticFact] = []
        rng = np.random.RandomState(42)

        # Process episodes in order, merging similar transitions
        for ep in episodes[:self._max_facts_per_cycle]:
            if ep.state_before is None or ep.state_after is None:
                continue

            # Compute signature vector for similarity matching
            sig = np.concatenate([
                ep.state_before.values,
                ep.action_taken if ep.action_taken is not None else np.zeros(1),
                ep.state_after.values[:4],  # first few dims of result
            ])

            # Try to merge with existing fact
            merged = False
            for fact in facts:
                if fact.state_pattern is None:
                    continue
                fact_sig = fact.state_pattern.values[:min(len(fact.state_pattern.values), len(sig))]
                sig_trunc = sig[:len(fact_sig)]
                sim = self._cosine_similarity(sig_trunc, fact_sig)
                if sim > self._similarity_threshold:
                    fact.frequency += 1
                    fact.confidence = min(1.0, fact.confidence + 0.05)
                    merged = True
                    break

            if not merged and len(facts) < self._max_facts_per_cycle:
                # Create new fact
                pattern = StateVector(
                    values=sig[:self._state_dim].astype(np.float32),
                    precision=np.ones(self._state_dim, dtype=np.float32) * 0.5,
                )
                fact_type = "transition"
                if ep.prediction_error > 0.5:
                    fact_type = "novelty"
                elif ep.confidence > 0.8:
                    fact_type = "well_known"

                facts.append(SemanticFact(
                    fact_id=f"fact_{self._total_facts + len(facts)}",
                    source_episode_id=ep.episode_id,
                    fact_type=fact_type,
                    state_pattern=pattern,
                    confidence=0.3,
                    frequency=1,
                ))

        return facts

    def _store_facts(self, facts: List[SemanticFact]) -> int:
        """Store extracted facts in the in-memory S-Stream fact store.

        Phase 3.2: In-memory storage. Phase 3.3+: Durable S-Stream store.

        Args:
            facts: Facts to store.

        Returns:
            Number of facts stored.
        """
        self._semantic_facts.extend(facts)
        if len(self._semantic_facts) > 10_000:
            # Prune lowest-confidence facts
            self._semantic_facts.sort(key=lambda f: f.confidence)
            self._semantic_facts = self._semantic_facts[-5_000:]
        return len(facts)

    # ── Queries ──────────────────────────────────────────────

    def get_semantic_facts(
        self,
        fact_type: Optional[str] = None,
        min_confidence: float = 0.0,
        max_results: int = 100,
    ) -> List[SemanticFact]:
        """Retrieve stored semantic facts with optional filters.

        Args:
            fact_type: Filter by fact type (None = all).
            min_confidence: Minimum confidence threshold.
            max_results: Maximum facts to return.

        Returns:
            Filtered list of SemanticFact objects.
        """
        results = self._semantic_facts
        if fact_type is not None:
            results = [f for f in results if f.fact_type == fact_type]
        results = [f for f in results if f.confidence >= min_confidence]
        return results[:max_results]

    def get_stats(self) -> Dict[str, Any]:
        """Get consolidation statistics.

        Returns:
            Dict with total_processed, total_facts, last_report, history_length.
        """
        return {
            "total_episodes_processed": self._total_processed,
            "total_facts_stored": self._total_facts,
            "last_consolidation_cycle": self._last_consolidation_cycle,
            "last_report": self._history[-1] if self._history else None,
            "history_length": len(self._history),
            "interval": self._consolidation_interval,
        }

    def reset(self) -> None:
        """Reset consolidation state."""
        self._last_consolidation_cycle = 0
        self._current_snapshot = None
        self._semantic_facts.clear()
        self._total_processed = 0
        self._total_facts = 0
        self._history.clear()

    # ── Helpers ─────────────────────────────────────────────

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a < 1e-8 or norm_b < 1e-8:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))
