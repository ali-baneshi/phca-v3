"""
PHCA v3.0 — Consolidation Scheduler (Sleep-Cycle Analogue).

Phase 3.2: Periodic E→S transfer consolidating unprocessed M3 episodes
into the S-Stream (statistical memory) via MVCC snapshot isolation.

Cycle steps 16-18 (blueprint xa77.B):
  Step 16: Create M3 snapshot for consolidation
  Step 17: Process snapshot → extract statistical facts
  Step 18: Write facts to S-Stream (atomic transaction)

Step 20 (sleep cycle):
  Periodic full consolidation cycle every N cognitive cycles.

Note: "Statistical" not "Semantic" — facts are extracted via frequency-based
pattern matching (cosine similarity × confidence), not genuine semantic
understanding. See G-007 in the Phase 4 gap report for details.

v3.0 Reference: v3.0 Patch §2.3, §3.1 Table, Theorem 3.3
"""

from __future__ import annotations

import copy
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from phca.config import StateVector, StreamID
from phca.memory.m3_episodic import M3EpisodicMemory, EpisodeRecord
from phca.logging import logger, _log

# M4 write-lock constants (v3.0 Patch §2.3.2)
# Phase 7 / B2: cap tightened from 10_000/5_000 to 1_000/500 so the retention
# bound engages within a 10k-cycle soak (measured ~0.15 facts/cyc → 1000 facts
# at ~6700 cyc, then confidence-ranked FIFO prune oscillates 500–1000). Bounds
# M4 to ~1000 facts × ~400 B ≈ 400 KB max. D-108.
M4_LOCK_TIMEOUT = 0.050       # 50ms max wait for M4 write lock
M4_MAX_FACTS = 1_000          # max facts before pruning low-confidence
M4_PRUNE_TARGET = 500         # target count after pruning


@dataclass
class ConsolidationReport:
    """Result of a consolidation cycle.

    Attributes:
        cycles_since_last: Cognitive cycles since last consolidation.
        episodes_processed: Number of episodes consolidated this cycle.
        facts_generated: Number of statistical facts extracted.
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
    """A single statistical fact extracted from episodes during consolidation.

    Note: "Statistical" not "Semantic" — facts are extracted via
    frequency-based pattern matching (cosine similarity × confidence),
    not genuine semantic understanding. See G-007 in the Phase 4 gap
    report for architectural context.

    Attributes:
        fact_id: Unique identifier for this fact.
        source_episode_id: Episode this fact was extracted from.
        fact_type: Type of statistical knowledge (transition, novelty, well_known).
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
        - Process unconsolidated episodes into statistical facts
        - Update S-Stream (via TSPL S-Stream parameters)
        - Log consolidation progress

    Phase 3.3+:
        - Distributed consolidation across sleep cycles
        - Hierarchical fact extraction (episode → event → narrative)

    Note: Produces "statistical" facts (frequency-based pattern matching),
    not "semantic" facts. The term "semantic" is aspirational and reserved
    for Phase 4 when a proper embedding layer is added.
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
            max_facts_per_cycle: Max statistical facts to generate per cycle.
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

        # M4 write-lock protected store (v3.0 Patch §2.3.2)
        self._m4_lock: threading.Lock = threading.Lock()
        self._committed_facts: List[SemanticFact] = []
        self._staging_buffer: Optional[List[SemanticFact]] = None

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
            # Flush pending M3 writes so episodes are durable before marking
            self.m3.flush()
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
        """Extract statistical facts from a batch of episodes.

        Groups similar episodes by state similarity and extracts:
        - Transition facts: (state_before, action) → state_after patterns
        - Reward facts: high-prediction-error episodes (outliers)
        - Frequency facts: frequently-visited state regions

        Note: These are "statistical" facts — frequency-based pattern matching
        via cosine similarity, not genuine semantic understanding.

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
        """Store extracted facts in the S-Stream with M4 write-lock atomicity.

        Implements v3.0 Patch §2.3.2 M4 write-lock semantics:
        1. Acquires exclusive write lock with timeout
        2. Builds complete new fact set in staging buffer (atomic transaction)
        3. Atomically swaps staging buffer into committed store
        4. Releases write lock
        5. Readers see last committed state (never blocked)

        Args:
            facts: Facts to store.

        Returns:
            Number of facts stored.
        """
        # Build the new fact set in a staging buffer
        n_new = 0
        staging = list(self._committed_facts)
        all_facts = staging + list(facts)
        n_new = len(facts)

        # Prune if over capacity
        if len(all_facts) > M4_MAX_FACTS:
            all_facts.sort(key=lambda f: f.confidence)
            all_facts = all_facts[-M4_PRUNE_TARGET:]

        # Acquire write lock with timeout (v3.0 §2.3.2 C.5)
        acquired = self._m4_lock.acquire(timeout=M4_LOCK_TIMEOUT)
        if not acquired:
            _log(logger, "warning", "consolidation.m4_lock_timeout",
                 facts_pending=n_new,
                 msg="M4 write lock timeout — consolidation cycle skipped")
            return 0

        try:
            self._staging_buffer = all_facts
            self._committed_facts = self._staging_buffer
            self._staging_buffer = None
        finally:
            self._m4_lock.release()

        _log(logger, "debug", "consolidation.m4_commit",
             facts_written=n_new, total_facts=len(self._committed_facts))
        return n_new

    # ── Queries ──────────────────────────────────────────────

    def get_relevant_facts(
        self,
        state: StateVector,
        n: int = 5,
        min_confidence: float = 0.3,
    ) -> List[SemanticFact]:
        """Return top-N facts most relevant to the given state.

        Uses cosine similarity between fact state_patterns and the
        given state to find the most relevant statistical facts.
        Facts are sorted by relevance (cosine sim * confidence).

        Args:
            state: Current state to find relevant facts for.
            n: Maximum number of facts to return.
            min_confidence: Minimum confidence threshold.

        Returns:
            List of SemanticFact objects ordered by relevance.
        """
        candidates = [
            f for f in self._committed_facts
            if f.confidence >= min_confidence and f.state_pattern is not None
        ]
        if not candidates:
            return []

        # Score each fact by cosine similarity to current state * confidence
        state_vec = state.values.astype(np.float64)
        scored = []
        for fact in candidates:
            fact_vec = fact.state_pattern.values[:len(state_vec)].astype(np.float64)
            sim = self._cosine_similarity(state_vec, fact_vec)
            scored.append((sim * fact.confidence, fact))

        # Sort by relevance score descending
        scored.sort(key=lambda x: x[0], reverse=True)
        return [f for _, f in scored[:n]]

    def get_semantic_facts(
        self,
        fact_type: Optional[str] = None,
        min_confidence: float = 0.0,
        max_results: int = 100,
    ) -> List[SemanticFact]:
        """Retrieve stored statistical facts with optional filters.

        Note: "Statistical" not "Semantic" — facts are extracted via
        frequency-based pattern matching, not genuine understanding.
        The method name is preserved for backward compatibility.

        Reads from committed store without lock (readers not blocked by
        concurrent writes — v3.0 §2.3.2 M4 write-lock semantics).

        Args:
            fact_type: Filter by fact type (None = all).
            min_confidence: Minimum confidence threshold.
            max_results: Maximum facts to return.

        Returns:
            Filtered list of SemanticFact objects.
        """
        results = list(self._committed_facts)
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
        with self._m4_lock:
            self._committed_facts.clear()
            self._staging_buffer = None
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
