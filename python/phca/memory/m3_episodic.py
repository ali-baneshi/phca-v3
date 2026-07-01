"""
PHCA v3.0 — M3 Episodic Memory (SQLite-backed).

Phase 3.2: SQLite-backed episodic memory store with MVCC concurrency.
Stores (state_before, action, state_after, prediction_error, timestamp)
per episode. Supports snapshot-based reads for consolidation.

v3.0 Reference: v3.0 Patch §2.3 (Memory concurrency), §3.1 Table
"""

from __future__ import annotations

import sqlite3
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from phca.config import StateVector
from phca.logging import logger, _log

# Default database path (in-memory for testing, file for persistence)
_DEFAULT_DB_PATH = ":memory:"


@dataclass
class EpisodeRecord:
    """A single episode record stored in M3.

    Attributes:
        episode_id: Auto-increment primary key.
        version: MVCC version for snapshot isolation.
        state_before: State before action was taken (serialized).
        action_taken: Action vector (serialized).
        state_after: Resulting state (serialized).
        prediction_error: L2² error from PEU.
        confidence: Prediction confidence at time of storage.
        timestamp: Monotonic cycle count.
        consolidated: 0 = pending, 1 = consolidated to M4.
        drive_id: Which MDIM drive generated the goal (None if unknown).
    """
    episode_id: int = 0
    version: int = 1
    state_before: StateVector | None = None
    action_taken: np.ndarray | None = None
    state_after: StateVector | None = None
    prediction_error: float = 0.0
    confidence: float = 0.0
    timestamp: int = 0
    consolidated: int = 0
    drive_id: int | None = None


@dataclass
class ConsolidationSnapshot:
    """MVCC snapshot for safe consolidation reads.

    Attributes:
        snapshot_version: Frozen version number.
        snapshot_id: Unique snapshot identifier.
        episodes: Episodes at snapshot time (list of records).
    """
    snapshot_version: int = 0
    snapshot_id: str = ""
    episodes: List[EpisodeRecord] = field(default_factory=list)


# SQLite schema for M3
M3_SCHEMA_SQL = """
-- Episodic memory store (M3)
CREATE TABLE IF NOT EXISTS episodes (
    episode_id INTEGER PRIMARY KEY AUTOINCREMENT,
    version INTEGER NOT NULL DEFAULT 1,
    state_before BLOB NOT NULL,
    action_taken BLOB NOT NULL,
    state_after BLOB NOT NULL,
    prediction_error REAL NOT NULL,
    confidence REAL NOT NULL DEFAULT 0.0,
    timestamp INTEGER NOT NULL,
    consolidated INTEGER DEFAULT 0,
    drive_id INTEGER DEFAULT NULL
);

-- Index for consolidation scan
CREATE INDEX IF NOT EXISTS idx_episodes_consolidated
    ON episodes(consolidated, timestamp);

-- Consolidation log (for idempotent processing)
CREATE TABLE IF NOT EXISTS consolidation_log (
    log_id INTEGER PRIMARY KEY AUTOINCREMENT,
    snapshot_version INTEGER NOT NULL,
    episodes_processed INTEGER NOT NULL,
    facts_generated INTEGER NOT NULL,
    started_at INTEGER NOT NULL,
    completed_at INTEGER,
    status TEXT DEFAULT 'in_progress'
);

-- (schema_version table removed in Phase 3.3 gap audit G-014 —
--  migration framework was never wired into _init_db())
"""


class M3EpisodicMemory:
    """M3 Episodic Memory — SQLite-backed episodic store with MVCC.

    Phase 3.2: Full episodic memory with:
      - SQLite storage (file or :memory:)
      - MVCC snapshot isolation for consolidation
      - Indexed queries by consolidated status, timestamp
      - Batch reads for experience replay

    Attributes:
        db_path: Path to SQLite database file (":memory:" for testing).
        _max_episodes: Maximum episodes before FIFO eviction.
    """

    def __init__(
        self,
        db_path: str = _DEFAULT_DB_PATH,
        max_episodes: int = 10_000,
        state_dim: int = 84,
        action_dim: int = 5,
    ):
        """Initialize M3 Episodic Memory.

        Args:
            db_path: SQLite database path. ":memory:" for in-memory store.
            max_episodes: Maximum episodes before FIFO eviction.
            state_dim: Dimensionality of stored state vectors.
            action_dim: Dimensionality of stored action vectors.
        """
        self.db_path = db_path
        self._max_episodes = max_episodes
        self._state_dim = state_dim
        self._action_dim = action_dim
        self._lock = threading.Lock()
        self._current_version: int = 1
        self._pending_commits: int = 0
        self._commit_interval: int = 10
        self._episodes_since_vacuum: int = 0
        self._vacuum_interval: int = 1000  # VACUUM every 1000 episodes (G-010)

        # Initialize database
        self._conn: Optional[sqlite3.Connection] = None
        self._init_db()

        _log(logger, "info", "m3.init", db_path=db_path, max_episodes=max_episodes)

    def _init_db(self) -> None:
        """Initialize the SQLite database and create schema.

        Runs integrity check for file-backed databases (G-010). On
        integrity failure OR a corrupt/unreadable file, falls back to
        an in-memory database so the cognitive cycle can continue
        (data loss is logged critical).
        """
        try:
            self._conn = sqlite3.connect(self.db_path)
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.executescript(M3_SCHEMA_SQL)
            self._conn.commit()
        except sqlite3.DatabaseError as e:
            # Corrupt or non-SQLite file: fall back to in-memory.
            if self.db_path == ":memory:":
                raise  # cannot fall back further
            _log(logger, "critical", "m3.init_failure",
                 error=str(e), db_path=self.db_path, fallback=":memory:")
            try:
                if self._conn is not None:
                    self._conn.close()
            except Exception:
                pass
            self._conn = sqlite3.connect(":memory:")
            self._conn.executescript(M3_SCHEMA_SQL)
            self._conn.commit()
            self.db_path = ":memory:"
            return

        # Run integrity check on persistent databases (G-010). On failure,
        # fall back to an in-memory store so the cycle can keep running.
        if self.db_path != ":memory:":
            try:
                cursor = self._conn.execute("PRAGMA integrity_check")
                result = cursor.fetchone()
                if result and result[0] != "ok":
                    _log(logger, "critical", "m3.integrity_failure",
                         result=str(result), fallback=":memory:")
                    try:
                        self._conn.close()
                    except Exception:
                        pass
                    self._conn = sqlite3.connect(":memory:")
                    self._conn.executescript(M3_SCHEMA_SQL)
                    self._conn.commit()
                    self.db_path = ":memory:"  # record the fallback
                else:
                    _log(logger, "info", "m3.integrity_check", result="ok")
            except Exception as e:
                _log(logger, "error", "m3.integrity_check_failed", error=str(e))

    @property
    def _connection(self) -> sqlite3.Connection:
        """Get or create the database connection (lazy init for pickle safety)."""
        if self._conn is None:
            self._init_db()
        assert self._conn is not None
        return self._conn

    # ── Write ─────────────────────────────────────────────────

    def store_episode(
        self,
        state_before: StateVector,
        action_taken: np.ndarray,
        state_after: StateVector,
        prediction_error: float,
        confidence: float = 0.0,
        drive_id: int | None = None,
        timestamp: int | None = None,
    ) -> int:
        """Store a single episode in M3.

        Args:
            state_before: State before action.
            action_taken: Action vector executed.
            state_after: Resulting state after action.
            prediction_error: L2² prediction error.
            confidence: Prediction confidence.
            drive_id: MDIM drive ID that generated the goal (optional).
            timestamp: Monotonic cycle count. If None, uses state_after.timestamp.

        Returns:
            Integer episode_id of the stored record.
        """
        if timestamp is None:
            timestamp = int(state_after.timestamp) if state_after.timestamp else 0

        with self._lock:
            cursor = self._connection.execute(
                """INSERT INTO episodes
                   (version, state_before, action_taken, state_after,
                    prediction_error, confidence, timestamp, drive_id)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    self._current_version,
                    state_before.to_bytes(),
                    action_taken.tobytes(),
                    state_after.to_bytes(),
                    float(prediction_error),
                    float(confidence),
                    timestamp,
                    drive_id,
                ),
            )
            episode_id = cursor.lastrowid
            self._pending_commits += 1
            if self._pending_commits >= self._commit_interval:
                self._connection.commit()
                self._pending_commits = 0

        # Evict oldest if over capacity
        self._evict_if_needed()
        # Periodic VACUUM to reclaim space from deleted episodes (G-010)
        self._vacuum_if_needed()

        _log(logger, "debug", "m3.store", episode_id=episode_id, timestamp=timestamp)
        return episode_id or 0

    def flush(self) -> None:
        """Force a commit of any pending writes.

        Called by the consolidation scheduler to ensure episodes are
        durable before marking them as consolidated.
        """
        if self._pending_commits > 0 and self._conn is not None:
            self._conn.commit()
            self._pending_commits = 0

    # ── Read ──────────────────────────────────────────────────

    def count(self, consolidated: Optional[int] = None) -> int:
        """Count episodes, optionally filtered by consolidated status.

        Args:
            consolidated: If set, count only episodes with this status.

        Returns:
            Integer count of matching episodes.
        """
        if consolidated is not None:
            cursor = self._connection.execute(
                "SELECT COUNT(*) FROM episodes WHERE consolidated = ?",
                (consolidated,),
            )
        else:
            cursor = self._connection.execute("SELECT COUNT(*) FROM episodes")
        return cursor.fetchone()[0] or 0

    # ── Observability v4: cheap indexed reads for the Memory tab ──

    def recent_episodes(self, n: int = 5) -> List[EpisodeRecord]:
        """Return the ``n`` most recent episodes (newest first).

        Indexed by ``idx_episodes_consolidated(timestamp)`` — a bounded tail
        scan, cheap even at the 10k FIFO cap. Used by the Memory & Belief tab.
        """
        try:
            n = max(0, int(n))
            if n == 0:
                return []
            cursor = self._connection.execute(
                "SELECT * FROM episodes ORDER BY timestamp DESC, episode_id DESC LIMIT ?",
                (n,),
            )
            return [r for r in (self._row_to_episode(row) for row in cursor.fetchall())
                    if r is not None]
        except Exception as e:
            _log(logger, "warning", "m3.recent_episodes_failed", error=str(e))
            return []

    def top_error_episodes(self, n: int = 5) -> List[EpisodeRecord]:
        """Return the ``n`` highest-prediction-error episodes (worst first).

        Used by the Memory & Belief tab to surface the agent's most surprising
        recent transitions. Bounded ``LIMIT`` scan over the FIFO table.
        """
        try:
            n = max(0, int(n))
            if n == 0:
                return []
            cursor = self._connection.execute(
                "SELECT * FROM episodes ORDER BY prediction_error DESC, "
                "timestamp DESC LIMIT ?",
                (n,),
            )
            return [r for r in (self._row_to_episode(row) for row in cursor.fetchall())
                    if r is not None]
        except Exception as e:
            _log(logger, "warning", "m3.top_error_episodes_failed", error=str(e))
            return []

    # ── MVCC Snapshots ────────────────────────────────────────

    def create_snapshot(self) -> ConsolidationSnapshot:
        """Create an MVCC snapshot for safe consolidation reads.

        The snapshot freezes the current version and reads all
        unconsolidated episodes at that version.

        Returns:
            ConsolidationSnapshot with frozen episode records.
        """
        with self._lock:
            snapshot_version = self._current_version
            snapshot_id = f"snap_v{snapshot_version}_t{int(np.floor(snapshot_version * 1000))}"

            # Read all unconsolidated episodes at this version
            cursor = self._connection.execute(
                "SELECT * FROM episodes WHERE consolidated = 0 AND version <= ? "
                "ORDER BY timestamp ASC",
                (snapshot_version,),
            )
            episodes = [self._row_to_episode(row) for row in cursor.fetchall()]

        _log(logger, "info", "m3.snapshot",
             snapshot_id=snapshot_id, version=snapshot_version,
             n_episodes=len(episodes))

        return ConsolidationSnapshot(
            snapshot_version=snapshot_version,
            snapshot_id=snapshot_id,
            episodes=episodes,
        )

    def mark_consolidated(self, episode_ids: List[int]) -> int:
        """Mark episodes as consolidated (moved to M4).

        Args:
            episode_ids: List of episode IDs to mark.

        Returns:
            Number of episodes successfully marked.
        """
        if not episode_ids:
            return 0

        with self._lock:
            placeholders = ",".join("?" for _ in episode_ids)
            cursor = self._connection.execute(
                f"UPDATE episodes SET consolidated = 1 "
                f"WHERE episode_id IN ({placeholders})",
                episode_ids,
            )
            self._connection.commit()
            return cursor.rowcount

    def increment_version(self) -> int:
        """Increment the MVCC version counter for new writes.

        Returns:
            The new current version.
        """
        with self._lock:
            self._current_version += 1
        return self._current_version

    # ── Maintenance ───────────────────────────────────────────

    def _evict_if_needed(self) -> None:
        """Evict oldest episodes if over max_episodes."""
        total = self.count()
        if total > self._max_episodes:
            excess = total - self._max_episodes
            with self._lock:
                # Find the oldest non-consolidated episodes and delete them
                self._connection.execute(
                    "DELETE FROM episodes WHERE episode_id IN ("
                    "SELECT episode_id FROM episodes ORDER BY timestamp ASC LIMIT ?"
                    ")", (excess,),
                )
                self._connection.commit()
                _log(logger, "debug", "m3.evict", count=excess)
                # Track deletions for VACUUM scheduling (G-010)
                self._episodes_since_vacuum += excess

    def _vacuum_if_needed(self) -> None:
        """Run VACUUM periodically to reclaim space from deleted episodes.

        SQLite does not automatically reclaim space from DELETEd rows.
        After enough deletions (evictions or consolidations), the database
        grows unbounded. VACUUM rebuilds the database, reclaiming space.

        Phase 7 / B1: VACUUM now also runs for in-memory (`:memory:`)
        databases. The in-memory backing store uses a pager cache whose
        deleted pages go to a free-list but are not returned to the OS, so
        without VACUUM a long-running `:memory:` M3 (the nightly-stress
        configuration) fragments and RSS keeps growing after the 10k FIFO
        cap engages. VACUUM rebuilds the in-memory page cache too (G-010 +
        D-108 Phase 7 retention fix).
        """
        if self._episodes_since_vacuum < self._vacuum_interval:
            return
        if self._conn is None:
            return
        try:
            self._conn.execute("VACUUM")
            self._episodes_since_vacuum = 0
            _log(logger, "info", "m3.vacuum", interval=self._vacuum_interval,
                 db_path=self.db_path)
        except Exception as e:
            _log(logger, "error", "m3.vacuum_failed", error=str(e))

    def close(self) -> None:
        """Close the database connection.

        For persistent (file-backed) databases, runs
        `PRAGMA wal_checkpoint(TRUNCATE)` first so the WAL file is
        merged into the main database and truncated, preventing
        unbounded WAL growth across runs (G-010 / D-082).
        """
        if self._conn is not None:
            if self.db_path != ":memory:":
                try:
                    self._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                    _log(logger, "info", "m3.wal_checkpoint", db_path=self.db_path)
                except Exception as e:
                    _log(logger, "error", "m3.wal_checkpoint_failed", error=str(e))
            self._conn.close()
            self._conn = None

    def __del__(self) -> None:
        """Cleanup on garbage collection."""
        self.close()

    # ── Helpers ───────────────────────────────────────────────

    @staticmethod
    def _row_to_episode(row: Optional[Tuple]) -> Optional[EpisodeRecord]:
        """Convert a SQLite row tuple to an EpisodeRecord.

        Args:
            row: SQLite result row (tuple of column values).

        Returns:
            EpisodeRecord if row is valid, None otherwise.
        """
        if row is None:
            return None

        # Column order: episode_id, version, state_before, action_taken,
        #               state_after, prediction_error, confidence, timestamp,
        #               consolidated, drive_id
        return EpisodeRecord(
            episode_id=row[0],
            version=row[1],
            state_before=StateVector.from_bytes(row[2]),
            action_taken=np.frombuffer(row[3], dtype=np.float32).copy(),
            state_after=StateVector.from_bytes(row[4]),
            prediction_error=float(row[5]),
            confidence=float(row[6]),
            timestamp=int(row[7]),
            consolidated=int(row[8]),
            drive_id=int(row[9]) if row[9] is not None else None,
        )
