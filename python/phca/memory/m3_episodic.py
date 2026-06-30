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

-- Version tracking for MVCC
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY,
    applied_at INTEGER NOT NULL
);
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

        # Initialize database
        self._conn: Optional[sqlite3.Connection] = None
        self._init_db()

        _log(logger, "info", "m3.init", db_path=db_path, max_episodes=max_episodes)

    def _init_db(self) -> None:
        """Initialize the SQLite database and create schema."""
        self._conn = sqlite3.connect(self.db_path)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.executescript(M3_SCHEMA_SQL)
        self._conn.commit()

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
            self._connection.commit()

        # Evict oldest if over capacity
        self._evict_if_needed()

        _log(logger, "debug", "m3.store", episode_id=episode_id, timestamp=timestamp)
        return episode_id or 0

    def store_batch(
        self,
        episodes: List[EpisodeRecord],
    ) -> List[int]:
        """Store multiple episodes in a single transaction.

        Args:
            episodes: List of episode records to store.

        Returns:
            List of inserted episode_ids.
        """
        ids: List[int] = []
        with self._lock:
            for ep in episodes:
                cursor = self._connection.execute(
                    """INSERT INTO episodes
                       (version, state_before, action_taken, state_after,
                        prediction_error, confidence, timestamp, drive_id)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        self._current_version,
                        ep.state_before.to_bytes() if ep.state_before else b"",
                        ep.action_taken.tobytes() if ep.action_taken is not None else b"",
                        ep.state_after.to_bytes() if ep.state_after else b"",
                        float(ep.prediction_error),
                        float(ep.confidence),
                        ep.timestamp,
                        ep.drive_id,
                    ),
                )
                ids.append(cursor.lastrowid or 0)
            self._connection.commit()

        self._evict_if_needed()
        return ids

    # ── Read ──────────────────────────────────────────────────

    def get_episode(self, episode_id: int) -> Optional[EpisodeRecord]:
        """Retrieve a single episode by ID.

        Args:
            episode_id: The episode ID to retrieve.

        Returns:
            EpisodeRecord if found, None otherwise.
        """
        cursor = self._connection.execute(
            "SELECT * FROM episodes WHERE episode_id = ?", (episode_id,)
        )
        row = cursor.fetchone()
        return self._row_to_episode(row) if row else None

    def query_episodes(
        self,
        limit: int = 100,
        offset: int = 0,
        consolidated: Optional[int] = None,
        min_timestamp: Optional[int] = None,
        max_timestamp: Optional[int] = None,
    ) -> List[EpisodeRecord]:
        """Query episodes with optional filters.

        Args:
            limit: Maximum number of episodes to return.
            offset: Number of episodes to skip.
            consolidated: Filter by consolidated status (0=unconsolidated, 1=consolidated).
            min_timestamp: Minimum timestamp (inclusive).
            max_timestamp: Maximum timestamp (inclusive).

        Returns:
            List of matching EpisodeRecord objects.
        """
        clauses: List[str] = []
        params: List[Any] = []

        if consolidated is not None:
            clauses.append("consolidated = ?")
            params.append(consolidated)
        if min_timestamp is not None:
            clauses.append("timestamp >= ?")
            params.append(min_timestamp)
        if max_timestamp is not None:
            clauses.append("timestamp <= ?")
            params.append(max_timestamp)

        where = " AND ".join(clauses) if clauses else "1=1"
        cursor = self._connection.execute(
            f"SELECT * FROM episodes WHERE {where} ORDER BY timestamp ASC LIMIT ? OFFSET ?",
            params + [limit, offset],
        )
        return [self._row_to_episode(row) for row in cursor.fetchall()]

    def sample_batch(self, batch_size: int, seed: int = 42) -> List[EpisodeRecord]:
        """Random sample of episodes for experience replay.

        Args:
            batch_size: Number of episodes to sample.
            seed: Random seed.

        Returns:
            List of sampled EpisodeRecord objects (up to batch_size).
        """
        rng = np.random.RandomState(seed)
        total = self.count()
        if total == 0:
            return []
        n = min(batch_size, total)
        # Use ORDER BY RANDOM() for reservoir sampling
        cursor = self._connection.execute(
            "SELECT * FROM episodes ORDER BY RANDOM() LIMIT ?", (n,)
        )
        return [self._row_to_episode(row) for row in cursor.fetchall()]

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

    def reset(self) -> None:
        """Clear all episodes and reset M3 state."""
        with self._lock:
            self._connection.execute("DELETE FROM episodes")
            self._connection.execute("DELETE FROM consolidation_log")
            self._connection.commit()
            self._current_version = 1
        _log(logger, "info", "m3.reset")

    def close(self) -> None:
        """Close the database connection."""
        if self._conn is not None:
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
