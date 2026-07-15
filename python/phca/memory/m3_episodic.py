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
from typing import List, Optional, Tuple

import numpy as np

from phca.config import PER_ALPHA, PER_BETA_INIT, PER_EPSILON, StateVector
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
        task_id: Continual-learning task tag (optional).
        priority: PER priority (higher = more likely to be replayed).
        stored_error: Prediction error at last storage/replay (for error-reduction rate).
        is_weight: Importance-sampling weight (set by PER sampling, 1.0 when unused).
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
    task_id: int | None = None
    priority: float = 1.0
    stored_error: float = 1.0
    is_weight: float = 1.0


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
    drive_id INTEGER DEFAULT NULL,
    task_id INTEGER DEFAULT NULL
);

-- Index for consolidation scan
CREATE INDEX IF NOT EXISTS idx_episodes_consolidated
    ON episodes(consolidated, timestamp);

-- Index for task-stratified sampling (P1-2)
CREATE INDEX IF NOT EXISTS idx_episodes_task_id
    ON episodes(task_id);

-- (consolidation_log table removed in P2-4 — ConsolidationScheduler
--  tracks its own reports; this table was never written to)
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
        seed: int = 0,
    ):
        """Initialize M3 Episodic Memory.

        Args:
            db_path: SQLite database path. ":memory:" for in-memory store.
            max_episodes: Maximum episodes before FIFO eviction.
            state_dim: Dimensionality of stored state vectors.
            action_dim: Dimensionality of stored action vectors.
            seed: Random seed for PER sampling.
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
        self._vacuum_interval: int = 1000  # VACUUM every 1000 evictions/purges (D-056)
        self.rng = np.random.RandomState(seed=seed)

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
            self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.execute("PRAGMA busy_timeout=5000")
            self._conn.executescript(M3_SCHEMA_SQL)
            self._conn.commit()
            self._migrate_schema()
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
                _log(logger, "debug", "m3.close_failed", db_path=self.db_path)
            self._conn = sqlite3.connect(":memory:", check_same_thread=False)
            self._conn.execute("PRAGMA busy_timeout=5000")
            self._conn.executescript(M3_SCHEMA_SQL)
            self._conn.commit()
            self._migrate_schema()
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
                    self._conn = sqlite3.connect(":memory:", check_same_thread=False)
                    self._conn.execute("PRAGMA busy_timeout=5000")
                    self._conn.executescript(M3_SCHEMA_SQL)
                    self._conn.commit()
                    self.db_path = ":memory:"  # record the fallback
                else:
                    _log(logger, "info", "m3.integrity_check", result="ok")
            except Exception as e:
                _log(logger, "error", "m3.integrity_check_failed", error=str(e))

    def _migrate_schema(self) -> None:
        """Add optional columns for older databases."""
        if self._conn is None:
            return
        try:
            cols = {
                row[1]
                for row in self._conn.execute("PRAGMA table_info(episodes)").fetchall()
            }
            if "task_id" not in cols:
                self._conn.execute(
                    "ALTER TABLE episodes ADD COLUMN task_id INTEGER DEFAULT NULL"
                )
            if "priority" not in cols:
                self._conn.execute(
                    "ALTER TABLE episodes ADD COLUMN priority REAL DEFAULT 1.0"
                )
            if "stored_error" not in cols:
                self._conn.execute(
                    "ALTER TABLE episodes ADD COLUMN stored_error REAL DEFAULT 1.0"
                )
            self._conn.commit()

            # Ensure task_id index exists (P1-2)
            indexes = {
                row[0]
                for row in self._conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='episodes'"
                ).fetchall()
            }
            if "idx_episodes_task_id" not in indexes:
                self._conn.execute(
                    "CREATE INDEX idx_episodes_task_id ON episodes(task_id)"
                )
                self._conn.commit()
        except sqlite3.OperationalError:
            pass

    @property
    def _connection(self) -> sqlite3.Connection:
        """Get the database connection (eagerly initialized in ``__init__``).

        For async use, the connection is created with ``check_same_thread=False``
        and the init race is prevented by double-checked locking over ``_lock``.
        """
        if self._conn is None:
            with self._lock:
                if self._conn is None:
                    self._init_db()
        if self._conn is None:
            raise RuntimeError("M3 connection not initialized")
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
        task_id: int | None = None,
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
                    prediction_error, confidence, timestamp, drive_id, task_id,
                    priority, stored_error)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    self._current_version,
                    state_before.to_bytes(),
                    action_taken.tobytes(),
                    state_after.to_bytes(),
                    float(prediction_error),
                    float(confidence),
                    timestamp,
                    drive_id,
                    task_id,
                    1.0,
                    float(prediction_error),
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

    def count_for_task(self, task_id: int) -> int:
        """Count episodes tagged with ``task_id``."""
        cursor = self._connection.execute(
            "SELECT COUNT(*) FROM episodes WHERE task_id = ?",
            (int(task_id),),
        )
        return int(cursor.fetchone()[0] or 0)

    def sample_episodes(
        self,
        n: int,
        task_id: int | None = None,
    ) -> List[EpisodeRecord]:
        """Sample up to ``n`` episodes, optionally filtered by ``task_id``."""
        n = max(0, int(n))
        if n == 0:
            return []
        try:
            if task_id is not None:
                cursor = self._connection.execute(
                    "SELECT * FROM episodes WHERE task_id = ? "
                    "ORDER BY RANDOM() LIMIT ?",
                    (task_id, n),
                )
            else:
                cursor = self._connection.execute(
                    "SELECT * FROM episodes ORDER BY RANDOM() LIMIT ?",
                    (n,),
                )
            return [
                r for r in (self._row_to_episode(row) for row in cursor.fetchall())
                if r is not None
            ]
        except sqlite3.OperationalError as e:
            _log(logger, "warning", "m3.sample_episodes_busy", error=str(e))
            return []
        except Exception as e:
            _log(logger, "warning", "m3.sample_episodes_failed", error=str(e))
            return []

    def sample_prior_task_episodes(
        self,
        n: int,
        before_task_id: int,
    ) -> List[EpisodeRecord]:
        """Sample up to ``n`` episodes, stratified across prior tasks.

        Each prior task (0 .. ``before_task_id - 1``) gets at least one
        episode when budget allows, preventing one task from dominating
        the replay batch.
        """
        n = max(0, int(n))
        if n == 0 or before_task_id <= 0:
            return []
        prior_ids = list(range(int(before_task_id)))
        result: List[EpisodeRecord] = []
        remain = n

        # Phase 1: allocate one per task, round-robin for surplus
        while remain > 0 and prior_ids:
            found_any = False
            for tid in prior_ids:
                if remain <= 0:
                    break
                try:
                    cursor = self._connection.execute(
                        "SELECT * FROM episodes WHERE task_id = ? "
                        "ORDER BY RANDOM() LIMIT 1",
                        (tid,),
                    )
                    row = cursor.fetchone()
                    if row is not None:
                        ep = self._row_to_episode(row)
                        if ep is not None:
                            result.append(ep)
                            remain -= 1
                            found_any = True
                except Exception:
                    _log(logger, "warning", "m3.sample_prior_task_failed",
                         task_id=tid, exc_info=True)
                    continue
            if not found_any:
                break

        # Phase 2: fill remaining budget, excluding Phase-1 episode_ids (P2-2)
        if remain > 0 and result:
            seen_ids = {ep.episode_id for ep in result}
            try:
                cursor = self._connection.execute(
                    "SELECT * FROM episodes WHERE task_id IS NOT NULL "
                    "AND task_id < ? ORDER BY RANDOM() LIMIT ?",
                    (int(before_task_id), remain + len(seen_ids)),
                )
                for row in cursor.fetchall():
                    ep = self._row_to_episode(row)
                    if ep is not None and ep.episode_id not in seen_ids:
                        result.append(ep)
                        seen_ids.add(ep.episode_id)
                        remain -= 1
                        if remain <= 0:
                            break
            except Exception:
                pass

        if result:
            return result

        # Fallback: original single-query approach
        try:
            cursor = self._connection.execute(
                "SELECT * FROM episodes WHERE task_id IS NOT NULL "
                "AND task_id < ? ORDER BY RANDOM() LIMIT ?",
                (int(before_task_id), n),
            )
            return [
                r for r in (self._row_to_episode(row) for row in cursor.fetchall())
                if r is not None
            ]
        except Exception as e:
            _log(logger, "warning", "m3.sample_prior_task_episodes_failed", error=str(e))
            return []

    # ── PER (Prioritized Experience Replay) ─────────────────────

    def sample_episodes_per(
        self,
        n: int,
        alpha: float = PER_ALPHA,
        beta: float = PER_BETA_INIT,
        task_id: int | None = None,
    ) -> List[EpisodeRecord]:
        """Sample episodes with priority-proportional probabilities (PER).

        Each returned EpisodeRecord has ``.is_weight`` set for importance-sampling
        correction.  Priority is based on error-reduction rate: episodes where the
        model recently improved get higher probability.

        Uses a two-phase query to avoid materialising all episode blobs:
        1. Fetch only (episode_id, priority) lightweight column scan.
        2. Sample n episode_ids, then fetch only those rows.

        Args:
            n: Number of episodes to sample.
            alpha: Prioritization exponent (0 = uniform, 1 = full priority).
            beta: Importance-sampling correction exponent.
            task_id: If set, only sample episodes from this task.

        Returns:
            List of EpisodeRecord with ``is_weight`` populated.
        """
        n = max(0, int(n))
        if n == 0:
            return []
        try:
            if task_id is not None:
                cursor = self._connection.execute(
                    "SELECT episode_id, priority FROM episodes WHERE task_id = ? "
                    "AND priority >= ? ORDER BY priority DESC",
                    (task_id, PER_EPSILON),
                )
            else:
                cursor = self._connection.execute(
                    "SELECT episode_id, priority FROM episodes WHERE priority >= ? "
                    "ORDER BY priority DESC",
                    (PER_EPSILON,),
                )
            id_prio = cursor.fetchall()  # lightweight: only two columns
        except sqlite3.OperationalError as e:
            _log(logger, "warning", "m3.sample_per_busy",
                 error=str(e), fallback="uniform_sample")
            return self.sample_episodes(n, task_id=task_id)
        except Exception as e:
            _log(logger, "warning", "m3.sample_per_failed", error=str(e))
            return self.sample_episodes(n, task_id=task_id)

        if not id_prio:
            return self.sample_episodes(n, task_id=task_id)

        ids = [r[0] for r in id_prio]
        priorities = np.array([max(PER_EPSILON, float(r[1])) for r in id_prio], dtype=np.float64)

        N = len(ids)
        if N <= n:
            # Return all eligible episodes
            placeholders = ",".join("?" * N)
            cursor = self._connection.execute(
                f"SELECT * FROM episodes WHERE episode_id IN ({placeholders})",
                ids,
            )
            episodes = [self._row_to_episode(r) for r in cursor.fetchall()]
            episodes = [ep for ep in episodes if ep is not None]
            for ep in episodes:
                ep.is_weight = 1.0
            return episodes

        probs = priorities ** alpha
        probs /= probs.sum()

        chosen_idx = self.rng.choice(N, size=n, replace=False, p=probs)
        chosen_ids = [ids[i] for i in chosen_idx]
        chosen_probs = probs[chosen_idx]

        # Phase 2: fetch only the chosen rows
        placeholders = ",".join("?" * n)
        cursor = self._connection.execute(
            f"SELECT * FROM episodes WHERE episode_id IN ({placeholders})",
            chosen_ids,
        )
        rows_map = {r[0]: r for r in cursor.fetchall()}
        chosen = [self._row_to_episode(rows_map[eid]) for eid in chosen_ids]
        chosen = [ep for ep in chosen if ep is not None]

        is_weights = (1.0 / (N * chosen_probs)) ** beta
        is_weights /= is_weights.max()  # normalize for stability
        for ep, w in zip(chosen, is_weights):
            ep.is_weight = float(w)

        return chosen

    def update_priority(
        self, episode_id: int, current_error: float,
        eps: float = PER_EPSILON,
    ) -> None:
        """Update PER priority for a single episode using error-reduction rate.

        ``priority = max(eps, (stored_error - current_error) / (stored_error + eps))``

        Args:
            episode_id: Target episode.
            current_error: Current model prediction error on this episode.
            eps: Floor to keep stale episodes sampleable.
        """
        try:
            cursor = self._connection.execute(
                "SELECT stored_error FROM episodes WHERE episode_id = ?",
                (episode_id,),
            )
            row = cursor.fetchone()
            if row is None:
                return
            stored = float(row[0])
            improvement = (stored - current_error) / (stored + eps)
            priority = max(eps, improvement)
            self._connection.execute(
                "UPDATE episodes SET priority = ?, stored_error = ? "
                "WHERE episode_id = ?",
                (priority, current_error, episode_id),
            )
            self._pending_commits += 1
            if self._pending_commits >= self._commit_interval:
                self._connection.commit()
                self._pending_commits = 0
        except sqlite3.OperationalError as e:
            _log(logger, "warning", "m3.update_priority_busy",
                 episode_id=episode_id, error=str(e))
        except Exception as e:
            _log(logger, "warning", "m3.update_priority_failed",
                 episode_id=episode_id, error=str(e))

    def batch_update_priorities(
        self, updates: List[Tuple[int, float]],
    ) -> None:
        """Batch-update PER priorities for multiple episodes.

        Args:
            updates: List of ``(episode_id, current_error)`` tuples.
        """
        for episode_id, current_error in updates:
            self.update_priority(episode_id, current_error)

    def last_inserted_id(self) -> int:
        """Return the last inserted episode_id (0 if none)."""
        try:
            cursor = self._connection.execute(
                "SELECT MAX(episode_id) FROM episodes"
            )
            row = cursor.fetchone()
            return int(row[0]) if row and row[0] else 0
        except sqlite3.OperationalError:
            return 0
        except Exception:
            return 0

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
        except sqlite3.OperationalError as e:
            _log(logger, "warning", "m3.recent_episodes_busy", error=str(e))
            return []
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
        """Evict oldest episodes if over max_episodes.

        Task-aware eviction: maintains a per-task quota so that early-task
        episodes are not starved by FIFO ordering (NEW-02 fix 2026-07-11).

        Prefers evicting consolidated episodes over unconsolidated to avoid
        losing experience before M4 extraction.
        """
        with self._lock:
            total = self.count()
            if total <= self._max_episodes:
                return
            # Get all task_ids that have episodes
            cursor = self._connection.execute(
                "SELECT DISTINCT task_id FROM episodes"
            )
            rows = cursor.fetchall()
            task_ids = [r[0] for r in rows if r[0] is not None]
            has_null = any(r[0] is None for r in rows)
            if not task_ids and not has_null:
                return
            groups = len(task_ids) + (1 if has_null else 0)
            max_per_group = self._max_episodes // groups
            deleted = 0

            def _evict_group(group_filter: str, group_params: tuple, target: int) -> int:
                """Evict from a single task (or NULL) group down to ``target``."""
                n = 0
                while True:
                    cnt = self._connection.execute(
                        f"SELECT COUNT(*) FROM episodes WHERE {group_filter}",
                        group_params,
                    ).fetchone()[0] or 0
                    if cnt <= target:
                        break
                    excess = cnt - target
                    # Phase 1: consolidated
                    cons = self._connection.execute(
                        f"SELECT episode_id FROM episodes WHERE {group_filter} AND consolidated = 1 "
                        "ORDER BY timestamp ASC LIMIT ?",
                        (*group_params, excess),
                    ).fetchall()
                    cons_ids = [r[0] for r in cons]
                    if cons_ids:
                        ph = ",".join("?" for _ in cons_ids)
                        self._connection.execute(
                            f"DELETE FROM episodes WHERE episode_id IN ({ph})", cons_ids,
                        )
                        n += len(cons_ids)
                        excess -= len(cons_ids)
                    if excess <= 0:
                        break
                    # Phase 2: unconsolidated
                    unc = self._connection.execute(
                        f"SELECT episode_id FROM episodes WHERE {group_filter} AND consolidated = 0 "
                        "ORDER BY timestamp ASC LIMIT ?",
                        (*group_params, excess),
                    ).fetchall()
                    unc_ids = [r[0] for r in unc]
                    if not unc_ids:
                        break
                    ph = ",".join("?" for _ in unc_ids)
                    self._connection.execute(
                        f"DELETE FROM episodes WHERE episode_id IN ({ph})", unc_ids,
                    )
                    n += len(unc_ids)
                return n

            for tid in task_ids:
                deleted += _evict_group("task_id = ?", (tid,), max_per_group)
            if has_null:
                deleted += _evict_group("task_id IS NULL", (), max_per_group)
            if deleted > 0:
                self._connection.commit()
                self._pending_commits = 0
                _log(logger, "debug", "m3.evict", count=deleted)
                self._episodes_since_vacuum += deleted
                self._vacuum_if_needed()

    def purge_consolidated(self, keep_recent: int = 500) -> int:
        """Delete consolidated episodes, retaining the most recent (P1-4 retention).

        Args:
            keep_recent: Minimum consolidated episodes to retain.

        Returns:
            Number of episodes deleted.
        """
        with self._lock:
            total_cons = self.count(consolidated=1)
            if total_cons <= keep_recent:
                return 0
            to_delete = total_cons - keep_recent
            cursor = self._connection.execute(
                "DELETE FROM episodes WHERE episode_id IN ("
                "SELECT episode_id FROM episodes WHERE consolidated = 1 "
                "ORDER BY timestamp ASC LIMIT ?"
                ")", (to_delete,),
            )
            self._connection.commit()
            deleted = cursor.rowcount
        if deleted > 0:
            self._episodes_since_vacuum += deleted
            self._vacuum_if_needed()
            _log(logger, "debug", "m3.purge_consolidated", deleted=deleted)
        return deleted

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
        #               consolidated, drive_id, task_id, priority, stored_error
        task_id = int(row[10]) if len(row) > 10 and row[10] is not None else None
        priority = float(row[11]) if len(row) > 11 and row[11] is not None else 1.0
        stored_error = float(row[12]) if len(row) > 12 and row[12] is not None else 1.0
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
            task_id=task_id,
            priority=priority,
            stored_error=stored_error,
        )
