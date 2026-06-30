"""
PHCA v3.0 — M3 Schema Migration v1.

Initial M3 schema: episodes table with MVCC version tracking,
consolidation log, and schema version tracking.

Applied automatically by M3EpisodicMemory.__init__().
"""

from __future__ import annotations

import sqlite3
import time
from typing import Any

M3_SCHEMA_V1 = """
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

CREATE INDEX IF NOT EXISTS idx_episodes_consolidated
    ON episodes(consolidated, timestamp);

CREATE TABLE IF NOT EXISTS consolidation_log (
    log_id INTEGER PRIMARY KEY AUTOINCREMENT,
    snapshot_version INTEGER NOT NULL,
    episodes_processed INTEGER NOT NULL,
    facts_generated INTEGER NOT NULL,
    started_at INTEGER NOT NULL,
    completed_at INTEGER,
    status TEXT DEFAULT 'in_progress'
);

CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY,
    applied_at INTEGER NOT NULL
);
"""


def apply_v1(connection: sqlite3.Connection) -> None:
    """Apply schema v1 migration.

    Args:
        connection: SQLite connection.
    """
    connection.executescript(M3_SCHEMA_V1)
    connection.execute(
        "INSERT OR REPLACE INTO schema_version (version, applied_at) VALUES (1, ?)",
        (int(time.time()),),
    )
    connection.commit()
