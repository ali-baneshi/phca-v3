"""
PHCA v3.0 — Metrics Store (Thread-Safe Ring Buffer for Live Monitoring).

Provides a lock-guarded, bounded ring buffer of CycleMetrics objects
that can be written from the cognitive cycle thread and read from a
separate dashboard thread without blocking the cycle.

Zero external dependencies — uses only Python stdlib (threading, deque).

Usage:
    store = MetricsStore(maxlen=1000)
    cycle = CognitiveCycle.build_for_env(metrics_store=store, ...)
    # In another thread:
    latest = store.latest()
    history = store.snapshot()
"""

from __future__ import annotations

import threading
from collections import deque
from typing import List, Optional

from phca.core.cycle import CycleMetrics


class MetricsStore:
    """Thread-safe ring buffer of CycleMetrics for live monitoring.

    One thread (cognitive cycle) calls push() every cycle.
    Another thread (dashboard) calls snapshot() or latest() every ~500ms.
    Both operations are lock-guarded but fast (< 5 μs each).

    Attributes:
        maxlen: Maximum number of metrics entries to retain.
    """

    def __init__(self, maxlen: int = 1000):
        """Initialize the metrics store.

        Args:
            maxlen: Maximum entries in the ring buffer.
        """
        self._deque: deque = deque(maxlen=maxlen)
        self._lock = threading.Lock()

    def push(self, metrics: CycleMetrics) -> None:
        """Push one metrics entry (called by CognitiveCycle.step()).

        This is a non-blocking O(1) operation that acquires a lock
        for < 1 μs. It cannot raise an exception.

        Args:
            metrics: CycleMetrics object from the current cognitive cycle.
        """
        with self._lock:
            self._deque.append(metrics)

    def latest(self) -> Optional[CycleMetrics]:
        """Return the most recent metrics entry, or None if empty.

        Returns:
            The most recent CycleMetrics, or None if no cycles have run.
        """
        with self._lock:
            if self._deque:
                return self._deque[-1]
            return None

    def snapshot(self) -> List[CycleMetrics]:
        """Return a copy of all entries (for rendering statistics).

        Returns:
            A new list containing all entries in the buffer.
            Modifying the returned list does not affect the store.
        """
        with self._lock:
            return list(self._deque)

    def __len__(self) -> int:
        """Return the number of entries currently stored."""
        with self._lock:
            return len(self._deque)
