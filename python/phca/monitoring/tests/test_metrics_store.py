"""Tests for thread-safe MetricsStore ring buffer."""
from __future__ import annotations

import threading

from phca.core.cycle import CycleMetrics
from phca.monitoring.metrics_store import MetricsStore


def test_push_latest_snapshot():
    store = MetricsStore(maxlen=3)
    m0 = CycleMetrics(cycle_id=0, prediction_error=1.0)
    m1 = CycleMetrics(cycle_id=1, prediction_error=2.0)
    store.push(m0)
    store.push(m1)
    assert store.latest() is m1
    snap = store.snapshot()
    assert len(snap) == 2
    assert snap[0].cycle_id == 0
    snap.append(CycleMetrics(cycle_id=99))
    assert len(store.snapshot()) == 2


def test_ring_buffer_evicts_oldest():
    store = MetricsStore(maxlen=2)
    for i in range(4):
        store.push(CycleMetrics(cycle_id=i))
    assert len(store) == 2
    snap = store.snapshot()
    assert [m.cycle_id for m in snap] == [2, 3]


def test_concurrent_push_and_snapshot():
    store = MetricsStore(maxlen=200)
    errors: list[str] = []

    def writer():
        try:
            for i in range(100):
                store.push(CycleMetrics(cycle_id=i, latency_ms=float(i)))
        except Exception as exc:
            errors.append(str(exc))

    def reader():
        try:
            for _ in range(100):
                snap = store.snapshot()
                if snap:
                    _ = snap[-1].cycle_id
        except Exception as exc:
            errors.append(str(exc))

    threads = [
        threading.Thread(target=writer),
        threading.Thread(target=reader),
        threading.Thread(target=reader),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5.0)
        assert not t.is_alive()
    assert not errors
    assert len(store) <= 200
