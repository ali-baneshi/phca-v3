"""Tests for Retention & Resources tab."""
from __future__ import annotations

import pytest

from phca.monitoring.observability import ObservabilityFrame
from phca.monitoring.qt_dashboard import (
    RBTABoundsView,
    RetentionView,
    ViolationTable,
    _retention_score,
    make_app,
)


@pytest.fixture(scope="module")
def qt_app():
    app = make_app()
    yield app
    app.processEvents()


def _ret_frame(**kwargs) -> ObservabilityFrame:
    f = ObservabilityFrame()
    f.cycle_id = kwargs.get("cycle_id", 1)
    f.episode_count = kwargs.get("episode_count", 10)
    f.fact_count = kwargs.get("fact_count", 5)
    f.m3_cap = kwargs.get("m3_cap", 100)
    f.m4_cap = kwargs.get("m4_cap", 200)
    f.rss_bytes = kwargs.get("rss_bytes", 50e6)
    f.latency_ms = kwargs.get("latency_ms", 12.0)
    f.module_timings = kwargs.get("module_timings", {})
    f.memory_log = kwargs.get("memory_log", {})
    f.energy_log = kwargs.get("energy_log", {})
    f.rbta_bounds = kwargs.get("rbta_bounds", {})
    f.rbta_violations = kwargs.get("rbta_violations", [])
    return f


def test_retention_score_decay():
    assert _retention_score(0, 10) == pytest.approx(1.0)
    assert _retention_score(10, 10) < _retention_score(0, 10)


def test_retention_rebuild_histories(qt_app):
    view = RetentionView()
    frames = [_ret_frame(cycle_id=i, episode_count=i, fact_count=i) for i in range(5)]
    view.rebuild_histories(frames)
    assert len(view.m3) == 5
    assert len(view.rss) == 5
    view.set_frame(_ret_frame(cycle_id=99), histories_done=True)
    assert len(view.m3) == 5


def test_retention_envelope_over_cap(qt_app):
    view = RetentionView()
    f = _ret_frame(episode_count=120, m3_cap=100)
    view.set_frame(f)
    ok, segs = view._envelope_status()
    assert any(s[2] for s in segs if s[0] == "M3")
    assert ok is False


def test_violation_table_rebuild_dedupes(qt_app):
    tbl = ViolationTable()
    v = {"module_id": "ACTION", "bound_type": "time", "measured": 2.5, "allowed": 2.0}
    frames = [
        _ret_frame(cycle_id=1, rbta_violations=[v]),
        _ret_frame(cycle_id=1, rbta_violations=[v]),
        _ret_frame(cycle_id=2, rbta_violations=[v]),
    ]
    tbl.rebuild_from_frames(frames)
    assert tbl.rowCount() == 2
    assert tbl.violations_seen() == 2


def test_retention_prune_events_recorded(qt_app):
    view = RetentionView()
    view.set_frame(_ret_frame(cycle_id=1, episode_count=50, fact_count=20))
    view.set_frame(_ret_frame(cycle_id=2, episode_count=45, fact_count=20))
    view.set_frame(_ret_frame(cycle_id=3, episode_count=45, fact_count=18))
    assert view.m3_events == [1]
    assert view.m4_events == [2]


def test_measured_sparkline_uses_history(qt_app):
    from collections import deque

    from phca.monitoring.qt_dashboard import RBTABoundsView

    view = RBTABoundsView()
    key = "ACTION:time"
    view._hist[key] = deque([1.0, 2.0, 3.0, 2.5], maxlen=60)
    assert len(view._hist[key]) == 4
    # smoke: helper accepts deque without raising during paint prep
    hist = view._hist[key]
    assert float(hist[-1]) == pytest.approx(2.5)


def test_rbta_bounds_rebuild_histories_no_paint_append(qt_app):
    view = RBTABoundsView()
    frames = [
        _ret_frame(
            cycle_id=i,
            module_timings={"tspl": 0.01 * i},
            rbta_bounds={"TSPL-P": {"time": 0.02}},
        )
        for i in range(1, 4)
    ]
    view.rebuild_histories(frames)
    key = "TSPL-P:time"
    assert len(view._hist[key]) == 3
    view.set_frame(frames[-1], histories_done=True)
    assert len(view._hist[key]) == 3
    view.set_frame(_ret_frame(
        cycle_id=4,
        module_timings={"tspl": 0.04},
        rbta_bounds={"TSPL-P": {"time": 0.02}},
    ))
    assert len(view._hist[key]) == 4
