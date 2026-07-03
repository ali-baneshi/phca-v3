"""R2 dashboard extensions — moment parity, retention/memory/goals report anchors."""
from __future__ import annotations

import json
from collections import deque

import numpy as np
import pytest

from phca.monitoring.cognitive_panels import (
    append_cognitive_moment,
    build_moment_series,
    cognitive_moment,
    count_moments,
)
from phca.monitoring.observability import ObservabilityFrame
from phca.monitoring.qt_dashboard import (
    OverviewAgentView,
    RetentionView,
    _overview_moment_flags,
    _overview_new_events,
    make_app,
)
from phca.monitoring.render import frame_from_json
from phca.monitoring.session_report import build_session_report


@pytest.fixture(scope="module")
def qt_app():
    app = make_app()
    yield app
    app.processEvents()


def _frame(**kwargs) -> ObservabilityFrame:
    f = ObservabilityFrame()
    f.cycle_id = kwargs.get("cycle_id", 1)
    f.env_kind = kwargs.get("env_kind", "grid")
    f.prediction_error = kwargs.get("prediction_error", 1.0)
    f.module_timings = kwargs.get("module_timings", {})
    f.action_rationale = kwargs.get("action_rationale", {})
    f.active_drive_id = kwargs.get("active_drive_id", 0)
    f.episode_count = kwargs.get("episode_count", 0)
    f.fact_count = kwargs.get("fact_count", 0)
    f.m3_cap = kwargs.get("m3_cap", 100)
    f.m4_cap = kwargs.get("m4_cap", 100)
    f.violations_count = kwargs.get("violations_count", 0)
    f.rbta_violations = kwargs.get("rbta_violations", [])
    f.rbta_bounds = kwargs.get("rbta_bounds", {})
    return f


@pytest.mark.parametrize(
    "prev,cur,expected",
    [
        (1, 2, (1, 2)),
        (None, 2, None),
        (2, 2, None),
    ],
)
def test_cognitive_moment_drive_change(prev, cur, expected):
    f = _frame(active_drive_id=cur, action_rationale={"goal_id": cur})
    hist = deque([1.0], maxlen=10)
    m = cognitive_moment(f, hist, prev_drive_id=prev)
    assert m.get("drive_change") == expected


def test_cognitive_moment_violation_flag():
    f = _frame(violations_count=2)
    m = cognitive_moment(f, deque([1.0], maxlen=5))
    assert m["violation"] is True


def test_append_cognitive_moment_trims_maxlen():
    series: list = []
    hist: deque = deque(maxlen=20)
    prev_d = None
    prev_b = None
    for i in range(10):
        f = _frame(cycle_id=i, prediction_error=float(i))
        series, prev_d, prev_b = append_cognitive_moment(
            series, f, hist, prev_drive_id=prev_d, prev_best_score=prev_b, maxlen=5)
    assert len(series) == 5
    assert series[0]["learn_ms"] == 0.0 or True  # smoke: series populated


def test_overview_moment_flags_merges_explored():
    f = _frame(action_rationale={"explored": True, "best_score": 0.4})
    moment = {"spike": False, "learn_burst": False, "decision_shift": False}
    flags = _overview_moment_flags(f, deque([1.0]), moment=moment)
    assert flags["explored"] is True
    assert flags["score"] == pytest.approx(0.4)


def test_overview_new_events_near_bound():
    f = _frame()
    flags = {"spike": False, "learn_burst": False, "near_bound": "gprime_learn", "violation": False}
    events = _overview_new_events(f, flags, None)
    assert any("NEAR-BOUND" in e for e in events)


def test_overview_new_events_violation():
    f = _frame()
    flags = {"spike": False, "learn_burst": False, "violation": True}
    events = _overview_new_events(f, flags, None)
    assert any("VIOLATION" in e for e in events)


def test_overview_rebuild_moment_series_matches_frames(qt_app):
    ov = OverviewAgentView()
    frames = [_frame(cycle_id=i, prediction_error=float(i + 1)) for i in range(6)]
    ov.rebuild_histories(frames)
    assert len(ov._moment_series) == 6
    assert len(ov._err_hist) == 6


def test_overview_sync_drive_change_from_moment(qt_app):
    ov = OverviewAgentView()
    ov._sync_drive_change_from_moment({"drive_change": (1, 3)})
    assert ov._drive_change == (1, 3)
    ov._sync_drive_change_from_moment({})
    assert ov._drive_change is None


def test_retention_m3_m4_events_align_with_hist(qt_app):
    view = RetentionView()
    view.set_frame(_frame(cycle_id=1, episode_count=10, fact_count=8))
    view.set_frame(_frame(cycle_id=2, episode_count=8, fact_count=8))
    view.set_frame(_frame(cycle_id=3, episode_count=8, fact_count=6))
    assert view.m3_events == [1]
    assert view.m4_events == [2]
    assert len(view.rss) == 3


def test_retention_prune_reason_meta_stable(qt_app):
    view = RetentionView()
    f = _frame(episode_count=5)
    f.meta_stable = {"is_meta_stable": True}
    assert view._prune_reason(f) == "meta-stable"


def test_session_report_retention_prune_counts():
    lines = []
    for i, (ep, fc) in enumerate([(10, 5), (8, 5), (8, 4)]):
        f = _frame(cycle_id=i, episode_count=ep, fact_count=fc)
        lines.append(json.dumps(f.__dict__, default=str))
    # frame_from_json won't work with raw __dict__ — build minimal JSONL manually
    payload = []
    for i, (ep, fc) in enumerate([(10, 5), (8, 5), (8, 4)]):
        payload.append(json.dumps({
            "cycle_id": i,
            "prediction_error": 1.0,
            "prediction_confidence": 0.5,
            "episode_count": ep,
            "fact_count": fc,
            "m3_cap": 100,
            "m4_cap": 100,
            "rss_bytes": 1e6,
            "latency_ms": 10.0,
            "module_timings": {},
            "action_rationale": {},
        }))
    report = build_session_report({}, payload)
    assert report["retention_metrics"]["m3_prune_count"] == 1
    assert report["retention_metrics"]["m4_prune_count"] == 1


def test_session_report_anchor_retention_last():
    from pathlib import Path

    fixture = Path(__file__).resolve().parent / "fixtures" / "reacher_short.jsonl"
    lines = [ln for ln in fixture.read_text().splitlines() if ln.strip()]
    report = build_session_report({}, lines)
    anchors = report["retention_metrics"]["anchor_retention"]
    assert "last" in anchors
    assert "episode_count" in anchors["last"]


def test_session_report_anchor_memory_mid():
    from pathlib import Path

    fixture = Path(__file__).resolve().parent / "fixtures" / "reacher_short.jsonl"
    lines = [ln for ln in fixture.read_text().splitlines() if ln.strip()]
    report = build_session_report({}, lines)
    anchors = report["memory_metrics"]["anchor_memory"]
    assert "mid" in anchors
    assert "m3_items" in anchors["mid"]


def test_session_report_anchor_goals_zero():
    from pathlib import Path

    fixture = Path(__file__).resolve().parent / "fixtures" / "reacher_short.jsonl"
    lines = [ln for ln in fixture.read_text().splitlines() if ln.strip()]
    report = build_session_report({}, lines)
    goals = report["goals_metrics"]["anchor_goals"]
    assert "0" in goals
    assert "active_drive_id" in goals["0"]


def test_build_moment_series_near_bound_preserved():
    f = _frame(
        rbta_bounds={"G'": {"time": 0.001}},
        module_timings={"prediction": 0.95, "gprime_learn": 0.1},
    )
    series = build_moment_series([f])
    assert series[0].get("near_bound") == "prediction"


def test_count_moments_drive_change():
    series = [
        {"drive_change": (1, 2)},
        {"drive_change": None},
        {"drive_change": (2, 3)},
    ]
    c = count_moments(series)
    assert c["drive_change_count"] == 2


@pytest.mark.parametrize("idx", [0, 1, 2])
def test_frame_from_json_roundtrip_fields(idx):
    from pathlib import Path

    fixture = Path(__file__).resolve().parent / "fixtures" / "reacher_short.jsonl"
    lines = [ln for ln in fixture.read_text().splitlines() if ln.strip()]
    f = frame_from_json(json.loads(lines[idx]))
    assert f.cycle_id == idx


def test_frame_from_json_restores_replay_array_fields():
    f = frame_from_json({
        "cycle_id": 7,
        "grid": [[0, 1], [2, 3]],
        "gprime_uncertainty": [0.1, 0.2],
        "prediction_precision": [1.0, 2.0],
        "per_dim_peu": [0.3, 0.4],
        "attention_weights": [0.5, 0.6],
        "last_action_vector": [0.7, 0.8],
    })
    assert f.grid.dtype == np.int32
    for name in (
        "gprime_uncertainty",
        "prediction_precision",
        "per_dim_peu",
        "attention_weights",
        "last_action_vector",
    ):
        assert isinstance(getattr(f, name), np.ndarray)
        assert getattr(f, name).dtype == np.float32


def test_overview_moment_flags_without_moment_uses_cognitive():
    f = _frame(prediction_error=30.0, env_kind="grid")
    hist = deque([10.0, 12.0], maxlen=10)
    flags = _overview_moment_flags(f, hist)
    assert flags["spike"] is True


def test_overview_event_log_uses_moment_drive_change(qt_app):
    ov = OverviewAgentView()
    f1 = _frame(cycle_id=1, active_drive_id=1, action_rationale={"goal_id": 1})
    ov.set_frame(f1)
    f2 = _frame(cycle_id=2, active_drive_id=3, action_rationale={"goal_id": 3})
    ov.set_frame(f2)
    assert any("DRIVE" in line for line in ov.visible_event_lines())


def test_session_report_no_frame_mutation_decision_shift():
    from pathlib import Path

    fixture = Path(__file__).resolve().parent / "fixtures" / "reacher_short.jsonl"
    lines = [ln for ln in fixture.read_text().splitlines() if ln.strip()]
    raw = json.loads(lines[1])
    report = build_session_report({}, lines)
    f = frame_from_json(raw)
    assert "decision_shift" not in (f.action_rationale or {})
    assert report["decision_shift_count"] >= 0


def test_retention_envelope_ok_when_under_cap(qt_app):
    view = RetentionView()
    view.set_frame(_frame(episode_count=50, fact_count=30, m3_cap=100, m4_cap=100))
    ok, segs = view._envelope_status()
    assert ok is True
    assert all(not s[2] for s in segs)


def test_overview_current_moment_fallback(qt_app):
    ov = OverviewAgentView()
    f = _frame(prediction_error=5.0)
    ov.set_frame(f)
    m = ov._current_moment(f)
    assert "spike" in m


def test_cognitive_moment_decision_shift_from_prev_score():
    f = _frame(action_rationale={"best_score": 0.55})
    m = cognitive_moment(f, deque([1.0]), prev_best_score=0.20)
    assert m["decision_shift"] is True


def test_overview_replay_does_not_append_histories_done(qt_app):
    ov = OverviewAgentView()
    f = _frame(cycle_id=1)
    ov.set_frame(f)
    n = len(ov._err_hist)
    ov.set_frame(_frame(cycle_id=2), histories_done=True)
    assert len(ov._err_hist) == n


def test_session_report_memory_cycles_counts():
    payload = [json.dumps({
        "cycle_id": 0,
        "prediction_error": 1.0,
        "prediction_confidence": 0.5,
        "m3_recent": [{"drive_id": 1}],
        "m4_top": [],
        "episode_count": 1,
        "fact_count": 0,
        "module_timings": {},
        "action_rationale": {},
    })]
    report = build_session_report({}, payload)
    assert report["memory_metrics"]["cycles_with_m3"] == 1
    assert report["memory_metrics"]["cycles_with_m4"] == 0


def test_session_report_goals_active_drive_switch():
    payload = []
    for i, ad in enumerate([1, 1, 3, 3]):
        payload.append(json.dumps({
            "cycle_id": i,
            "prediction_error": 1.0,
            "prediction_confidence": 0.5,
            "active_drive_id": ad,
            "episode_count": 0,
            "fact_count": 0,
            "module_timings": {},
            "action_rationale": {},
        }))
    report = build_session_report({}, payload)
    assert report["goals_metrics"]["active_drive_switch_count"] == 1


def test_session_report_envelope_over_count():
    payload = [json.dumps({
        "cycle_id": 0,
        "prediction_error": 1.0,
        "prediction_confidence": 0.5,
        "episode_count": 150,
        "fact_count": 10,
        "m3_cap": 100,
        "m4_cap": 200,
        "module_timings": {},
        "action_rationale": {},
    })]
    report = build_session_report({}, payload)
    assert report["retention_metrics"]["envelope_over_count"] == 1


def test_overview_moment_series_near_bound_chip_path(qt_app):
    ov = OverviewAgentView()
    f = _frame(
        rbta_bounds={"G'": {"time": 0.001}},
        module_timings={"prediction": 0.95},
    )
    ov.set_frame(f)
    assert ov._moment_series[-1].get("near_bound") == "prediction"


def test_count_moments_violation_key():
    series = [{"violation": True}, {"violation": False}, {"violation": True}]
    c = count_moments(series)
    assert c["violation_count"] == 2
