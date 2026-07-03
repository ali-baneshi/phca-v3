"""Behavioral tests for observability correctness (units, cache, replay check)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from phca.monitoring.cognitive_panels import (
    RBTA_TO_FLOW,
    flow_near_bound_modules,
    flow_timing_ratio,
    rbta_time_bound_ms,
)
from phca.monitoring.observability import ObservabilityFrame, clear_snap_cache
from phca.monitoring.render import frame_from_json
from phca.monitoring.session_report import build_session_report

_REPO = Path(__file__).resolve().parents[4]
_SCRIPTS = _REPO / "scripts"


@pytest.fixture
def qt_app():
    from PyQt5 import QtWidgets
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    return app


def _frame(**kw) -> ObservabilityFrame:
    f = ObservabilityFrame()
    for k, v in kw.items():
        setattr(f, k, v)
    return f


def test_rbta_time_bound_ms_normalization():
    bounds = {"ASI": {"time": 0.005}}
    assert rbta_time_bound_ms("sanitize", bounds) == pytest.approx(5.0)
    ratio = flow_timing_ratio("sanitize", {"sanitize": 4.0}, bounds)
    assert ratio == pytest.approx(0.8)


def test_rbta_alias_asi_sanitize():
    assert RBTA_TO_FLOW["ASI"] == "sanitize"
    bounds = {"ASI": {"time": 0.01}}
    ratio = flow_timing_ratio("sanitize", {"sanitize": 8.0}, bounds)
    assert ratio == pytest.approx(0.8)


def test_scale_state_idempotent(qt_app):
    from phca.monitoring.qt_dashboard import ScaleState

    sc = ScaleState(contract=0.05, head=0.10)
    lo1, hi1 = sc.update(0.0, 1.0)
    lo2, hi2 = sc.update(0.0, 1.0)
    assert lo1 == pytest.approx(lo2)
    assert hi1 == pytest.approx(hi2)
    for _ in range(8):
        lo_n, hi_n = sc.update(0.0, 1.0)
    assert lo_n == pytest.approx(lo2)
    assert hi_n == pytest.approx(hi2)


def test_phase_peu_page_index_safe(qt_app):
    from PyQt5 import QtGui

    from phca.monitoring.qt_dashboard import PANEL_BG, _PhasePortraitView

    view = _PhasePortraitView()
    view.page_size = 5
    d = 50
    obs = np.linspace(0, 1, d, dtype=np.float32)
    pred = obs + 0.1
    peu = np.linspace(0.1, 1.0, 10, dtype=np.float32)
    f = _frame(
        cycle_id=0,
        predicted_state=pred,
        obs_vector=obs,
        per_dim_peu=peu,
        gprime_uncertainty=np.full(d, 0.05, dtype=np.float32),
    )
    view.set_frame(f, replay=True)
    view.set_page(1)
    pm = QtGui.QPixmap(640, 280)
    pm.fill(PANEL_BG)
    p = QtGui.QPainter(pm)
    view._draw(p)
    p.end()


def test_replay_check_fails_empty(tmp_path):
    import importlib.util

    d = tmp_path / "sess"
    d.mkdir()
    (d / "meta.json").write_text(json.dumps({"cycles": 10}))
    (d / "timeseries.jsonl").write_text("")
    spec = importlib.util.spec_from_file_location("phca_replay_empty", _SCRIPTS / "phca_replay.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod._check(str(d)) == 1
    assert mod._check(str(d), allow_incomplete=True) == 1


def test_replay_check_fails_incomplete_count(tmp_path):
    import importlib.util

    d = tmp_path / "sess2"
    d.mkdir()
    (d / "meta.json").write_text(json.dumps({"cycles": 10}))
    lines = [json.dumps({"cycle_id": i}) for i in range(3)]
    (d / "timeseries.jsonl").write_text("\n".join(lines) + "\n")
    spec = importlib.util.spec_from_file_location("phca_replay2", _SCRIPTS / "phca_replay.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod._check(str(d)) == 1
    assert mod._check(str(d), allow_incomplete=True) == 0


def test_snap_cache_object_isolation():
    clear_snap_cache()

    class _Obj:
        def __init__(self, val):
            self._val = val

        def bounds_snapshot(self):
            return {"X": {"time": self._val}}

    from phca.monitoring.observability import _cached_snap

    a = _Obj(0.01)
    b = _Obj(0.99)
    sa = _cached_snap("rbta_bounds", a, "bounds_snapshot", 10.0)
    sb = _cached_snap("rbta_bounds", b, "bounds_snapshot", 10.0)
    assert sa["X"]["time"] == pytest.approx(0.01)
    assert sb["X"]["time"] == pytest.approx(0.99)


def test_frame_json_roundtrip_no_alias():
    f = _frame(
        cycle_id=3,
        module_timings={"prediction": 2.5},
        rbta_bounds={"G'": {"time": 0.005}},
        action_rationale={"best_score": 0.4, "chosen_idx": 1},
    )
    raw = f.to_json()
    restored = frame_from_json(raw)
    raw["action_rationale"]["best_score"] = 9.9
    assert restored.action_rationale["best_score"] == pytest.approx(0.4)


def test_session_report_near_bound_units():
    lines = [
        json.dumps({
            "cycle_id": 0,
            "module_timings": {"prediction": 4.2},
            "rbta_bounds": {"G'": {"time": 0.005}},
            "action_rationale": {},
            "prediction_error": 0.1,
            "env_kind": "grid",
        })
    ]
    report = build_session_report({}, lines)
    assert report["flow_metrics"]["near_bound_cycle_count"] >= 1


def test_rebuild_histories_resets_scales(qt_app):
    from phca.monitoring.qt_dashboard import CandidateScoreView, ScaleState

    view = CandidateScoreView()
    for i in range(5):
        view._score_scale.update(float(i), float(i + 10))
    before = (view._score_scale.lo, view._score_scale.hi)
    view.rebuild_histories([_frame(cycle_id=j) for j in range(3)])
    assert view._score_scale._have is False
    assert (view._score_scale.lo, view._score_scale.hi) != before or not view._score_scale._have
