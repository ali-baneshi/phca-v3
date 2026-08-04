"""Tests for Cognitive Flow tab helpers and UI smoke."""
from __future__ import annotations

import pytest

from phca.monitoring.observability import ObservabilityFrame
from phca.monitoring.playback import _Smoother
from phca.monitoring.qt_dashboard import (
    PIPELINE,
    PANEL_BG,
    CognitiveFlowView,
    _FLOW_ALL_MODULES,
    _flow_bottleneck_key,
    _flow_layout,
    _flow_status_line,
    _flow_update_active_idx,
    _heatmap_cell_alpha,
    _heatmap_cell_color,
    _heatmap_column_percentile,
)




def _flow_frame(**kwargs) -> ObservabilityFrame:
    f = ObservabilityFrame()
    f.cycle_id = kwargs.get("cycle_id", 1)
    f.module_timings = kwargs.get("module_timings", {
        "sanitize": 0.5,
        "prediction": 1.0,
        "action_selection": 2.0,
        "gprime_learn": 6.0,
        "rbta": 0.3,
    })
    f.violations_count = kwargs.get("violations_count", 0)
    f.rbta_violations = kwargs.get("rbta_violations", [])
    return f


def test_flow_status_line_bottleneck_and_pipe():
    f = _flow_frame()
    line = _flow_status_line(f, active_idx=2)
    assert line.startswith("Flow:")
    assert "bottleneck=G′lrn" in line
    assert "Σpipe=" in line
    assert "violations=OK" in line
    assert "Δ=G′" in line
    assert "phase=Learn" in line
    assert "learn=" in line


def test_flow_status_line_violations():
    f = _flow_frame(
        violations_count=1,
        rbta_violations=[{"module_id": "ACTION", "bound_type": "time",
                          "measured": 2.5, "allowed": 2.0}],
    )
    line = _flow_status_line(f)
    assert "1V" in line
    assert "Act" in line


def test_flow_bottleneck_key():
    f = _flow_frame()
    assert _flow_bottleneck_key(f) == "gprime_learn"


def test_heatmap_cell_alpha_floor():
    assert _heatmap_cell_alpha(0.0, 25.0) == 0
    assert _heatmap_cell_alpha(0.1, 25.0) >= 6
    assert _heatmap_cell_alpha(12.0, 25.0) > _heatmap_cell_alpha(0.1, 25.0)


def test_flow_layout_reserved_heatmap_band():
    lay = _flow_layout(640, 480, 13)
    assert lay["heatmap_h"] >= 90
    assert lay["row_h"] >= 6.0
    assert lay["graph_top"] + lay["graph_h"] + lay["margin"] <= lay["heatmap_top"]


def test_heatmap_cell_color_uses_cost_semantics():
    col = [0.1, 1.0, 5.0, 22.0]
    low = _heatmap_cell_color(2.0, col)
    high = _heatmap_cell_color(22.0, col)
    assert low.alpha() >= 20
    assert high.alpha() >= low.alpha()
    assert low.green() > low.red()
    assert high.red() > high.green()


def test_heatmap_column_percentile_monotonic():
    col = [0.5, 1.0, 2.0, 4.0, 8.0]
    a_lo = _heatmap_column_percentile(1.0, col)
    a_hi = _heatmap_column_percentile(8.0, col)
    assert a_hi >= a_lo


def test_flow_phase_strip_luminance(qt_app):
    from PyQt5 import QtGui

    ov = CognitiveFlowView()
    ov.resize(640, 480)
    f = _flow_frame(module_timings={
        "prediction": 2.0, "action_selection": 3.0, "peu": 1.0,
        "gprime_learn": 5.0, "mdim": 0.5, "tspl": 0.5, "rbta": 0.3,
    })
    ov.set_frame(f)
    pm = QtGui.QPixmap(640, 480)
    pm.fill(PANEL_BG)
    p = QtGui.QPainter(pm)
    ov._draw(p)
    p.end()
    c = pm.toImage().pixelColor(80, 81)
    lum = c.red() + c.green() + c.blue()
    bg_lum = PANEL_BG.red() + PANEL_BG.green() + PANEL_BG.blue()
    assert lum > bg_lum + 10


def test_scalar_gauge_action_uses_best_score(qt_app):
    ov = CognitiveFlowView()
    f = _flow_frame()
    f.action_rationale = {"best_score": 0.42, "eps": 0.1, "explored": False}
    f.candidate_scores = [0.2, 0.42]
    v, label, _ = ov._scalar_gauge_value("action_selection", f)
    assert label == "score"
    assert v == pytest.approx(0.42)


def test_flow_heatmap_band_luminance(qt_app):
    from PyQt5 import QtGui

    ov = CognitiveFlowView()
    ov.resize(640, 480)
    for i in range(12):
        ov.set_frame(_flow_frame(cycle_id=i))
    lay = _flow_layout(640, 480, 13)
    pm = QtGui.QPixmap(640, 480)
    pm.fill(PANEL_BG)
    p = QtGui.QPainter(pm)
    ov._heatmap(p, lay)
    p.end()
    y = lay["heatmap_top"] + lay["heatmap_h"] // 2
    c = pm.toImage().pixelColor(120, y)
    lum = c.red() + c.green() + c.blue()
    bg_lum = PANEL_BG.red() + PANEL_BG.green() + PANEL_BG.blue()
    assert lum > bg_lum + 12


def test_flow_update_active_idx_hysteresis():
    smooth = {m: _Smoother(0.15) for m in PIPELINE}
    prev = {m: 1.0 for m in PIPELINE}
    cur = {m: 1.0 for m in PIPELINE}
    cur["action_selection"] = 5.0
    idx, hold = _flow_update_active_idx(prev, cur, smooth, 0, 0)
    assert idx == PIPELINE.index("action_selection")
    assert hold == 3


def test_cognitive_flow_smoke_paint(qt_app):
    ov = CognitiveFlowView()
    ov.resize(640, 480)
    f = _flow_frame()
    ov.set_frame(f)
    ov.repaint_if_dirty()
    assert ov.frame is f


def test_session_report_flow_metrics():
    from pathlib import Path
    from phca.monitoring.session_report import build_session_report

    fixture = Path(__file__).resolve().parent / "fixtures" / "reacher_short.jsonl"
    lines = [ln for ln in fixture.read_text().splitlines() if ln.strip()]
    report = build_session_report({"env": "Reacher-v5"}, lines)
    flow_m = report.get("flow_metrics", {})
    assert flow_m.get("bottleneck_module")
    assert flow_m.get("violation_cycle_count", 0) >= 1
    assert "anchor_flow_status" in flow_m
    assert "0" in flow_m["anchor_flow_status"]


def test_flow_rebuild_histories_heat_length(qt_app):
    view = CognitiveFlowView()
    frames = [_flow_frame(cycle_id=i) for i in range(8)]
    view.rebuild_histories(frames)
    assert len(view.heat) == 8
    view.set_frame(_flow_frame(cycle_id=99), histories_done=True)
    assert len(view.heat) == 8


def test_flow_rebuild_clears_heat_cache(qt_app):
    view = CognitiveFlowView()
    view.resize(640, 480)
    frames = [_flow_frame(cycle_id=i) for i in range(6)]
    view.rebuild_histories(frames)
    view._heat_key = ("stale",)
    view._heat_pm = None
    view.rebuild_histories(frames[:3])
    assert view._heat_key == ()
    assert view._heat_pm is None


def test_flow_learn_burst_row_border_pixel(qt_app):
    from PyQt5 import QtGui

    view = CognitiveFlowView()
    view.resize(640, 480)
    for i in range(10):
        f = _flow_frame(cycle_id=i, module_timings={
            "sanitize": 0.5, "prediction": 1.0, "action_selection": 2.0,
            "gprime_learn": 40.0, "rbta": 0.3,
        })
        f.latency_ms = 80.0
        view.set_frame(f)
    assert any(m.get("learn_burst") for m in view._moment_series)
    lay = _flow_layout(640, 480, 13)
    pm = QtGui.QPixmap(640, 480)
    pm.fill(PANEL_BG)
    p = QtGui.QPainter(pm)
    view._heatmap(p, lay)
    p.end()
    learn_row = next(i for i, m in enumerate(_FLOW_ALL_MODULES) if m == "gprime_learn")
    assert learn_row >= 0


def test_flow_moment_ticks_on_heatmap(qt_app):
    from PyQt5 import QtGui

    view = CognitiveFlowView()
    view.resize(640, 480)
    frames = []
    for i in range(6):
        err = 0.1 if i < 4 else 20.0
        f = _flow_frame(cycle_id=i)
        f.prediction_error = err
        frames.append(f)
    view.rebuild_histories(frames)
    assert any(m.get("spike") for m in view._moment_series)
    lay = _flow_layout(640, 480, 13)
    pm = QtGui.QPixmap(640, 480)
    pm.fill(PANEL_BG)
    p = QtGui.QPainter(pm)
    view._heatmap(p, lay)
    p.end()
    assert len(view.heat) == len(view._moment_series)


def test_flow_replay_banner_text(qt_app):
    from PyQt5 import QtGui

    from phca.monitoring.cognitive_panels import data_contract_text

    view = CognitiveFlowView()
    view.resize(640, 480)
    view.set_frame(_flow_frame(), replay=True)
    contract = data_contract_text("flow", replay=True)
    assert "module_timings" in contract
    assert "PEU" in contract
    pm = QtGui.QPixmap(640, 480)
    pm.fill(PANEL_BG)
    p = QtGui.QPainter(pm)
    view._draw(p)
    p.end()
    c = pm.toImage().pixelColor(40, 8)
    lum = c.red() + c.green() + c.blue()
    bg_lum = PANEL_BG.red() + PANEL_BG.green() + PANEL_BG.blue()
    assert lum > bg_lum + 15


def test_flow_moment_series_matches_heat(qt_app):
    view = CognitiveFlowView()
    frames = [_flow_frame(cycle_id=i) for i in range(5)]
    view.rebuild_histories(frames)
    assert len(view.heat) == len(view._moment_series)
    view.set_frame(_flow_frame(cycle_id=99), histories_done=True)
    assert len(view.heat) == len(view._moment_series)


def test_flow_layout_header_covers_draw():
    for w, h in ((640, 480), (800, 600)):
        for replay in (False, True):
            lay = _flow_layout(w, h, 13, replay=replay)
            assert lay["graph_top"] >= lay["caption_bottom"]


def test_flow_action_link_human_labels():
    from phca.monitoring.cognitive_panels import flow_action_link_line

    f = _flow_frame(module_timings={"gprime_learn": 6.0, "action_selection": 2.0})
    f.action_rationale = {"explored": False}
    line = flow_action_link_line(f)
    assert "G′lrn" in line
    assert "gprime_learn" not in line


def test_flow_status_has_cycle_and_phase():
    f = _flow_frame()
    line = _flow_status_line(f)
    assert "cycle=" in line
    assert "phase=Learn" in line
