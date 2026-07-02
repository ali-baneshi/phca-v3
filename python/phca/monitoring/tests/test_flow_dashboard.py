"""Tests for Cognitive Flow tab helpers and UI smoke."""
from __future__ import annotations

import pytest

from phca.monitoring.observability import ObservabilityFrame
from phca.monitoring.playback import _Smoother
from phca.monitoring.qt_dashboard import (
    PIPELINE,
    PANEL_BG,
    CognitiveFlowView,
    _flow_bottleneck_key,
    _flow_layout,
    _flow_status_line,
    _flow_update_active_idx,
    _heatmap_cell_alpha,
    _heatmap_cell_color,
    make_app,
)


@pytest.fixture(scope="module")
def qt_app():
    app = make_app()
    yield app
    app.processEvents()


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
    assert "active=G′" in line


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
    low = _heatmap_cell_color(2.0, 25.0)
    high = _heatmap_cell_color(22.0, 25.0)
    assert low.alpha() >= 6
    assert high.alpha() >= 6
    assert low.green() > low.red()
    assert high.red() > high.green()


def test_scalar_gauge_fallback_best_score(qt_app):
    ov = CognitiveFlowView()
    f = _flow_frame()
    f.gprime_mutual_info = 0.0
    f.action_rationale = {"best_score": 0.42}
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
