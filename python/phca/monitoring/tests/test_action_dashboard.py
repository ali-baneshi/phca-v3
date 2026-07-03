"""Tests for Action Selection tab helpers."""
from __future__ import annotations

import numpy as np
import pytest

from phca.monitoring.observability import ObservabilityFrame
from phca.monitoring.qt_dashboard import (
    CandidateScoreView,
    PANEL_BG,
    PANEL_BG_ALT,
    _action_chosen_idx,
    _action_layout,
    _action_score_margin,
    _action_status_line,
    make_app,
)


@pytest.fixture(scope="module")
def qt_app():
    app = make_app()
    yield app
    app.processEvents()


def _action_frame(**kwargs) -> ObservabilityFrame:
    f = ObservabilityFrame()
    f.cycle_id = kwargs.get("cycle_id", 1)
    f.action_rationale = kwargs.get("action_rationale", {
        "explored": False,
        "best_score": 0.55,
        "goal_id": 2,
        "eps": 0.1,
        "k_candidates": 8,
        "continuous": True,
    })
    f.candidate_scores = kwargs.get("candidate_scores", [0.2, 0.35, 0.55, 0.4])
    f.continuous_action = kwargs.get("continuous_action", np.array([0.2, -0.1], dtype=np.float32))
    return f


def test_action_score_margin():
    assert _action_score_margin([0.55, 0.35, 0.2]) == pytest.approx(0.20)
    assert _action_score_margin([0.5]) is None


def test_action_status_line_exploit():
    f = _action_frame()
    scores = [float(x) for x in f.candidate_scores]
    chosen = int(np.argmax(scores))
    line = _action_status_line(f, scores, chosen)
    assert "EXPLOIT" in line
    assert "k=8" in line
    assert "Δ2nd=" in line
    assert "rollouts=0" in line


def test_action_status_line_explore():
    f = _action_frame(
        action_rationale={"explored": True, "eps": 0.15, "k_candidates": 8,
                          "best_score": None, "continuous": True},
        candidate_scores=[],
    )
    line = _action_status_line(f, [], -1)
    assert "EXPLORE" in line
    assert "ε=0.150" in line
    assert "score=—" in line


def test_action_layout_regions():
    lay = _action_layout(640, 480)
    assert lay["left_w"] > 100
    assert lay["right_w"] > 100
    assert lay["tau_slot_y"] + lay["tau_slot_h"] <= lay["body_top"] + lay["body_h"] + 4
    assert lay["epsilon_y"] + lay["epsilon_h"] <= lay["body_top"] + lay["body_h"] + 8


def test_action_chosen_idx_overrides_argmax():
    f = _action_frame(
        action_rationale={"chosen_idx": 1, "explored": False, "best_score": 0.35,
                        "eps": 0.1, "k_candidates": 8, "continuous": True},
        candidate_scores=[0.2, 0.35, 0.55, 0.4],
    )
    assert _action_chosen_idx(f, [0.2, 0.35, 0.55, 0.4]) == 1


def test_replay_banner_visible(qt_app):
    from PyQt5 import QtGui

    view = CandidateScoreView()
    view.resize(640, 480)
    view.set_frame(_action_frame(), replay=True)
    pm = QtGui.QPixmap(640, 480)
    pm.fill(PANEL_BG)
    p = QtGui.QPainter(pm)
    view._draw(p)
    p.end()
    c = pm.toImage().pixelColor(40, 8)
    lum = c.red() + c.green() + c.blue()
    bg_lum = PANEL_BG.red() + PANEL_BG.green() + PANEL_BG.blue()
    assert lum > bg_lum + 15


def test_epsilon_strip_pixels(qt_app):
    from PyQt5 import QtGui

    view = CandidateScoreView()
    view.resize(640, 480)
    for i in range(8):
        f = _action_frame(cycle_id=i)
        view.set_frame(f)
    lay = _action_layout(640, 480)
    pm = QtGui.QPixmap(640, 480)
    pm.fill(PANEL_BG)
    p = QtGui.QPainter(pm)
    view._epsilon_strip(p, lay["right_x"], lay["epsilon_y"], lay["right_w"], lay["epsilon_h"])
    p.end()
    c = pm.toImage().pixelColor(lay["right_x"] + 20, lay["epsilon_y"] + 10)
    lum = c.red() + c.green() + c.blue()
    bg_lum = PANEL_BG.red() + PANEL_BG.green() + PANEL_BG.blue()
    assert lum > bg_lum + 8


def test_action_chosen_row_luminance(qt_app):
    from PyQt5 import QtGui

    view = CandidateScoreView()
    view.resize(640, 480)
    view.set_frame(_action_frame())
    pm = QtGui.QPixmap(640, 480)
    pm.fill(PANEL_BG)
    p = QtGui.QPainter(pm)
    view._draw(p)
    p.end()
    lay = _action_layout(640, 480)
    c = pm.toImage().pixelColor(lay["left_x"] + 30, lay["left_y"] + 24)
    lum = c.red() + c.green() + c.blue()
    bg_lum = PANEL_BG.red() + PANEL_BG.green() + PANEL_BG.blue()
    assert lum > bg_lum + 8


def test_action_bar_fill_differs_from_track(qt_app):
    from PyQt5 import QtGui

    view = CandidateScoreView()
    view.resize(640, 480)
    view.set_frame(_action_frame())
    pm = QtGui.QPixmap(640, 480)
    pm.fill(PANEL_BG)
    p = QtGui.QPainter(pm)
    view._draw(p)
    p.end()
    lay = _action_layout(640, 480)
    bar_x = lay["left_x"] + 110
    # Second ranked row (not chosen): partial bar fill leaves visible track on the right.
    row_y = lay["left_y"] + 6 + 34
    fill_c = pm.toImage().pixelColor(bar_x + 4, row_y + 8)
    track_c = pm.toImage().pixelColor(bar_x + 100, row_y + 8)
    fill_lum = fill_c.red() + fill_c.green() + fill_c.blue()
    track_lum = track_c.red() + track_c.green() + track_c.blue()
    assert fill_lum > track_lum
    alt_lum = PANEL_BG_ALT.red() + PANEL_BG_ALT.green() + PANEL_BG_ALT.blue()
    assert abs(track_lum - alt_lum) < 30


def test_explore_branch_tau_heatmap_visible(qt_app):
    from PyQt5 import QtGui

    view = CandidateScoreView()
    view.resize(640, 480)
    f = _action_frame(
        action_rationale={"explored": True, "eps": 0.15, "k_candidates": 8,
                          "best_score": None, "continuous": True},
        candidate_scores=[],
        continuous_action=np.array([0.6, -0.4], dtype=np.float32),
    )
    view.set_frame(f)
    pm = QtGui.QPixmap(640, 480)
    pm.fill(PANEL_BG)
    p = QtGui.QPainter(pm)
    view._draw(p)
    p.end()
    lay = _action_layout(640, 480)
    c = pm.toImage().pixelColor(lay["left_x"] + 40, lay["tau_slot_y"] + 20)
    lum = c.red() + c.green() + c.blue()
    bg_lum = PANEL_BG.red() + PANEL_BG.green() + PANEL_BG.blue()
    assert lum > bg_lum + 10


def test_ranked_list_order():
    scores = [0.2, 0.55, 0.35, 0.4]
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    assert order[0] == 1
    assert scores[order[0]] - scores[order[1]] == pytest.approx(0.15)


def test_candidate_score_view_smoke_paint(qt_app):
    view = CandidateScoreView()
    view.resize(640, 480)
    f = _action_frame()
    view.set_frame(f)
    view.repaint_if_dirty()
    assert view.frame is f


def test_action_metrics_in_session_report():
    from pathlib import Path
    from phca.monitoring.session_report import build_session_report

    fixture = Path(__file__).resolve().parent / "fixtures" / "reacher_short.jsonl"
    lines = [ln for ln in fixture.read_text().splitlines() if ln.strip()]
    report = build_session_report({"env": "Reacher-v5"}, lines)
    action_m = report.get("action_metrics", {})
    assert action_m.get("cycles_with_scores", 0) >= 8
    assert action_m.get("cycles_explore_empty_scores", 0) >= 2
    assert action_m.get("score_margin_median") is not None
    assert "anchor_action_status" in action_m
    assert "0" in action_m["anchor_action_status"]
