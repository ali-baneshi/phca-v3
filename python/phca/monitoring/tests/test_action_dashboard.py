"""Tests for Action Selection tab helpers."""
from __future__ import annotations

import numpy as np
import pytest

from phca.monitoring.observability import ObservabilityFrame
from phca.monitoring.qt_dashboard import (
    CandidateScoreView,
    PANEL_BG,
    PANEL_BG_ALT,
    _ACTION_SPARK_GAP,
    _action_chosen_idx,
    _action_layout,
    _action_score_margin,
    _action_status_line,
    _draw_mechanism_stacked_bar,
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
    assert "chosen=" in line


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


def test_action_layout_no_ctx_overlap():
    for w, h in ((640, 480), (800, 600)):
        for replay in (False, True):
            lay = _action_layout(w, h, replay=replay)
            assert lay["status_y"] >= lay["title_y"]
            assert lay["chips_y"] >= lay["status_y"] + 10
            assert lay["decision_y"] >= lay["chips_y"] + 10
            assert lay["ctx_y"] >= lay["decision_y"] + 20
            assert lay["mech_y"] >= lay["ctx_y"] + 20
            assert lay["left_y"] >= lay["body_top"]
            assert lay["footer_y"] >= lay["cloud_bot"]
            assert lay["epsilon_y"] >= lay["footer_y"] + lay["footer_h"] - 2


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


def test_action_rebuild_histories_eps(qt_app):
    view = CandidateScoreView()
    frames = [_action_frame(cycle_id=i) for i in range(5)]
    view.rebuild_histories(frames)
    assert len(view.eps_hist) == 5
    assert len(view.score_hist) == 5
    view.set_frame(_action_frame(cycle_id=99), histories_done=True)
    assert len(view.eps_hist) == 5


def test_action_rollout_cache_clear_on_scrub(qt_app):
    view = CandidateScoreView()
    view._rollout_cache.drawn = [("stale",)]
    frames = [_action_frame(cycle_id=i) for i in range(3)]
    view.rebuild_histories(frames)
    assert view._rollout_cache.drawn == []


def test_explore_hides_stale_scores(qt_app):
    view = CandidateScoreView()
    view.last_scores = [0.9, 0.1]
    view._last_scores_cycle = 0
    f = _action_frame(
        cycle_id=5,
        action_rationale={"explored": True, "eps": 0.15, "k_candidates": 8,
                          "best_score": None, "continuous": True},
        candidate_scores=[],
    )
    view.set_frame(f)
    assert view.last_scores == [] or view._last_scores_cycle == 5


def test_action_chosen_idx_in_rollout_label(qt_app):
    from PyQt5 import QtGui
    from phca.monitoring.qt_dashboard import BeliefProjection

    view = CandidateScoreView()
    view.resize(640, 480)
    proj = BeliefProjection(window=16)
    for i in range(6):
        ff = _action_frame(cycle_id=i)
        proj.update(ff)
        proj.push_history(proj.project(ff.continuous_action))
    view.set_projection(proj)
    f = _action_frame(
        action_rationale={"chosen_idx": 1, "explored": False, "best_score": 0.35,
                          "eps": 0.1, "k_candidates": 8, "continuous": True},
        candidate_scores=[0.2, 0.35, 0.55, 0.4],
    )
    f.candidate_rollouts = [
        {"predicted": f.continuous_action, "score": 0.35, "chosen": True},
    ]
    view.set_frame(f)
    lay = _action_layout(640, 480)
    pm = QtGui.QPixmap(640, 480)
    pm.fill(PANEL_BG)
    p = QtGui.QPainter(pm)
    view._rollout_cloud(
        p, f, lay["right_x"], lay["cloud_top"],
        lay["right_x"] + lay["right_w"], lay["cloud_bot"],
        lay["footer_y"], lay["footer_h"],
        True, has_scores=True)
    p.end()
    assert view.last_chosen == 1


def test_action_sparkline_moment_ticks(qt_app):
    from PyQt5 import QtGui

    view = CandidateScoreView()
    view.resize(640, 480)
    for i in range(8):
        view.set_frame(_action_frame(cycle_id=i))
    view._moment_series = [{"decision_shift": i == 7} for i in range(8)]
    _action_layout(640, 480)
    pm = QtGui.QPixmap(640, 480)
    pm.fill(PANEL_BG)
    p = QtGui.QPainter(pm)
    view._best_score_spark(p, 640 - 190, 6, 180, 26)
    p.end()
    x = 640 - 12
    c = pm.toImage().pixelColor(x, 18)
    lum = c.red() + c.green() + c.blue()
    bg_lum = PANEL_BG.red() + PANEL_BG.green() + PANEL_BG.blue()
    assert lum > bg_lum + 10


def test_action_replay_status_rollouts_caveat():
    f = _action_frame()
    scores = [float(x) for x in f.candidate_scores]
    line = _action_status_line(f, scores, 2, replay=True)
    assert "rollouts=replay" in line


def test_action_pred_err_hist_on_rebuild(qt_app):
    view = CandidateScoreView()
    frames = []
    for i in range(4):
        f = _action_frame(cycle_id=i)
        f.prediction_error = 0.1 * i
        frames.append(f)
    view.rebuild_histories(frames)
    assert len(view._pred_err_hist) == 4
    assert len(view._moment_series) == 4


def test_action_rebuild_decision_shift_parity(qt_app):
    view = CandidateScoreView()
    frames = []
    scores = [0.20, 0.55, 0.56, 0.30]
    for i, bs in enumerate(scores):
        f = _action_frame(
            cycle_id=i,
            action_rationale={
                "explored": False, "best_score": bs, "goal_id": 2,
                "eps": 0.1, "k_candidates": 8, "continuous": True,
            },
        )
        frames.append(f)
    view.rebuild_histories(frames)
    shifts = [m.get("decision_shift") for m in view._moment_series]
    assert shifts[0] is False
    assert shifts[1] is True
    assert shifts[3] is True


def test_flow_action_link_line():
    from phca.monitoring.cognitive_panels import flow_action_link_line

    f = _action_frame()
    f.module_timings = {"gprime_learn": 6.0, "action_selection": 2.0}
    line = flow_action_link_line(f)
    assert "Flow bottleneck=" in line
    assert "EXPLOIT" in line


def test_action_review_prefix_in_status():
    f = _action_frame(cycle_id=42)
    scores = [float(x) for x in f.candidate_scores]
    line = _action_status_line(f, scores, 2, review=True, prefix_len=120)
    assert "prefix=120" in line


def test_action_mechanism_line_elided(qt_app):
    from PyQt5 import QtGui

    view = CandidateScoreView()
    view.resize(640, 480)
    frames = []
    for i in range(6):
        f = _action_frame(cycle_id=i)
        if i % 2 == 0:
            f.action_rationale = {
                "explored": True, "eps": 0.15, "k_candidates": 8,
                "best_score": None, "continuous": True,
            }
            f.candidate_scores = []
        frames.append(f)
    view.rebuild_histories(frames)
    view.set_frame(frames[-1], histories_done=True, review=True)
    line = view._mechanism_line()
    assert line.startswith("mechanism (last")
    assert "explore" in line
    pm = QtGui.QPixmap(640, 480)
    pm.fill(PANEL_BG)
    p = QtGui.QPainter(pm)
    view._draw(p)
    p.end()


def test_rollout_cloud_clip_smoke(qt_app):
    from PyQt5 import QtGui
    from phca.monitoring.qt_dashboard import BeliefProjection

    view = CandidateScoreView()
    view.resize(640, 400)
    proj = BeliefProjection(window=16)
    for i in range(6):
        ff = _action_frame(cycle_id=i)
        proj.update(ff)
        proj.push_history(proj.project(ff.continuous_action))
    view.set_projection(proj)
    f = _action_frame(cycle_id=6)
    f.candidate_rollouts = [
        {"predicted": f.continuous_action, "score": 0.35, "chosen": True},
        {"predicted": f.continuous_action + 0.1, "score": 0.55, "chosen": False},
    ]
    view.set_frame(f)
    lay = _action_layout(640, 400)
    pm = QtGui.QPixmap(640, 400)
    pm.fill(PANEL_BG)
    p = QtGui.QPainter(pm)
    view._draw(p)
    cloud_bot = lay["cloud_bot"]
    view._rollout_cloud(
        p, f, lay["right_x"], lay["cloud_top"],
        lay["right_x"] + lay["right_w"], cloud_bot,
        lay["footer_y"], lay["footer_h"],
        True, has_scores=True)
    p.end()
    assert view.last_chosen >= 0
    assert lay["footer_y"] < lay["epsilon_y"]


def test_action_layout_bands_no_overlap():
    for w, h in ((640, 480), (800, 600)):
        lay = _action_layout(w, h)
        rows = [
            ("title_y", "status_y", 10),
            ("status_y", "chips_y", 10),
            ("chips_y", "decision_y", 12),
            ("decision_y", "ctx_y", 20),
            ("ctx_y", "mech_y", 20),
            ("mech_y", "body_top", 4),
        ]
        for top_k, bot_k, gap in rows:
            assert lay[bot_k] >= lay[top_k] + gap - 2


def test_action_pred_err_spark_visible(qt_app):
    from PyQt5 import QtGui

    view = CandidateScoreView()
    view.resize(640, 480)
    for i in range(6):
        f = _action_frame(cycle_id=i)
        f.prediction_error = 0.1 * i
        view.set_frame(f)
    lay = _action_layout(640, 480)
    pm = QtGui.QPixmap(640, 480)
    pm.fill(PANEL_BG)
    p = QtGui.QPainter(pm)
    sx = lay["spark_x0"] + 2 * (lay["spark_w"] + _ACTION_SPARK_GAP)
    view._pred_err_spark(p, sx, lay["spark_y0"], lay["spark_w"], lay["spark_h"])
    p.end()
    c = pm.toImage().pixelColor(sx + 20, lay["spark_y0"] + 18)
    lum = c.red() + c.green() + c.blue()
    bg_lum = PANEL_BG.red() + PANEL_BG.green() + PANEL_BG.blue()
    assert lum > bg_lum + 10


def test_action_mechanism_stacked_bar(qt_app):
    from PyQt5 import QtGui

    view = CandidateScoreView()
    frames = []
    for i in range(8):
        f = _action_frame(cycle_id=i)
        if i % 2 == 0:
            f.action_rationale = {
                "explored": True, "eps": 0.15, "k_candidates": 8,
                "best_score": None, "continuous": True,
            }
            f.candidate_scores = []
        frames.append(f)
    view.rebuild_histories(frames)
    pm = QtGui.QPixmap(400, 30)
    pm.fill(PANEL_BG)
    p = QtGui.QPainter(pm)
    _draw_mechanism_stacked_bar(p, view._mech_frames, 10, 4, 380, 20)
    p.end()
    found_blue = found_green = False
    for x in range(10, 200, 4):
        c = pm.toImage().pixelColor(x, 14)
        if c.blue() > c.red() + 30:
            found_blue = True
        if c.green() > c.red() + 30:
            found_green = True
    assert found_blue or found_green
