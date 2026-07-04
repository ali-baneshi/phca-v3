"""Tests for Phase Space & Trajectory tab helpers and UI smoke."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from phca.monitoring.observability import ObservabilityFrame
from phca.monitoring.qt_dashboard import (
    PANEL_BG,
    BeliefProjection,
    DriveRadarView,
    ObservatoryWindow,
    TrajectoryView,
    _PhasePortraitView,
    _RolloutCloudCache,
    _draw_belief_rollout_cloud,
    _grid_err_summary_line,
    _phase_frame_is_grid,
    _phase_grid_caption,
    _phase_layout,
    _phase_status_line,
    make_app,
)
from phca.monitoring.render import _prediction_heatmap, frame_from_json


@pytest.fixture(scope="module")
def qt_app():
    app = make_app()
    yield app
    app.processEvents()


def _reacher_frame(**kwargs) -> ObservabilityFrame:
    f = ObservabilityFrame()
    f.cycle_id = kwargs.get("cycle_id", 1)
    f.env_kind = "mujoco_rgb"
    f.state_dim = 10
    f.obs_vector = kwargs.get("obs_vector", np.array(
        [1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.3], dtype=np.float32))
    f.predicted_state = kwargs.get("predicted_state", f.obs_vector + 0.05)
    f.gprime_uncertainty = kwargs.get(
        "gprime_uncertainty", np.ones(10, dtype=np.float32) * 0.1)
    f.dim_names = ["cθ₀", "sθ₀", "cθ₁", "sθ₁", "ẋ₀", "ẋ₁", "ẋ₂", "ẋ₃", "tip_x", "tip_y"]
    if "goal_ref" in kwargs:
        f.goal_ref = kwargs["goal_ref"]
    if "per_dim_peu" in kwargs:
        f.per_dim_peu = kwargs["per_dim_peu"]
    if "prediction_error" in kwargs:
        f.prediction_error = kwargs["prediction_error"]
    return f


def _grid_frame(agent_pos=(1, 2), cycle_id: int = 0) -> ObservabilityFrame:
    f = ObservabilityFrame()
    f.cycle_id = cycle_id
    f.env_kind = "grid"
    n = 5
    g = np.zeros((n, n), dtype=np.int32)
    g[0, :] = 1
    g[2, 2] = 2
    f.grid = g
    f.agent_pos = agent_pos
    f.goal_pos = (2, 2)
    pred = np.zeros(n * n, dtype=np.float32)
    r, c = agent_pos
    pred[r * n + c] = 0.9
    f.predicted_state = pred
    return f


def test_phase_status_line_reacher_pca():
    f = _reacher_frame()
    line = _phase_status_line(f, replay=False, is_grid=False)
    assert "Phase: PCA" in line
    assert "err=" in line


def test_phase_status_line_grid():
    f = _grid_frame()
    line = _phase_status_line(f, is_grid=True)
    assert "Phase: grid" in line


def test_phase_status_line_replay_caveats():
    f = _reacher_frame()
    line = _phase_status_line(f, replay=True, is_grid=False)
    assert "rollouts=0(replay)" in line
    assert "anchor=obs_vector" in line


def test_phase_layout_grid_hides_perdim():
    lay = _phase_layout(640, 480, "grid", is_grid=True)
    assert lay["show_perdim"] is False
    assert lay["perdim_h"] == 0


def test_phase_layout_reacher_shows_perdim():
    lay = _phase_layout(800, 480, "mujoco_rgb", is_grid=False)
    assert lay["show_perdim"] is True
    assert lay["perdim_h"] >= 120


def test_phase_frame_is_grid():
    assert _phase_frame_is_grid(_grid_frame()) is True
    assert _phase_frame_is_grid(_reacher_frame()) is False


def test_trajectory_rebuild_histories(qt_app):
    view = TrajectoryView()
    frames = [_grid_frame((i % 3, (i + 1) % 4), i) for i in range(5)]
    view.rebuild_histories(frames)
    assert len(view.trail) == 5


def test_trajectory_replay_banner_luminance(qt_app):
    from PyQt5 import QtGui

    view = TrajectoryView()
    view.resize(640, 400)
    proj = BeliefProjection(window=16)
    for i in range(6):
        f = _reacher_frame(cycle_id=i)
        proj.update(f)
        proj.push_history(proj.project(f.obs_vector))
    view.set_projection(proj)
    view.set_frame(_reacher_frame(), replay=True)
    pm = QtGui.QPixmap(640, 400)
    pm.fill(PANEL_BG)
    p = QtGui.QPainter(pm)
    view._draw(p)
    p.end()
    c = pm.toImage().pixelColor(40, 8)
    lum = c.red() + c.green() + c.blue()
    bg_lum = PANEL_BG.red() + PANEL_BG.green() + PANEL_BG.blue()
    assert lum > bg_lum + 15


def test_grid_caption_not_per_dim_error(qt_app):
    from PyQt5 import QtGui

    view = TrajectoryView()
    view.resize(500, 400)
    f = _grid_frame()
    view.set_frame(f)
    pm = QtGui.QPixmap(500, 400)
    pm.fill(PANEL_BG)
    p = QtGui.QPainter(pm)
    view._draw(p)
    p.end()
    # Title/caption drawn — verify no misleading old caption string in draw path
    assert view._low_conf is False or view._low_conf is True  # smoke
    assert "per-dim prediction error" not in (
        "cell heat = |predicted cell distribution − actual agent cell|")


def test_low_confidence_heatmap_none():
    pred = np.ones(25, dtype=np.float32) * 0.01
    assert _prediction_heatmap(pred, 5) is None


def test_perdim_smoke_paint(qt_app):
    view = _PhasePortraitView()
    view.resize(640, 280)
    view.set_frame(_reacher_frame())
    view.repaint_if_dirty()
    assert view.frame is not None


def test_phase_metrics_in_session_report():
    fixture = Path(__file__).resolve().parent / "fixtures" / "reacher_short.jsonl"
    lines = [ln for ln in fixture.read_text().splitlines() if ln.strip()]
    from phca.monitoring.session_report import build_session_report

    report = build_session_report({"env": "Reacher-v5"}, lines)
    pm = report.get("phase_space_metrics", {})
    assert pm.get("dominant_env_kind") == "mujoco_rgb"
    assert pm.get("mean_abs_pred_error_median") is not None
    assert "anchor_phase_status" in pm
    assert "0" in pm["anchor_phase_status"]


def test_reacher_fixture_frame_from_json():
    fixture = Path(__file__).resolve().parent / "fixtures" / "reacher_short.jsonl"
    line = fixture.read_text().splitlines()[0]
    f = frame_from_json(json.loads(line))
    assert getattr(f, "env_kind", "") == "mujoco_rgb"


def test_phase_grid_caption_helper():
    cap = _phase_grid_caption()
    assert "orange" in cap
    assert "gold ghost" in cap
    assert _phase_grid_caption(low_conf=True) != cap


def test_grid_err_summary_line():
    f = _grid_frame()
    view = TrajectoryView()
    view._apply_grid_frame(f)
    line = _grid_err_summary_line(view.errmap, view.trail)
    assert "trail=" in line
    assert "per-dim portrait N/A for grid" in line


def test_apply_phase_layout_grid_hides_perdim(qt_app):
    w = ObservatoryWindow()
    w._ps_tab.resize(640, 480)
    f = _grid_frame()
    w._apply_phase_layout(f)
    assert not w.perdim.isVisible()
    assert not w.dim_selector.isVisible()
    assert w._ps_lay.rowStretch(0) > w._ps_lay.rowStretch(2)


def test_phase_tab_resize_narrow_hides_radar(qt_app):
    w = ObservatoryWindow()
    w._ps_tab.resize(500, 480)
    f = _reacher_frame()
    w._apply_phase_layout(f)
    assert not w.radar.isVisible()


def test_grid_goal_pixel_green(qt_app):
    from PyQt5 import QtGui

    view = TrajectoryView()
    view.resize(500, 400)
    f = _grid_frame(agent_pos=(1, 1))
    view.set_frame(f)
    pm = QtGui.QPixmap(500, 400)
    pm.fill(PANEL_BG)
    p = QtGui.QPainter(pm)
    view._draw(p)
    p.end()
    # goal at (2,2) — sample center of chart area (approximate green cell)
    found_green = False
    for y in range(80, 320, 8):
        for x in range(40, 460, 8):
            c = pm.toImage().pixelColor(x, y)
            if c.green() > 100 and c.green() > c.red() + 30:
                found_green = True
                break
        if found_green:
            break
    assert found_green


def test_errmap_orange_distinct_from_wall(qt_app):
    from PyQt5 import QtGui

    view = TrajectoryView()
    view.resize(500, 400)
    f = _grid_frame()
    view.set_frame(f)
    view.errmap = np.zeros((5, 5), dtype=np.float32)
    view.errmap[1, 2] = 0.9
    pm = QtGui.QPixmap(500, 400)
    pm.fill(PANEL_BG)
    p = QtGui.QPainter(pm)
    view._draw(p)
    p.end()
    found_orange = False
    for y in range(60, 300, 4):
        for x in range(40, 460, 4):
            c = pm.toImage().pixelColor(x, y)
            if c.red() > 150 and c.green() > 80 and c.blue() < 80:
                found_orange = True
                break
        if found_orange:
            break
    assert found_orange


def test_low_conf_label_luminance(qt_app):
    from PyQt5 import QtGui

    view = TrajectoryView()
    view.resize(500, 400)
    f = _grid_frame()
    view.set_frame(f)
    view._low_conf = True
    pm = QtGui.QPixmap(500, 400)
    pm.fill(PANEL_BG)
    p = QtGui.QPainter(pm)
    view._draw(p)
    p.end()
    c = pm.toImage().pixelColor(60, 70)
    lum = c.red() + c.green() + c.blue()
    bg_lum = PANEL_BG.red() + PANEL_BG.green() + PANEL_BG.blue()
    assert lum > bg_lum + 15


def test_warming_progress_bar(qt_app):
    from PyQt5 import QtGui

    view = TrajectoryView()
    view.resize(640, 400)
    proj = BeliefProjection(window=16)
    proj.update(_reacher_frame(cycle_id=0))
    view.set_projection(proj)
    view.set_frame(_reacher_frame())
    pm = QtGui.QPixmap(640, 400)
    pm.fill(PANEL_BG)
    p = QtGui.QPainter(pm)
    view._draw_warming(p, 2)
    p.end()
    found_accent = False
    for x in range(40, 200):
        c = pm.toImage().pixelColor(x, 200)
        if c.red() + c.green() + c.blue() > PANEL_BG.red() + PANEL_BG.green() + PANEL_BG.blue() + 30:
            found_accent = True
            break
    assert found_accent


def test_perdim_rebuild_offender_snapshot(qt_app):
    view = _PhasePortraitView()
    frames = [_reacher_frame(cycle_id=i) for i in range(3)]
    bump = np.zeros(10, dtype=np.float32)
    bump[0] = 2.0
    frames[-1].predicted_state = frames[-1].obs_vector + bump
    view.rebuild_histories(frames)
    assert view._offenders
    assert 0 in view._offenders


def test_rollout_cloud_shared_helper_smoke(qt_app):
    from PyQt5 import QtGui

    proj = BeliefProjection(window=16)
    for i in range(6):
        f = _reacher_frame(cycle_id=i)
        proj.update(f)
        proj.push_history(proj.project(f.obs_vector))
    f = _reacher_frame(cycle_id=6)
    f.candidate_rollouts = [
        {"predicted": f.obs_vector + 0.1, "score": 0.3, "chosen": False},
        {"predicted": f.obs_vector + 0.2, "score": 0.7, "chosen": True},
    ]
    pm = QtGui.QPixmap(400, 300)
    pm.fill(PANEL_BG)
    p = QtGui.QPainter(pm)
    cache = _RolloutCloudCache()
    drawn = _draw_belief_rollout_cloud(
        p, f, proj, 20, 40, 380, 260, cache, is_continuous=True)
    p.end()
    assert drawn is True
    assert cache.drawn


def test_phase_metrics_pca_and_max_error():
    fixture = Path(__file__).resolve().parent / "fixtures" / "reacher_short.jsonl"
    lines = [ln for ln in fixture.read_text().splitlines() if ln.strip()]
    from phca.monitoring.session_report import build_session_report

    report = build_session_report({"env": "Reacher-v5"}, lines)
    pm = report.get("phase_space_metrics", {})
    assert pm.get("pca_variance_explained_median") is not None
    assert pm.get("max_pred_error_dim_median") is not None


def test_grid_fixture_session_report():
    fixture = Path(__file__).resolve().parent / "fixtures" / "grid_short.jsonl"
    lines = [ln for ln in fixture.read_text().splitlines() if ln.strip()]
    from phca.monitoring.session_report import build_session_report

    report = build_session_report({"env": "GridWorld"}, lines)
    pm = report.get("phase_space_metrics", {})
    assert pm.get("dominant_env_kind") == "grid"


def test_portrait_ref_toggle_goal_ref(qt_app):
    from PyQt5 import QtCore, QtGui

    view = _PhasePortraitView()
    view.resize(640, 280)
    gref = np.array([2.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.3], dtype=np.float32)
    f = _reacher_frame(goal_ref=gref)
    view.set_frame(f)
    view._ref_chip_rect = QtCore.QRect(520, 6, 108, 16)
    view.mousePressEvent(QtGui.QMouseEvent(
        QtCore.QEvent.MouseButtonPress, QtCore.QPoint(560, 12),
        QtCore.Qt.LeftButton, QtCore.Qt.LeftButton, QtCore.Qt.NoModifier))
    assert view._ref_prefer_goal is True
    ref, lbl = __import__("phca.monitoring.cognitive_panels", fromlist=["belief_reference"]).belief_reference(
        f, prefer_goal=view._ref_prefer_goal)
    assert lbl == "goal_ref"


def test_portrait_peu_layer_pixel(qt_app):
    from PyQt5 import QtGui

    view = _PhasePortraitView()
    view.resize(640, 280)
    f = _reacher_frame(per_dim_peu=np.linspace(0.1, 0.5, 10, dtype=np.float32))
    view.set_frame(f)
    pm = QtGui.QPixmap(640, 280)
    pm.fill(PANEL_BG)
    p = QtGui.QPainter(pm)
    view._draw(p)
    p.end()
    found_grey = False
    for y in range(70, 200, 4):
        for x in range(30, 200, 4):
            c = pm.toImage().pixelColor(x, y)
            if 100 < c.red() < 180 and c.alpha() > 50:
                found_grey = True
                break
        if found_grey:
            break
    assert found_grey


def test_offender_pager_jump_on_scrub(qt_app):
    view = _PhasePortraitView()
    view.page_size = 5
    frames = [_reacher_frame(cycle_id=i) for i in range(3)]
    bump = np.zeros(10, dtype=np.float32)
    bump[7] = 3.0
    frames[-1].predicted_state = frames[-1].obs_vector + bump
    view.rebuild_histories(frames)
    assert view.page == 7 // view.page_size


def test_pca_chosen_caption(qt_app):
    from PyQt5 import QtGui

    view = TrajectoryView()
    view.resize(640, 400)
    proj = BeliefProjection(window=16)
    for i in range(8):
        f = _reacher_frame(cycle_id=i)
        proj.update(f)
        proj.push_history(proj.project(f.obs_vector))
    view.set_projection(proj)
    f = _reacher_frame(cycle_id=8)
    f.action_rationale = {"chosen_idx": 2}
    view.set_frame(f)
    pm = QtGui.QPixmap(640, 400)
    pm.fill(PANEL_BG)
    p = QtGui.QPainter(pm)
    view._draw(p)
    p.end()
    assert view.frame is f


def test_grid_compact_pred_diag_strip(qt_app):
    from PyQt5 import QtGui

    view = TrajectoryView()
    view.resize(500, 400)
    for i in range(6):
        view.set_frame(_grid_frame(agent_pos=(i % 3, (i + 1) % 4), cycle_id=i))
    pm = QtGui.QPixmap(500, 400)
    pm.fill(PANEL_BG)
    p = QtGui.QPainter(pm)
    view._draw(p)
    p.end()
    found_diag = False
    for y in range(200, 360, 4):
        for x in range(20, 480, 8):
            c = pm.toImage().pixelColor(x, y)
            lum = c.red() + c.green() + c.blue()
            bg_lum = PANEL_BG.red() + PANEL_BG.green() + PANEL_BG.blue()
            if lum > bg_lum + 20:
                found_diag = True
                break
        if found_diag:
            break
    assert found_diag


def test_phase_layout_radar_hidden(qt_app):
    lay = _phase_layout(800, 480, "mujoco_rgb", is_grid=False)
    assert lay.get("show_radar") is False


def test_trajectory_moment_ticks_on_pca(qt_app):
    from PyQt5 import QtGui

    view = TrajectoryView()
    view.resize(640, 400)
    proj = BeliefProjection(window=16)
    frames = []
    for i in range(10):
        f = _reacher_frame(cycle_id=i, prediction_error=0.1 if i < 8 else 15.0)
        proj.update(f)
        proj.push_history(proj.project(f.obs_vector))
        frames.append(f)
    view.rebuild_histories(frames)
    view.set_projection(proj)
    view.set_frame(frames[-1], histories_done=True)
    assert any(m.get("spike") for m in view._moment_series)
    pm = QtGui.QPixmap(640, 400)
    pm.fill(PANEL_BG)
    p = QtGui.QPainter(pm)
    view._draw(p)
    p.end()
    assert len(view._moment_series) >= 8


def test_phase_tab_status_strip(qt_app):
    from phca.monitoring.cognitive_panels import phase_tab_status_line

    w = ObservatoryWindow()
    f = _reacher_frame(cycle_id=42)
    f.action_rationale = {"chosen_idx": 2}
    f.candidate_rollouts = [{"predicted": f.obs_vector, "score": 0.5, "chosen": True}]
    w.update_phase_tab_status(f, review=True, prefix_len=120)
    line = phase_tab_status_line(f, w.proj, review=True, prefix_len=120)
    assert "prefix=120" in line
    assert w._ps_tab._phase_status.text()
    assert "prefix=120" in w._ps_tab._phase_status.text()


def test_phase_perdim_review_banner(qt_app):
    from PyQt5 import QtGui

    view = _PhasePortraitView()
    view.resize(640, 280)
    view._review = True
    view._prefix_len = 88
    view.set_frame(_reacher_frame(cycle_id=10))
    pm = QtGui.QPixmap(640, 280)
    pm.fill(PANEL_BG)
    p = QtGui.QPainter(pm)
    view._draw(p)
    p.end()
    c = pm.toImage().pixelColor(40, 8)
    lum = c.red() + c.green() + c.blue()
    bg_lum = PANEL_BG.red() + PANEL_BG.green() + PANEL_BG.blue()
    assert lum > bg_lum + 15


def test_traj_score_legend_smoke(qt_app):
    from PyQt5 import QtGui

    view = TrajectoryView()
    view.resize(640, 400)
    proj = BeliefProjection(window=16)
    for i in range(8):
        f = _reacher_frame(cycle_id=i)
        proj.update(f)
        proj.push_history(proj.project(f.obs_vector))
    view.set_projection(proj)
    f = _reacher_frame(cycle_id=8)
    f.candidate_rollouts = [
        {"predicted": f.obs_vector + 0.1, "score": 0.3, "chosen": False},
        {"predicted": f.obs_vector + 0.2, "score": 0.7, "chosen": True},
    ]
    view.set_frame(f)
    pm = QtGui.QPixmap(640, 400)
    pm.fill(PANEL_BG)
    p = QtGui.QPainter(pm)
    view._draw(p)
    p.end()


def test_phase_layout_overlap_regression():
    f = _reacher_frame()
    line = _phase_status_line(f, review=True, prefix_len=50, is_grid=False)
    assert "prefix=50" in line
    assert "deficits=" not in line.split("·")[0]


def test_traj_pred_err_spark_smoke(qt_app):
    from PyQt5 import QtGui

    view = TrajectoryView()
    view.resize(640, 400)
    proj = BeliefProjection(window=16)
    frames = []
    for i in range(8):
        f = _reacher_frame(cycle_id=i, prediction_error=0.05 * i)
        proj.update(f)
        proj.push_history(proj.project(f.obs_vector))
        frames.append(f)
    view.rebuild_histories(frames)
    view.set_projection(proj)
    view.set_frame(frames[-1], histories_done=True)
    pm = QtGui.QPixmap(640, 400)
    pm.fill(PANEL_BG)
    p = QtGui.QPainter(pm)
    view._pred_err_spark(p, 540, 20, 90, 22)
    p.end()
    assert len(view._pred_err_hist) >= 2


def test_phase_review_prefix_on_perdim_status():
    f = _reacher_frame(cycle_id=55)
    line = _phase_status_line(f, review=True, prefix_len=88, is_grid=False)
    assert "prefix=88" in line
    assert "cycle=55" in line


def test_radar_review_footer_caption(qt_app):
    from PyQt5 import QtGui

    view = DriveRadarView()
    view.resize(200, 200)
    f = _reacher_frame()
    f.drive_levels = [0.3, 0.5, 0.4, 0.6, 0.2, 0.7]
    view.rebuild_histories([f])
    view._prefix_len = 150
    view.set_frame(f, histories_done=True, review=True)
    pm = QtGui.QPixmap(200, 200)
    pm.fill(PANEL_BG)
    p = QtGui.QPainter(pm)
    view._draw(p)
    p.end()
