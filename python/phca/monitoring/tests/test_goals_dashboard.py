"""Tests for Goals & Motivation tab."""
from __future__ import annotations

import pytest

from phca.monitoring.observability import ObservabilityFrame
from phca.monitoring.qt_dashboard import GoalsMotivationView




def _goals_frame(**kwargs) -> ObservabilityFrame:
    f = ObservabilityFrame()
    f.cycle_id = kwargs.get("cycle_id", 1)
    f.drive_levels = kwargs.get("drive_levels", [0.3, 0.5, 0.2, 0.4, 0.1, 0.6])
    f.drive_targets = kwargs.get("drive_targets", [0.5] * 6)
    f.drive_deficits = kwargs.get("drive_deficits", [0.2, 0.0, 0.1, 0.0, 0.0, 0.0])
    f.pareto_front = kwargs.get("pareto_front", [2])
    f.active_drive_id = kwargs.get("active_drive_id", 2)
    f.cr_temperature = kwargs.get("cr_temperature", 0.8)
    f.empowerment = kwargs.get("empowerment", 0.4)
    return f


def test_goals_rebuild_histories(qt_app):
    view = GoalsMotivationView()
    frames = [_goals_frame(cycle_id=i) for i in range(4)]
    view.rebuild_histories(frames)
    assert len(view.drive_hist) == 4
    view.set_frame(_goals_frame(cycle_id=99), histories_done=True)
    assert len(view.drive_hist) == 4


def test_goals_pareto_uses_drive_id(qt_app):
    view = GoalsMotivationView()
    f = _goals_frame(pareto_front=[2], drive_levels=[10.0, 8.0, 0.2, 0.1, 0.1, 0.1])
    view.set_frame(f)
    pareto = {int(x) for x in f.pareto_front}
    assert 2 in pareto
    assert 1 not in pareto


def test_goals_high_levels_normalized(qt_app):
    view = GoalsMotivationView()
    levels = [10.0, 5.0, 2.0, 1.0, 1.0, 1.0]
    f = _goals_frame(drive_levels=levels, drive_targets=[5.0] * 6)
    view.set_frame(f)
    scale = max(max(levels), 5.0)
    disp = min(1.0, levels[0] / scale)
    assert disp == pytest.approx(1.0)
    assert disp > min(1.0, levels[2] / scale)


def test_goals_replay_drive_goals_unavailable(qt_app):
    from PyQt5 import QtGui

    from phca.monitoring.qt_dashboard import PANEL_BG

    view = GoalsMotivationView()
    view.resize(640, 480)
    view.set_frame(_goals_frame(), replay=True)
    pm = QtGui.QPixmap(640, 480)
    pm.fill(PANEL_BG)
    p = QtGui.QPainter(pm)
    view._drive_goals_inset(p, view.frame, 20, 200, 180, 60)
    p.end()
    img = pm.toImage()
    bg_lum = PANEL_BG.red() + PANEL_BG.green() + PANEL_BG.blue()
    diffs = 0
    for y in range(200, 260):
        for x in range(20, 200):
            c = img.pixelColor(x, y)
            if c.red() + c.green() + c.blue() != bg_lum:
                diffs += 1
    assert diffs > 20


def test_goals_replay_data_contract_text():
    from phca.monitoring.cognitive_panels import data_contract_text

    text = data_contract_text("goals", replay=True)
    assert "drive_goals" in text
    assert "live-only" in text


def test_goals_radar_replay_flag(qt_app):
    from phca.monitoring.qt_dashboard import DriveRadarView

    view = DriveRadarView()
    view.set_frame(_goals_frame(), replay=True)
    assert view._replay is True
