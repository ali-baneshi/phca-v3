"""v8 Overview 1.5 — unified agent card offscreen smoke."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

pytest.importorskip("PyQt5")

from phca.monitoring.observability import ObservabilityFrame
from phca.monitoring.qt_dashboard import (
    ObservatoryWindow,
    OverviewAgentView,
    _OVERVIEW_HEADER_H,
    _OVERVIEW_RIBBON_H,
    _OVERVIEW_MARGIN,
    make_app,
    set_autoscale_frozen,
)


def _reacher_frame(cycle_id: int = 1) -> ObservabilityFrame:
    f = ObservabilityFrame()
    f.cycle_id = cycle_id
    f.env_kind = "mujoco_rgb"
    f.action_kind = "continuous"
    f.state_dim = 11
    f.action_dim = 2
    f.env_frame = np.zeros((120, 160, 3), dtype=np.uint8)
    f.env_frame[:, :, 1] = 80
    f.continuous_action = np.array([0.42, -0.31], dtype=np.float32)
    f.dim_names = ["shoulder", "elbow"]
    f.prediction_error = 12.5
    f.prediction_confidence = 0.73
    f.drive_levels = [0.2, 0.5, 0.1, 0.3, 0.15, 0.4]
    f.drive_deficits = [0.05] * 6
    f.active_drive_id = 2
    f.empowerment = 0.55
    f.goal_priority = 0.8
    f.cr_temperature = 0.12
    f.gprime_mutual_info = 1.2
    f.rss_bytes = 256_000_000
    f.latency_ms = 8.4
    f.action_rationale = {
        "goal_id": 2,
        "explored": False,
        "best_score": 0.91,
        "note": "exploit best candidate",
    }
    return f


@pytest.fixture(scope="module")
def qt_app():
    app = make_app()
    yield app
    app.processEvents()


def test_overview_unified_widget(qt_app):
    win = ObservatoryWindow()
    assert hasattr(win, "overview")
    assert isinstance(win.overview, OverviewAgentView)
    assert win.overview.minimumHeight() >= 480
    win.resize(1320, 840)
    win.show()
    qt_app.processEvents()


def test_overview_reacher_smoke_paint(qt_app):
    win = ObservatoryWindow()
    win.resize(1320, 840)
    win.show()
    f = _reacher_frame()
    for c in range(1, 6):
        fr = _reacher_frame(c)
        fr.prediction_error = 12.5 - c
        fr.prediction_confidence = 0.5 + c * 0.04
        win.controller.update(fr)
    win.overview.repaint_if_dirty()
    qt_app.processEvents()
    pm = win.grab()
    assert not pm.isNull()
    assert pm.width() > 0


def test_overview_body_area_layout(qt_app):
    ov = OverviewAgentView()
    ov.resize(1320, 840)
    ov.show()
    ov.set_frame(_reacher_frame())
    qt_app.processEvents()
    body = ov.main_body_rect()
    main_h = 840 - _OVERVIEW_HEADER_H - _OVERVIEW_RIBBON_H - _OVERVIEW_MARGIN * 3
    assert body.height() >= int(main_h * 0.95)
    assert body.height() >= int(840 * 0.40)


def test_overview_breath_frozen_on_pause(qt_app):
    ov = OverviewAgentView()
    ov.resize(800, 600)
    ov.show()
    ov.set_frame(_reacher_frame())
    set_autoscale_frozen(True)
    ov.repaint_if_dirty()
    qt_app.processEvents()
    set_autoscale_frozen(False)


def test_overview_camera_no_frame(qt_app):
    ov = OverviewAgentView()
    ov.resize(800, 600)
    ov.show()
    f = _reacher_frame()
    f.env_frame = None
    ov.set_frame(f)
    ov.repaint_if_dirty()
    qt_app.processEvents()
