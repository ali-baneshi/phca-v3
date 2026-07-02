"""v8 Overview 1.5 — unified agent card offscreen smoke."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

pytest.importorskip("PyQt5")

from phca.monitoring.camera_render import (
    is_glitchy_rgb_frame,
    rgb_frame_to_pixmap,
    rgb_frame_to_qimage,
)
from phca.monitoring.observability import ObservabilityFrame
from phca.monitoring.qt_dashboard import (
    ObservatoryWindow,
    OverviewAgentView,
    _OVERVIEW_HEADER_H,
    _OVERVIEW_RIBBON_H,
    _OVERVIEW_MARGIN,
    _rgb_frame_to_qimage,
    make_app,
    set_autoscale_frozen,
)

_LEGACY_TAU_PURPLE = (155, 89, 182)


def _reacher_frame(cycle_id: int = 1) -> ObservabilityFrame:
    f = ObservabilityFrame()
    f.cycle_id = cycle_id
    f.env_kind = "mujoco_rgb"
    f.action_kind = "continuous"
    f.state_dim = 11
    f.action_dim = 2
    f.env_frame = np.zeros((120, 160, 3), dtype=np.uint8)
    f.env_frame[:, :, 1] = 80
    f.obs_vector = np.linspace(-1.0, 1.0, 11, dtype=np.float32)
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


def test_ppm_qimage_not_magenta_with_reacher_frame(qt_app):
    from PyQt5 import QtCore, QtGui

    f = _reacher_frame()
    f.env_frame = np.random.randint(20, 200, (240, 320, 3), dtype=np.uint8)
    qimg = rgb_frame_to_qimage(f.env_frame)
    assert not qimg.isNull()
    pm = QtGui.QPixmap(320, 240)
    pm.fill(QtCore.Qt.black)
    p = QtGui.QPainter(pm)
    p.drawPixmap(0, 0, QtGui.QPixmap.fromImage(qimg.scaled(320, 240)))
    p.end()
    c = pm.toImage().pixelColor(160, 120)
    assert not (c.red() > 250 and c.blue() > 250 and c.green() < 10), "magenta placeholder"
    # qt_dashboard re-export should match camera_render
    assert not _rgb_frame_to_qimage(f.env_frame).isNull()


def test_glitchy_purple_frame_detected():
    pr, pg, pb = _LEGACY_TAU_PURPLE
    slab = np.full((64, 80, 3), (pr, pg, pb), dtype=np.uint8)
    assert is_glitchy_rgb_frame(slab)
    good = np.random.randint(20, 200, (64, 80, 3), dtype=np.uint8)
    assert not is_glitchy_rgb_frame(good)


def test_overview_caches_camera_pixmap(qt_app):
    ov = OverviewAgentView()
    ov.resize(800, 600)
    ov.show()
    f = _reacher_frame()
    f.env_frame = np.random.randint(30, 180, (120, 160, 3), dtype=np.uint8)
    ov.set_frame(f)
    assert ov._camera_pixmap is not None
    assert not ov._camera_pixmap.isNull()
    f.env_frame = None
    ov.set_frame(f)
    assert ov._camera_pixmap is None


def test_overview_live_reacher_not_mostly_glitch(qt_app):
    from phca.core.cycle import CognitiveCycle
    from phca.monitoring.observability import ObservabilityStore

    store = ObservabilityStore(maxlen=20)
    cyc = CognitiveCycle.build_for_mujoco(
        "Reacher-v5", seed=42, use_mlp=True,
        observability_store=store, render_mode="rgb_array")
    for _ in range(6):
        cyc.step()
    fr = store.latest()
    ov = OverviewAgentView()
    ov.resize(1320, 840)
    ov.show()
    ov.set_frame(fr)
    ov.repaint_if_dirty()
    qt_app.processEvents()
    body = ov.main_body_rect()
    img = ov.grab().toImage()
    tau_top = body.bottom() - 50
    samples = []
    for x in range(body.x() + 10, body.right() - 10, 35):
        for y in range(body.y() + 10, tau_top, 35):
            c = img.pixelColor(x, y)
            samples.append((c.red(), c.green(), c.blue()))
    assert samples, "no body samples"
    pr, pg, pb = _LEGACY_TAU_PURPLE
    tol = 35
    magenta = sum(1 for r, g, b in samples if r > 250 and b > 250 and g < 10)
    purple = sum(
        1 for r, g, b in samples
        if abs(r - pr) < tol and abs(g - pg) < tol and abs(b - pb) < tol
    )
    glitch = magenta + purple
    assert glitch < len(samples) * 0.2, f"body mostly glitch ({glitch}/{len(samples)})"

    ov2 = OverviewAgentView()
    ov2.resize(800, 600)
    ov2.show()
    f = _reacher_frame()
    f.env_frame = None
    ov2.set_frame(f)
    ov2.repaint_if_dirty()
    qt_app.processEvents()
    assert ov2._camera_pixmap is None
