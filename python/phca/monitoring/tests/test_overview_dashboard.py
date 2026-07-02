"""v8 Overview 1.5 — unified agent card offscreen smoke."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

pytest.importorskip("PyQt5")

from phca.monitoring.camera_render import (
    is_glitchy_rgb_frame,
    is_uniform_rgb_frame,
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
    f.env_frame[:, :, 0] = np.linspace(20, 160, 160, dtype=np.uint8)[np.newaxis, :]
    f.env_frame[:, :, 1] = np.linspace(30, 140, 120, dtype=np.uint8)[:, np.newaxis]
    f.env_frame[:, :, 2] = 60
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
    good = np.zeros((120, 160, 3), dtype=np.uint8)
    good[:, :, 0] = np.linspace(20, 160, 160, dtype=np.uint8)[np.newaxis, :]
    good[:, :, 1] = np.linspace(30, 140, 120, dtype=np.uint8)[:, np.newaxis]
    good[:, :, 2] = 60
    win.set_camera_provider(lambda: good.copy())
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
    good = np.random.randint(30, 180, (120, 160, 3), dtype=np.uint8)
    ov.set_camera_provider(lambda: good.copy())
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


def test_uniform_green_frame_rejected():
    slab = np.full((64, 80, 3), (0, 255, 0), dtype=np.uint8)
    assert is_uniform_rgb_frame(slab)
    assert is_glitchy_rgb_frame(slab)
    noisy = slab.copy()
    noisy[::17, ::19] = (2, 250, 1)
    assert is_glitchy_rgb_frame(noisy)


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
    good = np.random.randint(30, 180, (120, 160, 3), dtype=np.uint8)
    ov.set_camera_provider(lambda: good.copy())
    ov.set_frame(f)
    assert ov._camera_pixmap is not None
    assert not ov._camera_pixmap.isNull()
    prev = ov._camera_pixmap
    ov.set_camera_provider(lambda: None)
    ov.set_frame(f)
    assert ov._camera_pixmap is prev
    assert ov._camera_stale is True
    bad = np.full((120, 160, 3), (0, 255, 0), dtype=np.uint8)
    ov.set_camera_provider(lambda: bad.copy())
    ov.set_frame(f)
    assert ov._camera_pixmap is None
    assert ov._camera_stale is False


def test_overview_live_reacher_not_mostly_glitch(qt_app):
    from phca.core.cycle import CognitiveCycle
    from phca.monitoring.observability import ObservabilityStore

    store = ObservabilityStore(maxlen=20)
    cyc = CognitiveCycle.build_for_mujoco(
        "Reacher-v5", seed=42, use_mlp=True,
        observability_store=store, enable_camera=True)
    for _ in range(6):
        cyc.step()
    fr = store.latest()
    ov = OverviewAgentView()
    ov.resize(1320, 840)
    ov.show()
    ov.set_camera_provider(cyc.env.render_rgb)
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
    ov2.set_camera_provider(lambda: None)
    ov2.set_frame(f)
    ov2.repaint_if_dirty()
    qt_app.processEvents()
    assert ov2._camera_pixmap is None


def test_overview_live_reacher_shows_real_camera(qt_app):
    from collections import Counter
    from phca.core.cycle import CognitiveCycle
    from phca.monitoring.observability import ObservabilityStore

    store = ObservabilityStore(maxlen=20)
    cyc = CognitiveCycle.build_for_mujoco(
        "Reacher-v5", seed=42, use_mlp=True,
        observability_store=store, enable_camera=True)
    for _ in range(6):
        cyc.step()
    fr = store.latest()
    assert fr.env_frame is None, "env_frame deferred to main-thread camera_provider"
    live = cyc.env.render_rgb()
    assert live is not None, "main-thread render_rgb should capture"
    assert not is_glitchy_rgb_frame(live)

    ov = OverviewAgentView()
    ov.resize(1320, 840)
    ov.show()
    ov.set_camera_provider(cyc.env.render_rgb)
    ov.set_frame(fr)
    ov._dirty = True
    ov.update()
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
    rs = [r for r, _g, _b in samples]
    gs = [g for _r, g, _b in samples]
    bs = [b for _r, _g, b in samples]
    color_std = float(np.std(rs) + np.std(gs) + np.std(bs)) / 3.0
    assert color_std > 15.0, f"body too uniform for real camera (std={color_std:.1f})"
    top = Counter(samples).most_common(1)[0][0]
    teal = (26, 188, 156)
    amber = (243, 156, 18)
    assert top != teal and top != amber, f"body looks like obs fallback bars ({top})"


def test_paint_prefers_live_over_green_cache(qt_app):
    """Cached green pixmap must not mask a valid live frame from provider."""
    from PyQt5 import QtGui

    ov = OverviewAgentView()
    ov.resize(800, 600)
    ov.show()
    f = _reacher_frame()
    green = np.full((120, 160, 3), (0, 255, 0), dtype=np.uint8)
    ov._camera_pixmap = QtGui.QPixmap.fromImage(
        rgb_frame_to_qimage(green).scaled(160, 120))
    ov._camera_numpy = green
    live = np.random.randint(20, 200, (120, 160, 3), dtype=np.uint8)
    ov.set_camera_provider(lambda: live.copy())
    ov.set_frame(f)
    ov.repaint_if_dirty()
    qt_app.processEvents()
    assert ov._camera_numpy is not None
    assert not is_glitchy_rgb_frame(ov._camera_numpy)
    assert not np.array_equal(ov._camera_numpy, green)


def test_reacher_schematic_fallback(qt_app):
    """When provider fails, draw 2D schematic instead of solid green."""
    ov = OverviewAgentView()
    ov.resize(800, 600)
    ov.show()
    f = _reacher_frame()
    f.env_frame = None
    ov.set_camera_provider(lambda: None)
    ov.set_frame(f)
    ov._camera_fail_count = 2
    ov.repaint_if_dirty()
    qt_app.processEvents()
    body = ov.main_body_rect()
    img = ov.grab().toImage()
    samples = []
    for x in range(body.x() + 20, body.right() - 20, 30):
        for y in range(body.y() + 20, body.bottom() - 60, 30):
            c = img.pixelColor(x, y)
            samples.append((c.red(), c.green(), c.blue()))
    assert samples
    solid_green = sum(1 for r, g, b in samples if g > 240 and r < 20 and b < 20)
    assert solid_green < len(samples) * 0.5, "body should not be solid green slab"


def test_camera_schematic_mode_forces_2d(qt_app):
    ov = OverviewAgentView()
    ov.resize(800, 600)
    ov.show()
    ov.set_camera_provider(
        lambda: np.full((64, 80, 3), (0, 255, 0), dtype=np.uint8),
        mode="schematic",
    )
    ov.set_frame(_reacher_frame())
    ov.repaint_if_dirty()
    qt_app.processEvents()
    img = ov.grab().toImage()
    body = ov.main_body_rect()
    samples = []
    for x in range(body.x() + 20, body.right() - 20, 30):
        for y in range(body.y() + 20, body.bottom() - 60, 30):
            c = img.pixelColor(x, y)
            samples.append((c.red(), c.green(), c.blue()))
    solid_green = sum(1 for r, g, b in samples if g > 240 and r < 20 and b < 20)
    assert solid_green < len(samples) * 0.5
    assert ov._camera_gl_disabled is True


def test_camera_provider_called_on_set_frame(qt_app):
    import threading

    calls = {"n": 0, "threads": []}

    def _provider():
        calls["n"] += 1
        calls["threads"].append(threading.current_thread().name)
        arr = np.random.randint(30, 180, (64, 80, 3), dtype=np.uint8)
        return arr

    ov = OverviewAgentView()
    ov.resize(400, 300)
    ov.show()
    ov.set_camera_provider(_provider)
    ov.set_frame(_reacher_frame())
    qt_app.processEvents()
    assert calls["n"] >= 1
    assert all("Main" in t or t == "MainThread" for t in calls["threads"])
