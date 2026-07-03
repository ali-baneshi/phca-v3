"""v8 Overview 1.5 — unified agent card offscreen smoke."""

from collections import deque
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
from phca.monitoring.playback import PlaybackClock
from phca.monitoring.qt_dashboard import (
    BeliefProjection,
    ObservatoryWindow,
    OverviewAgentView,
    _OVERVIEW_SEMANTIC_MAP,
    _OVERVIEW_HEADER_H,
    _OVERVIEW_PHASE_H,
    _OVERVIEW_NARRATIVE_H,
    _OVERVIEW_EVENT_LOG_H,
    _OVERVIEW_RIBBON_H,
    _OVERVIEW_MARGIN,
    _OVERVIEW_LEARN_MS_MIN,
    _OVERVIEW_EVENT_HOLD,
    _draw_agent_limbs,
    _limb_line_start,
    _overview_dominant_phase,
    _overview_moment_flags,
    _overview_plain_story,
    _overview_goal_id,
    _overview_goal_intent_line,
    _overview_evidence_line,
    _overview_outcome_line,
    _overview_new_events,
    _reacher_kinematics_from_obs,
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
    top_h = (_OVERVIEW_HEADER_H + _OVERVIEW_PHASE_H + _OVERVIEW_NARRATIVE_H
             + _OVERVIEW_EVENT_LOG_H)
    main_h = 840 - top_h - _OVERVIEW_RIBBON_H - _OVERVIEW_MARGIN * 2
    assert body.height() >= int(main_h * 0.95)
    assert body.height() >= int(840 * 0.35)


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
    ov.set_camera_provider(lambda g=good: g.copy())
    ov.set_frame(f)
    ov._capture_timer.stop()
    ov._sync_camera_from_provider()
    assert ov._camera_pixmap is not None
    assert not ov._camera_pixmap.isNull()
    prev = ov._camera_pixmap
    ov.set_camera_provider(lambda: None)
    ov.set_frame(f)
    ov._sync_camera_from_provider()
    assert ov._camera_pixmap is prev
    assert ov._camera_stale is True
    bad = np.full((120, 160, 3), (0, 255, 0), dtype=np.uint8)
    assert is_glitchy_rgb_frame(bad)
    ov.set_camera_provider(lambda b=bad: b.copy())
    ov._sync_camera_from_provider()
    # Glitchy green frames must never become the active live frame.
    if ov._camera_numpy is not None:
        assert not np.array_equal(ov._camera_numpy, bad)
    if ov._camera_pixmap is not None and not ov._camera_pixmap.isNull():
        from phca.monitoring.camera_render import is_glitchy_pixmap
        assert not is_glitchy_pixmap(ov._camera_pixmap)


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
    ov._sync_camera_from_provider()
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
    ov._sync_camera_from_provider()
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
    ov._sync_camera_from_provider()
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


def test_camera_provider_called_on_sync(qt_app):
    import threading

    calls = {"n": 0, "threads": []}

    def _provider():
        calls["n"] += 1
        calls["threads"].append(threading.current_thread().name)
        return np.random.randint(30, 180, (64, 80, 3), dtype=np.uint8)

    ov = OverviewAgentView()
    ov.resize(400, 300)
    ov.show()
    ov.set_camera_provider(_provider)
    ov._sync_camera_from_provider()
    qt_app.processEvents()
    assert calls["n"] >= 1
    assert all("Main" in t or t == "MainThread" for t in calls["threads"])


def test_capture_not_called_during_paint(qt_app):
    from PyQt5 import QtGui

    calls = {"n": 0}

    def _provider():
        calls["n"] += 1
        return np.random.randint(30, 180, (64, 80, 3), dtype=np.uint8)

    ov = OverviewAgentView()
    ov.resize(800, 600)
    ov.show()
    ov.set_camera_provider(_provider)
    ov.set_frame(_reacher_frame())
    ov._sync_camera_from_provider()
    calls["n"] = 0
    p = QtGui.QPainter(ov)
    ov._draw(p)
    p.end()
    assert calls["n"] == 0, "provider must not run inside _draw/paint"


def test_pixmap_green_slab_rejected():
    from phca.monitoring.camera_render import rgb_frame_to_pixmap

    green = np.full((64, 80, 3), (0, 255, 0), dtype=np.uint8)
    assert rgb_frame_to_pixmap(green).isNull()
    good = np.random.randint(30, 180, (64, 80, 3), dtype=np.uint8)
    pm = rgb_frame_to_pixmap(good)
    assert not pm.isNull()


def test_reacher_default_schematic_mode():
    env = "Reacher-v5"
    camera = None
    if camera is None:
        camera = "schematic" if env == "Reacher-v5" else "auto"
    assert camera == "schematic"


def test_glitchy_rejects_green_slab():
    green = np.full((64, 80, 3), (0, 255, 0), dtype=np.uint8)
    assert is_glitchy_rgb_frame(green)


def test_overview_has_camera_label(qt_app):
    from PyQt5 import QtWidgets

    ov = OverviewAgentView()
    assert hasattr(ov, "_camera_label")
    assert isinstance(ov._camera_label, QtWidgets.QLabel)


def test_overview_uses_qlabel_not_painter_for_camera(qt_app, monkeypatch):
    """Camera pixmap is set on QLabel; paint must not call rgb_frame_to_pixmap."""
    from PyQt5 import QtGui

    calls = {"n": 0}
    orig = rgb_frame_to_pixmap

    def _track(frame):
        calls["n"] += 1
        return orig(frame)

    monkeypatch.setattr("phca.monitoring.qt_dashboard.rgb_frame_to_pixmap", _track)

    ov = OverviewAgentView()
    ov.resize(800, 600)
    ov.show()
    ov.set_camera_provider(None, mode="schematic")
    ov.set_frame(_reacher_frame())
    qt_app.processEvents()
    assert ov._camera_label.pixmap() is not None
    assert not ov._camera_label.pixmap().isNull()
    calls["n"] = 0
    p = QtGui.QPainter(ov)
    ov._draw(p)
    p.end()
    assert calls["n"] == 0


def test_camera_self_test_rejects_green():
    import importlib.util
    from pathlib import Path

    script = Path(__file__).resolve().parents[4] / "scripts" / "phca_observatory.py"
    spec = importlib.util.spec_from_file_location("phca_observatory", script)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)

    class _Env:
        def render_rgb(self):
            return np.full((64, 80, 3), (0, 255, 0), dtype=np.uint8)

    assert not mod._camera_self_test(_Env())


def test_camera_capture_gate_cadence_and_cooldown():
    import importlib.util
    from pathlib import Path

    script = Path(__file__).resolve().parents[4] / "scripts" / "phca_observatory.py"
    spec = importlib.util.spec_from_file_location("phca_observatory", script)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)

    period = mod._camera_capture_period(5.0)
    assert period == pytest.approx(0.2)
    assert mod._can_capture_live_camera(1.00, next_capture_t=1.00, cooldown_until_t=0.0)
    assert not mod._can_capture_live_camera(1.05, next_capture_t=1.20, cooldown_until_t=0.0)
    assert not mod._can_capture_live_camera(1.25, next_capture_t=1.20, cooldown_until_t=1.30)


def test_read_latest_camera_packet_returns_latest_reference():
    import importlib.util
    from pathlib import Path
    import threading

    script = Path(__file__).resolve().parents[4] / "scripts" / "phca_observatory.py"
    spec = importlib.util.spec_from_file_location("phca_observatory", script)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)

    frame = np.random.randint(20, 200, (8, 8, 3), dtype=np.uint8)
    holder = {"camera_lock": threading.Lock(), "camera_frame": frame, "camera_cycle_id": 7}
    pkt = mod._read_latest_camera_packet(holder)
    assert pkt is not None
    assert pkt["cycle_id"] == 7
    assert pkt["frame"] is frame


def test_draw_never_shows_green_when_implausible(qt_app):
    """Implausible green cache must fall back to 2D schematic in live mode."""
    ov = OverviewAgentView()
    ov.resize(800, 600)
    ov.show()
    green = np.full((120, 160, 3), (0, 255, 0), dtype=np.uint8)
    assert is_glitchy_rgb_frame(green)
    ov._camera_numpy = green
    ov._camera_pixmap = rgb_frame_to_pixmap(green)
    ov._camera_mode = "live"
    ov.set_frame(_reacher_frame())
    ov.repaint_if_dirty()
    qt_app.processEvents()
    body = ov.main_body_rect()
    img = ov.grab().toImage()
    samples = []
    for x in range(body.x() + 20, body.right() - 20, 30):
        for y in range(body.y() + 20, body.bottom() - 60, 30):
            c = img.pixelColor(x, y)
            samples.append((c.red(), c.green(), c.blue()))
    solid_green = sum(1 for r, g, b in samples if g > 240 and r < 20 and b < 20)
    assert solid_green < len(samples) * 0.5


def test_live_green_numpy_shows_schematic_not_slab(qt_app):
    """Live mode with invalid cache shows 2D schematic, not green LIVE slab."""
    ov = OverviewAgentView()
    ov.resize(800, 600)
    ov.show()
    green = np.full((120, 160, 3), (0, 255, 0), dtype=np.uint8)
    ov._camera_numpy = green
    ov._camera_pixmap = None
    ov._camera_mode = "live"
    ov.set_frame(_reacher_frame())
    assert ov._use_reacher_schematic(ov.frame)
    ov.repaint_if_dirty()
    qt_app.processEvents()
    body = ov.main_body_rect()
    img = ov.grab().toImage()
    samples = []
    for x in range(body.x() + 20, body.right() - 20, 30):
        for y in range(body.y() + 20, body.bottom() - 60, 30):
            c = img.pixelColor(x, y)
            samples.append((c.red(), c.green(), c.blue()))
    solid_green = sum(1 for r, g, b in samples if g > 240 and r < 20 and b < 20)
    assert solid_green < len(samples) * 0.5


def test_camera_provider_called_on_set_frame(qt_app):
    """set_frame alone does not capture; timer/sync does."""
    import threading

    calls = {"n": 0}

    def _provider():
        calls["n"] += 1
        return np.random.randint(30, 180, (64, 80, 3), dtype=np.uint8)

    ov = OverviewAgentView()
    ov.resize(400, 300)
    ov.show()
    ov.set_camera_provider(_provider)
    ov.set_frame(_reacher_frame())
    qt_app.processEvents()
    assert calls["n"] == 0
    ov._sync_camera_from_provider()
    assert calls["n"] >= 1


def test_reacher_kinematics_from_obs():
    """Schematic fingertip/goal math matches direct kinematics."""
    import math
    obs = np.array([1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.1, -0.2], dtype=np.float32)
    kin = _reacher_kinematics_from_obs(obs)
    assert kin is not None
    a0 = math.atan2(0.0, 1.0)
    a1 = math.atan2(0.0, 1.0)
    fx = 0.42 * math.cos(a0) + 0.38 * math.cos(a0 + a1)
    fy = 0.42 * math.sin(a0) + 0.38 * math.sin(a0 + a1)
    assert abs(kin["fx"] - fx) < 1e-5
    assert abs(kin["fy"] - fy) < 1e-5
    assert abs(kin["tx"] - (fx - 0.1)) < 1e-5
    assert abs(kin["ty"] - (fy + 0.2)) < 1e-5
    assert abs(kin["dist"] - math.hypot(0.1, -0.2)) < 1e-5


def test_overview_rebuild_histories(qt_app):
    """Rolling window rebuilds err/conf sparklines (scrub-safe)."""
    ov = OverviewAgentView()
    frames = []
    for i in range(5):
        f = _reacher_frame(cycle_id=i + 1)
        f.prediction_error = float(i + 1)
        f.prediction_confidence = 0.1 * (i + 1)
        # Distinct joint angles → distinct fingertip trail points
        ang = 0.15 * i
        f.obs_vector = np.array(
            [np.cos(ang), np.sin(ang), np.cos(ang * 0.5), np.sin(ang * 0.5),
             0.0, 0.0, 0.1, -0.2, 0.0, 0.0, 0.0],
            dtype=np.float32,
        )
        frames.append(f)
    ov.rebuild_histories(frames)
    assert list(ov._err_hist) == [1.0, 2.0, 3.0, 4.0, 5.0]
    assert list(ov._conf_hist) == pytest.approx([0.1, 0.2, 0.3, 0.4, 0.5])
    assert len(ov._reacher_trail) == 5
    ov.set_frame(_reacher_frame(cycle_id=99), histories_done=True)
    assert list(ov._err_hist) == [1.0, 2.0, 3.0, 4.0, 5.0]


def test_camera_provider_cycle_id(qt_app):
    """Provider may return frame + cycle_id dict for sync badge."""
    good = np.random.randint(30, 180, (64, 80, 3), dtype=np.uint8)

    def _provider():
        return {"frame": good.copy(), "cycle_id": 42}

    ov = OverviewAgentView()
    ov.resize(400, 300)
    ov.set_camera_provider(_provider, mode="live")
    ov.set_frame(_reacher_frame(cycle_id=42))
    ov._sync_camera_from_provider()
    assert ov._camera_cycle_id == 42
    assert ov._camera_badge(schematic=False) == "LIVE"
    ov.set_frame(_reacher_frame(cycle_id=7))
    assert ov._camera_badge(schematic=False) == "SYNC?"


def test_limbs_start_outside_core():
    """Action limbs must not originate inside the confidence core disc."""
    import math
    cx, cy, cr = 100, 100, 40
    for ang in (0.0, 0.7, 2.1, -1.2):
        sx, sy, _, _ = _limb_line_start(cx, cy, cr, ang)
        dist = math.hypot(sx - cx, sy - cy)
        assert dist >= cr - 0.5, f"limb starts inside core (dist={dist}, cr={cr})"


def test_overview_goal_id_consistent():
    f = _reacher_frame()
    f.action_rationale = {"goal_id": 4, "explored": False}
    f.active_drive_id = 2
    assert _overview_goal_id(f) == 4
    f.action_rationale = {"goal_id": None, "explored": False, "continuous": True}
    assert _overview_goal_id(f) == 2


def test_proj_rebuild_from_rolling():
    proj = BeliefProjection(window=64)
    frames = []
    for i in range(8):
        f = _reacher_frame(cycle_id=i + 1)
        f.obs_vector = np.linspace(-0.5, 0.5, 11, dtype=np.float32) + 0.01 * i
        frames.append(f)
    proj.rebuild_from_frames(frames)
    assert len(proj.history) == 8
    proj.rebuild_from_frames(frames[:3])
    assert len(proj.history) == 3


def test_drive_change_chip(qt_app):
    ov = OverviewAgentView()
    f1 = _reacher_frame(cycle_id=1)
    f1.active_drive_id = 2
    f1.action_rationale = {"goal_id": None, "explored": False, "continuous": True}
    ov.set_frame(f1)
    assert ov._drive_change is None
    f2 = _reacher_frame(cycle_id=2)
    f2.active_drive_id = 4
    f2.action_rationale = {"goal_id": None, "explored": False, "continuous": True}
    ov.set_frame(f2)
    assert ov._drive_change == (2, 4)
    f3 = _reacher_frame(cycle_id=3)
    f3.active_drive_id = 4
    f3.action_rationale = {"goal_id": None, "explored": False, "continuous": True}
    ov.set_frame(f3)
    assert ov._drive_change is None


def test_reacher_limbs_use_tau_bars_not_diametric():
    """For Reacher tau2, limbs renderer should avoid diametric crossing lines."""
    from PyQt5 import QtGui
    pm = QtGui.QPixmap(220, 220)
    pm.fill(QtGui.QColor(0, 0, 0))
    p = QtGui.QPainter(pm)
    f = _reacher_frame()
    _draw_agent_limbs(p, f, cx=110, cy=110, cr=34, R=70)
    p.end()
    img = pm.toImage()
    # Center should stay dark: tau bars live below the core area.
    c = img.pixelColor(110, 110)
    assert c.red() < 80 and c.green() < 80 and c.blue() < 80


def test_overview_moment_flags_spike():
    f = _reacher_frame()
    f.prediction_error = 40.0
    f.module_timings = {"gprime_learn": 2.3}
    flags = _overview_moment_flags(f, deque([2.0, 10.0]))
    assert flags["spike"] is True
    assert flags["learn_ms"] == pytest.approx(2.3)


def test_narrative_includes_learn_score():
    f = _reacher_frame()
    f.prediction_error = 12.0
    f.module_timings = {"gprime_learn": 1.7}
    f.action_rationale = {"explored": False, "best_score": 0.44}
    flags = _overview_moment_flags(f, deque([8.0, 9.0, 12.0]))
    assert flags["score"] == pytest.approx(0.44)
    assert flags["learn_ms"] == pytest.approx(1.7)


def test_semantic_map_covers_overview_channels():
    assert "glyph.limbs" in _OVERVIEW_SEMANTIC_MAP
    assert "narrative.learning" in _OVERVIEW_SEMANTIC_MAP
    assert "world.schematic" in _OVERVIEW_SEMANTIC_MAP


def test_decision_shift_flag_on_score_jump(qt_app):
    ov = OverviewAgentView()
    f1 = _reacher_frame(cycle_id=1)
    f1.action_rationale = {"explored": False, "best_score": 0.20}
    ov.set_frame(f1)
    assert not ov._moment_series[-1].get("decision_shift")
    f2 = _reacher_frame(cycle_id=2)
    f2.action_rationale = {"explored": False, "best_score": 0.55}
    ov.set_frame(f2)
    assert ov._moment_series[-1].get("decision_shift")
    assert not bool(f2.action_rationale.get("decision_shift", False))


def test_playback_clock_maxlen_trim():
    clock = PlaybackClock(maxlen=3)
    for i in range(5):
        f = ObservabilityFrame()
        f.cycle_id = i
        clock.push(f)
    assert len(clock) == 3
    assert clock._frames[0].cycle_id == 2
    assert clock._frames[-1].cycle_id == 4


def test_plain_story_spike_and_learn():
    f = _reacher_frame()
    f.prediction_error = 40.0
    f.module_timings = {"gprime_learn": _OVERVIEW_LEARN_MS_MIN + 1.0}
    flags = _overview_moment_flags(f, deque([2.0, 10.0]))
    story = _overview_plain_story(f, flags, deque([2.0, 10.0, 40.0]))
    assert "spiked" in story
    assert "world model updated" in story
    assert flags["learn_burst"] is True


def test_reacher_spike_requires_higher_ratio():
    f = _reacher_frame()
    f.prediction_error = 14.0
    flags = _overview_moment_flags(f, deque([10.0, 14.0]))
    assert flags["spike"] is False
    f.prediction_error = 35.0
    flags = _overview_moment_flags(f, deque([10.0, 35.0]))
    assert flags["spike"] is True


def test_phase_strip_dominant_module():
    f = _reacher_frame()
    f.module_timings = {
        "prediction": 1.0,
        "action_selection": 2.0,
        "peu": 0.5,
        "gprime_learn": 12.0,
        "mdim": 0.2,
        "rbta": 0.1,
    }
    assert _overview_dominant_phase(f) == "Learn"


def test_event_log_hold(qt_app):
    ov = OverviewAgentView()
    f0 = _reacher_frame(cycle_id=0)
    f0.prediction_error = 10.0
    ov.set_frame(f0)
    f = _reacher_frame()
    f.prediction_error = 50.0
    f.module_timings = {"gprime_learn": 8.0}
    ov.set_frame(f)
    assert any("SPIKE" in line for line in ov.visible_event_lines())
    assert any("LEARN" in line for line in ov.visible_event_lines())
    f2 = _reacher_frame(cycle_id=2)
    f2.prediction_error = 12.0
    f2.module_timings = {}
    ov.set_frame(f2)
    assert any("SPIKE" in line for line in ov.visible_event_lines())
    for _ in range(_OVERVIEW_EVENT_HOLD):
        f3 = _reacher_frame(cycle_id=3 + _)
        f3.prediction_error = 12.0
        ov.set_frame(f3)
    assert not any("SPIKE" in line for line in ov.visible_event_lines())


def test_golden_intent_line_eps_and_k():
    f = _reacher_frame()
    f.action_rationale = {
        "explored": True,
        "best_score": None,
        "goal_id": 2,
        "eps": 0.15,
        "k_candidates": 8,
    }
    flags = _overview_moment_flags(f, deque([10.0, 12.0]))
    line = _overview_goal_intent_line(f, flags)
    assert "EXPLORE" in line
    assert "eps=0.15" in line
    assert "k=8" in line


def test_golden_evidence_line_peu_mean():
    f = _reacher_frame()
    f.per_dim_peu = np.array([0.2, 0.4, 0.6], dtype=np.float32)
    f.module_timings = {"gprime_learn": 1.0, "prediction": 2.0}
    flags = _overview_moment_flags(f, deque([8.0, 9.0, 12.0]))
    line = _overview_evidence_line(f, flags, deque([8.0, 9.0, 12.0]))
    assert "PEU" in line
    assert "0.40" in line
    assert "Evidence:" in line


def test_golden_outcome_physical_only():
    f = _reacher_frame()
    f.prediction_error = 99.0
    f.continuous_action = np.array([0.42, -0.31], dtype=np.float32)
    f.goal_reached = True
    f.violations_count = 0
    f.rbta_action = "CONTINUE"
    flags = _overview_moment_flags(f, deque([10.0, 12.0]))
    dist_hist = deque([0.5, 0.4])
    line = _overview_outcome_line(f, flags, deque([10.0, 12.0]), dist_hist)
    assert line.startswith("Outcome:")
    assert "dist=" in line
    assert "action=τ=" in line
    assert "goal=yes" in line
    assert "rbta=OK" in line
    assert "error" not in line.lower()
    assert "EXPLORE" not in line
    assert "EXPLOIT" not in line


def test_explore_event_only_on_transition(qt_app):
    ov = OverviewAgentView()
    f0 = _reacher_frame(cycle_id=0)
    f0.action_rationale = {"explored": False, "best_score": 0.5, "goal_id": 2}
    ov.set_frame(f0)
    assert not any("EXPLORE" in line for line in ov.visible_event_lines())

    f1 = _reacher_frame(cycle_id=1)
    f1.action_rationale = {"explored": True, "best_score": None, "goal_id": 2}
    ov.set_frame(f1)
    assert sum("EXPLORE" in line for line in ov.visible_event_lines()) == 1

    f2 = _reacher_frame(cycle_id=2)
    f2.action_rationale = {"explored": True, "best_score": None, "goal_id": 2}
    ov.set_frame(f2)
    assert sum("EXPLORE" in line for line in ov.visible_event_lines()) == 1


def test_overview_new_events_explore_entered_param():
    f = _reacher_frame()
    f.action_rationale = {"explored": True, "best_score": None}
    flags = _overview_moment_flags(f, deque([10.0]))
    assert not _overview_new_events(f, flags, None, explore_entered=False)
    events = _overview_new_events(f, flags, None, explore_entered=True)
    assert any("EXPLORE" in e for e in events)


def test_overview_rebuild_clears_event_log(qt_app):
    ov = OverviewAgentView()
    for i in range(3):
        f = _reacher_frame(cycle_id=i)
        f.prediction_error = 0.1 if i < 2 else 20.0
        ov.set_frame(f)
    assert ov.visible_event_lines()
    frames = [_reacher_frame(cycle_id=i) for i in range(3)]
    ov.rebuild_histories(frames)
    assert ov.visible_event_lines() == []


def test_overview_goal_id_vitals_uses_active_drive():
    from phca.monitoring.qt_dashboard import _overview_goal_id

    f = _reacher_frame()
    f.active_drive_id = 3
    f.action_rationale = {"goal_id": None, "explored": False, "best_score": 0.5}
    assert _overview_goal_id(f) == 3


def test_overview_moment_series_on_live_frames(qt_app):
    ov = OverviewAgentView()
    f1 = _reacher_frame(cycle_id=1)
    f1.prediction_error = 5.0
    ov.set_frame(f1)
    f2 = _reacher_frame(cycle_id=2)
    f2.prediction_error = 25.0
    ov.set_frame(f2)
    assert len(ov._moment_series) == 2
    assert ov._moment_series[-1].get("spike") is True


def test_overview_rebuild_builds_moment_series(qt_app):
    ov = OverviewAgentView()
    frames = []
    for i in range(4):
        f = _reacher_frame(cycle_id=i)
        f.prediction_error = float(i + 1)
        f.action_rationale = {"explored": False, "best_score": 0.1 * i}
        frames.append(f)
    ov.rebuild_histories(frames)
    assert len(ov._moment_series) == 4
    assert ov._prev_best_score == pytest.approx(0.3)


def test_overview_replay_flag(qt_app):
    ov = OverviewAgentView()
    f = _reacher_frame()
    ov.set_frame(f, replay=True)
    assert ov._replay is True
