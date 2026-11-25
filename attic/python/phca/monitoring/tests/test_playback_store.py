"""Tests for playback rebuild semantics and observability-store drains."""
from __future__ import annotations

import json


from phca.monitoring.observability import ObservabilityFrame, ObservabilityStore
from phca.monitoring.playback import PlaybackClock
from phca.monitoring.render import frame_from_json


def _frame(cycle_id: int) -> ObservabilityFrame:
    f = ObservabilityFrame()
    f.cycle_id = cycle_id
    return f


def _rich_frame(cycle_id: int) -> ObservabilityFrame:
    import numpy as np

    f = _frame(cycle_id)
    f.env_kind = "grid"
    f.grid = np.zeros((5, 5), dtype=np.int32)
    f.agent_pos = (cycle_id % 5, (cycle_id + 1) % 5)
    f.goal_pos = (4, 4)
    f.predicted_state = np.zeros(25, dtype=np.float32)
    f.predicted_state[f.agent_pos[0] * 5 + f.agent_pos[1]] = 0.8
    f.prediction_error = float(cycle_id)
    f.prediction_confidence = 0.5
    f.module_timings = {"prediction": 1.0 + cycle_id, "action_selection": 0.5}
    f.rbta_bounds = {"G'": {"time": 0.01}, "ACTION": {"time": 0.01}}
    f.action_rationale = {"chosen_idx": 0, "best_score": 0.2 + cycle_id, "explored": False}
    f.candidate_scores = [0.2 + cycle_id, 0.1]
    f.drive_levels = [0.1] * 6
    f.drive_targets = [0.2] * 6
    f.drive_deficits = [0.1] * 6
    f.goal_history = [cycle_id % 6 + 1]
    f.episode_count = cycle_id
    f.fact_count = cycle_id * 2
    f.m3_cap = 100
    f.m4_cap = 200
    f.rss_bytes = 10_000 + cycle_id
    f.latency_ms = 5.0
    f.belief_entropies = {"G'": 0.5}
    f.gprime_uncertainty = np.ones(25, dtype=np.float32) * 0.1
    f.obs_vector = np.zeros(25, dtype=np.float32)
    f.obs_vector[f.agent_pos[0] * 5 + f.agent_pos[1]] = 1.0
    f.sanitized_state = f.obs_vector.copy()
    return f


def test_observability_store_frames_after_not_tail_limited():
    store = ObservabilityStore(maxlen=400)
    for i in range(300):
        store.push(_frame(i))
    frames = store.frames_after(0)
    assert len(frames) == 299
    assert frames[0].cycle_id == 1
    assert frames[-1].cycle_id == 299


def test_playback_clock_rebuilds_only_on_seek_or_jump():
    frames = [_frame(i) for i in range(5)]
    clock = PlaybackClock(mode="replay")
    clock.set_frames(frames)
    events = []

    def on_update(frame, rolling, error):
        events.append((
            frame.cycle_id,
            rolling is not None,
            len(rolling) if rolling is not None else 0,
        ))

    clock.on_update = on_update
    clock.seek(0)
    clock.step()
    clock.seek(3)
    clock.step()
    clock.step()

    assert events == [
        (0, True, 1),
        (1, False, 0),
        (3, True, 4),
        (4, False, 0),
        (4, False, 0),
    ]


def test_playback_clock_seek_uses_rolling_window_prefix():
    frames = [_frame(i) for i in range(250)]
    clock = PlaybackClock(mode="replay")
    clock.set_frames(frames)
    events = []

    def on_update(frame, rolling, error):
        events.append((
            frame.cycle_id,
            rolling[0].cycle_id if rolling else None,
            rolling[-1].cycle_id if rolling else None,
            len(rolling) if rolling else 0,
        ))

    clock.on_update = on_update
    clock.seek(249)
    clock.seek(100)
    clock.step()

    assert events == [
        (249, 49, 249, 201),
        (100, 0, 100, 101),
        (101, None, None, 0),
    ]


def test_playback_clock_review_mode_uses_full_prefix():
    from phca.monitoring.cognitive_panels import rolling_prefix

    frames = [_frame(i) for i in range(500)]
    assert len(rolling_prefix(frames, 280, review_mode=False)) == 201
    assert len(rolling_prefix(frames, 280, review_mode=True)) == 281

    clock = PlaybackClock(mode="live")
    clock.review_mode = True
    clock.set_frames(frames)
    events = []

    def on_update(frame, rolling, error):
        events.append(len(rolling) if rolling else 0)

    clock.on_update = on_update
    clock.seek(280)
    assert events == [281]


def test_review_scrub_full_prefix_rebuilds_all_tabs(qt_app):
    from phca.monitoring.qt_dashboard import ObservatoryWindow

    frames = [_rich_frame(i) for i in range(500)]
    win = ObservatoryWindow()
    win._review_mode = True
    clock = PlaybackClock(mode="live")
    clock.review_mode = True
    clock.set_frames(frames)
    clock.on_update = lambda f, rolling, err: win.controller.update(f, rolling, err)
    win._transport = type("_T", (), {"clock": clock})()

    clock.seek(280)
    qt_app.processEvents()
    assert len(win.proj.history) == 256  # BeliefProjection window maxlen
    assert win.proj.history[-1] is not None
    assert len(win.retention.m3) == 200  # deque maxlen
    assert win.retention.m3[-1] == 280
    assert win.goals.frame.cycle_id == 280
    assert len(win.goals.drive_hist) == 200
    assert win.goals._prefix_len == 281


def test_replay_scrub_keeps_lazy_200_window(qt_app):
    """OBS-002: replay scrub rebuilds all tabs; Goals history capped at 200."""
    from phca.monitoring.qt_dashboard import ObservatoryWindow

    frames = [_rich_frame(i) for i in range(500)]
    win = ObservatoryWindow()
    clock = PlaybackClock(mode="replay")
    clock.set_frames(frames)
    clock.on_update = lambda f, rolling, err: win.controller.update(f, rolling, err)
    win._transport = type("_T", (), {"clock": clock})()

    win._tabs.setCurrentIndex(0)
    clock.seek(280)
    qt_app.processEvents()
    assert len(win.proj.history) == 201
    assert not win.controller._pending_tab_rebuilds
    win._tabs.setCurrentIndex(6)
    win.controller.on_tab_changed(6)
    assert win.goals.frame.cycle_id == 280
    assert len(win.goals.drive_hist) == 200


def test_live_scrub_rebuilds_all_tabs(qt_app):
    """OBS-002: live pause-scrub rebuilds all tabs without tab switch."""
    from phca.monitoring.qt_dashboard import ObservatoryWindow

    frames = [_rich_frame(i) for i in range(500)]
    win = ObservatoryWindow()
    clock = PlaybackClock(mode="live")
    clock.set_frames(frames)
    clock.paused = True
    clock.on_update = lambda f, rolling, err: win.controller.update(f, rolling, err)
    win._transport = type("_T", (), {"clock": clock})()

    win._tabs.setCurrentIndex(0)
    clock.seek(280)
    qt_app.processEvents()
    assert not win.controller._pending_tab_rebuilds
    assert win.goals.frame.cycle_id == 280
    assert len(win.goals.drive_hist) == 200


def test_goals_scrub_updates_heatmap_on_visible_tab(qt_app):
    from phca.monitoring.qt_dashboard import ObservatoryWindow

    frames = [_rich_frame(i) for i in range(120)]
    win = ObservatoryWindow()
    win._review_mode = True
    clock = PlaybackClock(mode="live")
    clock.review_mode = True
    clock.set_frames(frames)
    clock.on_update = lambda f, rolling, err: win.controller.update(f, rolling, err)
    win._transport = type("_T", (), {"clock": clock})()
    win._tabs.setCurrentIndex(6)

    clock.seek(100)
    qt_app.processEvents()
    assert win.goals.frame.cycle_id == 100
    assert len(win.goals.drive_hist) == 101

    clock.seek(50)
    qt_app.processEvents()
    assert win.goals.frame.cycle_id == 50
    assert len(win.goals.drive_hist) == 51


def test_review_scrub_3000_under_budget(qt_app):
    import time

    from phca.monitoring.qt_dashboard import ObservatoryWindow

    frames = [_rich_frame(i) for i in range(3000)]
    win = ObservatoryWindow()
    win._review_mode = True
    clock = PlaybackClock(mode="live")
    clock.review_mode = True
    clock.set_frames(frames)
    clock.on_update = lambda f, rolling, err: win.controller.update(f, rolling, err)
    win._transport = type("_T", (), {"clock": clock})()

    t0 = time.monotonic()
    for idx in (0, 1500, 2999, 500):
        clock.seek(idx)
        qt_app.processEvents()
    elapsed = time.monotonic() - t0
    assert elapsed < 4.0, f"review scrub took {elapsed:.2f}s (budget 4s)"
    assert len(win.proj.history) <= 2001
    assert win.goals.frame.cycle_id == 500


def test_dashboard_controller_scrub_rebuilds_visible_tab(qt_app):
    from phca.monitoring.qt_dashboard import ObservatoryWindow

    frames = [_rich_frame(i) for i in range(12)]
    win = ObservatoryWindow()
    clock = PlaybackClock(mode="replay")
    clock.set_frames(frames)
    clock.on_update = lambda f, rolling, err: win.controller.update(f, rolling, err)
    win._transport = type("_T", (), {"clock": clock})()

    clock.seek(9)
    assert len(win.overview._err_hist) == 10
    win._tabs.setCurrentIndex(1)
    win.controller.on_tab_changed(1)
    assert len(win.flow.heat) == 10
    win._tabs.setCurrentIndex(2)
    win.controller.on_tab_changed(2)
    assert len(win.cand.score_hist) == 10
    win._tabs.setCurrentIndex(3)
    win.controller.on_tab_changed(3)
    assert len(win.traj.trail) == 10
    win._tabs.setCurrentIndex(4)
    win.controller.on_tab_changed(4)
    assert len(win.retention.m3) == 10
    assert len(win.rbta_bounds._hist["G':time"]) == 10
    win._tabs.setCurrentIndex(5)
    win.controller.on_tab_changed(5)
    assert win.memory.frame is frames[9]
    win._tabs.setCurrentIndex(6)
    win.controller.on_tab_changed(6)
    assert len(win.goals.drive_hist) == 10

    win._tabs.setCurrentIndex(4)
    before = len(win.retention.m3)
    clock.step()
    win.controller.on_tab_changed(4)
    assert len(win.retention.m3) == before + 1
    clock.seek(3)
    win.controller.on_tab_changed(4)
    assert len(win.retention.m3) == 4
    assert list(win.retention.m3) == [0, 1, 2, 3]


def test_dashboard_controller_scrub_500_jsonl_frames_no_mutation(qt_app):
    from phca.monitoring.qt_dashboard import ObservatoryWindow

    raw_frames = [_rich_frame(i).to_json() for i in range(500)]
    raw_before = json.loads(json.dumps(raw_frames))
    frames = [frame_from_json(json.loads(json.dumps(obj))) for obj in raw_frames]
    frame_zero_before = frames[0].to_json()
    frame_last_before = frames[-1].to_json()
    win = ObservatoryWindow()
    clock = PlaybackClock(mode="replay")
    clock.set_frames(frames)
    clock.on_update = lambda f, rolling, err: win.controller.update(f, rolling, err)
    win._transport = type("_T", (), {"clock": clock})()

    for idx in (0, 250, 499, 10):
        clock.seek(idx)
        qt_app.processEvents()

    win._tabs.setCurrentIndex(4)
    win.controller.on_tab_changed(4)
    assert len(win.retention.m3) == 11
    assert list(win.retention.m3) == list(range(11))
    assert len(win.rbta_bounds._hist["G':time"]) == 11
    assert raw_frames == raw_before
    assert frames[0].to_json() == frame_zero_before
    assert frames[-1].to_json() == frame_last_before


def test_playback_clock_surfaces_update_error():
    clock = PlaybackClock(mode="replay")
    clock.set_frames([_frame(0), _frame(1)])

    def boom(frame, rolling, error):
        raise RuntimeError("bad frame")

    clock.on_update = boom
    clock.seek(0)
    assert clock.error is not None
    assert "bad frame" in clock.error


def test_scrub_tab_badges_show_spike(qt_app):
    from phca.monitoring.qt_dashboard import ObservatoryWindow

    frames = []
    for i in range(5):
        f = _rich_frame(i)
        f.prediction_error = 1.0
        frames.append(f)
    frames[4].prediction_error = 5.0

    win = ObservatoryWindow()
    clock = PlaybackClock(mode="replay")
    clock.set_frames(frames)
    clock.on_update = lambda f, rolling, err: win.controller.update(f, rolling, err)
    win._transport = type("_T", (), {"clock": clock})()

    clock.seek(4)
    qt_app.processEvents()

    tabs = win._tabs
    assert any("SPIKE" in tabs.tabText(i) for i in range(tabs.count()))


def test_abort_status_strip_and_incomplete_banner(qt_app):
    from phca.monitoring.cognitive_panels import data_contract_text
    from phca.monitoring.qt_dashboard import ObservatoryWindow

    win = ObservatoryWindow()
    win.set_session_context({"env": "Reacher-v5", "target_cycles": 1000, "recording": True})
    win.set_cycle_error("Action dimension mismatch. Expected (2,), found ()")
    win.mark_session_incomplete(recorded=167, requested=1000)
    win.set_verify_status("FAIL")
    win.enter_review_mode(resync_only=True)
    win.refresh_session_strip(jsonl_count=167)
    text = win._status_strip.text()
    assert "ABORTED" in text
    assert "Action dimension mismatch" in text
    assert win._session_incomplete is True
    assert data_contract_text(
        "overview", replay=False, review=True, incomplete=True,
    ).startswith("ABORTED —")


def test_scrub_3000_frames_under_budget(qt_app):
    import time

    from phca.monitoring.qt_dashboard import ObservatoryWindow

    frames = [_rich_frame(i) for i in range(3000)]
    win = ObservatoryWindow()
    clock = PlaybackClock(mode="replay")
    clock.set_frames(frames)
    clock.on_update = lambda f, rolling, err: win.controller.update(f, rolling, err)
    win._transport = type("_T", (), {"clock": clock})()

    t0 = time.monotonic()
    for idx in (0, 1500, 2999, 500):
        clock.seek(idx)
        qt_app.processEvents()
    elapsed = time.monotonic() - t0
    assert elapsed < 2.0, f"scrub took {elapsed:.2f}s (budget 2s)"


def test_rebuild_all_histories_populates_all_tabs(qt_app):
    from phca.monitoring.qt_dashboard import ObservatoryWindow

    frames = [_rich_frame(i) for i in range(12)]
    win = ObservatoryWindow()
    win.controller.rebuild_all_histories(frames)
    assert len(win.overview._err_hist) == 12
    assert len(win.flow.heat) == 12
    assert len(win.cand.score_hist) == 12
    assert len(win.retention.m3) == 12
    assert len(win.goals.drive_hist) == 12


def test_enter_review_resyncs_transport_range(qt_app):
    from phca.monitoring.qt_dashboard import ObservatoryWindow, _TransportBar

    frames = [_rich_frame(i) for i in range(50)]
    win = ObservatoryWindow()
    clock = PlaybackClock(mode="live")
    clock.set_frames(frames)
    transport = _TransportBar(clock)
    win.install_transport(transport)
    win.enter_review_mode(resync_only=True)
    assert transport.slider.maximum() == 49
    assert clock.cursor_int == 49
    assert clock.review_mode is True


def test_enter_review_resync_no_emit(qt_app):
    from phca.monitoring.qt_dashboard import ObservatoryWindow, _TransportBar

    frames = [_rich_frame(i) for i in range(10)]
    win = ObservatoryWindow()
    clock = PlaybackClock(mode="live")
    clock.set_frames(frames)
    calls = []
    clock.on_update = lambda f, r, e: calls.append(1)
    transport = _TransportBar(clock)
    win.install_transport(transport)
    win.enter_review_mode(resync_only=True)
    assert calls == []


def test_rebuild_all_histories_once_in_review_flow(qt_app):
    from phca.monitoring.qt_dashboard import ObservatoryWindow

    frames = [_rich_frame(i) for i in range(20)]
    win = ObservatoryWindow()
    win._review_mode = True
    counts = {"n": 0}
    orig = win.controller.rebuild_all_histories

    def spy(rolling):
        counts["n"] += 1
        return orig(rolling)

    win.controller.rebuild_all_histories = spy  # type: ignore[method-assign]
    win.enter_review_mode(resync_only=True)
    win.controller.rebuild_all_histories(frames)
    assert counts["n"] == 1


def test_retention_draw_small_height_no_paint_error(qt_app):
    from phca.monitoring.qt_dashboard import RetentionView

    view = RetentionView()
    view.resize(400, 100)
    frames = [_rich_frame(i) for i in range(5)]
    view.rebuild_histories(frames)
    view.set_frame(frames[-1], histories_done=True)
    view.show()
    qt_app.processEvents()
    assert view.repaint_count >= 1


def test_trajectory_draw_small_height_no_paint_error(qt_app):
    from phca.monitoring.qt_dashboard import ObservatoryWindow, TrajectoryView

    frames = [_rich_frame(i) for i in range(8)]
    win = ObservatoryWindow()
    win.proj.rebuild_from_frames(frames)
    view = TrajectoryView()
    view.set_projection(win.proj)
    f = frames[-1]
    f.env_kind = "reacher"
    f.grid = None
    f.agent_pos = None
    view.set_frame(f, review=True)
    view.resize(200, 80)
    view.show()
    qt_app.processEvents()
    assert view.repaint_count >= 1


def test_trajectory_chart_rect_within_widget(qt_app):
    from phca.monitoring.qt_dashboard import _traj_canvas_layout

    for h in (120, 180, 240):
        cl = _traj_canvas_layout(400, h, y0=16, is_grid=False)
        assert cl["chart_bot"] <= h
        assert cl["chart_top"] < cl["chart_bot"]
        assert cl["legend_y"] <= h
        assert cl["axis_y"] < cl["legend_y"]


def test_map_pt_clamps_to_chart(qt_app):
    from phca.monitoring.qt_dashboard import _map_pt

    bounds = (0.0, 1.0, 0.0, 1.0)
    x, y = _map_pt((2.5, -1.0), bounds, 10, 20, 110, 120)
    assert 10 <= x <= 110
    assert 20 <= y <= 120


def test_perdim_draw_small_height_no_paint_error(qt_app):
    from phca.monitoring.qt_dashboard import _PhasePortraitView

    frames = [_rich_frame(i) for i in range(12)]
    view = _PhasePortraitView()
    view.rebuild_histories(frames)
    view.set_frame(frames[-1], histories_done=True)
    view.resize(400, 180)
    view.show()
    qt_app.processEvents()
    assert view.repaint_count >= 1


def test_on_tab_changed_skips_lazy_rebuild_in_review(qt_app):
    from phca.monitoring.qt_dashboard import ObservatoryWindow

    frames = [_rich_frame(i) for i in range(30)]
    win = ObservatoryWindow()
    win._review_mode = True
    win.controller._pending_tab_rebuilds = {6}
    win.controller._last_rolling = frames[:5]
    calls = {"n": 0}
    orig = win.controller._rebuild_tab_histories

    def spy(rolling, tab_idx):
        calls["n"] += 1
        return orig(rolling, tab_idx)

    win.controller._rebuild_tab_histories = spy  # type: ignore[method-assign]
    win.controller.on_tab_changed(6)
    assert calls["n"] == 0
    assert len(win.goals.drive_hist) == 0


def test_rebuild_all_histories_sets_last_cycle(qt_app):
    from phca.monitoring.qt_dashboard import ObservatoryWindow

    frames = [_rich_frame(i) for i in range(20)]
    win = ObservatoryWindow()
    win._review_mode = True
    win.controller.rebuild_all_histories(frames)
    assert win.controller._last_cycle == 19


def test_rbta_bounds_small_height_no_paint_error(qt_app):
    from phca.monitoring.qt_dashboard import RBTABoundsView

    view = RBTABoundsView()
    view.resize(400, 80)
    frames = [_rich_frame(i) for i in range(5)]
    view.rebuild_histories(frames)
    view.set_frame(frames[-1], histories_done=True)
    view.show()
    qt_app.processEvents()
    assert view.repaint_count >= 1


def test_rbta_bounds_narrow_width_no_paint_error(qt_app):
    from phca.monitoring.qt_dashboard import RBTABoundsView

    view = RBTABoundsView()
    view.resize(400, 180)
    frames = [_rich_frame(i) for i in range(5)]
    view.rebuild_histories(frames)
    view.set_frame(frames[-1], histories_done=True)
    view.show()
    qt_app.processEvents()
    assert view.repaint_count >= 1


def test_footer_labels_within_widget_bounds(qt_app):
    from phca.monitoring.qt_dashboard import (
        RetentionView, TrajectoryView, _PhasePortraitView,
    )

    frames = [_rich_frame(i) for i in range(8)]
    from phca.monitoring.qt_dashboard import ObservatoryWindow
    win = ObservatoryWindow()
    win.proj.rebuild_from_frames(frames)
    traj = TrajectoryView()
    traj.set_projection(win.proj)
    f = frames[-1]
    f.env_kind = "reacher"
    f.grid = None
    f.agent_pos = None
    traj.set_frame(f, review=True)
    traj.resize(320, 140)
    traj.show()
    perdim = _PhasePortraitView()
    perdim.rebuild_histories(frames)
    perdim.set_frame(frames[-1], histories_done=True)
    perdim.resize(400, 200)
    perdim.show()
    ret = RetentionView()
    ret.rebuild_histories(frames)
    ret.set_frame(frames[-1], histories_done=True, review=True)
    ret.resize(320, 180)
    ret.show()
    qt_app.processEvents()
    assert traj.repaint_count >= 1
    assert perdim.repaint_count >= 1
    assert ret.repaint_count >= 1


def test_caption_elide_does_not_overflow_rect(qt_app):
    from phca.monitoring.qt_dashboard import RetentionView

    view = RetentionView()
    view.resize(320, 200)
    frames = [_rich_frame(i) for i in range(8)]
    view.rebuild_histories(frames)
    view.set_frame(frames[-1], histories_done=True, review=True)
    view.show()
    qt_app.processEvents()
    assert view.repaint_count >= 1
