"""Tests for playback rebuild semantics and observability-store drains."""
from __future__ import annotations

import json

import pytest

from phca.monitoring.observability import ObservabilityFrame, ObservabilityStore
from phca.monitoring.playback import PlaybackClock
from phca.monitoring.render import frame_from_json


@pytest.fixture(scope="module")
def qt_app():
    from phca.monitoring.qt_dashboard import make_app

    app = make_app()
    yield app
    app.processEvents()


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


def test_dashboard_controller_scrub_rebuilds_all_panels(qt_app):
    from phca.monitoring.qt_dashboard import ObservatoryWindow

    frames = [_rich_frame(i) for i in range(12)]
    win = ObservatoryWindow()
    clock = PlaybackClock(mode="replay")
    clock.set_frames(frames)
    clock.on_update = lambda f, rolling, err: win.controller.update(f, rolling, err)
    win._transport = type("_T", (), {"clock": clock})()

    clock.seek(9)
    assert len(win.overview._err_hist) == 10
    assert len(win.flow.heat) == 10
    assert len(win.cand.score_hist) == 10
    assert len(win.traj.trail) == 10
    assert len(win.retention.m3) == 10
    assert len(win.rbta_bounds._hist["G':time"]) == 10
    assert win.memory.frame is frames[9]
    assert len(win.goals.drive_hist) == 10

    before = len(win.retention.m3)
    clock.step()
    assert len(win.retention.m3) == before + 1
    clock.seek(3)
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

    assert raw_frames == raw_before
    assert frames[0].to_json() == frame_zero_before
    assert frames[-1].to_json() == frame_last_before
    assert len(win.retention.m3) == 11
    assert list(win.retention.m3) == list(range(11))
    assert len(win.rbta_bounds._hist["G':time"]) == 11
