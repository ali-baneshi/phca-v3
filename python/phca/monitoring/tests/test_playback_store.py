"""Tests for playback rebuild semantics and observability-store drains."""
from __future__ import annotations

from phca.monitoring.observability import ObservabilityFrame, ObservabilityStore
from phca.monitoring.playback import PlaybackClock


def _frame(cycle_id: int) -> ObservabilityFrame:
    f = ObservabilityFrame()
    f.cycle_id = cycle_id
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
