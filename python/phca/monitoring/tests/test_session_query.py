"""Phase 18 session query engine and CLI tests."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path


from phca.monitoring.observability import ObservabilityFrame
from phca.monitoring.session_query import (
    MomentQuery,
    export_matches_jsonl,
    navigate_match,
    query_frames,
)

_ROOT = Path(__file__).resolve().parents[4]
_FIXTURE_JSONL = Path(__file__).resolve().parent / "fixtures" / "reacher_short.jsonl"
_QUERY = _ROOT / "scripts" / "phca_query.py"


def _frame(**kwargs) -> ObservabilityFrame:
    f = ObservabilityFrame()
    f.cycle_id = kwargs.get("cycle_id", 0)
    f.env_kind = kwargs.get("env_kind", "mujoco_rgb")
    f.prediction_error = kwargs.get("prediction_error", 0.1)
    f.module_timings = kwargs.get("module_timings", {
        "prediction": 1.0, "action_selection": 2.0,
    })
    f.action_rationale = kwargs.get("action_rationale", {})
    f.rbta_bounds = kwargs.get("rbta_bounds", {})
    f.violations_count = kwargs.get("violations_count", 0)
    f.active_drive_id = kwargs.get("active_drive_id", 1)
    return f


def _spike_series(n: int = 10) -> list[ObservabilityFrame]:
    frames = []
    for i in range(n):
        if i < 5:
            err = 0.1
        elif i == 5:
            err = 12.0
        else:
            err = 11.0
        frames.append(_frame(cycle_id=i, prediction_error=err))
    return frames


def test_query_spike_on_synthetic():
    frames = _spike_series()
    matches = query_frames(frames, MomentQuery(spike=True))
    assert len(matches) == 1
    assert matches[0].cycle_id == 5


def test_query_and_filters():
    frames = [
        _frame(cycle_id=0, prediction_error=12.0, violations_count=0),
        _frame(cycle_id=1, prediction_error=12.0, violations_count=1),
        _frame(cycle_id=2, prediction_error=0.5, violations_count=1),
    ]
    # Only cycle 1 is spike AND violation
    frames[0].env_kind = "mujoco_rgb"
    frames[1].env_kind = "mujoco_rgb"
    frames[2].env_kind = "mujoco_rgb"
    # Fix spike: need history - use build_moment_series path via full series
    series_frames = _spike_series(8)
    series_frames.append(_frame(cycle_id=8, prediction_error=35.0, violations_count=1))
    matches = query_frames(series_frames, MomentQuery(spike=True, violation=True))
    assert all(m.flags.get("spike") and m.flags.get("violation") for m in matches)


def test_query_drive_and_cycle_range():
    frames = [
        _frame(cycle_id=i, action_rationale={"goal_id": 2 if i < 5 else 3})
        for i in range(10)
    ]
    q = MomentQuery(drive_id=3, cycle_min=5, cycle_max=8)
    matches = query_frames(frames, q)
    assert [m.cycle_id for m in matches] == [5, 6, 7, 8]


def test_query_near_bound_module():
    frames = [
        _frame(
            cycle_id=0,
            module_timings={"prediction": 3.0},
            rbta_bounds={"G'": {"time": 0.001}},
        ),
        _frame(cycle_id=1, module_timings={"prediction": 0.5}),
    ]
    matches = query_frames(frames, MomentQuery(near_bound_module="prediction"))
    assert len(matches) == 1
    assert matches[0].cycle_id == 0


def test_query_does_not_mutate_frames():
    frames = _spike_series(6)
    before = [f.to_json() for f in frames]
    query_frames(frames, MomentQuery(spike=True))
    after = [f.to_json() for f in frames]
    assert before == after


def test_query_3000_cycles_under_budget():
    frames = [_frame(cycle_id=i, prediction_error=0.1 + (i % 50) * 0.01) for i in range(3000)]
    frames[1500].prediction_error = 40.0
    t0 = time.monotonic()
    matches = query_frames(frames, MomentQuery(spike=True))
    elapsed = time.monotonic() - t0
    assert elapsed < 2.0
    assert len(matches) >= 1


def test_navigate_match():
    from phca.monitoring.session_query import MomentMatch

    matches = [MomentMatch(index=i, cycle_id=i) for i in (2, 5, 9)]
    assert navigate_match(matches, 0, 1) == 2
    assert navigate_match(matches, 3, 1) == 5
    assert navigate_match(matches, 9, 1) == 2
    assert navigate_match(matches, 5, -1) == 2


def _write_reacher_session(tmp_path: Path) -> Path:
    d = tmp_path / "reacher_sess"
    d.mkdir()
    lines = [ln for ln in _FIXTURE_JSONL.read_text().splitlines() if ln.strip()]
    meta = {
        "env": "Reacher-v5",
        "cycles": len(lines),
        "recorded_cycles": len(lines),
        "observability_schema_version": 1,
    }
    (d / "meta.json").write_text(json.dumps(meta))
    (d / "timeseries.jsonl").write_text("\n".join(lines) + "\n")
    return d


def test_phca_query_cli_fixture(tmp_path):
    sess = _write_reacher_session(tmp_path)
    env = {**dict(os.environ), "PYTHONPATH": str(_ROOT / "python")}
    r = subprocess.run(
        [sys.executable, str(_QUERY), str(sess), "--explore", "--count-only"],
        capture_output=True,
        text=True,
        env=env,
    )
    assert r.returncode == 0, r.stderr
    assert int(r.stdout.strip()) >= 1


def test_phca_query_cli_json(tmp_path):
    sess = _write_reacher_session(tmp_path)
    env = {**dict(os.environ), "PYTHONPATH": str(_ROOT / "python")}
    r = subprocess.run(
        [sys.executable, str(_QUERY), str(sess), "--violation", "--json"],
        capture_output=True,
        text=True,
        env=env,
    )
    assert r.returncode == 0, r.stderr
    payload = json.loads(r.stdout)
    assert payload["count"] >= 1
    assert payload["matches"]


def test_export_matches_jsonl(tmp_path):
    frames = _spike_series(4)
    matches = query_frames(frames, MomentQuery(spike=True))
    out = tmp_path / "out.jsonl"
    export_matches_jsonl(matches, frames, out)
    lines = out.read_text().strip().splitlines()
    assert len(lines) == len(matches)


def test_dashboard_moment_navigation(qt_app):
    from phca.monitoring.playback import PlaybackClock
    from phca.monitoring.qt_dashboard import ObservatoryWindow, _TransportBar

    frames = _spike_series(12)
    win = ObservatoryWindow()
    win._all_frames = frames
    win._review_mode = True
    clock = PlaybackClock(mode="replay")
    clock.set_frames(frames)
    clock.review_mode = True
    clock.set_paused(True)
    transport = _TransportBar(clock, pacer=None)
    win.install_transport(transport)
    transport.moment_filter.setCurrentText("Spike")
    win._rebuild_moment_matches()
    assert len(win._moment_matches) == 1
    clock.seek(0)
    win._jump_moment(1)
    assert clock.cursor_int == 5
