"""Phase 17 multi-agent Observatory tests."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from phca.monitoring.multi_agent import (
    DEFAULT_AGENT_ID,
    frames_for_agent,
    interleave_frames_for_record,
    is_multi_agent_session,
    session_agent_ids,
    validate_agent_cycle_contiguity,
    validate_aligned_timeline,
    validate_jsonl_step_major_order,
)
from phca.monitoring.observability import (
    ObservabilityFrame,
    normalize_observability_json,
)
from phca.monitoring.render import frame_from_json
from phca.monitoring.session_report import (
    build_session_report,
    compare_all_agents,
    compare_session_reports,
)

_ROOT = Path(__file__).resolve().parents[4]
_FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"
_MULTI_AGENT_FIXTURE = _FIXTURE_DIR / "multi_agent_short"
_REPLAY = _ROOT / "scripts" / "phca_replay.py"
_OBS = _ROOT / "scripts" / "phca_observatory.py"


def _grid_frame(cycle_id: int, agent_id: int, *, explored: bool = False) -> ObservabilityFrame:
    f = ObservabilityFrame(
        cycle_id=cycle_id,
        agent_id=agent_id,
        agent_label=f"agent_{agent_id}",
        timeline_step=cycle_id,
        env_kind="grid",
        agent_pos=(cycle_id % 5, (cycle_id + 1) % 5),
        goal_pos=(4, 4),
        grid=np.zeros((5, 5), dtype=np.int32),
        prediction_error=0.1 * (cycle_id + 1) * (agent_id + 1),
        action_rationale={
            "explored": explored,
            "best_score": 0.5,
            "decision_reason": "explore" if explored else "prediction",
            "mechanism": "explore" if explored else "prediction",
        },
        drive_levels=[0.5] * 6,
    )
    return f


def _build_two_agent_jsonl(cycles_per_agent: int = 6) -> list[str]:
    lines: list[str] = []
    for step in range(cycles_per_agent):
        for aid in (0, 1):
            explored = aid == 1 and step % 2 == 0
            f = _grid_frame(step, aid, explored=explored)
            lines.append(json.dumps(f.to_json()))
    return lines


def _write_fixture_session(tmp_path: Path, cycles_per_agent: int = 6) -> Path:
    lines = _build_two_agent_jsonl(cycles_per_agent)
    d = tmp_path / "multi_sess"
    d.mkdir()
    meta = {
        "env": "gridworld",
        "cycles": len(lines),
        "cycles_per_agent": cycles_per_agent,
        "agent_count": 2,
        "agents": [
            {"agent_id": 0, "label": "agent_0", "recorded_cycles": cycles_per_agent},
            {"agent_id": 1, "label": "agent_1", "recorded_cycles": cycles_per_agent},
        ],
        "timeline_mode": "aligned",
        "recording_layout": "single_jsonl",
        "recorded_cycles": len(lines),
        "observability_schema_version": 1,
    }
    (d / "meta.json").write_text(json.dumps(meta, indent=2))
    (d / "timeseries.jsonl").write_text("\n".join(lines) + "\n")
    return d


def test_frame_agent_defaults_legacy():
    obj = normalize_observability_json({"cycle_id": 3, "schema_version": 1})
    assert obj["agent_id"] == DEFAULT_AGENT_ID
    assert obj["agent_label"] == ""
    assert obj["timeline_step"] == -1
    f = frame_from_json(obj)
    assert f.agent_id == 0


def test_fixture_two_agent_session():
    lines = _build_two_agent_jsonl()
    frames = [frame_from_json(json.loads(ln)) for ln in lines]
    assert session_agent_ids(frames) == [0, 1]
    assert len(frames_for_agent(frames, 0)) == 6
    assert len(frames_for_agent(frames, 1)) == 6
    assert is_multi_agent_session(frames=frames)


def test_validate_agent_cycle_contiguity_pass():
    parsed = [json.loads(ln) for ln in _build_two_agent_jsonl()]
    ok, err = validate_agent_cycle_contiguity(parsed)
    assert ok, err


def test_validate_agent_cycle_contiguity_fail():
    parsed = [json.loads(ln) for ln in _build_two_agent_jsonl()]
    parsed[3]["cycle_id"] = 9
    ok, err = validate_agent_cycle_contiguity(parsed)
    assert not ok
    assert "agent_id=1" in (err or "")


def test_validate_aligned_timeline():
    parsed = [json.loads(ln) for ln in _build_two_agent_jsonl()]
    ok, err = validate_aligned_timeline(parsed)
    assert ok, err


def test_validate_aligned_timeline_fail():
    parsed = [json.loads(ln) for ln in _build_two_agent_jsonl()]
    parsed.pop(1)
    ok, err = validate_aligned_timeline(parsed)
    assert not ok
    assert "timeline_step=0" in (err or "")


def test_interleave_frames_for_record_step_major():
    batched = [_grid_frame(step, 0) for step in range(3)]
    batched.extend(_grid_frame(step, 1) for step in range(3))
    fixed = interleave_frames_for_record(batched)
    assert [(f.timeline_step, f.agent_id) for f in fixed] == [
        (0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1),
    ]


def test_validate_jsonl_step_major_order_pass():
    parsed = [json.loads(ln) for ln in _build_two_agent_jsonl()]
    ok, err = validate_jsonl_step_major_order(parsed)
    assert ok, err


def test_validate_jsonl_step_major_order_fail():
    parsed = [json.loads(ln) for ln in _build_two_agent_jsonl()]
    parsed = [parsed[i] for i in (0, 2, 4, 6, 8, 10, 1, 3, 5, 7, 9, 11)]
    ok, err = validate_jsonl_step_major_order(parsed)
    assert not ok
    assert "step-major" in (err or "")


def test_replay_check_multi_agent(tmp_path):
    sess = _write_fixture_session(tmp_path)
    env = {**dict(os.environ), "PYTHONPATH": str(_ROOT / "python")}
    r = subprocess.run(
        [sys.executable, str(_REPLAY), "--check", str(sess)],
        capture_output=True,
        text=True,
        env=env,
    )
    assert r.returncode == 0, r.stdout + r.stderr
    assert "per-agent cycle_id contiguous" in r.stdout
    assert "step-major JSONL line order" in r.stdout


def test_replay_check_committed_multi_agent_fixture():
    """CI/reproduce gate: tracked multi_agent_short session dir."""
    assert _MULTI_AGENT_FIXTURE.is_dir()
    env = {**dict(os.environ), "PYTHONPATH": str(_ROOT / "python")}
    r = subprocess.run(
        [sys.executable, str(_REPLAY), "--check", str(_MULTI_AGENT_FIXTURE)],
        capture_output=True,
        text=True,
        env=env,
    )
    assert r.returncode == 0, r.stdout + r.stderr
    assert "per-agent cycle_id contiguous" in r.stdout
    assert "step-major JSONL line order" in r.stdout
    assert "Overall: PASS" in r.stdout


def test_replay_check_multi_agent_fail(tmp_path):
    sess = _write_fixture_session(tmp_path)
    jsonl = sess / "timeseries.jsonl"
    lines = jsonl.read_text().splitlines()
    obj = json.loads(lines[1])
    obj["cycle_id"] = 99
    lines[1] = json.dumps(obj)
    jsonl.write_text("\n".join(lines) + "\n")
    env = {**dict(os.environ), "PYTHONPATH": str(_ROOT / "python")}
    r = subprocess.run(
        [sys.executable, str(_REPLAY), "--check", str(sess)],
        capture_output=True,
        text=True,
        env=env,
    )
    assert r.returncode != 0


def test_session_report_per_agent(tmp_path):
    sess = _write_fixture_session(tmp_path)
    lines = (sess / "timeseries.jsonl").read_text().splitlines()
    meta = json.loads((sess / "meta.json").read_text())
    report = build_session_report(meta, [ln for ln in lines if ln.strip()])
    assert "agents" in report
    assert "0" in report["agents"]
    assert "1" in report["agents"]
    assert report["agents"]["1"]["explore_ratio"] > report["agents"]["0"]["explore_ratio"]


def test_compare_session_reports_agent_slice():
    base = build_session_report({}, _build_two_agent_jsonl(4))
    cur_lines = _build_two_agent_jsonl(4)
    # inflate agent 1 errors
    objs = [json.loads(ln) for ln in cur_lines]
    for o in objs:
        if int(o.get("agent_id", 0)) == 1:
            o["prediction_error"] = float(o.get("prediction_error", 0)) + 5.0
    cur = build_session_report({}, [json.dumps(o) for o in objs])
    cmp0 = compare_session_reports(cur, base, agent_id=0)
    cmp1 = compare_session_reports(cur, base, agent_id=1)
    assert cmp0["agent_id"] == 0
    assert cmp1["agent_id"] == 1
    all_cmp = compare_all_agents(cur, base)
    assert "0" in all_cmp["agents"]
    assert "1" in all_cmp["agents"]


def test_scrub_immutability_multi_agent(qt_app):
    from phca.monitoring.playback import PlaybackClock
    from phca.monitoring.qt_dashboard import ObservatoryWindow, _TransportBar

    lines = _build_two_agent_jsonl(20)
    raw_before = [json.loads(ln) for ln in lines]
    frames = [frame_from_json(json.loads(ln)) for ln in lines]
    win = ObservatoryWindow()
    win.load_multi_agent_frames(frames, {"agent_count": 2})
    projected0 = win.project_frames_for_agent(frames)
    clock = PlaybackClock(mode="replay")
    clock.set_frames(projected0)
    clock.on_update = lambda f, rolling, err: win.controller.update(f, rolling, err)
    transport = _TransportBar(clock, pacer=None)
    win.install_transport(transport)
    for idx in (0, 10, 19):
        clock.seek(idx)
        qt_app.processEvents()
    win.select_agent(1)
    projected1 = win.project_frames_for_agent(frames)
    clock.set_frames(projected1)
    for idx in (0, 5, 19):
        clock.seek(idx)
        qt_app.processEvents()
    assert [json.loads(ln) for ln in lines] == raw_before


def test_aligned_runner_smoke():
    env = {
        **dict(os.environ),
        "PYTHONPATH": str(_ROOT / "python"),
        "QT_QPA_PLATFORM": "offscreen",
    }
    r = subprocess.run(
        [
            sys.executable,
            str(_OBS),
            "--agents=2",
            "--cycles=3",
            "--no-record",
            "--no-verify",
            "--close-at-end",
        ],
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
    )
    assert r.returncode == 0, r.stdout + r.stderr


def test_export_csv_agent_columns(tmp_path):
    from phca.monitoring.export import export_session_csv, SESSION_SCALAR_COLUMNS

    sess = _write_fixture_session(tmp_path, cycles_per_agent=2)
    out = export_session_csv(sess)
    text = out.read_text()
    assert "agent_id" in SESSION_SCALAR_COLUMNS
    assert "agent_label" in text
