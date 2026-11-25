"""Tests for session recovery and production hardening (Phase 16)."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


from phca.monitoring.observability import ObservabilityFrame, SessionRecorder
from phca.monitoring.session_recovery import (
    detect_session_state,
    finalize_session,
    read_latest_pointer,
    recover_session,
)

_REPO = Path(__file__).resolve().parents[4]
_SCRIPTS = _REPO / "scripts"
_FIXTURE_JSONL = Path(__file__).resolve().parent / "fixtures" / "reacher_short.jsonl"


def _load_replay():
    spec = importlib.util.spec_from_file_location("phca_replay", _SCRIPTS / "phca_replay.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _load_supervisor():
    spec = importlib.util.spec_from_file_location(
        "phca_observatory_supervisor", _SCRIPTS / "phca_observatory_supervisor.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _write_partial_session(tmp_path: Path, *, n: int = 3, cycles: int = 10,
                           status: str = "running") -> Path:
    d = tmp_path / "partial"
    d.mkdir()
    meta = {
        "env": "Reacher-v5",
        "cycles": cycles,
        "status": status,
        "observability_schema_version": 1,
    }
    if status == "complete":
        meta["recorded_cycles"] = n
    (d / "meta.json").write_text(json.dumps(meta))
    lines = [ln for ln in _FIXTURE_JSONL.read_text().splitlines() if ln.strip()][:n]
    (d / "timeseries.jsonl").write_text("\n".join(lines) + "\n")
    return d


def test_session_recorder_start_writes_running_and_latest(tmp_path):
    rec = SessionRecorder(root=str(tmp_path / "sessions"), record=True)
    session_dir = rec.start({"cycles": 5, "env": "gridworld"})
    assert session_dir is not None
    meta = json.loads((session_dir / "meta.json").read_text())
    assert meta["status"] == "running"
    assert meta.get("started_at")
    assert read_latest_pointer(tmp_path / "sessions") == session_dir.resolve()
    rec.record(ObservabilityFrame(cycle_id=0))
    rec.close()
    meta = json.loads((session_dir / "meta.json").read_text())
    assert meta["status"] == "complete"
    assert meta["recorded_cycles"] == 1
    assert meta.get("closed_at")
    assert read_latest_pointer(tmp_path / "sessions") is None


def test_session_recorder_abort_clears_latest(tmp_path):
    rec = SessionRecorder(root=str(tmp_path / "sessions"), record=True)
    session_dir = rec.start({"cycles": 5})
    rec.record(ObservabilityFrame(cycle_id=0))
    rec.abort("test_crash")
    meta = json.loads((session_dir / "meta.json").read_text())
    assert meta["status"] == "incomplete"
    assert meta["recorded_cycles"] == 1
    assert meta["recovery_reason"] == "test_crash"
    assert read_latest_pointer(tmp_path / "sessions") is None


def test_session_recorder_close_incomplete_never_complete(tmp_path):
    """OBS-D05: short/aborted runs must not write status=complete."""
    rec = SessionRecorder(root=str(tmp_path / "sessions"), record=True)
    session_dir = rec.start({"cycles": 1000, "env": "Reacher-v5"})
    rec.record(ObservabilityFrame(cycle_id=0))
    rec.close(incomplete=True, reason="cycle_error")
    meta = json.loads((session_dir / "meta.json").read_text())
    assert meta["status"] == "incomplete"
    assert meta["recorded_cycles"] == 1
    assert meta["recovery_reason"] == "cycle_error"
    assert read_latest_pointer(tmp_path / "sessions") is None


def test_detect_session_state_complete(tmp_path):
    d = _write_partial_session(tmp_path, n=3, cycles=3, status="complete")
    state = detect_session_state(d)
    assert state["state"] == "complete"
    assert state["jsonl_lines"] == 3


def test_detect_session_state_running_partial(tmp_path):
    d = _write_partial_session(tmp_path, n=3, cycles=10, status="running")
    state = detect_session_state(d)
    assert state["state"] in ("running", "incomplete")
    assert state["jsonl_lines"] == 3


def test_detect_session_state_corrupt(tmp_path):
    d = _write_partial_session(tmp_path, n=2, cycles=10)
    text = (d / "timeseries.jsonl").read_text() + "{not valid json\n"
    (d / "timeseries.jsonl").write_text(text)
    state = detect_session_state(d)
    assert state["state"] == "corrupt"
    assert state["corrupt_lines"] >= 1


def test_finalize_session_partial_writes_report(tmp_path):
    d = _write_partial_session(tmp_path, n=3, cycles=10, status="running")
    summary = finalize_session(d, reason="test_finalize", write_report=True)
    meta = json.loads((d / "meta.json").read_text())
    assert meta["recorded_cycles"] == 3
    assert meta["status"] == "incomplete"
    assert summary["report_path"] is not None
    assert (d / "session_report.json").exists()


def test_replay_check_strict_vs_allow_incomplete_on_partial(tmp_path):
    replay = _load_replay()
    d = _write_partial_session(tmp_path, n=3, cycles=10, status="running")
    assert replay._check(str(d)) == 1
    assert replay._check(str(d), allow_incomplete=True) == 0


def test_recover_session_with_verify_allow_incomplete(tmp_path, monkeypatch):
    d = _write_partial_session(tmp_path, n=3, cycles=10, status="running")

    def fake_check(session_dir, *, allow_incomplete=False):
        return (0, "PASS") if allow_incomplete else (1, "FAIL")

    import phca.monitoring.session_recovery as sr
    monkeypatch.setattr(sr, "_run_replay_check", fake_check)
    rc, summary = recover_session(d, verify=True, allow_incomplete=True)
    assert rc == 0
    assert summary["jsonl_lines"] == 3


def _crash_stub_script(tmp_path: Path) -> str:
    root = tmp_path / "sessions"
    return f"""
import os, sys
sys.path.insert(0, {str(_REPO / "python")!r})
from phca.monitoring.observability import ObservabilityFrame, SessionRecorder
rec = SessionRecorder(root={str(root)!r}, record=True)
rec.start({{"cycles": 10, "env": "test"}})
for i in range(3):
    rec.record(ObservabilityFrame(cycle_id=i, env_kind="mujoco_rgb"))
rec.flush()
os._exit(137)
"""


def test_supervisor_recovers_crashed_child(tmp_path):
    sup = _load_supervisor()
    log_path = tmp_path / "supervisor.jsonl"
    record_dir = tmp_path / "sessions"
    stub = _crash_stub_script(tmp_path)
    rc = sup.run_supervisor(
        [sys.executable, "-c", stub],
        record_dir=str(record_dir),
        supervisor_log=log_path,
        no_recover=False,
        no_verify=True,
        strict_verify=False,
    )
    assert rc == 0
    sessions = [p for p in record_dir.iterdir() if p.is_dir()]
    assert len(sessions) >= 1
    sess = sessions[0]
    meta = json.loads((sess / "meta.json").read_text())
    assert meta["recorded_cycles"] == 3
    assert meta["status"] == "crashed"
    assert (sess / "session_report.json").exists()
    events = [json.loads(ln) for ln in log_path.read_text().splitlines() if ln.strip()]
    kinds = {e["event"] for e in events}
    assert "child_exit" in kinds
    assert "recover_done" in kinds


def test_supervisor_propagates_clean_exit(tmp_path, monkeypatch):
    sup = _load_supervisor()
    log_path = tmp_path / "supervisor.jsonl"
    record_dir = tmp_path / "sessions"

    class _Proc:
        pid = 4242

        def wait(self):
            return 0

    monkeypatch.setattr(sup.subprocess, "Popen", lambda *a, **k: _Proc())
    monkeypatch.setattr(sup, "read_latest_pointer", lambda _rd: None)
    rc = sup.run_supervisor(
        ["--cycles=1"],
        record_dir=str(record_dir),
        supervisor_log=log_path,
    )
    assert rc == 0
    events = [json.loads(ln) for ln in log_path.read_text().splitlines()]
    assert events[-1]["event"] == "recover_skip"


def test_supervisor_preserve_child_exit_masks_recovery_success(tmp_path, monkeypatch):
    sup = _load_supervisor()
    log_path = tmp_path / "supervisor.jsonl"
    record_dir = tmp_path / "sessions"
    sess = _write_partial_session(tmp_path, n=2, cycles=10, status="running")

    class _Proc:
        pid = 9999

        def wait(self):
            return 137

    monkeypatch.setattr(sup.subprocess, "Popen", lambda *a, **k: _Proc())
    monkeypatch.setattr(sup, "read_latest_pointer", lambda _rd: sess)
    monkeypatch.setattr(
        sup,
        "recover_session",
        lambda *a, **k: (0, {"status": "incomplete", "jsonl_lines": 2}),
    )
    rc_default = sup.run_supervisor(
        ["--cycles=10"],
        record_dir=str(record_dir),
        supervisor_log=log_path,
        no_verify=True,
    )
    assert rc_default == 0
    rc_preserve = sup.run_supervisor(
        ["--cycles=10"],
        record_dir=str(record_dir),
        supervisor_log=log_path,
        no_verify=True,
        preserve_child_exit=True,
    )
    assert rc_preserve == 137
