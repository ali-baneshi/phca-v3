"""Tests for phca_observatory post-run verify + session_report pipeline."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

_REPO = Path(__file__).resolve().parents[4]
_SCRIPTS = _REPO / "scripts"


def _load_observatory():
    spec = importlib.util.spec_from_file_location("phca_observatory", _SCRIPTS / "phca_observatory.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _minimal_session(tmp_path: Path, *, n: int = 3) -> Path:
    d = tmp_path / "sess"
    d.mkdir()
    meta = {"env": "Reacher-v5", "cycles": n, "recorded_cycles": n}
    (d / "meta.json").write_text(json.dumps(meta))
    lines = [
        json.dumps({"cycle_id": i, "env_kind": "mujoco_rgb", "schema_version": 1})
        for i in range(n)
    ]
    (d / "timeseries.jsonl").write_text("\n".join(lines) + "\n")
    return d


def test_post_run_pipeline_writes_report_and_verifies(tmp_path, monkeypatch, capsys):
    mod = _load_observatory()
    session = _minimal_session(tmp_path)
    calls = []

    def fake_run(cmd, env=None):
        calls.append(cmd)
        return type("R", (), {"returncode": 0})()

    monkeypatch.setattr(mod.subprocess, "run", fake_run)
    rc, status, report_path = mod._post_run_pipeline(
        session,
        n_lines=3,
        expected=3,
        verify=True,
        warnings=[],
    )
    captured = capsys.readouterr()
    assert rc == 0
    assert "PASS" in status
    assert report_path is not None
    assert report_path.exists()
    assert any("--check" in str(c) for c in calls)
    assert "=== Session summary ===" in captured.out


def test_post_run_pipeline_verify_fail_nonzero_exit(tmp_path, monkeypatch):
    mod = _load_observatory()
    session = _minimal_session(tmp_path)

    def fake_run(cmd, env=None):
        return type("R", (), {"returncode": 1})()

    monkeypatch.setattr(mod.subprocess, "run", fake_run)
    rc, status, _ = mod._post_run_pipeline(
        session,
        n_lines=3,
        expected=3,
        verify=True,
        warnings=[],
    )
    assert rc != 0
    assert status == "FAIL"


def test_load_optional_json_missing(tmp_path):
    mod = _load_observatory()
    assert mod._load_optional_json(None) is None
    assert mod._load_optional_json(str(tmp_path / "nope.json")) is None


def test_run_session_verify_allow_incomplete_flag(tmp_path, monkeypatch):
    mod = _load_observatory()
    session = _minimal_session(tmp_path)
    calls = []

    def fake_run(cmd, env=None):
        calls.append(list(cmd))
        return type("R", (), {"returncode": 0})()

    monkeypatch.setattr(mod.subprocess, "run", fake_run)
    rc = mod._run_session_verify(session, allow_incomplete=True)
    assert rc == 0
    assert "--allow-incomplete" in calls[0]


def test_post_run_pipeline_recorder_error_fails_verify(tmp_path, monkeypatch, capsys):
    mod = _load_observatory()
    session = _minimal_session(tmp_path)
    calls = []

    def fake_run(cmd, env=None):
        calls.append(cmd)
        return type("R", (), {"returncode": 0})()

    monkeypatch.setattr(mod.subprocess, "run", fake_run)
    rc, status, _ = mod._post_run_pipeline(
        session,
        n_lines=1,
        expected=3,
        verify=True,
        warnings=[],
        recorder_error="disk full",
    )
    captured = capsys.readouterr()
    assert rc != 0
    assert "recorder error" in status
    assert "JSONL write error: disk full" in captured.err
    assert "JSONL write error: disk full" in captured.out
    assert not any("--check" in str(c) for c in calls)


def test_post_run_pipeline_recorder_error_fails_without_verify(tmp_path):
    mod = _load_observatory()
    session = _minimal_session(tmp_path)
    rc, status, _ = mod._post_run_pipeline(
        session,
        n_lines=1,
        expected=3,
        verify=False,
        warnings=[],
        recorder_error="disk full",
    )
    assert rc != 0
    assert "recorder error" in status


def test_post_run_pipeline_empty_session_skips_report(tmp_path, monkeypatch, capsys):
    mod = _load_observatory()
    session = tmp_path / "empty"
    session.mkdir()
    (session / "meta.json").write_text(
        json.dumps({"env": "gridworld", "cycles": 10, "recorded_cycles": 0})
    )
    (session / "timeseries.jsonl").write_text("")

    def fail_if_called(_session):
        raise AssertionError("empty sessions must not build reports")

    monkeypatch.setattr(
        "phca.monitoring.session_report.write_session_report",
        fail_if_called,
    )
    monkeypatch.setattr(
        mod.subprocess,
        "run",
        lambda cmd, env=None: type("R", (), {"returncode": 1})(),
    )
    rc, status, report_path = mod._post_run_pipeline(
        session, n_lines=0, expected=10, verify=True, warnings=[]
    )
    captured = capsys.readouterr()
    assert rc != 0
    assert status == "FAIL"
    assert report_path is None
    assert "no frames recorded" in captured.err


def test_session_summary_recorder_error(tmp_path, capsys):
    mod = _load_observatory()
    session = _minimal_session(tmp_path)
    mod._session_summary(session, 1, 3, recorder_error="I/O error")
    captured = capsys.readouterr()
    assert "JSONL write failure: I/O error" in captured.err


def test_session_recorder_stores_first_write_error(tmp_path):
    from phca.monitoring.observability import ObservabilityFrame, SessionRecorder

    rec = SessionRecorder(root=str(tmp_path), record=True)
    rec.start({"env": "grid", "cycles": 2})
    f = ObservabilityFrame(cycle_id=0)
    rec.record(f)

    class _BrokenIO:
        def write(self, _data):
            raise OSError("disk full")

    rec._jsonl = _BrokenIO()
    rec.record(ObservabilityFrame(cycle_id=1))
    assert rec.error == "disk full"
    rec.record(ObservabilityFrame(cycle_id=2))
    assert rec.error == "disk full"
