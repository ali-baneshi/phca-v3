"""Tests for Phase 13 session anomaly detection."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from phca.monitoring.observability import ObservabilityFrame
from phca.monitoring.render import frame_from_json
from phca.monitoring.retention_slope import compute_rss_slopes
from phca.monitoring.session_anomalies import (
    DEFAULT_THRESHOLDS,
    anomaly_overall_pass,
    anomaly_strict_fail,
    detect_session_anomalies,
)
from phca.monitoring.session_report import build_session_report

_FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"
_FIXTURE_JSONL = _FIXTURE_DIR / "reacher_short.jsonl"
_REPO = Path(__file__).resolve().parents[4]
_SCRIPTS = _REPO / "scripts"


def _load_fixture_frames():
    lines = [ln for ln in _FIXTURE_JSONL.read_text().splitlines() if ln.strip()]
    return [frame_from_json(json.loads(ln)) for ln in lines]


def _frame(**kw) -> ObservabilityFrame:
    f = ObservabilityFrame()
    for k, v in kw.items():
        setattr(f, k, v)
    return f


def test_reacher_fixture_healthy():
    frames = _load_fixture_frames()
    result = detect_session_anomalies(frames)
    assert anomaly_overall_pass(result)
    assert result["flags"]["spike"] is False
    assert result["flags"]["drift"] is False
    assert result["flags"]["leak"] is False
    assert result["flags"]["goal_instability"] is False


def test_spike_cycle_marker():
    frames = _load_fixture_frames()
    result = detect_session_anomalies(frames)
    spike_markers = [
        m for m in result["cycle_markers"]
        if "spike" in m.get("flags", [])
    ]
    assert any(m["cycle_id"] == 5 for m in spike_markers)


def test_drift_synthetic():
    frames = []
    for i in range(100):
        err = 1.0 + i * 0.15
        frames.append(_frame(
            cycle_id=i,
            prediction_error=err,
            action_rationale={"goal_id": 1},
            active_drive_id=1,
        ))
    result = detect_session_anomalies(frames)
    assert result["flags"]["drift"] is True


def test_leak_synthetic():
    n = 250
    base = 200_000_000.0
    frames = []
    for i in range(n):
        rss = base + i * 5500.0
        frames.append(_frame(
            cycle_id=i,
            prediction_error=1.0,
            rss_bytes=int(rss),
            action_rationale={"goal_id": 1},
            active_drive_id=1,
        ))
    result = detect_session_anomalies(frames)
    assert result["flags"]["leak"] is True
    assert anomaly_strict_fail(result)


def test_goal_instability_synthetic():
    frames = []
    for i in range(10):
        drive = 1 + (i % 4)
        frames.append(_frame(
            cycle_id=i,
            prediction_error=1.0,
            action_rationale={"goal_id": drive},
            active_drive_id=drive,
        ))
    result = detect_session_anomalies(frames)
    assert result["flags"]["goal_instability"] is True


def test_spike_density_synthetic():
    frames = []
    for i in range(50):
        err = 50.0 if i % 2 == 0 else 1.0
        frames.append(_frame(
            cycle_id=i,
            prediction_error=err,
            env_kind="grid",
            action_rationale={"goal_id": 1},
            active_drive_id=1,
        ))
    result = detect_session_anomalies(frames)
    assert result["flags"]["spike"] is True


def test_build_session_report_includes_anomalies():
    meta = {"env": "Reacher-v5", "cycles": 12}
    lines = [ln for ln in _FIXTURE_JSONL.read_text().splitlines() if ln.strip()]
    report = build_session_report(meta, lines)
    anom = report.get("anomalies") or {}
    assert "flags" in anom
    assert "metrics" in anom
    assert "cycle_markers" in anom
    for kind in ("spike", "drift", "leak", "goal_instability"):
        assert kind in anom["flags"]


def test_no_frame_mutation():
    raw = json.loads(_FIXTURE_JSONL.read_text().splitlines()[1])
    snapshot = json.dumps(raw, sort_keys=True)
    f = frame_from_json(raw)
    detect_session_anomalies([f])
    assert json.dumps(raw, sort_keys=True) == snapshot
    assert "decision_shift" not in (f.action_rationale or {})


def test_retention_slope_shared_with_nightly():
    n = 2000
    base = 229_000_000
    xs = np.array([i * 100 for i in range(n // 100)], dtype=np.float64)
    ys = np.array([base + (i * 100) * 4000 for i in range(n // 100)], dtype=np.float64)
    slopes = compute_rss_slopes(xs.tolist(), ys.tolist(), total_cycles=n)
    half = len(xs) // 2
    late_legacy = float(np.polyfit(xs[half:], ys[half:], 1)[0])
    assert slopes["late_slope"] == pytest.approx(late_legacy, rel=1e-6)
    assert slopes["late_slope"] >= 1600.0


def _load_replay():
    spec = importlib.util.spec_from_file_location("phca_replay_anom", _SCRIPTS / "phca_replay.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _minimal_session(tmp_path, lines, *, meta=None):
    d = tmp_path / "sess"
    d.mkdir()
    m = meta or {"env": "test", "cycles": len(lines), "recorded_cycles": len(lines)}
    (d / "meta.json").write_text(json.dumps(m))
    (d / "timeseries.jsonl").write_text("\n".join(lines) + "\n")
    return d


def test_replay_check_anomaly_summary(tmp_path, capsys):
    mod = _load_replay()
    lines = [ln for ln in _FIXTURE_JSONL.read_text().splitlines() if ln.strip()]
    d = _minimal_session(tmp_path, lines, meta={"env": "Reacher-v5", "cycles": len(lines)})
    rc = mod._check(str(d))
    captured = capsys.readouterr()
    assert rc == 0
    assert "anomalies  :" in captured.out
    assert "Anomaly overall: PASS" in captured.out


def test_replay_check_anomaly_strict_leak(tmp_path, capsys):
    mod = _load_replay()
    n = 250
    lines = []
    base = 200_000_000.0
    for i in range(n):
        obj = {
            "cycle_id": i,
            "schema_version": 1,
            "prediction_error": 1.0,
            "rss_bytes": int(base + i * 5500),
            "action_rationale": {"goal_id": 1},
            "active_drive_id": 1,
            "module_timings": {},
        }
        lines.append(json.dumps(obj))
    d = _minimal_session(tmp_path, lines, meta={"env": "gridworld", "cycles": n})
    rc = mod._check(str(d), anomaly_strict=True)
    captured = capsys.readouterr()
    assert rc == 1
    assert "leak=FAIL" in captured.out


def test_cycle_markers_cap():
    th = dict(DEFAULT_THRESHOLDS)
    th["cycle_markers_cap"] = 2
    frames = []
    for i in range(20):
        err = 50.0 if i % 2 == 0 else 1.0
        frames.append(_frame(
            cycle_id=i,
            prediction_error=err,
            env_kind="grid",
            action_rationale={"goal_id": 1},
            active_drive_id=1,
        ))
    result = detect_session_anomalies(frames, thresholds=th)
    assert len(result["cycle_markers"]) <= 2
