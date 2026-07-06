"""Tests for Phase 13 session anomaly detection."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from phca.monitoring.observability import ObservabilityFrame
from phca.monitoring.render import frame_from_json
from phca.monitoring.retention_slope import compute_rss_slopes, rss_leak_flagged
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
    assert result["metrics"]["drift_evaluated"] is False


def test_short_session_drift_skipped_in_check_output(capsys):
    """Short fixtures must not emit drift=FAIL in phca_replay --check."""
    import importlib.util

    frames = _load_fixture_frames()
    result = detect_session_anomalies(frames)
    assert result["metrics"]["drift_evaluated"] is False
    from phca.monitoring.session_anomalies import print_anomaly_summary

    print_anomaly_summary(result, stream=__import__("sys").stdout)
    captured = capsys.readouterr()
    assert "drift=SKIP (short session)" in captured.out
    assert "Anomaly overall: PASS" in captured.out

    d = Path(__file__).resolve().parent / "fixtures" / "multi_agent_short"
    spec = importlib.util.spec_from_file_location("phca_replay_drift_skip", _SCRIPTS / "phca_replay.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    rc = mod._check(str(d))
    captured2 = capsys.readouterr()
    assert rc == 0
    assert "drift=SKIP (short session)" in captured2.out
    assert "Anomaly overall: PASS" in captured2.out


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


def test_anomalies_from_report_rolling_goal_parity():
    """Rolling goal flag must survive report-only re-evaluation (OBS-003)."""
    from phca.monitoring.session_anomalies import anomalies_from_report

    frames = []
    for i in range(100):
        drive = 1
        if 48 <= i <= 55:
            drive = 1 + ((i - 48) % 4)
        frames.append(_frame(
            cycle_id=i,
            prediction_error=1.0,
            action_rationale={"goal_id": drive},
            active_drive_id=drive,
        ))
    frame_result = detect_session_anomalies(frames)
    assert frame_result["metrics"]["goal_rolling_instability"] is True
    assert frame_result["flags"]["goal_instability"] is True
    assert frame_result["metrics"]["drive_switch_rate"] < 0.25

    report = {
        "cycles": 100,
        "spike_count": frame_result["metrics"]["spike_count"],
        "error_early_median": frame_result["metrics"]["error_early_median"],
        "error_late_median": frame_result["metrics"]["error_late_median"],
        "drive_switch_count": frame_result["metrics"]["drive_switch_count"],
        "goals_metrics": {
            "active_drive_switch_count": frame_result["metrics"]["active_drive_switch_count"],
        },
        "anomalies": frame_result,
    }
    report_only = anomalies_from_report(report)
    assert report_only["flags"]["goal_instability"] is True
    assert report_only["metrics"]["goal_rolling_instability"] is True


def test_anomalies_from_report_leak_requires_slope():
    """Report-only re-eval cannot detect leak without rss_late_slope in anomalies.metrics."""
    from phca.monitoring.session_anomalies import anomalies_from_report

    report_with_leak = {
        "cycles": 8000,
        "spike_count": 0,
        "error_early_median": 1.0,
        "error_late_median": 1.0,
        "drive_switch_count": 0,
        "goals_metrics": {"active_drive_switch_count": 0},
        "anomalies": {
            "metrics": {"rss_late_slope_bytes_per_cycle": 2000.0},
        },
    }
    assert anomalies_from_report(report_with_leak)["flags"]["leak"] is True

    report_stripped = dict(report_with_leak)
    report_stripped["anomalies"] = {"metrics": {}}
    assert anomalies_from_report(report_stripped)["flags"]["leak"] is False


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


def test_post_m3_slope_window_flat_after_cap():
    n = 11_000
    sample_every = 100
    base = 70_000_000
    xs, ys = [], []
    for i in range(1, n // sample_every + 1):
        cyc = i * sample_every
        xs.append(cyc)
        ys.append(base + cyc * 1500 if cyc <= 10_000 else base + 10_000 * 1500)
    slopes = compute_rss_slopes(xs, ys, total_cycles=n)
    assert slopes["late_slope_window"] == "post_m3"
    assert slopes["late_slope"] == pytest.approx(0.0, abs=1.0)
    assert not rss_leak_flagged(float(slopes["late_slope"]), n)


def test_post_m3_slope_ignores_pre_cap_tail_growth():
    n = 11_000
    sample_every = 100
    base = 70_000_000
    flat_post = base + 20_000_000
    xs, ys = [], []
    for i in range(1, n // sample_every + 1):
        cyc = i * sample_every
        xs.append(cyc)
        if cyc < 7500:
            ys.append(base + cyc * 100)
        elif cyc <= 10_000:
            ys.append(base + cyc * 3000)
        else:
            ys.append(flat_post + (cyc - 10_000) * 10)
    slopes = compute_rss_slopes(xs, ys, total_cycles=n)
    assert slopes["late_slope_window"] == "post_m3"
    assert float(slopes["late_slope"]) < 100.0
    assert not rss_leak_flagged(float(slopes["late_slope"]), n)


def test_10k_uses_tail_quarter_not_post_m3():
    n = 10_000
    sample_every = 100
    base = 70_000_000
    xs = [i * sample_every for i in range(1, n // sample_every + 1)]
    ys = [base + c * 1500 for c in xs]
    slopes = compute_rss_slopes(xs, ys, total_cycles=n)
    assert slopes["late_slope_window"] == "tail_quarter"


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
