"""Tests for offline session_report walking JSONL with Overview parity."""
from __future__ import annotations

import json
from collections import deque
from pathlib import Path

import numpy as np
import pytest

from phca.monitoring.qt_dashboard import _apply_decision_shift
from phca.monitoring.session_report import build_session_report, write_session_report

_FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"
_FIXTURE_JSONL = _FIXTURE_DIR / "reacher_short.jsonl"


def _load_fixture_lines() -> list[str]:
    return [ln for ln in _FIXTURE_JSONL.read_text().splitlines() if ln.strip()]


def test_apply_decision_shift_threshold():
    assert _apply_decision_shift(0.20, 0.55) is True
    assert _apply_decision_shift(0.20, 0.35) is False
    assert _apply_decision_shift(None, 0.5) is False


def test_build_session_report_metrics():
    meta = {"env": "Reacher-v5", "cycles": 12}
    report = build_session_report(meta, _load_fixture_lines())
    assert report["cycles"] == 12
    assert 0.0 <= report["explore_ratio"] <= 1.0
    assert report["spike_count"] >= 1
    assert report["learn_burst_count"] >= 1
    assert report["drive_switch_count"] >= 1
    assert "anchor_narratives" in report
    assert "0" in report["anchor_narratives"]
    assert "last" in report["anchor_narratives"]
    assert report["anchor_narratives"]["0"]["intent"].startswith("Intent:")
    assert report["anchor_narratives"]["0"]["outcome"].startswith("Outcome:")
    assert report["phase_budget_pct"]
    assert sum(report["phase_budget_pct"].values()) == pytest.approx(100.0, abs=0.1)


def test_decision_shift_parity_with_overview():
    """Offline report decision_shift_count matches _apply_decision_shift replay."""
    from phca.monitoring.render import frame_from_json

    lines = _load_fixture_lines()
    report = build_session_report({}, lines)

    prev_best_score = None
    offline_count = 0
    for ln in lines:
        f = frame_from_json(json.loads(ln))
        r = dict(getattr(f, "action_rationale", {}) or {})
        bs = r.get("best_score")
        cur = float(bs) if isinstance(bs, (int, float)) else None
        if _apply_decision_shift(prev_best_score, cur):
            offline_count += 1
        if cur is not None:
            prev_best_score = cur

    assert report["decision_shift_count"] == offline_count


def test_write_session_report(tmp_path):
    meta = {"env": "Reacher-v5", "cycles": 12}
    d = tmp_path / "session"
    d.mkdir()
    (d / "meta.json").write_text(json.dumps(meta))
    (d / "timeseries.jsonl").write_text(_FIXTURE_JSONL.read_text())
    report = write_session_report(d)
    out = d / "session_report.json"
    assert out.exists()
    loaded = json.loads(out.read_text())
    assert loaded["cycles"] == report["cycles"]
    assert loaded["notable_cycles"]


def test_notable_cycles_cap():
    meta = {"env": "Reacher-v5"}
    report = build_session_report(meta, _load_fixture_lines())
    assert len(report["notable_cycles"]) <= 20
