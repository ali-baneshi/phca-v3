"""Tests for offline session_report walking JSONL with Overview parity."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from phca.monitoring.cognitive_panels import apply_decision_shift
from phca.monitoring.qt_dashboard import _apply_decision_shift
from phca.monitoring.session_report import build_session_report, write_session_report

_FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"
_FIXTURE_JSONL = _FIXTURE_DIR / "reacher_short.jsonl"


def _load_fixture_lines() -> list[str]:
    return [ln for ln in _FIXTURE_JSONL.read_text().splitlines() if ln.strip()]


def test_apply_decision_shift_threshold():
    assert apply_decision_shift(0.20, 0.55) is True
    assert apply_decision_shift(0.20, 0.35) is False
    assert apply_decision_shift(None, 0.5) is False
    assert _apply_decision_shift(0.20, 0.55) is True


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
    assert "flow_metrics" in report
    assert "action_metrics" in report
    assert report["flow_metrics"].get("violation_cycle_count", 0) >= 1
    assert report["action_metrics"].get("cycles_with_scores", 0) >= 8
    assert "cycles_with_chosen_idx" in report["action_metrics"]
    assert report.get("phase_space_metrics", {}).get("dominant_env_kind")
    pm = report.get("phase_space_metrics", {})
    assert pm.get("pca_variance_explained_median") is not None
    assert pm.get("max_pred_error_dim_median") is not None
    cpm = report.get("cognitive_panels_metrics", {})
    assert cpm.get("spike_count", 0) >= 1
    assert cpm.get("learn_burst_count", 0) >= 1
    assert "anchor_moment_flags" in cpm
    assert "0" in cpm["anchor_moment_flags"]
    assert report["flow_metrics"].get("anchor_flow_moments")
    assert report["action_metrics"].get("anchor_action_moments")
    mh = report["action_metrics"].get("mechanism_histogram", {})
    assert sum(mh.values()) == report["cycles"]
    assert "mechanism_pct" in report["action_metrics"]
    assert "retention_metrics" in report
    assert "memory_metrics" in report
    assert "goals_metrics" in report
    assert "anchor_retention" in report["retention_metrics"]
    assert "anchor_memory" in report["memory_metrics"]
    assert "anchor_goals" in report["goals_metrics"]


def test_retention_memory_goals_metric_keys():
    meta = {"env": "Reacher-v5", "cycles": 12}
    report = build_session_report(meta, _load_fixture_lines())
    for key in ("m3_prune_count", "m4_prune_count", "envelope_over_count"):
        assert key in report["retention_metrics"]
    for key in ("cycles_with_m3", "cycles_with_m4"):
        assert key in report["memory_metrics"]
    assert "active_drive_switch_count" in report["goals_metrics"]


def test_cognitive_panels_metrics_keys():
    meta = {"env": "Reacher-v5", "cycles": 12}
    report = build_session_report(meta, _load_fixture_lines())
    cpm = report.get("cognitive_panels_metrics", {})
    for key in ("spike_count", "learn_burst_count", "decision_shift_count",
                "drive_change_count", "anchor_moment_flags"):
        assert key in cpm


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
        if apply_decision_shift(prev_best_score, cur):
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


def test_session_report_overview_results_parity():
    """Phase 10: offline report fields match Overview format_session_results_lines."""
    from phca.monitoring.cognitive_panels import format_session_results_lines

    meta = {"env": "Reacher-v5"}
    report = build_session_report(meta, _load_fixture_lines())
    lines = format_session_results_lines(report)
    joined = "\n".join(lines)
    assert lines
    assert str(report.get("cycles", 0)) in lines[0]
    assert "explore" in lines[0].lower()
    mech_pct = (report.get("action_metrics") or {}).get("mechanism_pct") or {}
    if mech_pct:
        top = max(mech_pct.items(), key=lambda kv: kv[1])
        assert top[0] in joined
    phase_pct = report.get("phase_budget_pct") or {}
    if phase_pct:
        top_phase = max(phase_pct.items(), key=lambda kv: kv[1])
        assert top_phase[0] in joined


def test_compare_session_reports_delta():
    """Phase 12: multi-session report comparison."""
    from phca.monitoring.session_report import compare_session_reports

    current = {
        "meta": {"env": "gridworld", "session_id": "b"},
        "cycles": 200,
        "error_late_median": 0.4,
        "explore_ratio": 0.1,
        "goal_reached_count": 5,
        "spike_count": 2,
        "flow_metrics": {"violation_cycle_count": 1},
        "action_metrics": {"score_margin_median": 0.2},
    }
    baseline = {
        "meta": {"env": "gridworld", "session_id": "a"},
        "cycles": 200,
        "error_late_median": 0.5,
        "explore_ratio": 0.08,
        "goal_reached_count": 4,
        "spike_count": 3,
        "flow_metrics": {"violation_cycle_count": 0},
        "action_metrics": {"score_margin_median": 0.15},
    }
    result = compare_session_reports(current, baseline)
    assert result["deltas"]["error_late_median"] == pytest.approx(-0.1)
    assert result["deltas"]["explore_ratio"] == pytest.approx(0.02)
    assert result["regression_flags"]["error_late_worse"] is False
    assert result["regression_flags"]["violations_increased"] is True
