"""Tests for Phase 14 action explainability."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from phca.monitoring.action_explain import (
    build_explain_chain,
    explain_anchor_bundle,
    explain_fields_present,
    explain_smoke_pass,
    infer_decision_reason,
)
from phca.monitoring.observability import ObservabilityFrame
from phca.monitoring.render import frame_from_json
from phca.monitoring.session_report import build_session_report

_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "reacher_short.jsonl"
_REPO = Path(__file__).resolve().parents[4]
_SCRIPTS = _REPO / "scripts"


@pytest.fixture
def qt_app():
    from PyQt5 import QtWidgets

    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    return app


def _frame(**kw) -> ObservabilityFrame:
    f = ObservabilityFrame()
    for k, v in kw.items():
        setattr(f, k, v)
    return f


def _enriched_rationale(**extra):
    base = {
        "explored": False,
        "eps": 0.04,
        "goal_id": 2,
        "drive_id": 2,
        "continuous": True,
        "best_score": 0.55,
        "k_candidates": 8,
        "chosen_idx": 2,
        "decision_reason": "continuous_mpc",
        "mechanism": "continuous",
        "score_components": {"confidence": 0.8, "ref_align": 0.6, "pga": 0.5},
        "relevant_facts_summary": [{"fact_id": "fact_1", "confidence": 0.7}],
        "chosen_label": "τ#2",
    }
    base.update(extra)
    return base


def test_infer_decision_reason_legacy_explore():
    assert infer_decision_reason({"explored": True, "continuous": False}) == "explore"


def test_infer_decision_reason_explicit():
    assert infer_decision_reason({"decision_reason": "prediction"}) == "prediction"


def test_build_explain_chain_enriched():
    f = _frame(
        cycle_id=5,
        active_drive_id=2,
        drive_levels=[0.1, 0.42, 0.3, 0.2, 0.1, 0.1],
        action_rationale=_enriched_rationale(),
        candidate_scores=[0.2, 0.3, 0.55, 0.4],
        continuous_action=[0.1, -0.2],
    )
    chain = build_explain_chain(f)
    assert chain
    assert any("drive" in line for line in chain)
    assert any("continuous_mpc" in line for line in chain)
    assert any("chosen" in line for line in chain)
    assert any("facts" in line for line in chain)


def test_explain_fields_present():
    assert explain_fields_present(_enriched_rationale())
    assert not explain_fields_present({"explored": True})


def test_json_roundtrip_preserves_explain_fields():
    f = _frame(
        cycle_id=1,
        action_rationale=_enriched_rationale(),
        candidate_scores=[0.1, 0.2, 0.55],
    )
    raw = f.to_json()
    restored = frame_from_json(raw)
    r = restored.action_rationale
    assert r["decision_reason"] == "continuous_mpc"
    assert r["mechanism"] == "continuous"
    assert r["score_components"]["confidence"] == pytest.approx(0.8)


def test_build_session_report_explain_metrics():
    lines = [ln for ln in _FIXTURE.read_text().splitlines() if ln.strip()]
    report = build_session_report({"env": "Reacher-v5"}, lines)
    explain_m = (report.get("action_metrics") or {}).get("explain_metrics") or {}
    assert "decision_reason_counts" in explain_m
    assert "anchor_explain" in explain_m
    assert "0" in explain_m["anchor_explain"]


def test_explain_anchor_bundle():
    f = _frame(
        cycle_id=3,
        action_rationale=_enriched_rationale(),
        candidate_scores=[0.2, 0.55],
    )
    bundle = explain_anchor_bundle(f)
    assert bundle["cycle_id"] == 3
    assert bundle["decision_reason"] == "continuous_mpc"
    assert bundle["chain"]


def test_explain_smoke_pass_legacy_false():
    f = _frame(action_rationale={"explored": False, "best_score": 0.2})
    assert explain_smoke_pass([f, f, f]) is False


def test_explain_smoke_pass_enriched_true():
    frames = [
        _frame(action_rationale=_enriched_rationale()),
        _frame(action_rationale=_enriched_rationale()),
        _frame(action_rationale=_enriched_rationale()),
    ]
    assert explain_smoke_pass(frames) is True


def test_candidate_score_view_explain_band(qt_app):
    from PyQt5 import QtGui
    from phca.monitoring.qt_dashboard import CandidateScoreView, PANEL_BG

    view = CandidateScoreView()
    view.resize(640, 480)
    f = _frame(
        cycle_id=1,
        active_drive_id=2,
        drive_levels=[0.1, 0.5, 0.3, 0.2, 0.1, 0.1],
        action_rationale=_enriched_rationale(),
        candidate_scores=[0.2, 0.3, 0.55],
        continuous_action=[0.1, -0.2],
    )
    view.set_frame(f, replay=True)
    pm = QtGui.QPixmap(640, 480)
    pm.fill(PANEL_BG)
    p = QtGui.QPainter(pm)
    view._draw(p)
    p.end()


def test_replay_check_explain_line(tmp_path, capsys):
    spec = importlib.util.spec_from_file_location("phca_replay_explain", _SCRIPTS / "phca_replay.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    d = tmp_path / "sess"
    d.mkdir()
    obj = {
        "cycle_id": 0,
        "schema_version": 1,
        "prediction_error": 1.0,
        "action_rationale": _enriched_rationale(),
        "candidate_scores": [0.2, 0.55],
        "module_timings": {},
        "active_drive_id": 2,
    }
    (d / "meta.json").write_text(json.dumps({"env": "test", "cycles": 1}))
    (d / "timeseries.jsonl").write_text(json.dumps(obj) + "\n")
    rc = mod._check(str(d))
    captured = capsys.readouterr()
    assert rc == 0
    assert "explain    :" in captured.out
