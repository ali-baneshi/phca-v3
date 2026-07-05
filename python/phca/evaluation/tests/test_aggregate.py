"""Tests for aggregate_validation hypothesis evaluation."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[4]
_spec = importlib.util.spec_from_file_location(
    "aggregate_validation", ROOT / "scripts" / "aggregate_validation.py"
)
_av = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_av)

evaluate_hypotheses = _av.evaluate_hypotheses
_extract_phi_iq = _av._extract_phi_iq


def test_extract_phi_iq_from_aggregate():
    data = {"aggregate": {"metrics": {"phi_iq": {"mean": 0.72}}}}
    assert _extract_phi_iq(data) == pytest.approx(0.72)


def test_evaluate_h006_validated():
    hypotheses = [{"id": "H006", "statement": "predictive validity"}]
    results = {
        "phi_iq_validation.json": {"predictive_validity": {"pearson_r": 0.75}},
    }
    verdicts = evaluate_hypotheses(hypotheses, results)
    assert verdicts[0]["status"] == "Validated"


def test_evaluate_h006_refuted():
    hypotheses = [{"id": "H006", "statement": "predictive validity"}]
    results = {
        "phi_iq_validation.json": {"predictive_validity": {"pearson_r": 0.5}},
    }
    verdicts = evaluate_hypotheses(hypotheses, results)
    assert verdicts[0]["status"] == "Refuted"


def test_desync_stage_order_matches_cycle_implementation():
    from phca.evaluation.interventions import DESYNC_STAGE_ORDER

    assert DESYNC_STAGE_ORDER.index("regulation") < DESYNC_STAGE_ORDER.index("prediction")
    assert DESYNC_STAGE_ORDER.index("feedback") > DESYNC_STAGE_ORDER.index("action")
