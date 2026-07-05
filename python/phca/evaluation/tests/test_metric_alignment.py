"""Regression tests for trace-index alignment in evaluation metrics."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from phca.evaluation.metrics.emergence import cross_context_reuse, behavioral_compression
from phca.evaluation.metrics.interaction import _aligned_feature_targets, surrogate_r2
from phca.evaluation.metrics.synergy import prediction_action_synergy
from phca.evaluation.trace import CycleTraceRecord

ROOT = Path(__file__).resolve().parents[4]
_spec = importlib.util.spec_from_file_location(
    "aggregate_validation", ROOT / "scripts" / "aggregate_validation.py"
)
_av = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_av)
_extract_metric = _av._extract_metric


def _trace_with_gaps(n: int = 50) -> list[CycleTraceRecord]:
    """Trace with invalid actions (-1) interleaved like continuous env markers."""
    records = []
    for i in range(n):
        action = -1 if i % 5 == 0 else i % 4
        records.append(
            CycleTraceRecord(
                cycle_id=i,
                action=action,
                prediction_error=0.1 * i,
                prediction_confidence=0.2 + 0.01 * (i % 10),
                goal_drive=1,
                goal_switched=(i == n // 2),
                goal_reached=False,
            )
        )
    return records


def test_aligned_feature_targets_pairs_same_cycle():
    trace = _trace_with_gaps(50)
    X, y = _aligned_feature_targets(trace)
    assert len(X) == len(y)
    idx = 0
    for i in range(1, len(trace)):
        if trace[i].action < 0:
            continue
        assert y[idx] == pytest.approx(float(trace[i].action))
        assert X[idx, 0] == pytest.approx(float(trace[i].prediction_error))
        idx += 1
    assert idx == len(y)


def test_surrogate_r2_with_filtered_actions_does_not_crash():
    trace = _trace_with_gaps(50)
    score = surrogate_r2(trace)
    assert 0.0 <= score <= 1.0


def test_synergy_uses_matching_confidence_for_valid_actions():
    trace = _trace_with_gaps(40)
    pairs = [
        (trace[i].prediction_confidence, trace[i + 1].action)
        for i in range(len(trace) - 1)
        if trace[i + 1].action >= 0
    ]
    score = prediction_action_synergy(trace)
    assert len(pairs) >= 10
    assert 0.0 <= score <= 1.0


def test_cross_context_reuse_uses_trace_windows():
    trace = _trace_with_gaps(60)
    score = cross_context_reuse(trace)
    assert 0.0 <= score <= 1.0


def test_behavioral_compression_is_deterministic():
    trace = _trace_with_gaps(50)
    assert behavioral_compression(trace) == behavioral_compression(trace)


def test_extract_metric_falls_back_to_per_run_transfer_efficiency():
    data = {
        "aggregate": {"metrics": {"phi_iq": {"mean": 0.7}}},
        "runs": [
            {"metrics": {"transfer_efficiency": 0.5}},
            {"metrics": {"transfer_efficiency": 0.7}},
        ],
    }
    assert _extract_metric(data, "transfer_efficiency") == pytest.approx(0.6)


def test_evaluate_h002_uses_transfer_efficiency_fallback():
    hypotheses = [{"id": "H002"}]
    results = {
        "ablations/no_prediction.json": {
            "aggregate": {"metrics": {"phi_iq": {"mean": 0.7}}},
            "runs": [{"metrics": {"transfer_efficiency": 0.6}}] * 5,
        },
        "ablations/full_system.json": {
            "aggregate": {"metrics": {"phi_iq": {"mean": 0.8}}},
            "runs": [{"metrics": {"transfer_efficiency": 0.4}}] * 5,
        },
    }
    verdicts = _av.evaluate_hypotheses(hypotheses, results)
    drop = verdicts[0]["observed"]["transfer_drop_fraction"]
    assert drop == pytest.approx((0.4 - 0.6) / 0.4)
