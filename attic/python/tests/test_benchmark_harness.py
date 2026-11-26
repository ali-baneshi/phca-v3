"""Regression tests for benchmark intervention propagation."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from phca.evaluation.interventions import InterventionConfig
from phca.evaluation.result_schema import BenchmarkConfig


def _load_benchmark_module():
    root = Path(__file__).resolve().parents[2]
    path = root / "scripts" / "benchmark.py"
    spec = importlib.util.spec_from_file_location("phca_benchmark", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_multiseed_benchmark_preserves_selector_intervention(monkeypatch):
    module = _load_benchmark_module()
    seen = []

    class FakeRunner:
        def __init__(self, config, interventions=None):
            seen.append(interventions)
            self.config = config

        def run_all(self, levels):
            return type(
                "Report",
                (),
                {"overall_phi_iq": 0.5, "results": []},
            )()

    monkeypatch.setattr(module, "BenchmarkRunner", FakeRunner)
    intervention = InterventionConfig(
        confidence_gated_selector=True,
        disable_blended_scorer=False,
    )
    module.run_multiseed(
        [0], BenchmarkConfig(n_cycles=1, use_mlp=False), 2,
        interventions=intervention,
    )

    assert seen == [intervention, intervention]


def test_multiseed_benchmark_aggregates_selector_evidence(monkeypatch):
    module = _load_benchmark_module()

    class FakeResult:
        def __init__(self, seed):
            self.level = 2
            self.phi_iq = float(seed)
            self.raw_metrics = {
                "action_selection": {
                    "n_records": 3,
                    "selector_mode_counts": {"confidence_gated_prediction": 2},
                    "decision_reason_counts": {"confidence_gated_prediction": 2},
                },
            }

    class FakeReport:
        def __init__(self, seed):
            self.overall_phi_iq = float(seed)
            self.results = [FakeResult(seed)]

    class FakeRunner:
        def __init__(self, config, interventions=None):
            pass

        def run_all(self, levels):
            return FakeReport(1)

    monkeypatch.setattr(module, "BenchmarkRunner", FakeRunner)
    _, multi_seed = module.run_multiseed(
        [2], BenchmarkConfig(n_cycles=1), 2,
        interventions=InterventionConfig(confidence_gated_selector=True),
    )
    evidence = multi_seed["action_selection"]["2"]
    assert evidence["n_records"] == 6
    assert evidence["selector_mode_counts"] == {
        "confidence_gated_prediction": 4,
    }


def test_long_horizon_trace_preserves_selector_evidence():
    root = Path(__file__).resolve().parents[2]
    path = root / "scripts" / "run_horizon.py"
    spec = importlib.util.spec_from_file_location("phca_horizon", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    trace = module.RollingTrace(window=4)
    trace.record(
        cycle_id=0,
        action=2,
        action_rationale={
            "selector_mode": "continuous_mpc",
            "decision_reason": "continuous_mpc",
        },
        candidate_scores=[0.4, 0.6],
    )
    record = trace.snapshot()[0]
    assert record.selector_mode == "continuous_mpc"
    assert record.decision_reason == "continuous_mpc"
    assert record.candidate_scores == [0.4, 0.6]
