"""
Tests for G' Forward Inference Engine (PHCA-3.1-008).

Cross-ref: v3.0 §2.2 Definition 2.5, §D.1
"""

from __future__ import annotations

import pytest
import numpy as np
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD

from phca.config import StateVector
from phca.world_model.inference import forward_inference, infer_next_state


@pytest.fixture
def chain_bn() -> DiscreteBayesianNetwork:
    """A→B→C deterministic chain Bayesian network."""
    bn = DiscreteBayesianNetwork([("A", "B"), ("B", "C")])

    cpd_a = TabularCPD(variable="A", variable_card=2, values=[[0.5], [0.5]])
    cpd_b = TabularCPD(
        variable="B", variable_card=2,
        values=[[0.9, 0.1], [0.1, 0.9]],
        evidence=["A"], evidence_card=[2],
    )
    cpd_c = TabularCPD(
        variable="C", variable_card=2,
        values=[[0.95, 0.05], [0.05, 0.95]],
        evidence=["B"], evidence_card=[2],
    )
    bn.add_cpds(cpd_a, cpd_b, cpd_c)
    assert bn.check_model()
    return bn


@pytest.fixture
def temporal_bn() -> DiscreteBayesianNetwork:
    """Simple s0_t → s0_t1 temporal model (80% stay, 20% flip)."""
    bn = DiscreteBayesianNetwork([("s0_t", "s0_t1")])

    cpd_t = TabularCPD(variable="s0_t", variable_card=2, values=[[0.5], [0.5]])
    cpd_t1 = TabularCPD(
        variable="s0_t1", variable_card=2,
        values=[[0.8, 0.2], [0.2, 0.8]],
        evidence=["s0_t"], evidence_card=[2],
    )
    bn.add_cpds(cpd_t, cpd_t1)
    assert bn.check_model()
    return bn


class TestForwardInference:
    """Core inference tests."""

    def test_exact_junction_tree_deterministic(self, chain_bn: DiscreteBayesianNetwork):
        """Chain A→B→C: given A=0, P(B=0) should be high."""
        result = forward_inference(
            chain_bn,
            evidence={"A": 0},
            variables=["B"],
            method="exact",
        )
        assert "B" in result
        probs = result["B"]
        assert len(probs) == 2
        # P(B=0 | A=0) = 0.9
        assert abs(float(probs[0]) - 0.9) < 0.001
        assert abs(float(probs[1]) - 0.1) < 0.001

    def test_exact_junction_tree_missing_evidence(self, chain_bn: DiscreteBayesianNetwork):
        """Missing evidence → prior marginal used."""
        result = forward_inference(
            chain_bn,
            evidence={},  # no evidence
            variables=["A"],
            method="exact",
        )
        assert "A" in result
        probs = result["A"]
        # Prior: P(A=0) = 0.5
        assert abs(float(probs[0]) - 0.5) < 0.001

    def test_temporal_prediction(self, temporal_bn: DiscreteBayesianNetwork):
        """Given s0_t=0, s0_t1 should have 80% chance of 0."""
        result = forward_inference(
            temporal_bn,
            evidence={"s0_t": 0},
            variables=["s0_t1"],
            method="exact",
        )
        assert "s0_t1" in result
        probs = result["s0_t1"]
        assert abs(float(probs[0]) - 0.8) < 0.001  # P(s0_t1=0 | s0_t=0) = 0.8

    def test_large_graph_fallback(self):
        """|V| > 100 → ValueError (Phase 3.2 deferred)."""
        # Create a chain of 101 nodes
        edges = [(f"X{i}", f"X{i+1}") for i in range(100)]
        large_bn = DiscreteBayesianNetwork(edges)
        for i in range(101):
            cpd = TabularCPD(
                variable=f"X{i}", variable_card=2,
                values=[[0.5], [0.5]],
            )
            large_bn.add_cpds(cpd)

        with pytest.raises(ValueError, match="exact inference requires \|V\| ≤ 100"):
            forward_inference(large_bn, evidence={}, variables=["X100"], method="exact")

    def test_unknown_method_raises(self, chain_bn: DiscreteBayesianNetwork):
        """Unknown inference method raises ValueError."""
        with pytest.raises(ValueError, match="Unknown inference method"):
            forward_inference(chain_bn, evidence={}, variables=["A"], method="invalid")

    def test_sampling_method_raises(self, chain_bn: DiscreteBayesianNetwork):
        """Sampling method raises NotImplementedError."""
        with pytest.raises(NotImplementedError, match="deferred to Phase 3.2"):
            forward_inference(chain_bn, evidence={}, variables=["A"], method="sampling")

    def test_empty_graph(self):
        """Empty graph → uniform distributions."""
        empty_bn = DiscreteBayesianNetwork()
        result = forward_inference(empty_bn, evidence={}, variables=["X"], method="exact")
        assert "X" in result
        assert abs(float(result["X"][0]) - 0.5) < 0.001


class TestInferNextState:
    """Convenience wrapper tests."""

    def test_infer_next_state_returns_state(self, temporal_bn: DiscreteBayesianNetwork):
        """infer_next_state returns a valid StateVector."""
        state = StateVector(
            values=np.array([0], dtype=np.float32),
            precision=np.ones(1, dtype=np.float32),
        )
        pred, confidence = infer_next_state(temporal_bn, state)
        assert isinstance(pred, StateVector)
        assert pred.values.shape == (1,)
        assert 0.0 <= confidence <= 1.0

    def test_infer_next_state_high_confidence(self, temporal_bn: DiscreteBayesianNetwork):
        """Deterministic temporal edge → high confidence."""
        state = StateVector(
            values=np.array([0], dtype=np.float32),
            precision=np.ones(1, dtype=np.float32),
        )
        pred, confidence = infer_next_state(temporal_bn, state)
        # P(s0_t1=0 | s0_t=0) = 0.8
        assert pred.values[0] == 0.0
        assert confidence >= 0.5

    def test_infer_next_state_fallback_on_no_targets(self):
        """BN with no _t1 targets → low-confidence identity fallback."""
        bn = DiscreteBayesianNetwork()
        state = StateVector(
            values=np.array([1.0], dtype=np.float32),
            precision=np.ones(1, dtype=np.float32),
        )
        pred, confidence = infer_next_state(bn, state)
        assert pred.values[0] == 1.0  # identity
        assert confidence == 0.0  # zero confidence
