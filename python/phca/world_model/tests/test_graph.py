"""
Tests for G' Probabilistic Graph (PHCA-3.1-007).

Cross-ref: v3.0 §2.2 Definition 2.4b, Definition 2.5, §D.3
"""

from __future__ import annotations

import pytest
import numpy as np

from phca.config import StateVector
from phca.world_model.graph import WorldModelGPrime, StateNode, TemporalEdge


@pytest.fixture
def empty_model() -> WorldModelGPrime:
    """Empty G' model (no nodes)."""
    return WorldModelGPrime(state_dim=4, action_dim=2, seed=42)


@pytest.fixture
def deterministic_chain() -> WorldModelGPrime:
    """A→B→C deterministic chain with 2-state discrete CPDs."""
    model = WorldModelGPrime(state_dim=1, action_dim=0, seed=42)

    model.add_node(StateNode(
        name="A_t", cpd_type="discrete", cardinality=2,
        params=np.array([[0.5], [0.5]]),  # uniform prior
    ))
    model.add_node(StateNode(
        name="B_t", cpd_type="discrete", parents=["A_t"], cardinality=2,
        params=np.array([[0.9, 0.1], [0.1, 0.9]]),  # mostly follow A
    ))
    model.add_node(StateNode(
        name="C_t", cpd_type="discrete", parents=["B_t"], cardinality=2,
        params=np.array([[0.95, 0.05], [0.05, 0.95]]),  # mostly follow B
    ))
    model.add_node(StateNode(
        name="s0_t", cpd_type="discrete", cardinality=2,
        params=np.array([[0.5], [0.5]]),
    ))
    model.add_node(StateNode(
        name="s0_t1", cpd_type="discrete", parents=["s0_t"], cardinality=2,
        params=np.array([[0.8, 0.2], [0.2, 0.8]]),  # 80% stay, 20% flip
    ))

    model.add_temporal_edge(TemporalEdge(source="s0_t", target="s0_t1"))
    model._build_graph()
    return model


class TestWorldModelGPrime:
    """Core G' tests."""

    def test_empty_graph_predict_returns_low_confidence(self, empty_model: WorldModelGPrime):
        """Empty graph returns identity prediction with 0.0 confidence."""
        state = StateVector(values=np.ones(4, dtype=np.float32), precision=np.ones(4, dtype=np.float32))
        pred, confidence = empty_model.predict(state, np.zeros(2))
        assert isinstance(pred, StateVector)
        assert pred.values.shape == (4,)
        assert confidence == 0.0

    def test_add_node_duplicate_raises(self, empty_model: WorldModelGPrime):
        """Adding duplicate node raises ValueError."""
        node = StateNode(name="s0_t", cpd_type="discrete", cardinality=2)
        empty_model.add_node(node)
        with pytest.raises(ValueError, match="already exists"):
            empty_model.add_node(node)

    def test_add_temporal_edge_missing_source_raises(self, empty_model: WorldModelGPrime):
        """Adding edge with missing source raises ValueError."""
        edge = TemporalEdge(source="nonexistent", target="s0_t1")
        with pytest.raises(ValueError, match="not in graph"):
            empty_model.add_temporal_edge(edge)

    def test_add_causal_edge_missing_target_raises(self, empty_model: WorldModelGPrime):
        """Adding causal edge with missing target raises ValueError."""
        empty_model.add_node(StateNode(name="a_t", cpd_type="discrete", cardinality=2))
        with pytest.raises(ValueError, match="not in graph"):
            empty_model.add_causal_edge("a_t", "nonexistent")

    def test_deterministic_chain_high_confidence(self, deterministic_chain: WorldModelGPrime):
        """Deterministic-ish chain → confidence ≈ high."""
        state = StateVector(
            values=np.array([0], dtype=np.float32),
            precision=np.ones(1, dtype=np.float32),
        )
        pred, confidence = deterministic_chain.predict(state, np.zeros(0))
        assert confidence > 0.5  # should be fairly confident

    def test_learn_updates_params(self, empty_model: WorldModelGPrime):
        """learn() stores state in history for similarity search."""
        state_t = StateVector(
            values=np.array([0, 1, 2, 3], dtype=np.float32),
            precision=np.ones(4, dtype=np.float32),
        )
        state_t1 = StateVector(
            values=np.array([1, 2, 3, 4], dtype=np.float32),
            precision=np.ones(4, dtype=np.float32),
        )

        empty_model.learn(state_t, np.zeros(2), state_t1, error=0.5)
        assert len(empty_model.state_history) == 1

        # Second learn adds another
        empty_model.learn(state_t1, np.zeros(2), state_t, error=0.3)
        assert len(empty_model.state_history) == 2

    def test_reset_clears_state(self, empty_model: WorldModelGPrime):
        """Reset clears state history and CPD params."""
        state_t = StateVector(
            values=np.zeros(4, dtype=np.float32),
            precision=np.ones(4, dtype=np.float32),
        )
        empty_model.learn(state_t, np.zeros(2), state_t, error=0.0)
        assert len(empty_model.state_history) == 1

        empty_model.reset()
        assert len(empty_model.state_history) == 0


