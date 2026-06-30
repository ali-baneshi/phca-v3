"""
Tests for k-NN Similarity Search (PHCA-3.1-007 — similarity component).

Cross-ref: v3.0 §D.3
"""

from __future__ import annotations

import pytest
import numpy as np

from phca.config import StateVector
from phca.world_model.similarity import knn_similarity


@pytest.fixture
def state_history() -> list[StateVector]:
    history = []
    for i in range(10):
        history.append(StateVector(
            values=np.full(4, float(i), dtype=np.float32),
            precision=np.ones(4, dtype=np.float32),
        ))
    return history


class TestKNN:
    """k-NN similarity search tests."""

    def test_empty_history_returns_empty(self):
        """Empty history → empty list."""
        query = StateVector(values=np.ones(4, dtype=np.float32), precision=np.ones(4, dtype=np.float32))
        results = knn_similarity(query, [], k=5)
        assert results == []

    def test_returns_k_results(self, state_history: list[StateVector]):
        """Request k results → exactly k returned."""
        query = StateVector(values=np.ones(4, dtype=np.float32), precision=np.ones(4, dtype=np.float32))
        results = knn_similarity(query, state_history, k=3)
        assert len(results) == 3

    def test_zero_k_returns_empty(self, state_history: list[StateVector]):
        """k=0 → empty list."""
        query = StateVector(values=np.ones(4, dtype=np.float32), precision=np.ones(4, dtype=np.float32))
        results = knn_similarity(query, state_history, k=0)
        assert results == []

    def test_negative_k_raises(self, state_history: list[StateVector]):
        """Negative k raises ValueError."""
        query = StateVector(values=np.ones(4, dtype=np.float32), precision=np.ones(4, dtype=np.float32))
        with pytest.raises(ValueError, match="k must be >= 0"):
            knn_similarity(query, state_history, k=-1)

    def test_sorted_by_similarity_descending(self, state_history: list[StateVector]):
        """Results sorted by similarity descending (most similar first)."""
        query = StateVector(values=np.full(4, 0.0, dtype=np.float32), precision=np.ones(4, dtype=np.float32))
        results = knn_similarity(query, state_history, k=5)
        scores = [score for _, score in results]
        assert all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1))

    def test_exact_match_has_score_1(self):
        """Exact match returns similarity = 1.0."""
        vals = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        query = StateVector(values=vals.copy(), precision=np.ones(4, dtype=np.float32))
        match = StateVector(values=vals.copy(), precision=np.ones(4, dtype=np.float32))
        history = [match]
        results = knn_similarity(query, history, k=1)
        assert abs(results[0][1] - 1.0) < 0.001

    def test_worst_match_has_low_score(self):
        """Very different state returns low similarity."""
        query = StateVector(
            values=np.zeros(4, dtype=np.float32),
            precision=np.ones(4, dtype=np.float32),
        )
        far = StateVector(
            values=np.full(4, 1000.0, dtype=np.float32),
            precision=np.ones(4, dtype=np.float32),
        )
        history = [far]
        results = knn_similarity(query, history, k=1)
        assert results[0][1] < 0.5  # far away → low similarity

    def test_unsupported_metric_raises(self, state_history: list[StateVector]):
        """Unsupported metric raises ValueError."""
        query = StateVector(values=np.ones(4, dtype=np.float32), precision=np.ones(4, dtype=np.float32))
        with pytest.raises(ValueError, match="Unsupported metric"):
            knn_similarity(query, state_history, k=3, metric="manhattan")

    def test_cosine_metric_returns_valid_scores(self):
        """Cosine metric returns 0-1 scores."""
        query = StateVector(
            values=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            precision=np.ones(4, dtype=np.float32),
        )
        match = StateVector(
            values=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            precision=np.ones(4, dtype=np.float32),
        )
        opposite = StateVector(
            values=np.array([-1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            precision=np.ones(4, dtype=np.float32),
        )
        history = [match, opposite]
        results = knn_similarity(query, history, k=2, metric="cosine")

        assert len(results) == 2
        # Match should have score ≈ 1.0
        assert abs(results[0][1] - 1.0) < 0.1
        # Opposite should have lower score
        assert results[0][1] > results[1][1]
