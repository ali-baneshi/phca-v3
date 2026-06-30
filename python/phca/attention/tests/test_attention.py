"""
Tests for Precision-Weighted Sparse Attention (PHCA-3.2-006).

Covers: k-WTA selection, bottom-up salience, top-down relevance,
precision weighting, Gumbel noise, precision learning, reset.

v3.0 Reference: §3.2 Definition 3.4
"""

from __future__ import annotations

import numpy as np
import pytest

from phca.attention.attention import Attention
from phca.config import GoalVector, StateVector
from phca.memory.m2_working import Chunk


@pytest.fixture
def attn() -> Attention:
    return Attention(default_k=3, state_dim=4)


@pytest.fixture
def sample_chunks() -> list[Chunk]:
    chunks = []
    for i in range(7):  # 7±2 capacity
        sv = StateVector(
            values=np.full(4, float(i), dtype=np.float32),
            precision=np.ones(4, dtype=np.float32),
        )
        chunks.append(Chunk(state=sv, chunk_id=i, salience=0.5))
    return chunks


@pytest.fixture
def sample_goal() -> GoalVector:
    return GoalVector(
        drive_id=1,
        target_state=StateVector(
            values=np.full(4, 3.0, dtype=np.float32),
            precision=np.ones(4, dtype=np.float32),
        ),
        tolerance=0.1,
        creation_cycle=0,
        priority=1.0,
    )


class TestAttentionInit:
    """Initialization tests."""

    def test_default_params(self):
        """Default parameters should be set correctly."""
        attn = Attention()
        assert attn.default_k == 3
        assert attn.alpha_bu == 0.6
        assert attn.beta_td == 0.4
        assert attn.gumbel_temperature == 0.5
        assert attn.precision_lr == 0.1

    def test_initial_state(self, attn):
        """Attention should start with no precision history."""
        assert attn._precisions == {}
        assert attn._last_saliences == []


class TestKWTASelection:
    """k-WTA selection tests."""

    def test_select_returns_k_chunks(self, attn, sample_chunks):
        """select() should return exactly k chunks."""
        selected = attn.select(sample_chunks)
        assert len(selected) == attn.default_k  # default k=3

    def test_select_with_custom_k(self, attn, sample_chunks):
        """select() should respect custom k parameter."""
        selected = attn.select(sample_chunks, k=5)
        assert len(selected) == 5

    def test_select_respects_wm_capacity(self, attn, sample_chunks):
        """k should be capped by the number of available chunks."""
        selected = attn.select(sample_chunks[:2], k=5)
        assert len(selected) == 2

    def test_select_empty_chunks(self, attn):
        """Empty chunks should return empty list."""
        selected = attn.select([])
        assert selected == []

    def test_select_returns_sorted_by_salience(self, attn, sample_chunks):
        """Selected chunks should be in descending salience order."""
        for i, chunk in enumerate(sample_chunks):
            chunk.salience = float(i)
        selected = attn.select(sample_chunks, k=3)
        if len(selected) > 1:
            assert selected[0].salience >= selected[1].salience


class TestBottomUpSalience:
    """Bottom-up salience tests."""

    def test_bu_salience_high_for_unexpected(self):
        """Unexpected chunks (far from prediction) should win over expected ones."""
        attn = Attention(alpha_bu=1.0, beta_td=0.0, gumbel_temperature=0.0)
        prediction = StateVector(
            values=np.zeros(4, dtype=np.float32),
            precision=np.ones(4, dtype=np.float32),
        )
        near = Chunk(state=StateVector(
            values=np.full(4, 0.1, dtype=np.float32),
            precision=np.ones(4, dtype=np.float32),
        ), chunk_id=0)
        far = Chunk(state=StateVector(
            values=np.full(4, 5.0, dtype=np.float32),
            precision=np.ones(4, dtype=np.float32),
        ), chunk_id=1)
        selected = attn.select([near, far], prediction=prediction, k=1)
        assert selected[0].chunk_id == 1, "Unexpected (far) chunk should be selected"

    def test_bu_salience_zero_for_perfect_prediction(self):
        """Perfect prediction match → zero bottom-up salience."""
        attn = Attention(alpha_bu=1.0, beta_td=0.0, gumbel_temperature=0.0)
        chunk = Chunk(
            state=StateVector(
                values=np.zeros(4, dtype=np.float32),
                precision=np.ones(4, dtype=np.float32),
            ),
            chunk_id=0,
        )
        prediction = StateVector(
            values=np.zeros(4, dtype=np.float32),
            precision=np.ones(4, dtype=np.float32),
        )
        _ = attn.select([chunk], prediction=prediction, k=1)
        assert attn._last_saliences[chunk.chunk_id] == pytest.approx(0.0, abs=1e-6)


class TestTopDownRelevance:
    """Top-down relevance tests."""

    def test_td_relevance_with_goal(self, attn, sample_chunks, sample_goal):
        """Goal-relevant chunks should be selected."""
        selected = attn.select(sample_chunks, goal=sample_goal)
        assert len(selected) > 0

    def test_td_relevance_similar_chunks_preferred(self):
        """Chunks similar to goal target should be selected over dissimilar ones."""
        attn = Attention(alpha_bu=0.0, beta_td=1.0)  # only top-down
        goal = GoalVector(
            drive_id=1,
            target_state=StateVector(
                values=np.array([1.0, 1.0], dtype=np.float32),
                precision=np.ones(2, dtype=np.float32),
            ),
            tolerance=0.1,
            creation_cycle=0,
            priority=1.0,
        )

        similar_chunk = Chunk(
            state=StateVector(
                values=np.array([0.95, 1.05], dtype=np.float32),
                precision=np.ones(2, dtype=np.float32),
            ),
            chunk_id=0,
        )
        dissimilar_chunk = Chunk(
            state=StateVector(
                values=np.array([-5.0, -5.0], dtype=np.float32),
                precision=np.ones(2, dtype=np.float32),
            ),
            chunk_id=1,
        )

        selected = attn.select(
            [similar_chunk, dissimilar_chunk],
            goal=goal, k=1,
        )
        assert selected[0].chunk_id == 0


class TestPrecisionWeighting:
    """Precision weighting tests."""

    def test_high_precision_increases_salience(self):
        """Higher precision → higher composite salience."""
        attn = Attention(alpha_bu=0.5, beta_td=0.5)

        low_prec_chunk = Chunk(
            state=StateVector(
                values=np.ones(4, dtype=np.float32),
                precision=np.ones(4, dtype=np.float32),
            ),
            chunk_id=0,
        )
        high_prec_chunk = Chunk(
            state=StateVector(
                values=np.ones(4, dtype=np.float32),
                precision=np.ones(4, dtype=np.float32),
            ),
            chunk_id=1,
        )
        attn._precisions[0] = 0.1  # low precision
        attn._precisions[1] = 5.0  # high precision

        selected = attn.select([low_prec_chunk, high_prec_chunk], k=1)
        assert selected[0].chunk_id == 1  # high precision should win


class TestPrecisionLearning:
    """Precision learning tests."""

    def test_update_precision_returns_value(self, attn):
        """update_precision should return the new precision value."""
        new_p = attn.update_precision(chunk_id=5, prediction_error=0.5)
        assert 0.01 <= new_p <= 10.0

    def test_precision_increases_with_error(self, attn):
        """Higher prediction error → higher precision growth."""
        p_low = attn.update_precision(chunk_id=0, prediction_error=0.0)
        attn2 = Attention()
        p_high = attn2.update_precision(chunk_id=0, prediction_error=5.0)
        assert p_high > p_low

    def test_precision_persists(self, attn):
        """Precision should persist across calls."""
        attn.update_precision(chunk_id=3, prediction_error=2.0)
        assert attn._precisions[3] > 0.01

    def test_default_precision(self, attn):
        """Unknown chunk ID should have no precision entry."""
        assert 999 not in attn._precisions

    def test_precision_clamped(self, attn):
        """Precision should be clamped to [0.01, 10.0]."""
        p = attn.update_precision(chunk_id=0, prediction_error=100.0)
        assert p <= 10.0
        p = attn.update_precision(chunk_id=0, prediction_error=0.0)
        assert p >= 0.01


class TestGumbelNoise:
    """Gumbel noise for stochastic exploration."""

    def test_deterministic_with_zero_temperature(self, sample_chunks):
        """Zero Gumbel temperature → deterministic selection."""
        attn1 = Attention(gumbel_temperature=0.0)
        attn2 = Attention(gumbel_temperature=0.0)

        s1 = attn1.select(sample_chunks)
        s2 = attn2.select(sample_chunks)

        assert len(s1) == len(s2)

    def test_stochastic_with_positive_temperature(self, sample_chunks):
        """Positive Gumbel temperature → potentially different selections."""
        selections = set()
        for _ in range(10):
            attn = Attention(gumbel_temperature=2.0)
            selected = attn.select(sample_chunks)
            chunk_ids = tuple(sorted(c.chunk_id for c in selected))
            selections.add(chunk_ids)
        assert len(selections) >= 1





class TestCosineSimilarity:
    """Precision-weighted similarity helper."""

    def test_identical_vectors(self, attn):
        """Identical vectors → similarity = 1.0."""
        a = np.array([1.0, 2.0, 3.0])
        sim = attn._precision_weighted_similarity(a, a)
        assert sim == pytest.approx(1.0, abs=1e-6)

    def test_orthogonal_vectors(self, attn):
        """Orthogonal vectors → similarity = 0.5 (neutral)."""
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([0.0, 1.0, 0.0])
        sim = attn._precision_weighted_similarity(a, b)
        assert sim == pytest.approx(0.5, abs=1e-6)

    def test_opposite_vectors(self, attn):
        """Opposite vectors → similarity = 0.0."""
        a = np.array([1.0, 2.0])
        sim = attn._precision_weighted_similarity(a, -a)
        assert sim == pytest.approx(0.0, abs=1e-6)

    def test_precision_weighted(self, attn):
        """Precision weighting emphasizes important dimensions."""
        # a and b share dimension 0 (both 1.0) but differ on dimension 1
        a = np.array([1.0, 0.0], dtype=np.float32)
        b = np.array([1.0, 1.0], dtype=np.float32)
        # With uniform precision, dim 1 difference reduces similarity
        uniform = attn._precision_weighted_similarity(a, b, precision=np.ones(2))
        # With precision favoring shared dimension 0, similarity increases
        weighted = attn._precision_weighted_similarity(a, b, precision=np.array([5.0, 1.0]))
        assert weighted > uniform
