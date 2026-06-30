"""
Tests for Gaussian Bayesian Network Inference (PHCA-3.2-001).

Covers:
- compute_joint_moments() for simple and conditional Gaussians
- posterior() analytic inference
- confidence_from_variance() mapping
- sample_posterior() for large graphs
- WorldModelGPrime.predict_continuous() integration

v3.0 Reference: §2.2 Definition 2.4b (Phase 3.2 Gaussian CPDs)
"""

from __future__ import annotations

import pytest
import numpy as np

from phca.config import StateVector
from phca.world_model.gaussian import (
    compute_joint_moments,
    posterior,
    confidence_from_variance,
    sample_posterior,
)
from phca.world_model.graph import WorldModelGPrime, StateNode


class TestComputeJointMoments:
    """Analytic joint moment computation for Gaussian BNs."""

    def test_single_root_node(self):
        """Single root node: X ~ N(0, 1)."""
        mu, cov = compute_joint_moments(
            node_order=["X"],
            betas={"X": [0.0]},
            sigmas={"X": 1.0},
            parents={"X": []},
        )
        assert mu.shape == (1,)
        assert cov.shape == (1, 1)
        assert abs(mu[0]) < 1e-10
        assert abs(cov[0, 0] - 1.0) < 1e-6

    def test_chain_two_nodes(self):
        """X ~ N(0,1), Y|X ~ N(0.5 + 1.0*X, 0.5)."""
        mu, cov = compute_joint_moments(
            node_order=["X", "Y"],
            betas={"X": [0.0], "Y": [0.5, 1.0]},
            sigmas={"X": 1.0, "Y": np.sqrt(0.5)},
            parents={"X": [], "Y": ["X"]},
        )
        # E[Y] = 0.5 + 1.0 * E[X] = 0.5
        assert abs(mu[1] - 0.5) < 1e-6
        # Var[Y] = Var[E[Y|X]] + E[Var[Y|X]] = 1.0^2 * 1.0 + 0.5 = 1.5
        assert abs(cov[1, 1] - 1.5) < 0.01

    def test_deterministic_chain(self):
        """X ~ N(0,1), Y = 2*X (deterministic)."""
        mu, cov = compute_joint_moments(
            node_order=["X", "Y"],
            betas={"X": [0.0], "Y": [0.0, 2.0]},
            sigmas={"X": 1.0, "Y": 0.001},  # near-deterministic
            parents={"X": [], "Y": ["X"]},
        )
        assert abs(mu[1]) < 1e-6  # E[Y] = 0
        assert abs(cov[0, 1] - 2.0) < 0.01  # Cov(X, Y) = 2 * Var(X)

    def test_three_node_diamond(self):
        """X → Y, X → Z (diamond with common cause)."""
        mu, cov = compute_joint_moments(
            node_order=["X", "Y", "Z"],
            betas={"X": [0.0], "Y": [0.0, 1.0], "Z": [0.0, 1.0]},
            sigmas={"X": 1.0, "Y": 0.5, "Z": 0.5},
            parents={"X": [], "Y": ["X"], "Z": ["X"]},
        )
        # E[Y] = E[Z] = 0
        assert abs(mu[1]) < 1e-6 and abs(mu[2]) < 1e-6
        # Cov(Y, Z) = Var(X) = 1.0 (both depend on X with coefficient 1.0)
        assert abs(cov[1, 2] - 1.0) < 0.05

    def test_order_independence(self):
        """Results should be invariant to topological ordering."""
        order1 = ["A", "B", "C"]
        order2 = ["B", "A", "C"]
        params = {
            "betas": {"A": [0.0], "B": [0.0, 0.5], "C": [0.0, 0.3]},
            "sigmas": {"A": 1.0, "B": 0.5, "C": 0.3},
            "parents": {"A": [], "B": ["A"], "C": ["A"]},
        }
        mu1, cov1 = compute_joint_moments(order1, **params)
        mu2, cov2 = compute_joint_moments(order2, **params)

        # Marginals for the shared variable should match
        assert abs(mu1[0] - mu2[0]) < 1e-6  # A
        assert abs(mu1[2] - mu2[2]) < 1e-6  # C


class TestPosterior:
    """Analytic posterior computation given evidence."""

    def test_single_evidence(self):
        """X ~ N(0,1), Y|X ~ N(X, 0.5). Given X=1, P(Y) should be N(1, 0.5)."""
        mu = np.array([0.0, 0.0])
        cov = np.array([[1.0, 1.0], [1.0, 1.5]])

        result = posterior(mu, cov, {"X": 1.0}, ["Y"], ["X", "Y"])
        assert "Y" in result
        post_mean, post_std = result["Y"]
        # E[Y|X=1] = 0 + (1.0/1.0)*(1-0) = 1.0
        assert abs(post_mean - 1.0) < 0.01
        # Var[Y|X=1] = 1.5 - 1.0^2/1.0 = 0.5
        assert abs(post_std**2 - 0.5) < 0.01

    def test_multiple_evidence(self):
        """Two evidence variables queried."""
        mu = np.array([0.0, 0.0, 5.0])
        cov = np.eye(3) * np.array([1.0, 2.0, 3.0])

        result = posterior(mu, cov, {"X": 1.0}, ["Y", "Z"], ["X", "Y", "Z"])
        assert "Y" in result
        assert "Z" in result

    def test_no_evidence(self):
        """No evidence → returns prior."""
        mu = np.array([0.0, 5.0])
        cov = np.array([[1.0, 0.0], [0.0, 4.0]])

        result = posterior(mu, cov, {}, ["X", "Y"], ["X", "Y"])
        assert "X" in result
        assert "Y" in result
        assert abs(result["X"][0]) < 1e-6  # prior mean
        assert abs(result["Y"][0] - 5.0) < 1e-6  # prior mean

    def test_missing_evidence_variable_raises(self):
        """Evidence for non-existent variable raises ValueError."""
        mu = np.array([0.0])
        cov = np.eye(1)

        with pytest.raises(ValueError, match="not found"):
            posterior(mu, cov, {"NONEXISTENT": 1.0}, ["X"], ["X"])

    def test_confidence_from_variance(self):
        """confidence = 1/(1+variance)."""
        assert abs(confidence_from_variance(0.0) - 1.0) < 1e-6
        assert abs(confidence_from_variance(1.0) - 0.5) < 1e-6
        assert confidence_from_variance(100.0) < 0.01
        assert 0.0 < confidence_from_variance(1e-12) <= 1.0


class TestSamplePosterior:
    """Sampling-based inference for large graphs."""

    def test_small_graph_falls_back_to_exact(self):
        """Graph with ≤ 200 nodes uses exact inference."""
        result = sample_posterior(
            node_order=["X", "Y"],
            betas={"X": [0.0], "Y": [0.5, 1.0]},
            sigmas={"X": 1.0, "Y": 0.5},
            parents={"X": [], "Y": ["X"]},
            evidence={"X": 1.0},
            query_vars=["Y"],
            n_samples=100,
        )
        assert "Y" in result
        post_mean, post_std = result["Y"]
        # E[Y|X=1] = 0.5 + 1.0*1.0 = 1.5
        assert abs(post_mean - 1.5) < 0.1

    def test_large_graph_sampling(self):
        """Graph with > 200 nodes uses rejection sampling."""
        n = 250
        node_order = [f"X{i}" for i in range(n)]
        betas = {f"X{i}": [0.0] for i in range(n)}
        sigmas = {f"X{i}": 1.0 for i in range(n)}
        parents = {f"X{i}": [] for i in range(n)}

        result = sample_posterior(
            node_order=node_order, betas=betas, sigmas=sigmas,
            parents=parents, evidence={"X0": 0.5}, query_vars=["X1"],
            n_samples=100, tolerance=1.0,
        )
        assert "X1" in result
        mean_val, std_val = result["X1"]
        # No evidence propagation between independent nodes
        assert abs(mean_val) < 2.0


class TestPredictContinuous:
    """Integration test: WorldModelGPrime.predict_continuous()."""

    def test_small_grid_predict_returns_state(self):
        """predict_continuous returns a valid StateVector."""
        model = WorldModelGPrime.build_gaussian_grid(
            state_dim=4, action_dim=2, transition_std=0.5, seed=42,
        )
        state = StateVector(
            values=np.zeros(4, dtype=np.float32),
            precision=np.ones(4, dtype=np.float32),
        )
        action = np.zeros(2, dtype=np.float32)

        pred, confidence = model.predict_continuous(state, action)
        assert isinstance(pred, StateVector)
        assert pred.values.shape == (4,)
        assert 0.0 <= confidence <= 1.0

    def test_confidence_high_with_low_noise(self):
        """Low transition std → high confidence."""
        model = WorldModelGPrime.build_gaussian_grid(
            state_dim=4, action_dim=2, transition_std=0.1, seed=42,
        )
        state = StateVector(
            values=np.ones(4, dtype=np.float32),
            precision=np.ones(4, dtype=np.float32),
        )
        action = np.zeros(2, dtype=np.float32)

        _, confidence = model.predict_continuous(state, action)
        # |V| = 4*2 + 2 = 10 nodes, low noise → high confidence
        assert confidence > 0.8

    def test_confidence_low_with_high_noise(self):
        """High transition std → lower confidence."""
        model_low = WorldModelGPrime.build_gaussian_grid(
            state_dim=4, action_dim=2, transition_std=0.1, seed=42,
        )
        model_high = WorldModelGPrime.build_gaussian_grid(
            state_dim=4, action_dim=2, transition_std=5.0, seed=42,
        )
        state = StateVector(
            values=np.ones(4, dtype=np.float32),
            precision=np.ones(4, dtype=np.float32),
        )
        action = np.zeros(2, dtype=np.float32)

        _, conf_low = model_low.predict_continuous(state, action)
        _, conf_high = model_high.predict_continuous(state, action)
        assert conf_low > conf_high

    def test_predict_follows_trend(self):
        """Prediction should be close to β₀ + β₁·state for identity-like model."""
        model = WorldModelGPrime.build_gaussian_grid(
            state_dim=4, action_dim=2, transition_std=0.01, seed=42,
        )
        state = StateVector(
            values=np.full(4, 2.0, dtype=np.float32),
            precision=np.ones(4, dtype=np.float32),
        )
        action = np.zeros(2, dtype=np.float32)

        pred, _ = model.predict_continuous(state, action)
        # For each dimension: β₀=0, β₁=0.95 → pred ≈ 0 + 0.95 * 2.0 = 1.9
        for i in range(4):
            assert abs(pred.values[i] - 1.9) < 0.1

    def test_empty_graph_returns_identity(self):
        """Empty graph → identity with zero confidence."""
        model = WorldModelGPrime(state_dim=4, action_dim=2, seed=42)
        state = StateVector(
            values=np.ones(4, dtype=np.float32),
            precision=np.ones(4, dtype=np.float32),
        )
        pred, conf = model.predict_continuous(state, np.zeros(2))
        assert np.allclose(pred.values, state.values)
        assert conf == 0.0

    def test_no_temporal_targets_returns_identity(self):
        """Graph with only _t nodes → identity with 0.5 confidence."""
        model = WorldModelGPrime(state_dim=2, action_dim=1, seed=42)
        model.add_node(StateNode(name="s0_t", cpd_type="gaussian", parents=[], std=1.0))
        model.add_node(StateNode(name="s1_t", cpd_type="gaussian", parents=[], std=1.0))

        state = StateVector(
            values=np.array([1.0, 2.0], dtype=np.float32),
            precision=np.ones(2, dtype=np.float32),
        )
        pred, conf = model.predict_continuous(state, np.zeros(1))
        assert np.allclose(pred.values, state.values)
        assert conf == 0.5

    def test_has_gaussian_nodes_true(self):
        """build_gaussian_grid creates a model with Gaussian nodes."""
        model = WorldModelGPrime.build_gaussian_grid(
            state_dim=4, action_dim=2, seed=42,
        )
        assert model.has_gaussian_nodes()

    def test_has_gaussian_nodes_false(self):
        """Discrete-only model returns False for has_gaussian_nodes()."""
        model = WorldModelGPrime(state_dim=4, action_dim=2, seed=42)
        model.add_node(StateNode(name="s0_t", cpd_type="discrete", cardinality=2))
        assert not model.has_gaussian_nodes()

    def test_predict_with_action_influence(self):
        """Action should influence the prediction."""
        model = WorldModelGPrime.build_gaussian_grid(
            state_dim=2, action_dim=2, transition_std=0.01, seed=42,
        )
        state = StateVector(
            values=np.zeros(2, dtype=np.float32),
            precision=np.ones(2, dtype=np.float32),
        )

        # No action
        pred_no_action, _ = model.predict_continuous(state, np.zeros(2))
        # Action = 1.0
        pred_action, _ = model.predict_continuous(
            state, np.array([1.0, 0.0], dtype=np.float32)
        )

        # Action should push prediction away from zero (β₂ = 0.1)
        # pred_no_action ≈ 0, pred_action ≈ 0.1
        assert abs(pred_action.values[0] - pred_no_action.values[0]) > 0.05
