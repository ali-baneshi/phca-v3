"""
PHCA v3.0 — World Model G' (Probabilistic Graph).

Phase 3.1: Simplified Bayesian network with discrete/gaussian CPDs.
Phase 3.2+: Full ensemble with VSA integration.

v3.0 References:
    - §2.2 Definition 2.4b (Phase 3.1 simplified G')
    - §2.2 Definition 2.5 (prediction via G' only)
    - §D.3 (similarity search replaces VSA)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD, DiscreteFactor
from pgmpy.inference import VariableElimination

from phca.config import StateVector
from phca.logging import logger, _log
from phca.world_model.gaussian import (
    compute_joint_moments,
    posterior,
    conditional_covariance,
    gaussian_mutual_information,
    confidence_from_variance,
    sample_posterior,
)


@dataclass
class StateNode:
    """A node in the G' probabilistic graph (v3.0 §2.2).

    Attributes:
        name: Unique node identifier (e.g., "pos_x_t", "pos_y_t1").
        cpd_type: Type of conditional probability distribution.
        parents: Names of parent nodes.
        cardinality: Number of discrete states (discrete CPDs only).
        params: CPD parameters — for discrete: probability table as 2D array.
        mean: Mean for Gaussian CPDs.
        std: Standard deviation for Gaussian CPDs.
    """
    name: str
    cpd_type: Literal["discrete", "gaussian", "conditional_gaussian"]
    parents: List[str] = field(default_factory=list)
    cardinality: int = 2
    params: Optional[np.ndarray] = None
    mean: float = 0.0
    std: float = 1.0


@dataclass
class TemporalEdge:
    """A temporal dependency between time-sliced variables (v3.0 §2.2).

    Attributes:
        source: Source variable name (e.g., "pos_x_t").
        target: Target variable name (e.g., "pos_x_t1").
        lag: Temporal lag (1 for X_i^(t) → X_i^(t+1)).
        params: Transition parameters (for learnable edges).
    """
    source: str
    target: str
    lag: int = 1
    params: Optional[np.ndarray] = None


class WorldModelGPrime:
    """G' — Probabilistic World Model.

    Phase 3.1: Single-model ensemble (G' only). Supports discrete Bayesian
    networks via pgmpy with exact inference (variable elimination).

    Phase 3.2+: Dual-model (G' + V) with meta-gradient weights.
        Also supports continuous Gaussian CPDs with closed-form analytic
        inference (see predict_continuous()).
    """

    def __init__(self, state_dim: int, action_dim: int, seed: int = 42):
        """Initialize an empty G' graph.

        Args:
            state_dim: Dimensionality of the state space.
            action_dim: Dimensionality of the action space.
            seed: Random seed for reproducibility.
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.rng = np.random.RandomState(seed)

        # Graph structure
        self.nodes: Dict[str, StateNode] = {}
        self.temporal_edges: List[TemporalEdge] = []
        self.causal_edges: List[Tuple[str, str]] = []

        # pgmpy model (built lazily on first predict/learn call)
        self._bn: Optional[DiscreteBayesianNetwork] = None

        # State history for similarity search
        self.state_history: List[StateVector] = []

        # Learned CPD parameters (updated by learn())
        self._cpd_params: Dict[str, np.ndarray] = {}

        # Continuous Gaussian model parameters
        self._gaussian_betas: Dict[str, List[float]] = {}
        self._gaussian_sigmas: Dict[str, float] = {}
        self._gaussian_parents: Dict[str, List[str]] = {}

        # Inference cache: topology and joint moments are static for a given graph
        # Avoids recomputing O(n³) matrix inversion on every predict() call.
        # Tuple stores (node_order, betas, sigmas, parents_dict) — no redundant boolean flag.
        self._cached_topology: Tuple[List[str], Dict[str, List[float]],
                                     Dict[str, float], Dict[str, List[str]]] | None = None
        self._cached_joint_moments: Tuple[np.ndarray, np.ndarray] | None = None
        # Observability v4: cache last per-dim posterior std (uncertainty portrait).
        self._last_pred_std: Optional[np.ndarray] = None

    # ── Graph Construction ────────────────────────────────────

    def add_node(self, node: StateNode) -> None:
        """Add a node to the graph.

        Args:
            node: StateNode to add.

        Raises:
            ValueError: If node name already exists.
        """
        if node.name in self.nodes:
            raise ValueError(f"Node '{node.name}' already exists")
        self.nodes[node.name] = node

    def add_temporal_edge(self, edge: TemporalEdge) -> None:
        """Add a temporal edge between nodes.

        Args:
            edge: TemporalEdge to add.

        Raises:
            ValueError: If source or target node doesn't exist.
        """
        if edge.source not in self.nodes:
            raise ValueError(f"Source node '{edge.source}' not in graph")
        if edge.target not in self.nodes:
            raise ValueError(f"Target node '{edge.target}' not in graph")
        self.temporal_edges.append(edge)

    def add_causal_edge(self, source: str, target: str) -> None:
        """Add a causal edge (action → state variable).

        Args:
            source: Source node name (action variable).
            target: Target node name (state variable).

        Raises:
            ValueError: If source or target doesn't exist.
        """
        if source not in self.nodes:
            raise ValueError(f"Source node '{source}' not in graph")
        if target not in self.nodes:
            raise ValueError(f"Target node '{target}' not in graph")
        self.causal_edges.append((source, target))

    def _build_graph(self) -> None:
        """Build (or rebuild) the pgmpy DiscreteBayesianNetwork from current structure.

        Adds all nodes and edges from temporal/causal/parent relationships,
        then attaches CPDs. Automatically creates edges for node parent
        relationships to keep CPDs consistent with the DAG.
        """
        # Collect all edges from temporal, causal, and parent relationships
        edges = set()
        for e in self.temporal_edges:
            edges.add((e.source, e.target))
        for source, target in self.causal_edges:
            edges.add((source, target))
        for node_name, node in self.nodes.items():
            for parent in node.parents:
                if parent in self.nodes:
                    edges.add((parent, node_name))

        self._bn = DiscreteBayesianNetwork()
        # Add all nodes first (including isolated nodes with no edges)
        for node_name in self.nodes:
            self._bn.add_node(node_name)
        # Add all edges
        for parent, child in edges:
            self._bn.add_edge(parent, child)

        # Build CPDs for each node
        cpds = []
        for node_name, node in self.nodes.items():
            parents = [p for p in node.parents if p in self.nodes]

            if node.cpd_type == "discrete":
                cpd = self._build_discrete_cpd(node, parents)
                cpds.append(cpd)

        # For Gaussian nodes, store parameters for analytic inference
        for node_name, node in self.nodes.items():
            if node.cpd_type in ("gaussian", "conditional_gaussian"):
                parents = [p for p in node.parents if p in self.nodes]
                if node.params is not None:
                    beta = node.params.tolist()
                else:
                    # Default: identity with some noise
                    beta = [0.0] + [1.0] * len(parents) if parents else [0.0]
                self._gaussian_betas[node_name] = beta
                self._gaussian_sigmas[node_name] = node.std
                self._gaussian_parents[node_name] = parents

        self._bn.add_cpds(*cpds)

    def _build_discrete_cpd(self, node: StateNode, parents: List[str]) -> TabularCPD:
        """Build a TabularCPD for a discrete node.

        If params is provided, use it directly. Otherwise, use a uniform
        distribution over the node's cardinality.
        """
        var_card = node.cardinality

        if parents:
            evidence_card = [self.nodes[p].cardinality for p in parents]
            if node.params is not None:
                values = node.params
            else:
                # Uniform distribution: [1/var_card] * var_card, repeated for each parent combo
                n_combos = int(np.prod(evidence_card))
                values = np.ones((var_card, n_combos), dtype=np.float32) / var_card
            return TabularCPD(
                variable=node.name,
                variable_card=var_card,
                values=values.tolist(),
                evidence=parents,
                evidence_card=evidence_card,
            )
        else:
            if node.params is not None:
                values = node.params
            else:
                values = np.ones(var_card, dtype=np.float32) / var_card
            return TabularCPD(
                variable=node.name,
                variable_card=var_card,
                values=values.reshape(-1, 1).tolist(),
            )

    # ── Prediction (Discrete / pgmpy) ─────────────────────────

    def predict(
        self, state: StateVector, action: np.ndarray
    ) -> Tuple[StateVector, float]:
        """Predict next state given current state and action.

        Auto-dispatches to:
          - predict_continuous() if the graph has Gaussian CPD nodes
          - pgmpy exact inference (variable elimination) for discrete-only graphs

        Args:
            state: Current state vector.
            action: Action vector.

        Returns:
            Tuple of (predicted_state, confidence):
                predicted_state: StateVector with predicted values.
                confidence: Prediction confidence (0.0 = uncertain, 1.0 = certain).
        """
        # Auto-dispatch to continuous inference if Gaussian nodes exist
        if self.has_gaussian_nodes():
            return self.predict_continuous(state, action)

        if self._bn is None and len(self.nodes) > 0:
            self._build_graph()

        if self._bn is None or len(self.nodes) == 0:
            # Empty or unbuildable graph: return copy of state with zero confidence
            return (
                StateVector(
                    values=state.values.copy(),
                    precision=np.zeros_like(state.precision),
                    timestamp=state.timestamp + 1.0,
                    grounding_level=state.grounding_level,
                ),
                0.0,
            )

        # Collect evidence from current state
        evidence: Dict[str, float] = {}
        for i in range(min(self.state_dim, len(state.values))):
            var_name = f"s{i}_t"
            if var_name in self.nodes:
                # round to nearest integer for discrete CPD evidence
                evidence[var_name] = float(int(round(state.values[i])))

        try:
            # Run variable elimination for t+1 variables
            target_vars = [n for n in self.nodes if n.endswith("_t1")]
            if not target_vars:
                # No temporal targets — predict identity
                return (
                    StateVector(
                        values=state.values.copy(),
                        precision=np.ones_like(state.precision) * 0.5,
                        timestamp=state.timestamp + 1.0,
                        grounding_level=state.grounding_level,
                    ),
                    0.5,
                )

            infer = VariableElimination(self._bn)
            result = infer.query(variables=target_vars, evidence=evidence)

            # Convert result to factor dict (handles pgmpy 1.1.2 API)
            factor_dict = _result_to_dict(result, target_vars)

            # Extract predicted values (most probable assignment)
            predicted_values = np.zeros(self.state_dim, dtype=np.float32)
            confidence_values = np.ones(self.state_dim, dtype=np.float32)
            var_count = 0

            for var_name in target_vars:
                if var_name in factor_dict:
                    probs = factor_dict[var_name]
                    # Most probable value
                    max_idx = int(np.argmax(probs))
                    # Confidence = probability of most likely value
                    confidence = float(probs[max_idx])

                    # Determine the state dimension index from variable name
                    idx = var_count % self.state_dim
                    predicted_values[idx] = float(max_idx)
                    confidence_values[idx] = confidence
                    var_count += 1

            # Average confidence across all predicted variables
            avg_confidence = float(np.mean(confidence_values)) if var_count > 0 else 0.0

            return (
                StateVector(
                    values=predicted_values,
                    precision=confidence_values,
                    timestamp=state.timestamp + 1.0,
                    grounding_level=state.grounding_level,
                ),
                avg_confidence,
            )

        except Exception:
            _log(logger, "warning", "gprime.inference_failed",
                 method="discrete", fallback="identity", exc_info=True)
            # Inference failed — return low-confidence identity
            return (
                StateVector(
                    values=state.values.copy(),
                    precision=np.ones_like(state.precision) * 0.01,
                    timestamp=state.timestamp + 1.0,
                    grounding_level=state.grounding_level,
                ),
                0.0,
            )

    # ── Prediction (Continuous / Gaussian) ───────────────────

    def has_gaussian_nodes(self) -> bool:
        """Check if the graph has any Gaussian/continuous nodes."""
        return any(
            n.cpd_type in ("gaussian", "conditional_gaussian")
            for n in self.nodes.values()
        )

    def _get_gaussian_topology(self) -> Tuple[List[str], Dict[str, List[float]],
                                              Dict[str, float], Dict[str, List[str]]]:
        """Extract Gaussian BN topology from current nodes.

        Results are cached because the graph structure is static after construction.
        Use invalidate_cache() to clear when the graph is modified.

        Returns:
            Tuple of (node_order, betas, sigmas, parents) for the
            Gaussian BN inference engine.
        """
        # Return cached topology if available
        if self._cached_topology is not None:                return self._cached_topology[0], self._cached_topology[1], self._cached_topology[2], self._cached_topology[3]

        # Get topological ordering from temporal + causal + parent edges
        node_order = list(self.nodes.keys())

        # Build adjacency for topological sort
        edges = set()
        for e in self.temporal_edges:
            edges.add((e.source, e.target))
        for source, target in self.causal_edges:
            edges.add((source, target))
        for node_name, node in self.nodes.items():
            for parent in node.parents:
                if parent in self.nodes:
                    edges.add((parent, node_name))

        # Simple topological sort via Kahn's algorithm
        in_degree = {n: 0 for n in node_order}
        for src, dst in edges:
            if src in in_degree and dst in in_degree:
                in_degree[dst] += 1

        queue = [n for n, d in in_degree.items() if d == 0]
        sorted_order = []
        while queue:
            n = queue.pop(0)
            sorted_order.append(n)
            for src, dst in edges:
                if src == n and dst in in_degree:
                    in_degree[dst] -= 1
                    if in_degree[dst] == 0:
                        queue.append(dst)

        # Add any remaining nodes (shouldn't happen for DAG)
        for n in node_order:
            if n not in sorted_order:
                sorted_order.append(n)

        # Build betas, sigmas, parents from node data
        betas: Dict[str, List[float]] = {}
        sigmas: Dict[str, float] = {}
        parents_dict: Dict[str, List[str]] = {}

        for node_name, node in self.nodes.items():
            parents_list = [p for p in node.parents if p in self.nodes]
            parents_dict[node_name] = parents_list

            if node.params is not None:
                betas[node_name] = node.params.tolist()
            else:
                # Default: identity transition
                if parents_list:
                    betas[node_name] = [0.0] + [1.0] * len(parents_list)
                else:
                    betas[node_name] = [0.0]

            sigmas[node_name] = max(node.std, 0.001)

        # Cache the topology
        self._cached_topology = (sorted_order, betas, sigmas, parents_dict)

        return sorted_order, betas, sigmas, parents_dict

    def predict_continuous(
        self,
        state: StateVector,
        action: np.ndarray,
        method: str = "analytic",
    ) -> Tuple[StateVector, float]:
        """Predict next state using Gaussian BN analytic inference.

        Phase 3.2: Closed-form posterior computation for Gaussian
        Bayesian networks. Uses precision matrix operations for
        exact inference (O(n³) for n nodes).

        Joint moments are cached after first computation because the
        graph structure is static — only evidence changes between calls.
        This avoids O(n³) recomputation on every predict() call.

        Args:
            state: Current state vector (provides evidence for _t nodes).
            action: Action vector (provides evidence for action nodes).
            method: "analytic" for exact closed-form (n ≤ 200),
                    "sampling" for approximate sampling (n > 200).

        Returns:
            Tuple of (predicted_state, confidence):
                predicted_state: StateVector with posterior means.
                confidence: Average confidence across all predicted dims.
        """
        if len(self.nodes) == 0:
            return (
                StateVector(
                    values=state.values.copy(),
                    precision=np.zeros_like(state.precision),
                    timestamp=state.timestamp + 1.0,
                    grounding_level=state.grounding_level,
                ),
                0.0,
            )

        # Build Gaussian topology (cached after first call)
        node_order, betas, sigmas, parents_dict = self._get_gaussian_topology()

        # Build evidence from current state and action
        evidence: Dict[str, float] = {}
        for i in range(min(self.state_dim, len(state.values))):
            name_t = f"s{i}_t"
            if name_t in self.nodes:
                evidence[name_t] = float(state.values[i])
        for i in range(min(self.action_dim, len(action))):
            name_a = f"a{i}_t"
            if name_a in self.nodes:
                evidence[name_a] = float(action[i])

        # Identify query variables (_t1 nodes)
        query_vars = [n for n in self.nodes if n.endswith("_t1")]

        if not query_vars:
            # No temporal targets — return identity
            return (
                StateVector(
                    values=state.values.copy(),
                    precision=np.ones_like(state.precision) * 0.5,
                    timestamp=state.timestamp + 1.0,
                    grounding_level=state.grounding_level,
                ),
                0.5,
            )

        try:
            if method == "analytic":
                # Compute or reuse cached joint moments
                if self._cached_joint_moments is None:
                    self._cached_joint_moments = compute_joint_moments(
                        node_order, betas, sigmas, parents_dict
                    )
                mu, cov = self._cached_joint_moments
                posteriors = posterior(
                    mu, cov, evidence, query_vars, node_order
                )
            else:
                posteriors = sample_posterior(
                    node_order, betas, sigmas, parents_dict,
                    evidence, query_vars,
                    n_samples=10_000,
                    seed=self.rng.randint(10000),
                )
        except (ValueError, np.linalg.LinAlgError) as e:
            _log(logger, "warning", "gprime.continuous_inference_failed",
                 error=str(e), fallback="identity")
            return (
                StateVector(
                    values=state.values.copy(),
                    precision=np.ones_like(state.precision) * 0.01,
                    timestamp=state.timestamp + 1.0,
                    grounding_level=state.grounding_level,
                ),
                0.0,
            )

        # Build output StateVector
        predicted_values = np.zeros(self.state_dim, dtype=np.float32)
        precision_values = np.ones(self.state_dim, dtype=np.float32)
        std_values = np.zeros(self.state_dim, dtype=np.float32)
        confidences: List[float] = []

        for var_name, (mean_val, std_val) in posteriors.items():
            # Extract dimension index from var name
            try:
                idx_str = var_name.split("s")[1].split("_t1")[0]
                idx = int(idx_str)
                if 0 <= idx < self.state_dim:
                    predicted_values[idx] = float(mean_val)
                    std_values[idx] = float(std_val)
                    conf = confidence_from_variance(std_val ** 2)
                    precision_values[idx] = conf
                    confidences.append(conf)
            except (IndexError, ValueError):
                continue

        # Observability v4: cache per-dim posterior std for the uncertainty portrait.
        self._last_pred_std = std_values

        avg_confidence = float(np.mean(confidences)) if confidences else 0.0

        return (
            StateVector(
                values=predicted_values,
                precision=precision_values,
                timestamp=state.timestamp + 1.0,
                grounding_level=state.grounding_level,
            ),
            avg_confidence,
        )

    def uncertainty_snapshot(self) -> Dict[str, Any]:
        """Observability v4: per-dim posterior std portrait (Gaussian G').

        Returns the cached per-dim posterior std from the last
        ``predict_continuous`` call (zero recomputation). For discrete-only
        graphs, ``per_dim_std`` is None.
        """
        kind = "gaussian" if self.has_gaussian_nodes() else "discrete"
        std = self._last_pred_std
        return {
            "kind": kind,
            "per_dim_std": (std.copy() if std is not None else None),
            "mutual_info": None,
        }

    def estimate_empowerment(
        self,
        state: Optional[StateVector] = None,
        action: Optional[np.ndarray] = None,
    ) -> float:
        """Compute closed-form empowerment I(S';A|S) for Gaussian BN (AF-002).

        Uses the Gaussian BN structure to compute mutual information
        between action A and next state S' given current state S:

            I(S';A|S) = H(S'|S) - H(S'|A,S)

        For Gaussian BNs, the conditional covariance does not depend on
        the specific evidence values — only on which variables are observed.
        Therefore, the state and action arguments are optional (used only
        for the conditional mean, which is not needed for MI).

        Returns:
            Empowerment in nats (≥ 0), clamped to [0.0, 1.0].
        """
        if not self.has_gaussian_nodes():
            return 0.0

        try:
            node_order, betas, sigmas, parents_dict = self._get_gaussian_topology()
            if self._cached_joint_moments is None:
                self._cached_joint_moments = compute_joint_moments(
                    node_order, betas, sigmas, parents_dict
                )
            mu, cov = self._cached_joint_moments
        except (ValueError, np.linalg.LinAlgError):
            return 0.3

        # Identify variable groups
        state_t_vars = sorted([n for n in node_order if n.endswith("_t") and not n.startswith("a")])
        action_vars = sorted([n for n in node_order if n.startswith("a") and n.endswith("_t")])
        state_t1_vars = sorted([n for n in node_order if n.endswith("_t1")])

        if not state_t1_vars or not state_t_vars:
            return 0.0

        try:
            # Σ_{S'|S}: covariance of S' given S (marginalized over A)
            Σ_S_given_S = conditional_covariance(
                mu, cov, state_t_vars, state_t1_vars, node_order
            )

            # Σ_{S'|S,A}: covariance of S' given S and A
            if action_vars:
                evidence_sa = state_t_vars + action_vars
            else:
                evidence_sa = state_t_vars
            Σ_S_given_SA = conditional_covariance(
                mu, cov, evidence_sa, state_t1_vars, node_order
            )

            empowerment = gaussian_mutual_information(Σ_S_given_S, Σ_S_given_SA)
            return float(np.clip(empowerment, 0.0, 1.0))

        except (ValueError, np.linalg.LinAlgError):
            return 0.3

    # ── Continuous GridWorld Builder ──────────────────────────

    @classmethod
    def build_gaussian_grid(
        cls,
        state_dim: int,
        action_dim: int = 5,
        transition_std: float = 0.5,
        seed: int = 42,
    ) -> WorldModelGPrime:
        """Build a Gaussian BN for GridWorld with continuous CPDs.

        Creates a 2-step temporal model: s{i}_t → s{i}_t1
        with action conditioning: a{j}_t influences all s{i}_t1.

        Each s{i}_t1 has CPD:
            P(s{i}_t1 | s{i}_t, a{0..4}_t) = N(β₀ + β₁·s{i}_t + β₂·a_t, σ²)

        Args:
            state_dim: Dimensionality of the state space (e.g., 84 for 5×5).
            action_dim: Dimensionality of the action space (default 5).
            transition_std: Standard deviation of the transition noise.
            seed: Random seed.

        Returns:
            WorldModelGPrime with Gaussian nodes configured for GridWorld.
        """
        model = cls(state_dim=state_dim, action_dim=action_dim, seed=seed)

        # Create state nodes at time t (evidence)
        for i in range(state_dim):
            model.add_node(StateNode(
                name=f"s{i}_t",
                cpd_type="gaussian",
                parents=[],
                cardinality=2,
                params=np.array([0.0], dtype=np.float32),  # β₀ = 0
                std=transition_std,
            ))

        # Create state nodes at time t+1 (query targets)
        for i in range(state_dim):
            # β₀ = 0 (intercept), β₁ = 0.95 (temporal persistence),
            # β₂..β_{1+action_dim} = 0.1 (action influence)
            beta = [0.0, 0.95] + [0.1] * action_dim
            model.add_node(StateNode(
                name=f"s{i}_t1",
                cpd_type="conditional_gaussian",
                parents=[f"s{i}_t"] + [f"a{j}_t" for j in range(action_dim)],
                cardinality=2,
                params=np.array(beta, dtype=np.float32),
                std=transition_std,
            ))
            model.add_temporal_edge(TemporalEdge(
                source=f"s{i}_t", target=f"s{i}_t1", lag=1,
            ))

        # Create action nodes at time t
        for j in range(action_dim):
            model.add_node(StateNode(
                name=f"a{j}_t",
                cpd_type="gaussian",
                parents=[],
                cardinality=2,
                params=np.array([0.0], dtype=np.float32),
                std=0.1,
            ))
            for i in range(state_dim):
                model.add_causal_edge(f"a{j}_t", f"s{i}_t1")

        return model

    # ── Cache Invalidation ────────────────────────────────────

    def invalidate_cache(self) -> None:
        """Invalidate the inference caches (topology + joint moments).

        Must be called whenever model parameters (betas, sigmas) change,
        because the cached joint moments depend on them.
        """
        self._cached_topology = None
        self._cached_joint_moments = None

    # ── Learning ──────────────────────────────────────────────

    def learn(
        self,
        state_t: StateVector,
        action: np.ndarray,
        state_t1: StateVector,
        error: float,
    ) -> None:
        """Update G' parameters from observed transition.

        Phase 3.1: Frequency-based CPD update for discrete graphs.
        Phase 3.2+: Delta-rule update for continuous Gaussian betas.
        Phase 3.3+: Gradient-based learning with TSPL streams.

        For Gaussian nodes (used in benchmarks):
            Updates betas via delta rule: β += lr · (observed - predicted) · parent_val
            Updates sigmas via running EMA of squared residual.

        For discrete nodes:
            Counts observed transitions and normalizes to probabilities.

        Args:
            state_t: State at time t.
            action: Action taken (one-hot vector).
            state_t1: Observed state at time t+1.
            error: Prediction error from PEU.
        """
        # Store observation in history
        self.state_history.append(state_t)

        updated_gaussian = False

        for i in range(min(self.state_dim, len(state_t.values))):
            t1_name = f"s{i}_t1"

            if t1_name not in self.nodes:
                continue

            node = self.nodes[t1_name]

            # ── Gaussian parameter update (continuous model) ──
            if node.cpd_type in ("gaussian", "conditional_gaussian"):
                if t1_name not in self._gaussian_betas:
                    continue

                beta = self._gaussian_betas[t1_name]
                parents_list = self._gaussian_parents.get(t1_name, [])

                # Collect parent values
                parent_vals: List[float] = []
                for p in parents_list:
                    if p.startswith("s"):
                        # State parent: extract index from name (e.g. "s3_t" → 3)
                        try:
                            p_idx = int(p.split("s")[1].split("_t")[0])
                            parent_vals.append(float(state_t.values[p_idx]) if p_idx < len(state_t.values) else 0.0)
                        except (IndexError, ValueError):
                            parent_vals.append(0.0)
                    elif p.startswith("a"):
                        # Action parent: extract index from name (e.g. "a2_t" → 2)
                        try:
                            p_idx = int(p.split("a")[1].split("_t")[0])
                            parent_vals.append(float(action[p_idx]) if p_idx < len(action) else 0.0)
                        except (IndexError, ValueError):
                            parent_vals.append(0.0)
                    else:
                        parent_vals.append(0.0)

                # Predicted value: β₀ + Σ β_{j+1} · parent_j
                predicted = beta[0]
                for j, pv in enumerate(parent_vals):
                    if j + 1 < len(beta):
                        predicted += beta[j + 1] * pv

                observed = float(state_t1.values[i]) if i < len(state_t1.values) else 0.0
                delta = observed - predicted

                # Delta-rule update of betas with error-modulated learning rate (A5 fix)
                lr_mod = float(np.clip(1.0 + abs(error) * 0.1, 0.5, 2.0))
                lr = 0.05 * lr_mod
                beta[0] += lr * delta * 1.0  # intercept
                for j, pv in enumerate(parent_vals):
                    if j + 1 < len(beta):
                        beta[j + 1] += lr * delta * pv

                # Write updated betas back to node.params so predict_continuous() reads them
                # (_get_gaussian_topology() reads from node.params, not _gaussian_betas)
                node.params = np.array(beta, dtype=np.float32)

                # Running EMA estimate of residual sigma
                sigma_old = self._gaussian_sigmas.get(t1_name, 0.5)
                lr_sigma = 0.01
                sigma_new = np.sqrt(max(0.0, (1.0 - lr_sigma) * sigma_old**2 + lr_sigma * delta**2))
                self._gaussian_sigmas[t1_name] = float(max(sigma_new, 0.01))
                # Also update node.std so _get_gaussian_topology() reads the learned value
                node.std = float(max(sigma_new, 0.01))

                updated_gaussian = True

            # ── Discrete parameter update (pgmpy model) ──
            elif node.cpd_type == "discrete":
                s_name = f"s{i}_t"
                if s_name not in self.nodes:
                    continue

                s_val = int(round(state_t.values[i]))
                s1_val = int(round(state_t1.values[i]))
                card = self.nodes[s_name].cardinality

                key = f"{s_name}->{t1_name}"
                if key not in self._cpd_params:
                    self._cpd_params[key] = np.ones((card, card), dtype=np.float32)

                self._cpd_params[key][s_val, s1_val] += 1.0

        # Normalize discrete CPDs if any were updated
        if any(n.cpd_type == "discrete" for n in self.nodes.values()):
            self._normalize_cpds()

        # Invalidate caches if Gaussian parameters changed
        if updated_gaussian:
            self.invalidate_cache()

        # Keep only recent history to bound memory
        if len(self.state_history) > 10_000:
            self.state_history = self.state_history[-5_000:]

    def _normalize_cpds(self) -> None:
        """Normalize frequency-count CPDs to probabilities and rebuild graph.

        Call this before inference to ensure CPDs are proper probability tables.
        """
        for node_name, node in self.nodes.items():
            if node.cpd_type != "discrete":
                continue

            parents = [p for p in node.parents if p in self.nodes]
            if not parents:
                continue

            # Build transition matrix from learned params
            card = node.cardinality
            parent_cards = [self.nodes[p].cardinality for p in parents]
            n_parent_combos = int(np.prod(parent_cards))

            params = np.zeros((card, n_parent_combos), dtype=np.float32)

            # For each parent combination, get the transition counts
            for parent_idx in range(n_parent_combos):
                for child_val in range(card):
                    # AF-003: use key from all parents (multi-parent support)
                    parent_key = "&".join(sorted(parents)) + "->" + node_name
                    if parent_key in self._cpd_params:
                        total = max(float(np.sum(self._cpd_params[parent_key][:, child_val])), 1.0)
                        params[child_val, parent_idx] = (
                            self._cpd_params[parent_key][parent_idx, child_val] / total
                        )
                    else:
                        params[child_val, parent_idx] = 1.0 / card

            # Normalize columns
            col_sums = params.sum(axis=0, keepdims=True)
            col_sums = np.where(col_sums > 0, col_sums, 1.0)
            params = params / col_sums

            node.params = params

        # Rebuild the pgmpy graph with updated CPDs
        if self._bn is not None:
            self._build_graph()

    def reset(self) -> None:
        """Reset G' to initial state (new episode)."""
        self.state_history.clear()
        self._cpd_params.clear()
        self._bn = None
        self._cached_topology = None
        self._cached_joint_moments = None


# ── Helper: pgmpy result conversion (pgmpy 1.1.2+ compat) ─────


def _result_to_dict(
    result: DiscreteFactor,
    expected_vars: List[str],
) -> Dict[str, np.ndarray]:
    """Convert a pgmpy DiscreteFactor query result to a var→probs dict.

    pgmpy 1.1.2+ returns a single DiscreteFactor from VariableElimination.query().
    This helper extracts per-variable marginals from the joint factor.

    Args:
        result: DiscreteFactor from VariableElimination.query().
        expected_vars: List of variable names the query requested.

    Returns:
        Dict mapping each variable name to its probability array.
    """
    scope = result.scope()
    values = np.array(result.values, dtype=np.float32)

    # Single-variable result
    if len(scope) == 1 and len(expected_vars) == 1:
        return {expected_vars[0]: values.flatten()}

    # Multi-variable result: extract marginals for each expected var
    result_dict: Dict[str, np.ndarray] = {}

    for var_name in expected_vars:
        if var_name in scope:
            var_idx = scope.index(var_name)
            other_axes = tuple(i for i in range(len(scope)) if i != var_idx)
            marginal = np.sum(values, axis=other_axes) if other_axes else values
            result_dict[var_name] = marginal.flatten()
        else:
            result_dict[var_name] = np.ones(2, dtype=np.float32) / 2.0

    return result_dict
