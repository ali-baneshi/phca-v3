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

    # ── Prediction ────────────────────────────────────────────

    def predict(
        self, state: StateVector, action: np.ndarray
    ) -> Tuple[StateVector, float]:
        """Predict next state given current state and action.

        Uses pgmpy exact inference (variable elimination).

        Args:
            state: Current state vector.
            action: Action vector.

        Returns:
            Tuple of (predicted_state, confidence):
                predicted_state: StateVector with predicted values.
                confidence: Prediction confidence (0.0 = uncertain, 1.0 = certain).
        """
        if self._bn is None or len(self.nodes) == 0:
            # Empty graph: return copy of state with zero confidence
            return (
                StateVector(
                    values=state.values.copy(),
                    precision=np.zeros_like(state.precision),
                    timestamp=state.timestamp + 1.0,
                    grounding_level=state.grounding_level,
                ),
                0.0,
            )

        # Build graph if needed
        if self._bn is None:
            self._build_graph()

        # Collect evidence from current state
        evidence = {}
        for i in range(min(self.state_dim, len(state.values))):
            var_name = f"s{i}_t"
            if var_name in self.nodes:
                evidence[var_name] = int(round(state.values[i]))

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

    # ── Learning ──────────────────────────────────────────────

    def learn(
        self,
        state_t: StateVector,
        action: np.ndarray,
        state_t1: StateVector,
        error: float,
    ) -> None:
        """Update G' parameters from observed transition.

        Phase 3.1: Frequency-based CPD update. Counts observed transitions
        and normalizes to probabilities.

        Phase 3.2+: Gradient-based learning with TSPL streams.

        Args:
            state_t: State at time t.
            action: Action taken.
            state_t1: Observed state at time t+1.
            error: Prediction error from PEU (for future gradient-based methods).
        """
        # Store observation in history
        self.state_history.append(state_t)

        # Extract discrete state values for node update
        for i in range(min(self.state_dim, len(state_t.values))):
            s_name = f"s{i}_t"
            s1_name = f"s{i}_t1"

            if s_name in self.nodes and s1_name in self.nodes:
                s_val = int(round(state_t.values[i]))
                s1_val = int(round(state_t1.values[i]))
                card = self.nodes[s_name].cardinality

                # Initialize CPD params if needed
                key = f"{s_name}->{s1_name}"
                if key not in self._cpd_params:
                    self._cpd_params[key] = np.ones((card, card), dtype=np.float32)

                # Frequency count update
                self._cpd_params[key][s_val, s1_val] += 1.0

        # Normalize CPDs and rebuild the pgmpy model with updated params
        self._normalize_cpds()

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
                    # Sum counts from CPD params
                    key = f"{parents[0]}->{node_name}"
                    if key in self._cpd_params:
                        total = max(float(np.sum(self._cpd_params[key][:, child_val])), 1.0)
                        params[child_val, parent_idx] = (
                            self._cpd_params[key][parent_idx % card, child_val] / total
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

    # ── Utility ───────────────────────────────────────────────

    def similarity_search(
        self, query: StateVector, k: int = 5
    ) -> List[Tuple[StateVector, float]]:
        """Find k most similar states from history (v3.0 §D.3).

        Phase 3.1: Euclidean distance k-NN over state_history.
        Phase 3.2+: VSA hyperdimensional computing (conditional).

        Args:
            query: Query state vector.
            k: Number of nearest neighbors to return.

        Returns:
            List of (state, similarity_score) tuples, sorted by
            similarity descending (most similar first).
        """
        if not self.state_history or k == 0:
            return []

        k = min(k, len(self.state_history))

        # Compute Euclidean distances
        distances = []
        for hist_state in self.state_history:
            dist = np.linalg.norm(query.values - hist_state.values)
            distances.append(dist)

        # Find k nearest indices
        idxs = np.argsort(distances)[:k]

        # Convert distances to similarity scores (0-1, higher = more similar)
        max_dist = max(distances) if distances else 1.0
        max_dist = max(max_dist, 1e-8)  # avoid division by zero

        results = []
        for idx in idxs:
            similarity = 1.0 - (distances[idx] / max_dist)
            results.append((self.state_history[idx], float(similarity)))

        return results

    def reset(self) -> None:
        """Reset G' to initial state (new episode)."""
        self.state_history.clear()
        self._cpd_params.clear()
        self._bn = None


# ── Helper: pgmpy result conversion (pgmpy 1.1.2+ compat) ─────


def _result_to_dict(
    result: DiscreteFactor,
    expected_vars: List[str],
) -> Dict[str, np.ndarray]:
    """Convert a pgmpy DiscreteFactor query result to a var→probs dict.

    pgmpy 1.1.2+ returns a single DiscreteFactor from VariableElimination.query(),
    even when querying multiple variables. This helper normalizes to the
    expected {var_name: np.ndarray} format for backward compatibility.

    Args:
        result: DiscreteFactor from VariableElimination.query().
        expected_vars: List of variable names the query requested.

    Returns:
        Dict mapping each variable name to its probability array.
    """
    if isinstance(result, dict):
        # Fallback for older pgmpy versions that return dict
        return {var: np.array(f.values, dtype=np.float32) for var, f in result.items()}

    if not isinstance(result, DiscreteFactor):
        # Unknown type — fall back to uniform
        return {var: np.ones(2, dtype=np.float32) / 2.0 for var in expected_vars}

    scope = result.scope()
    values = np.array(result.values, dtype=np.float32)

    # Single-variable result
    if len(scope) == 1 and len(expected_vars) == 1:
        return {expected_vars[0]: values.flatten()}

    # If exact scope match, extract marginals
    # pgmpy flatten order: variable order from `scope`
    result_dict: Dict[str, np.ndarray] = {}
    all_vars = list(scope)

    for var_name in expected_vars:
        if var_name in scope:
            # Phase 3.1 uses flat values for single-var queries.
            # Phase 3.2: marginalize over scope using var_idx = all_vars.index(var_name)
            # For Phase 3.1: use flat values since we mostly query single var
            result_dict[var_name] = values.flatten() if len(all_vars) == 1 else values
        else:
            result_dict[var_name] = np.ones(2, dtype=np.float32) / 2.0

    return result_dict
