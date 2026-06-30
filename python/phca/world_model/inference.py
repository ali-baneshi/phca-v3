"""
PHCA v3.0 — G' Forward Inference Engine.

Phase 3.1: Exact junction tree (pgmpy VariableElimination) for |V| ≤ 100.
Phase 3.2+: Importance sampling (pyro) for larger graphs.

v3.0 References:
    - §2.2 Definition 2.5 (prediction via G' only)
    - §D.1 (Bayesian inference framework)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from pgmpy.factors.discrete import DiscreteFactor

from phca.config import StateVector


def forward_inference(
    bn: Any,
    evidence: Dict[str, float],
    variables: List[str],
    method: str = "exact",
) -> Dict[str, np.ndarray]:
    """Run forward inference on G' discrete Bayesian network.

    Phase 3.1: Exact junction tree (pgmpy VariableElimination) for |V| ≤ 100.
    Phase 3.2+: Importance sampling (pyro) for |V| > 100.

    Args:
        bn: pgmpy DiscreteBayesianNetwork model (fitted with CPDs).
        evidence: Dict mapping variable names to observed values.
            Example: {"s0_t": 2, "s1_t": 3} for observed state variables.
        variables: List of target variable names to infer.
            Example: ["s0_t1", "s1_t1"] for next-state variables.
        method: Inference method — "exact" (Phase 3.1) or "sampling" (Phase 3.2+).

    Returns:
        Dict mapping each target variable name to a numpy array of
        posterior probabilities (summing to 1.0).

    Raises:
        ValueError: If method is "exact" but |V| > 100 (Phase 3.2 deferred).
        RuntimeError: If inference fails.
    """
    from pgmpy.inference import VariableElimination

    n_nodes = len(bn.nodes()) if bn is not None else 0

    if method == "exact":
        if n_nodes > 100:
            raise ValueError(
                f"Graph has {n_nodes} nodes; exact inference requires |V| ≤ 100. "
                "Phase 3.2 provides sampling inference for larger graphs."
            )

        if n_nodes == 0:
            # Empty graph — return uniform distributions
            return {var: np.ones(2) / 2.0 for var in variables}

        try:
            infer = VariableElimination(bn)
            result = infer.query(variables=variables, evidence=evidence)
        except Exception as e:
            raise RuntimeError(f"Variable elimination failed: {e}") from e

        return _inference_result_to_dict(result, variables)

    elif method == "sampling":
        raise NotImplementedError(
            "Sampling inference deferred to Phase 3.2. "
            "Use method='exact' for Phase 3.1."
        )

    else:
        raise ValueError(f"Unknown inference method: '{method}'. Use 'exact' or 'sampling'.")


def infer_next_state(
    bn: Any,
    state: StateVector,
    target_prefix: str = "s",
) -> Tuple[StateVector, float]:
    """Convenience wrapper: infer next state from current state vector.

    Builds evidence from state.values and queries all _t1 variables.

    Args:
        bn: pgmpy BayesianNetwork model.
        state: Current state vector.
        target_prefix: Prefix for state variable names (default "s").

    Returns:
        Tuple of (predicted_state, confidence):
            predicted_state: StateVector with most probable next values.
            confidence: Mean probability of most-likely assignments.
    """
    # Build evidence from current state
    evidence = {}
    for i in range(len(state.values)):
        var_name = f"{target_prefix}{i}_t"
        if bn is not None and var_name in bn.nodes():
            evidence[var_name] = int(round(state.values[i]))

    # Identify target variables
    target_vars = []
    if bn is not None:
        for node in bn.nodes():
            if node.endswith("_t1"):
                target_vars.append(node)

    if not target_vars:
        # Fallback: return identity
        return (
            StateVector(
                values=state.values.copy(),
                precision=np.zeros_like(state.precision),
                timestamp=state.timestamp + 1.0,
                grounding_level=state.grounding_level,
            ),
            0.0,
        )

    try:
        posteriors = forward_inference(bn, evidence, target_vars, method="exact")
    except (RuntimeError, ValueError):
        return (
            StateVector(
                values=state.values.copy(),
                precision=np.ones_like(state.precision) * 0.01,
                timestamp=state.timestamp + 1.0,
                grounding_level=state.grounding_level,
            ),
            0.0,
        )

    # Extract most probable values
    predicted = np.zeros(len(state.values), dtype=np.float32)
    confidence = np.ones(len(state.values), dtype=np.float32)

    for var_name, probs in posteriors.items():
        # Extract index from variable name (e.g., "s2_t1" → 2)
        try:
            idx_str = var_name.replace(f"{target_prefix}", "").replace("_t1", "")
            idx = int(idx_str)
            if 0 <= idx < len(state.values):
                max_idx = int(np.argmax(probs))
                predicted[idx] = float(max_idx)
                confidence[idx] = float(probs[max_idx])
        except (ValueError, IndexError):
            continue

    avg_confidence = float(np.mean(confidence))

    return (
        StateVector(
            values=predicted,
            precision=confidence,
            timestamp=state.timestamp + 1.0,
            grounding_level=state.grounding_level,
        ),
        avg_confidence,
    )


# ── Helper: pgmpy result conversion (pgmpy 1.1.2+ compat) ─────


def _inference_result_to_dict(
    result: Any,
    variables: List[str],
) -> Dict[str, np.ndarray]:
    """Convert a pgmpy query result to {var: probs_array} dict.

    pgmpy 1.1.2+ returns a single DiscreteFactor from query() even
    when requesting multiple variables. This normalizes to dict format.

    Args:
        result: DiscreteFactor or dict from VariableElimination.query().
        variables: The list of variable names that were requested.

    Returns:
        Dict mapping each variable name to its probability array.
    """
    if isinstance(result, dict):
        # Older pgmpy versions return dict
        return {var: np.array(f.values, dtype=np.float32) for var, f in result.items()}

    if not isinstance(result, DiscreteFactor):
        # Unknown type — fallback to uniform
        return {var: np.ones(2, dtype=np.float32) / 2.0 for var in variables}

    scope = result.scope()
    values = np.array(result.values, dtype=np.float32)

    # Single variable queried — flatten directly
    if len(scope) == 1:
        return {scope[0]: values.flatten()}

    # Multiple variables — for Phase 3.1 we handle the common case
    # where variables have cardinality 2
    output: Dict[str, np.ndarray] = {}
    for var in variables:
        if var in scope:
            output[var] = values.flatten() if len(variables) == 1 else values
        else:
            output[var] = np.ones(2, dtype=np.float32) / 2.0
    return output
