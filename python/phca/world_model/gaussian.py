"""
PHCA v3.0 — Gaussian Bayesian Network Inference Engine.

Phase 3.2: Closed-form analytic inference for Gaussian Bayesian networks.

A Gaussian BN represents each node as a linear Gaussian of its parents:
    P(X_i | pa(X_i)) = N(β_0 + Σ β_j · parent_j, σ_i²)

The joint distribution P(X) = N(μ, Σ) is multivariate Gaussian, where:
    - μ is computed recursively from the topological ordering
    - Σ = (I - B)^(-1) · diag(σ²) · (I - B)^(-T)
    where B is the weighted adjacency matrix (B_ij = β coefficient of j→i)

Given evidence X_E = e, the posterior P(X_Q | X_E = e) is also Gaussian:
    μ_Q|E = μ_Q + Σ_QE · Σ_EE^(-1) · (e - μ_E)
    Σ_Q|E = Σ_QQ - Σ_QE · Σ_EE^(-1) · Σ_EQ

This avoids any external pgmpy dependency for continuous CPDs.

v3.0 Reference: §2.2 Definition 2.4b (Phase 3.2 Gaussian CPDs)
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from phca.logging import logger, _log


@np.errstate(divide="raise", invalid="raise", over="ignore")
def compute_joint_moments(
    node_order: List[str],
    betas: Dict[str, List[float]],
    sigmas: Dict[str, float],
    parents: Dict[str, List[str]],
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute joint mean μ and covariance Σ for a Gaussian BN.

    Uses the topological ordering to compute μ recursively, then computes
    Σ = (I - B)^(-1) · diag(σ²) · (I - B)^(-T).

    Args:
        node_order: Topological ordering of all nodes.
        betas: Mapping node_name → [intercept, coefficient_for_parent1, ...].
            Root nodes have just [intercept].
        sigmas: Mapping node_name → standard deviation σ.
        parents: Mapping node_name → list of parent names.

    Returns:
        Tuple of (mean_vector, covariance_matrix):
            mean_vector: Shape (n,) float64 array of marginal means.
            covariance_matrix: Shape (n, n) float64 array of covariances.

    Raises:
        ValueError: If there's a mismatch in node references.
    """
    n = len(node_order)
    name_to_idx = {name: i for i, name in enumerate(node_order)}

    # Compute mean vector
    mu = np.zeros(n, dtype=np.float64)
    for name in node_order:
        i = name_to_idx[name]
        b = betas.get(name, [0.0])
        par = parents.get(name, [])
        mu[i] = b[0]  # intercept
        for j, p in enumerate(par):
            p_idx = name_to_idx.get(p)
            if p_idx is not None and j + 1 < len(b):
                mu[i] += b[j + 1] * mu[p_idx]

    # Build weighted adjacency matrix B where B[i,j] = coefficient of j→i
    B = np.zeros((n, n), dtype=np.float64)
    for name in node_order:
        i = name_to_idx[name]
        b = betas.get(name, [0.0])
        par = parents.get(name, [])
        for j, p in enumerate(par):
            p_idx = name_to_idx.get(p)
            if p_idx is not None and j + 1 < len(b):
                B[i, p_idx] = b[j + 1]

    # Compute Σ = (I - B)^(-1) · diag(σ²) · (I - B)^(-T)
    I_minus_B = np.eye(n, dtype=np.float64) - B
    try:
        inv_I_minus_B = np.linalg.inv(I_minus_B)
    except np.linalg.LinAlgError:
        # Fallback: pseudoinverse for singular matrices
        _log(logger, "warning", "gaussian.singular_matrix",
             method="inv", fallback="pinv", exc_info=True)
        inv_I_minus_B = np.linalg.pinv(I_minus_B)

    diag_sigma2 = np.diag(
        np.array([sigmas.get(name, 1.0) ** 2 for name in node_order], dtype=np.float64)
    )
    cov = inv_I_minus_B @ diag_sigma2 @ inv_I_minus_B.T

    return mu, cov


@np.errstate(divide="raise", invalid="raise", over="ignore")
def posterior(
    mu: np.ndarray,
    cov: np.ndarray,
    evidence: Dict[str, float],
    query_vars: List[str],
    all_vars: List[str],
) -> Dict[str, Tuple[float, float]]:
    """Compute posterior mean and variance given evidence.

    P(X_Q | X_E = e) = N(μ_Q|E, Σ_Q|E) where:
        μ_Q|E = μ_Q + Σ_QE · Σ_EE^(-1) · (e - μ_E)
        Σ_Q|E = Σ_QQ - Σ_QE · Σ_EE^(-1) · Σ_EQ

    Args:
        mu: Joint mean vector (n,), ordered by all_vars.
        cov: Joint covariance matrix (n, n), ordered by all_vars.
        evidence: Dict mapping variable names to observed values.
        query_vars: List of variable names to compute posterior for.
        all_vars: Ordered list of all variable names.

    Returns:
        Dict mapping each query variable name to (posterior_mean, posterior_std).

    Raises:
        ValueError: If evidence contains variables not in all_vars.
    """
    name_to_idx = {name: i for i, name in enumerate(all_vars)}

    # Validate evidence
    for var in evidence:
        if var not in name_to_idx:
            raise ValueError(f"Evidence variable '{var}' not found in graph")

    # Build indices
    e_idx = [name_to_idx[v] for v in evidence]
    q_idx = [name_to_idx[v] for v in query_vars if v in name_to_idx]

    if not q_idx:
        return {}

    e_vec = np.array([evidence[v] for v in evidence], dtype=np.float64)

    # Partition mean and covariance
    mu_E = mu[e_idx]
    mu_Q = mu[q_idx]

    Σ_EE = cov[np.ix_(e_idx, e_idx)]
    Σ_QQ = cov[np.ix_(q_idx, q_idx)]
    Σ_QE = cov[np.ix_(q_idx, e_idx)]

    # Compute posterior mean and covariance
    # Use pinv unconditionally — solve() can hang on near-singular matrices
    # (cond ~1e8) without raising LinAlgError, and pinv is robust at negligible
    # performance cost for the matrix sizes in use (~300x300).
    resid = e_vec - mu_E
    Σ_EE_inv = np.linalg.pinv(Σ_EE)
    mu_Q_given_E = mu_Q + Σ_QE @ Σ_EE_inv @ resid
    Σ_QQ_given_E = Σ_QQ - Σ_QE @ Σ_EE_inv @ Σ_QE.T

    # Extract diagonal (variances) and ensure non-negative
    variances = np.maximum(np.diag(Σ_QQ_given_E), 1e-12)

    result: Dict[str, Tuple[float, float]] = {}
    for i, var_name in enumerate(query_vars):
        if var_name in name_to_idx:
            idx = q_idx.index(name_to_idx[var_name])
            result[var_name] = (float(mu_Q_given_E[idx]), float(np.sqrt(variances[idx])))

    return result


def conditional_covariance(
    mu: np.ndarray,
    cov: np.ndarray,
    evidence_vars: List[str],
    query_vars: List[str],
    all_vars: List[str],
) -> np.ndarray:
    """Compute the full conditional covariance Σ_{Q|E} (Schur complement).

    Σ_{Q|E} = Σ_{QQ} - Σ_{QE} · Σ_{EE}^{-1} · Σ_{EQ}

    This is the covariance of the query variables after conditioning on
    the evidence variables. Unlike posterior(), this returns the FULL
    covariance matrix, not just the diagonal.

    Used for closed-form Gaussian mutual information (AF-002).

    Args:
        mu: Joint mean vector (n,), ordered by all_vars.
        cov: Joint covariance matrix (n, n), ordered by all_vars.
        evidence_vars: Names of observed (evidence) variables.
        query_vars: Names of query variables.
        all_vars: Ordered list of all variable names.

    Returns:
        Conditional covariance matrix of query variables given evidence.
        Shape (len(query_vars), len(query_vars)).

    Raises:
        ValueError: If evidence or query variables are not in all_vars.
    """
    name_to_idx = {name: i for i, name in enumerate(all_vars)}

    for var in evidence_vars:
        if var not in name_to_idx:
            raise ValueError(f"Evidence variable '{var}' not found in graph")
    for var in query_vars:
        if var not in name_to_idx:
            raise ValueError(f"Query variable '{var}' not found in graph")

    e_idx = [name_to_idx[v] for v in evidence_vars]
    q_idx = [name_to_idx[v] for v in query_vars]

    if not q_idx:
        return np.zeros((0, 0), dtype=np.float64)
    if not e_idx:
        # No evidence: return the full prior covariance of query vars
        return cov[np.ix_(q_idx, q_idx)]

    Σ_EE = cov[np.ix_(e_idx, e_idx)]
    Σ_QQ = cov[np.ix_(q_idx, q_idx)]
    Σ_QE = cov[np.ix_(q_idx, e_idx)]

    # Use pinv unconditionally — solve() can hang on near-singular matrices.
    solved = np.linalg.pinv(Σ_EE) @ Σ_QE.T
    Σ_QQ_given_E = Σ_QQ - Σ_QE @ solved
    return Σ_QQ_given_E


def gaussian_mutual_information(
    Σ_marginal: np.ndarray,
    Σ_conditional: np.ndarray,
    gamma: float = 1.0,
) -> float:
    """Compute mutual information I(Q; Δ | E) for two Gaussian covariances.

    I = 0.5 · log(det(2πe · Σ_marginal)) - 0.5 · log(det(2πe · Σ_conditional))
      = 0.5 · log(det(Σ_marginal) / det(Σ_conditional))

    where Σ_marginal is the covariance of Q given a reduced evidence set
    (without the variables of interest), and Σ_conditional is the covariance
    of Q given the full evidence set (with all variables).

    The gamma parameter allows soft scaling of the information gained
    from the additional variables.

    Args:
        Σ_marginal: Conditional covariance given reduced evidence (d, d).
        Σ_conditional: Conditional covariance given full evidence (d, d).
        gamma: Scaling factor for the MI (default 1.0).

    Returns:
        Mutual information in nats (≥ 0), scaled by gamma.

    Raises:
        ValueError: If matrices are not square or dimensions mismatch.
    """
    if Σ_marginal.shape != Σ_conditional.shape:
        raise ValueError(
            f"Matrix shape mismatch: {Σ_marginal.shape} vs {Σ_conditional.shape}"
        )
    d = Σ_marginal.shape[0]
    if d == 0:
        return 0.0

    try:
        _, logdet_marginal = np.linalg.slogdet(2.0 * np.pi * np.e * Σ_marginal)
        _, logdet_conditional = np.linalg.slogdet(2.0 * np.pi * np.e * Σ_conditional)
    except np.linalg.LinAlgError:
        _log(logger, "warning", "gaussian.slogdet_failed",
             fallback="mutual_info_zero", exc_info=True)
        return 0.0

    mutual_info = 0.5 * (logdet_marginal - logdet_conditional)
    return float(max(mutual_info * gamma, 0.0))


def confidence_from_variance(variance: float) -> float:
    """Convert posterior variance to a confidence score.

    confidence = 1 / (1 + variance)

    When variance → 0, confidence → 1.0 (certain).
    When variance → ∞, confidence → 0.0 (uncertain).

    Args:
        variance: Posterior variance (must be > 0).

    Returns:
        Confidence score in [0.0, 1.0].
    """
    return float(1.0 / (1.0 + max(variance, 1e-12)))


# ── Sampling-based inference for large graphs ────────────────


def sample_posterior(
    node_order: List[str],
    betas: Dict[str, List[float]],
    sigmas: Dict[str, float],
    parents: Dict[str, List[str]],
    evidence: Dict[str, float],
    query_vars: List[str],
    n_samples: int = 10_000,
    tolerance: float = 0.1,
    seed: int = 42,
) -> Dict[str, Tuple[float, float]]:
    """Approximate posterior via rejection sampling.

    For graphs too large for exact analytic inference (|V| > 200),
    use forward sampling + rejection to approximate the posterior.

    Args:
        node_order: Topological ordering of all nodes.
        betas: Mapping node_name → [intercept, coefficients].
        sigmas: Mapping node_name → standard deviation.
        parents: Mapping node_name → list of parent names.
        evidence: Dict mapping variable names to observed values.
        query_vars: List of variable names to query.
        n_samples: Number of forward samples to draw.
        tolerance: Acceptance radius for evidence matching (± tolerance).
        seed: Random seed.

    Returns:
        Dict mapping query variable → (posterior_mean, posterior_std).
    """
    rng = np.random.RandomState(seed)
    name_to_idx = {name: i for i, name in enumerate(node_order)}
    n = len(node_order)

    # Check if exact is feasible
    if n <= 200:
        try:
            mu, cov = compute_joint_moments(node_order, betas, sigmas, parents)
            return posterior(mu, cov, evidence, query_vars, node_order)
        except np.linalg.LinAlgError:
            _log(logger, "warning", "gaussian.mvn_singular",
                 fallback="rejection_sampling", exc_info=True)
            # Fall through to rejection sampling

    accepted_samples: List[np.ndarray] = []
    evidence_vars = list(evidence.keys())
    evidence_vals = np.array([evidence[v] for v in evidence_vars], dtype=np.float64)

    while len(accepted_samples) < n_samples:
        # Forward sample using topological order
        sample = np.zeros(n, dtype=np.float64)
        for name in node_order:
            i = name_to_idx[name]
            b = betas.get(name, [0.0])
            par = parents.get(name, [])
            mean_val = b[0]
            for j, p in enumerate(par):
                p_idx = name_to_idx.get(p)
                if p_idx is not None and j + 1 < len(b):
                    mean_val += b[j + 1] * sample[p_idx]
            sample[i] = mean_val + rng.randn() * sigmas.get(name, 1.0)

        # Check evidence match
        match = True
        for var, val in zip(evidence_vars, evidence_vals):
            idx = name_to_idx.get(var)
            if idx is not None:
                if abs(sample[idx] - val) > tolerance:
                    match = False
                    break

        if match:
            accepted_samples.append(sample)

        # Safety: avoid infinite loop for impossible evidence
        if len(accepted_samples) == 0 and len(evidence) > 0:
            # Increase tolerance exponentially if no samples accepted
            tolerance *= 1.5
            if tolerance > 10.0:
                break

    if not accepted_samples:
        # Fallback: return prior moments
        mu, cov = compute_joint_moments(node_order, betas, sigmas, parents)
        return posterior(mu, cov, {}, query_vars, node_order)

    samples = np.array(accepted_samples)
    result: Dict[str, Tuple[float, float]] = {}
    for var in query_vars:
        idx = name_to_idx.get(var)
        if idx is not None:
            vals = samples[:, idx]
            result[var] = (float(np.mean(vals)), float(np.std(vals)))

    return result
