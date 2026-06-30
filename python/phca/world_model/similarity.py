"""
PHCA v3.0 — Similarity Search (v3.0 §D.3).

Phase 3.1: k-NN Euclidean distance over state history.
Phase 3.2+: VSA hyperdimensional computing (conditional — see v3.0 §4.3).

This module replaces VSA during Phase 3.1. If similarity search proves
insufficient for analogical retrieval, activate V = python/phca/world_model/vsa.py.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from phca.config import StateVector


def knn_similarity(
    query: StateVector,
    history: List[StateVector],
    k: int = 5,
    metric: str = "euclidean",
) -> List[Tuple[StateVector, float]]:
    """Find k nearest neighbors to query in state history.

    Args:
        query: Query state vector.
        history: List of historical state vectors to search.
        k: Number of nearest neighbors to return.
        metric: Distance metric ('euclidean' or 'cosine').

    Returns:
        List of (neighbor_state, similarity_score) tuples, sorted by
        similarity score descending (most similar first).

    Raises:
        ValueError: If k < 0 or metric is unsupported.
    """
    if k < 0:
        raise ValueError(f"k must be >= 0, got {k}")
    if metric not in ("euclidean", "cosine"):
        raise ValueError(f"Unsupported metric '{metric}'; use 'euclidean' or 'cosine'")
    if not history or k == 0:
        return []

    k = min(k, len(history))
    query_vals = query.values.flatten()

    if metric == "euclidean":
        distances = np.array([
            np.linalg.norm(query_vals - s.values.flatten())
            for s in history
        ], dtype=np.float32)
    elif metric == "cosine":
        query_norm = np.linalg.norm(query_vals)
        if query_norm < 1e-10:
            return []  # zero vector has no direction
        similarities = np.array([
            np.dot(query_vals, s.values.flatten()) / (query_norm * max(np.linalg.norm(s.values.flatten()), 1e-10))
            for s in history
        ], dtype=np.float32)
        # For cosine, higher = more similar; convert to 0-1 range
        similarities = (similarities + 1.0) / 2.0
        # For cosine metric, sort by similarity descending
        idxs = np.argsort(-similarities)[:k]
        return [(history[idx], float(similarities[idx])) for idx in idxs]
    else:
        raise ValueError(f"Unsupported metric: {metric}")

    # Euclidean: convert distance to similarity (0-1, higher = more similar)
    max_dist = float(np.max(distances)) if len(distances) > 0 else 1.0
    max_dist = max(max_dist, 1e-8)
    idxs = np.argsort(distances)[:k]

    results = []
    for idx in idxs:
        similarity = 1.0 - (float(distances[idx]) / max_dist)
        results.append((history[idx], similarity))

    return results
