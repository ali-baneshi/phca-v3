"""
PHCA v3.0 — Precision-Weighted Sparse Attention.

Phase 3.2: k-WTA sparse selection with Gumbel noise, precision-weighted
salience (bottom-up + top-down), and online precision learning.

v3.0 Reference: §3.2 Definition 3.4, Phase 2 Architecture §3.4
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from phca.config import GoalVector, StateVector
from phca.memory.m2_working import Chunk


class Attention:
    """Precision-Weighted Sparse Attention — Phase 3.2 full implementation.

    Mechanism:
        1. Bottom-up salience: S_bu(chunk) = |chunk.state - prediction|
           — unexpected chunks are salient.
        2. Top-down relevance: S_td(chunk) = similarity(chunk.state, goal)
           — goal-relevant chunks are salient.
        3. Precision weighting: S = (α·S_bu + β·S_td) · precision
        4. k-WTA selection with Gumbel noise for stochasticity
        5. Precision learning: p(t+1) = p(t) + η·(|δ|-p(t))

    Phase 3.2: k defaults to 3 (for 7±2 WM capacity).
    Phase 3.3+: Full soft-attention with learned queries/keys/values.

    v3.0 Reference: §3.2 Definition 3.4
    """

    def __init__(
        self,
        default_k: int = 3,
        alpha_bu: float = 0.6,
        beta_td: float = 0.4,
        gumbel_temperature: float = 0.5,
        precision_lr: float = 0.1,
        state_dim: int = 4,
    ):
        """Initialize attention module.

        Args:
            default_k: Default number of chunks to select (k-WTA).
            alpha_bu: Bottom-up salience weight.
            beta_td: Top-down salience weight.
            gumbel_temperature: Temperature for Gumbel noise (0=deterministic).
            precision_lr: Learning rate for precision updates.
            state_dim: Dimensionality of state vectors for precision tracking.
        """
        self.default_k = default_k
        self.alpha_bu = alpha_bu
        self.beta_td = beta_td
        self.gumbel_temperature = gumbel_temperature
        self.precision_lr = precision_lr
        self.state_dim = state_dim

        # Precision per chunk: maps chunk_id → precision value
        self._precisions: Dict[int, float] = {}
        # RNG for Gumbel noise
        self._rng = np.random.RandomState(42)

        # Last selection state
        self._last_saliences: List[float] = []
        self._last_selected_indices: List[int] = []

    def select(
        self,
        wm_chunks: List[Chunk],
        goal: Any = None,
        k: Optional[int] = None,
        prediction: Optional[StateVector] = None,
    ) -> List[Chunk]:
        """Select k chunks from working memory via k-WTA.

        Computes composite salience for each chunk:
            S = (α·S_bu + β·S_td) · precision + Gumbel(0, temperature)

        Then selects the top-k chunks.

        Args:
            wm_chunks: List of chunks from working memory.
            goal: GoalVector for top-down relevance (optional).
            k: Number of chunks to select (default: self.default_k).
            prediction: Current prediction state for bottom-up salience
                (optional — uses chunk's state if None).

        Returns:
            List of selected chunks (sorted by salience descending).
        """
        if not wm_chunks:
            return []

        k = k or self.default_k
        k = min(k, len(wm_chunks))

        saliences: List[float] = []
        for chunk in wm_chunks:
            # Bottom-up salience: unexpectedness
            if prediction is not None:
                diff = chunk.state.values.astype(np.float64) - prediction.values.astype(np.float64)
                s_bu = float(np.mean(np.abs(diff)))
            else:
                s_bu = chunk.salience  # use chunk's existing salience as proxy

            # Top-down relevance: similarity to goal
            if goal is not None and goal.target_state is not None:
                sim = self._cosine_similarity(
                    chunk.state.values, goal.target_state.values
                )
                s_td = sim
            else:
                s_td = 0.5  # neutral when no goal

            # Precision weight
            precision = self._precisions.get(chunk.chunk_id, 1.0)

            # Composite salience
            raw_salience = (self.alpha_bu * s_bu + self.beta_td * s_td) * precision
            saliences.append(float(raw_salience))

        # Add Gumbel noise for stochastic exploration
        if self.gumbel_temperature > 0:
            gumbel_noise = self._rng.gumbel(0.0, self.gumbel_temperature, size=len(saliences))
            noisy_saliences = np.array(saliences, dtype=np.float64) + gumbel_noise
        else:
            noisy_saliences = np.array(saliences, dtype=np.float64)

        # k-WTA selection
        top_k_indices = np.argsort(noisy_saliences)[-k:][::-1]
        selected = [wm_chunks[i] for i in top_k_indices]

        # Update chunk saliences with computed composite
        for i, chunk in enumerate(wm_chunks):
            chunk.salience = float(noisy_saliences[i])

        self._last_saliences = list(noisy_saliences)
        self._last_selected_indices = list(top_k_indices)

        return selected

    # ── Precision Learning ───────────────────────────────────

    def update_precision(
        self,
        chunk_id: int,
        prediction_error: float,
    ) -> float:
        """Update precision for a chunk based on prediction error.

        Online delta rule:
            p(t+1) = p(t) + η · (|δ| - p(t))

        Where |δ| is the magnitude of prediction error.
        If |δ| > p, precision increases (chunk was more useful than expected).
        If |δ| < p, precision decreases.

        Args:
            chunk_id: Chunk identifier to update.
            prediction_error: Scalar prediction error δ.

        Returns:
            Updated precision value.
        """
        old_p = self._precisions.get(chunk_id, 1.0)
        delta_mag = min(abs(prediction_error), 10.0)
        new_p = old_p + self.precision_lr * (delta_mag - old_p)
        new_p = max(0.01, min(new_p, 10.0))  # clamp
        self._precisions[chunk_id] = new_p
        return new_p

    def get_precision(self, chunk_id: int) -> float:
        """Get the current precision for a chunk.

        Args:
            chunk_id: Chunk identifier.

        Returns:
            Precision value (default 1.0 if no history).
        """
        return self._precisions.get(chunk_id, 1.0)

    # ── Utility ──────────────────────────────────────────────

    def get_last_selection(self) -> Tuple[List[float], List[int]]:
        """Get the saliences and indices from the last selection.

        Returns:
            Tuple of (saliences, selected_indices).
        """
        return (self._last_saliences, self._last_selected_indices)

    def reset(self) -> None:
        """Reset attention state for a new training run."""
        self._precisions.clear()
        self._last_saliences.clear()
        self._last_selected_indices.clear()
        self._rng = np.random.RandomState(42)

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors.

        Args:
            a: First vector.
            b: Second vector.

        Returns:
            Cosine similarity in [-1, 1].
        """
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a < 1e-8 or norm_b < 1e-8:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))
