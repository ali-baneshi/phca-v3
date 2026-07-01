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
            S = (α(goal)·S_bu + β(goal)·S_td) · precision + Gumbel(0, temperature)

        α and β are drive-dependent blends where higher-priority goals get
        stronger top-down biasing. The similarity metric is precision-weighted
        so dimensions with higher goal specificity dominate.

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

        # ── Drive-dependent attention blend ──
        # When a goal with a target_state is present, the bottom-up/top-down
        # blend is modulated by drive type and goal priority:
        #   D1/D3 (error/competence): focus on prediction errors (bottom-up)
        #   D2/D4 (exploration): follow exploration goals (top-down)
        #   D5 (energy): seek minimal action states (top-down)
        #   D6 (empowerment): balanced
        # When no goal is present, instance-configured alpha_bu/beta_td are used.
        effective_alpha = self.alpha_bu
        effective_beta = self.beta_td
        if goal is not None and goal.target_state is not None:
            drive_id = getattr(goal, "drive_id", 1)
            if drive_id in (1, 3):
                effective_alpha = 0.7
                effective_beta = 0.3
            elif drive_id in (2, 4):
                effective_alpha = 0.3
                effective_beta = 0.7
            elif drive_id == 5:
                effective_alpha = 0.2
                effective_beta = 0.8
            else:  # D6 and default
                effective_alpha = 0.5
                effective_beta = 0.5

            # Scale top-down by goal priority: high-priority goals get stronger biasing
            goal_priority = getattr(goal, "priority", 0.5)
            effective_beta *= max(0.2, min(2.0, goal_priority * 2.0))
            # Re-normalize so alpha + beta = 1.0
            total = effective_alpha + effective_beta
            effective_alpha /= total
            effective_beta /= total

        saliences: List[float] = []
        for chunk in wm_chunks:
            # Bottom-up salience: unexpectedness
            if prediction is not None:
                diff = chunk.state.values.astype(np.float64) - prediction.values.astype(np.float64)
                s_bu = float(np.mean(np.abs(diff)))
            else:
                s_bu = chunk.salience  # use chunk's existing salience as proxy

            # Top-down relevance: precision-weighted similarity to goal
            if goal is not None and goal.target_state is not None:
                sim = self._precision_weighted_similarity(
                    chunk.state.values, goal.target_state.values,
                    goal.target_state.precision,
                )
                s_td = sim
            else:
                s_td = 0.5  # neutral when no goal

            # Precision weight (learned per-chunk precision from prediction error)
            precision = self._precisions.get(chunk.chunk_id, 1.0)

            # Composite salience with drive-dependent blend
            raw_salience = (effective_alpha * s_bu + effective_beta * s_td) * precision
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

    @staticmethod
    def _precision_weighted_similarity(
        a: np.ndarray, b: np.ndarray,
        precision: Optional[np.ndarray] = None,
    ) -> float:
        """Compute precision-weighted cosine similarity between two vectors.

        Dimensions with higher precision (specified goal targets) are weighted
        more heavily. If no precision is provided, falls back to standard
        cosine similarity.

        Args:
            a: First vector (chunk state).
            b: Second vector (goal target_state).
            precision: Per-dimension precision weights from goal.target_state
                (higher = more important dimension for the goal).

        Returns:
            Similarity score in [0.0, 1.0] (1.0 = exact match).
        """
        # Use minimum dimension to avoid shape mismatch
        min_dim = min(a.shape[0], b.shape[0])
        a = a[:min_dim].astype(np.float64)
        b = b[:min_dim].astype(np.float64)

        if precision is not None:
            w = precision[:min_dim].astype(np.float64)
        else:
            w = np.ones(min_dim, dtype=np.float64)

        # Weighted cosine similarity
        a_norm = np.linalg.norm(a * w)
        b_norm = np.linalg.norm(b * w)
        if a_norm < 1e-8 or b_norm < 1e-8:
            return 0.5

        cos_sim = float(np.dot(a * w, b * w) / (a_norm * b_norm))
        # Map [-1, 1] → [0, 1]
        return float(np.clip((cos_sim + 1.0) / 2.0, 0.0, 1.0))

    # ── Observability v4: additive snapshot ──

    def snapshot(self) -> Dict[str, Any]:
        """Read-only portrait of attention precisions for the dashboard."""
        return {
            "precisions": [float(p) for p in self._precisions.values()],
            "gumbel_temperature": float(self.gumbel_temperature),
        }
