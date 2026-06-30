"""
PHCA v3.0 — P-Stream TSPL (Three-Stream Predictive Learning).

Phase 3.1: P-Stream active, E-Stream and S-Stream are stubs.
Phase 3.2+: All three streams active with EWC/GEM consolidation.

v3.0 References:
    - §3.1 Definition 3.2 (P-Stream only, Phase 3.1)
    - §3.1 Definition 3.3.3 (skill compilation)
    - §3.1 Table 2 (stream-specific hyperparameters)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from phca.config import StateVector, StreamID


@dataclass
class StreamConfig:
    """Configuration for a TSPL learning stream (v3.0 §3.1 Table 2).

    Attributes:
        alpha: Learning rate (P-Stream highest, S-Stream lowest).
        lambda_: Elastic consolidation strength (P-Stream lowest, S-Stream highest).
        eta: Exploration noise stddev (P-Stream highest).
        accuracy_threshold: Skill compilation threshold (default 0.95).
    """
    alpha: float
    lambda_: float
    eta: float
    accuracy_threshold: float = 0.95


DEFAULT_STREAM_CONFIGS: Dict[StreamID, StreamConfig] = {
    StreamID.P_STREAM: StreamConfig(
        alpha=0.05, lambda_=0.01, eta=0.1, accuracy_threshold=0.95,
    ),
    StreamID.E_STREAM: StreamConfig(
        alpha=0.005, lambda_=0.1, eta=0.01, accuracy_threshold=0.90,
    ),
    StreamID.S_STREAM: StreamConfig(
        alpha=0.0005, lambda_=1.0, eta=0.001, accuracy_threshold=0.90,
    ),
}


class TSPL:
    """Three-Stream Predictive Learning module.

    Phase 3.1:
        - P-Stream (procedural): Fast learning, high exploration noise.
          Updates G' CPD parameters via prediction error δ_t.
        - E-Stream (episodic): Stub — returns theta unchanged.
        - S-Stream (semantic): Stub — returns theta unchanged.
        - Skill compilation: P-Stream parameters freeze at ≥95% accuracy.

    Phase 3.2+:
        - E-Stream active with GEM projection.
        - S-Stream active with EWC penalty.
        - Multi-stream consolidation scheduling.

    The unified learning rule (v3.0 Definition 3.2):
        θ_new = θ - α · ∇L - λ · (θ - θ_protected) + η · noise
    """

    def __init__(self, seed: int = 42):
        """Initialize TSPL with default stream configurations.

        Args:
            seed: Random seed for deterministic exploration noise.
        """
        self.configs = {
            StreamID.P_STREAM: DEFAULT_STREAM_CONFIGS[StreamID.P_STREAM],
            StreamID.E_STREAM: DEFAULT_STREAM_CONFIGS[StreamID.E_STREAM],
            StreamID.S_STREAM: DEFAULT_STREAM_CONFIGS[StreamID.S_STREAM],
        }

        # Learned parameters (mapping parameter name → numpy array)
        self.theta: Dict[str, np.ndarray] = {}

        # Protected parameters for elastic consolidation (v3.0 §3.1)
        self.theta_protected: Dict[str, np.ndarray] = {}

        # Skill compilation state
        self.skill_compiled: bool = False
        self.skill_accuracy: float = 0.0
        self.compiled_skill_ids: List[str] = []

        # Deterministic noise
        self.rng = np.random.RandomState(seed)

        # ── Phase 3.2: E-Stream (GEM) state ──
        # Reference gradients from previous tasks (for GEM projection)
        self._gem_reference_grads: List[Dict[str, np.ndarray]] = []
        # Number of GEM reference tasks seen
        self._gem_tasks_seen: int = 0
        # Max reference gradients to store (memory bound)
        self._gem_max_references: int = 100

        # ── Phase 3.2: S-Stream (EWC) state ──
        # Fisher information matrix diagonal (for EWC penalty)
        self._ewc_fisher: Dict[str, np.ndarray] = {}
        self._ewc_theta_star: Dict[str, np.ndarray] = {}  # optimal params before new task

    def init_parameters(self, name: str, shape: Tuple[int, ...]) -> None:
        """Initialize a parameter group with random values.

        Args:
            name: Parameter name (e.g., "gprime_cpd_transition").
            shape: Shape of the parameter array.
        """
        self.theta[name] = self.rng.randn(*shape).astype(np.float32) * 0.1

    def update(
        self,
        stream: StreamID,
        prediction_error: float,
        state: StateVector,
        prediction: StateVector,
        gradient: Optional[Dict[str, np.ndarray]] = None,
    ) -> Tuple[Dict[str, np.ndarray], bool]:
        """Unified TSPL update (v3.0 Definition 3.2).

        Applies the learning rule for the specified stream.
        For E-Stream and S-Stream stubs, returns theta unchanged.

        Args:
            stream: Which stream to update (P/E/S).
            prediction_error: δ_t from PredictionErrorUnit.
            state: Current state vector (for gradient computation).
            prediction: Predicted state vector (for accuracy estimation).
            gradient: Optional pre-computed gradients. If None, computed
                using a simple delta-rule approximation.

        Returns:
            Tuple of (updated_theta, skill_compiled):
                updated_theta: Dict of parameter arrays after update.
                skill_compiled: Whether a new skill was compiled this cycle.
        """
        config = self.configs[stream]

        if not self.theta:
            return {}, False

        # Compute gradient if not provided (simple delta-rule approximation)
        if gradient is None:
            gradient = self._compute_gradient(prediction_error, state, prediction)

        # Phase 3.2: E-Stream — apply GEM projection to prevent forgetting
        if stream == StreamID.E_STREAM:
            gradient = self._gem_project(gradient)

        # Phase 3.2: S-Stream — add EWC penalty gradient
        if stream == StreamID.S_STREAM:
            gradient = self._ewc_add_penalty(gradient)

        # Unfrozen parameter update
        theta_new: Dict[str, np.ndarray] = {}
        for key in self.theta:
            theta_new[key] = (
                self.theta[key]
                - config.alpha * gradient.get(key, np.zeros_like(self.theta[key]))
                - config.lambda_ * (self.theta[key] - self.theta_protected.get(key, self.theta[key]))
                + config.eta * self.rng.randn(*self.theta[key].shape).astype(np.float32)
            )

        self.theta = theta_new

        # Store reference gradient for E-Stream (GEM memory update)
        if stream == StreamID.E_STREAM:
            self._gem_add_reference(gradient)

        # Update Fisher information for S-Stream (EWC)
        if stream == StreamID.S_STREAM:
            self._ewc_update_fisher(gradient)

        # Skill compilation check (P-Stream only)
        skill_compiled = False
        if stream == StreamID.P_STREAM:
            accuracy = self._estimate_accuracy(prediction, state, prediction_error)
            self.skill_accuracy = accuracy
            if accuracy >= config.accuracy_threshold and not self.skill_compiled:
                self._compile_skill("p_stream_current")
                skill_compiled = True
                self.skill_compiled = True

        return dict(self.theta), skill_compiled

    # ── Internal Methods ──────────────────────────────────────

    def _compute_gradient(
        self,
        prediction_error: float,
        state: StateVector,
        prediction: StateVector,
    ) -> Dict[str, np.ndarray]:
        """Compute proper per-dimension delta-rule gradient.

        For each parameter θⱼ, the gradient is proportional to the signed
        prediction error for that dimension:
            ∇Lⱼ = (predictionⱼ - stateⱼ) · η_scale

        Unlike the Phase 3.1 placeholder (which used sign(diff) × scalar error × 0.001,
        discarding per-dimension error magnitude), this preserves the direction
        AND magnitude of each dimension's prediction error.

        The gradient is then shaped to match the parameter's dimensions via
        broadcasting so each parameter component receives a gradient
        proportional to the corresponding dimension's error.

        Phase 3.3+: Exact backpropagation through G' if differentiable.

        Args:
            prediction_error: Scalar error δ_t (unused — per-dim error used instead).
            state: Current state (target).
            prediction: Predicted state (output).

        Returns:
            Dict mapping parameter names to gradient arrays.
        """
        # Per-dimension signed error (preserves magnitude per dimension)
        error_per_dim = (
            prediction.values.astype(np.float64) - state.values.astype(np.float64)
        )
        # Delta-rule scaling: 0.1 gives ~0.005 effective step with α=0.05
        scaled_error = error_per_dim * 0.1

        gradient: Dict[str, np.ndarray] = {}
        for key in self.theta:
            param = self.theta[key]
            if param.ndim == 1 and param.shape[0] == scaled_error.shape[0]:
                # 1-D param: direct per-dimension gradient
                gradient[key] = scaled_error.astype(np.float32)
            elif param.ndim == 2 and param.shape[0] == scaled_error.shape[0]:
                # 2-D param (e.g., (state_dim, n_features)): broadcast error across columns
                grad = np.tile(
                    scaled_error[:, np.newaxis], (1, param.shape[1])
                )
                gradient[key] = grad.astype(np.float32)
            elif scaled_error.size == param.size:
                # Same total size, different shape: reshape error
                gradient[key] = scaled_error.astype(np.float32).reshape(param.shape)
            else:
                # Fallback: use mean absolute error across all param dims.
                # When state and param dimensionalities don't align (e.g.,
                # (state_dim=2) vs param_shape=(4,4)), fill with mean abs
                # error so every parameter component receives a signal
                # proportional to the aggregate prediction error magnitude.
                # Mean abs (not mean) avoids sign cancellation from
                # symmetric errors (e.g., [-0.002, 0.002] → mean=0 but |mean|=0.002).
                mean_abs_error = float(np.mean(np.abs(scaled_error)))
                gradient[key] = np.full_like(param, mean_abs_error, dtype=np.float32)
        return gradient

    def _estimate_accuracy(
        self,
        prediction: StateVector,
        state: StateVector,
        prediction_error: float,
    ) -> float:
        """Estimate prediction accuracy from error and confidence.

        Phase 3.1: accuracy = max(0, 1 - sqrt(δ_t / dim)).

        Args:
            prediction: Predicted state.
            state: Actual observed state.
            prediction_error: Scalar error δ_t.

        Returns:
            Accuracy in [0.0, 1.0].
        """
        dim = max(state.values.shape[0], 1)
        rmse = np.sqrt(prediction_error / dim)
        return float(max(0.0, 1.0 - rmse))

    def _compile_skill(self, skill_id: str) -> None:
        """Compile a skill by freezing current parameters.

        Phase 3.1: Simple flag + protected parameter snapshot.
        Phase 3.2+: Store in M5 (long-term procedural memory).

        Args:
            skill_id: Unique identifier for the skill.
        """
        self.theta_protected = {key: val.copy() for key, val in self.theta.items()}
        self.compiled_skill_ids.append(skill_id)

    def freeze_skill(self, skill_id: str) -> None:
        """External API to freeze a skill (v3.0 Definition 3.3.3).

        Args:
            skill_id: Unique identifier for the skill.
        """
        self._compile_skill(skill_id)

    # ── Phase 3.2: GEM Projection (E-Stream) ─────────────────

    def _gem_project(
        self, gradient: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """Apply GEM projection to prevent catastrophic forgetting.

        Projects the current gradient onto the feasible region where
        inner product with all stored reference gradients is ≥ 0.
        Uses the quadratic programming formulation from
        Lopez-Paz & Ranzato (2017).

        If no references or all inner products ≥ 0, returns gradient unchanged.

        Args:
            gradient: Current gradient dict (param_name → array).

        Returns:
            Projected gradient (or original if no projection needed).
        """
        if not self._gem_reference_grads:
            return gradient

        # Check if projection is needed
        needs_projection = False
        for ref in self._gem_reference_grads:
            dot_product = 0.0
            for key in gradient:
                if key in ref:
                    dot_product += float(np.sum(gradient[key] * ref[key]))
            if dot_product < 0:
                needs_projection = True
                break

        if not needs_projection:
            return gradient

        # Simple projection: average current gradient with nearest reference
        # that has negative inner product. For full QP, we'd need cvxopt.
        # Phase 3.3+: Full quadratic programming.
        proj: Dict[str, np.ndarray] = {}
        for key in gradient:
            proj[key] = gradient[key].copy()

        for ref in self._gem_reference_grads:
            dot_product = 0.0
            for key in gradient:
                if key in ref:
                    dot_product += float(np.sum(gradient[key] * ref[key]))
            if dot_product < 0:
                # Average with reference (linear interpolation toward feasible)
                for key in gradient:
                    if key in ref:
                        ref_norm = float(np.linalg.norm(ref[key]))
                        if ref_norm > 1e-8:
                            # Project onto feasible direction
                            alpha = min(1.0, abs(dot_product) / (  # type: ignore[operator]
                                float(np.linalg.norm(gradient[key])) * ref_norm + 1e-8
                            ))
                            proj[key] = (1.0 - alpha * 0.5) * proj[key] + alpha * 0.5 * ref[key]

        return proj

    def _gem_add_reference(self, gradient: Dict[str, np.ndarray]) -> None:
        """Add a reference gradient to GEM memory.

        Args:
            gradient: Gradient to store as reference.
        """
        if len(self._gem_reference_grads) >= self._gem_max_references:
            # Remove oldest
            self._gem_reference_grads.pop(0)
        self._gem_reference_grads.append(
            {k: v.copy() for k, v in gradient.items()}
        )
        self._gem_tasks_seen += 1

    # ── Phase 3.2: EWC Penalty (S-Stream) ────────────────────

    def _ewc_add_penalty(
        self, gradient: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """Add EWC penalty gradient to prevent catastrophic forgetting.

        EWC penalty: penalty = Σ (λ · Fisher · (θ - θ_star))
        The gradient of this penalty is added to the parameter update.

        Args:
            gradient: Current gradient dict.

        Returns:
            Gradient with EWC penalty added.
        """
        if not self._ewc_fisher or not self._ewc_theta_star:
            return gradient

        result: Dict[str, np.ndarray] = {}
        config = self.configs[StreamID.S_STREAM]

        for key in gradient:
            penalty = np.zeros_like(gradient[key])
            if key in self._ewc_fisher and key in self._ewc_theta_star:
                # EWC penalty gradient = λ · Fisher · (θ - θ*)
                fisher = self._ewc_fisher[key]
                theta_diff = self.theta[key] - self._ewc_theta_star[key]
                penalty = config.lambda_ * fisher * theta_diff
            result[key] = gradient[key] + penalty

        return result

    def _ewc_update_fisher(self, gradient: Dict[str, np.ndarray]) -> None:
        """Update Fisher information matrix diagonal for EWC.

        Fisher information is approximated as the squared gradient.
        Running average: F ← (1 - β) · F + β · g²

        Args:
            gradient: Current gradient dict.
        """
        beta = 0.1  # Fisher update rate

        for key in gradient:
            g_squared = gradient[key] ** 2
            if key in self._ewc_fisher:
                self._ewc_fisher[key] = (1.0 - beta) * self._ewc_fisher[key] + beta * g_squared
            else:
                self._ewc_fisher[key] = g_squared
                if key in self.theta:
                    self._ewc_theta_star[key] = self.theta[key].copy()

    def reset(self) -> None:
        """Reset TSPL state for a new training run."""
        self.theta.clear()
        self.theta_protected.clear()
        self.skill_compiled = False
        self.skill_accuracy = 0.0
        self.compiled_skill_ids.clear()
        self._gem_reference_grads.clear()
        self._gem_tasks_seen = 0
        self._ewc_fisher.clear()
        self._ewc_theta_star.clear()
