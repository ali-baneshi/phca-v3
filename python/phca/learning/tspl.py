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

from dataclasses import dataclass, asdict
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
        enabled: If False, the stream's update is a no-op (Phase 3.3a feature gate).
    """
    alpha: float
    lambda_: float
    eta: float
    accuracy_threshold: float = 0.95
    enabled: bool = True


DEFAULT_STREAM_CONFIGS: Dict[StreamID, StreamConfig] = {
    StreamID.P_STREAM: StreamConfig(
        alpha=0.08, lambda_=0.01, eta=0.1, accuracy_threshold=0.95,
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
        # Deep-copy configs so instance mutation never pollutes the global DEFAULT_STREAM_CONFIGS
        self.configs = {
            sid: StreamConfig(**asdict(cfg))
            for sid, cfg in DEFAULT_STREAM_CONFIGS.items()
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

        # Phase 3.3a: If stream is disabled (E/S-Stream by default), return theta unchanged.
        if not config.enabled:
            return dict(self.theta), False

        # Compute gradient if not provided (simple delta-rule approximation)
        if gradient is None:
            gradient = self._compute_gradient(prediction_error, state, prediction)

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
        # G4: NaN gate — return zero gradient if inputs are degenerate
        if not np.all(np.isfinite(state.values)):
            _log(logger, "warning", "tspl.nan_gradient_input",
                 source="state", fallback="zero_gradient")
            return {key: np.zeros_like(param) for key, param in self.theta.items()}
        if not np.all(np.isfinite(prediction.values)):
            _log(logger, "warning", "tspl.nan_gradient_input",
                 source="prediction", fallback="zero_gradient")
            return {key: np.zeros_like(param) for key, param in self.theta.items()}

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
        # G3: NaN gate — return 0.0 if prediction_error is degenerate
        if not np.isfinite(prediction_error):
            _log(logger, "warning", "tspl.nan_prediction_error",
                 value=prediction_error, fallback="0.0")
            return 0.0

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




