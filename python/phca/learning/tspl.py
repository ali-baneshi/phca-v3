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

        # Phase 3.1: E-Stream and S-Stream are stubs
        if stream in (StreamID.E_STREAM, StreamID.S_STREAM):
            return dict(self.theta), False

        # P-Stream update
        if not self.theta:
            return {}, False

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
        """Compute approximate gradient of prediction error w.r.t. parameters.

        Phase 3.1: Simple delta-rule approximation:
            ∇L ≈ (prediction - state) · δ_t · sign()

        Phase 3.2+: Exact backpropagation through G' if differentiable,
            or REINFORCE-style gradient estimation.

        Args:
            prediction_error: Scalar error δ_t (for scaling).
            state: Current state (target).
            prediction: Predicted state (output).

        Returns:
            Dict mapping parameter names to gradient arrays.
        """
        gradient: Dict[str, np.ndarray] = {}
        for key in self.theta:
            # Simple Hebbian-style gradient approximation
            diff = prediction.values.astype(np.float64) - state.values.astype(np.float64)
            grad = np.sign(diff) * np.abs(prediction_error) * 0.001

            # Broadcast or reshape to match parameter shape
            if grad.size == self.theta[key].size:
                gradient[key] = grad.astype(np.float32).reshape(self.theta[key].shape)
            else:
                # Fallback: small random gradient
                gradient[key] = (
                    self.rng.randn(*self.theta[key].shape).astype(np.float32) * 0.001
                )
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

    def reset(self) -> None:
        """Reset TSPL state for a new training run."""
        self.theta.clear()
        self.theta_protected.clear()
        self.skill_compiled = False
        self.skill_accuracy = 0.0
        self.compiled_skill_ids.clear()
