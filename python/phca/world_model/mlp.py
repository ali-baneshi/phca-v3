"""
PHCA v3.0 — MLP World Model G' (Phase 3.3b).

Replaces the Gaussian CPD Bayesian network with a learned
feedforward MLP for GridWorld state spaces (values ∈ {0, 1, 2}).

Architecture:  Input(89) → Linear(128) → ReLU → Linear(128) → ReLU → Linear(84)
Loss:          MSE (mean squared error) — supports any target value range.
Confidence:    exp(-mean_MSE)  (1.0 = perfect, → 0.0 as loss increases)

Pure NumPy — no PyTorch dependency. Manual forward/backward pass.
Output is LINEAR (no sigmoid) because GridWorld states contain values 0, 1, and 2.

v3.0 References:
    - §2.2 Definition 2.4b (G' interface)
    - Phase 3.3b Strategic Report §3.2
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from phca.config import StateVector
from phca.logging import logger, _log


_EPS = 1e-8


class WorldModelMLP:
    """MLP-based world model replacing Gaussian G'.

    Interface-compatible with WorldModelGPrime.predict() and learn(),
    so the cognitive cycle requires no structural changes.

    Attributes:
        state_dim: Dimensionality of state vectors.
        action_dim: Dimensionality of action vectors.
        hidden_dim: Number of hidden units per layer.
    """

    def __init__(
        self,
        state_dim: int = 84,
        action_dim: int = 5,
        hidden_dim: int = 64,
        seed: int = 42,
        lr: float = 0.2,
        replay_capacity: int = 500,
        batch_size: int = 64,
        train_steps: int = 8,
        dropout_rate: float = 0.1,
        mc_samples: int = 10,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.replay_capacity = replay_capacity
        self.batch_size = batch_size
        self.train_steps = train_steps
        self.dropout_rate = dropout_rate
        self.mc_samples = mc_samples
        self.rng = np.random.RandomState(seed)
        self._mc_rng = np.random.RandomState(seed + 1)  # separate RNG for MC Dropout

        input_dim = state_dim + action_dim

        # He initialization
        self.w1 = self.rng.normal(0, np.sqrt(2.0 / input_dim), (input_dim, hidden_dim)).astype(np.float32)
        self.b1 = np.zeros(hidden_dim, dtype=np.float32)
        self.w2 = self.rng.normal(0, np.sqrt(2.0 / hidden_dim), (hidden_dim, hidden_dim)).astype(np.float32)
        self.b2 = np.zeros(hidden_dim, dtype=np.float32)
        self.w3 = self.rng.normal(0, np.sqrt(2.0 / hidden_dim), (hidden_dim, state_dim)).astype(np.float32)
        self.b3 = np.zeros(state_dim, dtype=np.float32)

        # Cache for backward pass (set by predict, consumed by compute_gradient)
        self._last_input: Optional[np.ndarray] = None
        self._last_activations: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None

        # Phase-4 interface stubs (AF-001/002 compatibility)
        self._tspl_bias: np.ndarray = np.zeros(state_dim, dtype=np.float32)

        # Experience replay buffer
        self._replay_buffer: List[Tuple[np.ndarray, np.ndarray]] = []
        self._replay_idx: int = 0

    # ── Public Interface (G'-compatible) ──────────────────────

    def predict(
        self, state: StateVector, action: np.ndarray
    ) -> Tuple[StateVector, float]:
        """Predict next state given current state and action via MC Dropout.

        Runs *mc_samples* stochastic forward passes with dropout mask on
        hidden layers.  Returns the mean prediction and a confidence
        derived from prediction variance across the ensemble.

        Args:
            state: Current state vector (state_dim,).
            action: Action vector (action_dim,), typically one-hot.

        Returns:
            Tuple of (predicted_state, confidence):
                predicted_state: Mean prediction across MC samples.
                confidence: Prediction confidence in [0, 1], scaled by
                    epistemic uncertainty — lower when MC variance is high.
        """
        x = np.concatenate([state.values.astype(np.float32), action.astype(np.float32)])

        # MC Dropout forward passes
        mc_outs: List[np.ndarray] = []
        for _ in range(self.mc_samples):
            z1, z2, out = self._forward_mc(x)
            mc_outs.append(out)

        mean_out = np.mean(mc_outs, axis=0).astype(np.float32)

        # Cache last mean pass for backward compatibility
        self._last_input = x
        self._last_activations = (mean_out,)  # partial cache

        # Confidence: exp(-MSE) scaled by mutual information
        # (epistemic uncertainty from MC variance)
        var = np.var(mc_outs, axis=0).mean()
        # mutual information ≈ log(1 + var) clipped to [0, ~1.1]
        mutual_info = float(np.log1p(min(var, 2.0)))
        # base confidence = exp(-MSE on mean prediction)
        base_conf = self._compute_confidence(mean_out, state.values.astype(np.float32))
        # penalize confidence when mutual info is high (uncertain)
        confidence = float(np.clip(base_conf * (1.0 - 0.5 * mutual_info), 0.01, 1.0))

        # Apply TSPL learned bias (AF-002 compatibility, safe mode).
        # Only apply when bias norm is small (< 1.0) to avoid output corruption.
        _bias_norm = float(np.linalg.norm(self._tspl_bias))
        if _bias_norm < 1.0:
            biased_out = mean_out + self._tspl_bias
        else:
            biased_out = mean_out
            self._tspl_bias[:] = 0.0  # reset corrupted bias

        return (
            StateVector(
                values=biased_out.astype(np.float32),
                precision=np.full(self.state_dim, confidence, dtype=np.float32),
                timestamp=state.timestamp + 1.0,
                grounding_level=state.grounding_level,
            ),
            confidence,
        )

    def learn(
        self,
        state_t: StateVector,
        action: np.ndarray,
        state_t1: StateVector,
        error: float,
    ) -> None:
        """Learn from observed transition via SGD with experience replay.

        Stores the transition in a replay buffer and trains on a
        mini-batch of random samples from the buffer each cycle.

        Args:
            state_t: State at time t.
            action: Action taken.
            state_t1: Observed state at time t+1 (target).
            error: Scalar prediction error (unused — MSE gradient used instead).
        """
        # Redo forward pass with the actual action for immediate learning
        x = np.concatenate([state_t.values.astype(np.float32), action.astype(np.float32)])
        self._last_input = x
        z1, z2, out = self._forward(x)
        self._last_activations = (z1, z2, out)

        target = state_t1.values.astype(np.float32)

        # Store in replay buffer
        if len(self._replay_buffer) < self.replay_capacity:
            self._replay_buffer.append((x.copy(), target.copy()))
        else:
            self._replay_buffer[self._replay_idx % self.replay_capacity] = (x.copy(), target.copy())
        self._replay_idx += 1

        # Skip batch training if buffer not big enough yet
        if len(self._replay_buffer) < self.batch_size:
            grad = self._backward(x, z1, z2, out, target)
            self._apply_gradient(grad, lr=self.lr)
            return

        # Train on mini-batches from replay buffer (current transition included via buffer)
        for _ in range(self.train_steps):
                indices = self.rng.randint(0, len(self._replay_buffer), size=self.batch_size)
                avg_grad: Optional[Dict[str, np.ndarray]] = None
                for idx in indices:
                    bx, btarget = self._replay_buffer[idx]
                    bz1, bz2, bout = self._forward(bx)
                    bgrad = self._backward(bx, bz1, bz2, bout, btarget)
                    if avg_grad is None:
                        avg_grad = {k: v.copy() for k, v in bgrad.items()}
                    else:
                        for k in avg_grad:
                            avg_grad[k] += bgrad[k]
                if avg_grad is not None:
                    for k in avg_grad:
                        avg_grad[k] /= float(self.batch_size)
                    self._apply_gradient(avg_grad, lr=self.lr * 0.5)

    def reset(self) -> None:
        """Reset forward/backward cache. Weights and replay buffer persist across episodes."""
        self._last_input = None
        self._last_activations = None

    # ── Internal Methods ──────────────────────────────────────

    def _forward(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Forward pass.

        Args:
            x: Input vector (state_dim + action_dim,).

        Returns:
            Tuple of (z1, z2, out):
                z1: Pre-activation of layer 1.
                z2: Pre-activation of layer 2.
                out: Sigmoid output (state_dim,).
        """
        z1 = x @ self.w1 + self.b1
        a1 = np.maximum(0, z1)
        z2 = a1 @ self.w2 + self.b2
        a2 = np.maximum(0, z2)
        z3 = a2 @ self.w3 + self.b3
        out = z3  # linear output — GridWorld states contain values 0, 1, 2
        return z1, z2, out

    def _forward_mc(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Stochastic forward pass with dropout (inference only).

        Applies dropout mask to both hidden layers.  Each call uses a
        fresh mask sampled from a separate RNG so training determinism
        is unaffected.
        """
        z1 = x @ self.w1 + self.b1
        a1 = np.maximum(0, z1)
        mask1 = self._mc_rng.binomial(1, 1.0 - self.dropout_rate, size=a1.shape).astype(np.float32)
        a1 *= mask1 / (1.0 - self.dropout_rate)

        z2 = a1 @ self.w2 + self.b2
        a2 = np.maximum(0, z2)
        mask2 = self._mc_rng.binomial(1, 1.0 - self.dropout_rate, size=a2.shape).astype(np.float32)
        a2 *= mask2 / (1.0 - self.dropout_rate)

        out = a2 @ self.w3 + self.b3
        return z1, z2, out

    def _backward(
        self,
        x: np.ndarray,
        z1: np.ndarray,
        z2: np.ndarray,
        out: np.ndarray,
        target: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """Backward pass of MSE loss.

        Computes gradients of MSE = 0.5 * mean((out - target)²) w.r.t.
        all parameters, with gradient clipping to [-1, 1].

        Note: For linear output + MSE, dL/d(z) = (z - target) / state_dim,
        which is identical to the sigmoid+BCE combined gradient formula.
        The backward pass code is unchanged from the BCE version.

        Args:
            x: Input vector (state_dim + action_dim,).
            z1: Pre-activation of layer 1 (hidden_dim,).
            z2: Pre-activation of layer 2 (hidden_dim,).
            out: Linear output (state_dim,).
            target: Target vector (state_dim,).

        Returns:
            Dict mapping parameter keys to gradient arrays.
        """
        # dL/d(out) for MSE: (out - target) / state_dim
        n = float(out.shape[0])
        d_out = (out - target) / n

        # Add cross-entropy on agent position (first 25 dims)
        # Treats out[:25] as logits for a 25-class softmax.
        # This gives a strong gradient signal for learning position dynamics.
        pos_target = target[:25]
        pos_idx = int(np.argmax(pos_target))
        if pos_target.max() > 0.5:  # valid one-hot position in target
            logits = out[:25].copy()
            logits -= logits.max()  # numerical stability
            exp_l = np.exp(logits)
            probs = exp_l / (exp_l.sum() + 1e-8)
            ce_grad = probs.copy()
            ce_grad[pos_idx] -= 1.0  # = softmax - one_hot
            d_out[:25] += ce_grad / n  # same scaling as MSE

        a1 = np.maximum(0, z1)
        a2 = np.maximum(0, z2)

        # Layer 3: Linear(128 -> 84)
        d_z3 = d_out
        grad_w3 = np.outer(a2, d_z3)
        grad_b3 = d_z3

        # Layer 2: ReLU + Linear(128 -> 128)
        d_a2 = d_z3 @ self.w3.T
        d_z2 = d_a2 * (z2 > 0).astype(np.float32)
        grad_w2 = np.outer(a1, d_z2)
        grad_b2 = d_z2

        # Layer 1: ReLU + Linear(89 -> 128)
        d_a1 = d_z2 @ self.w2.T
        d_z1 = d_a1 * (z1 > 0).astype(np.float32)
        grad_w1 = np.outer(x, d_z1)
        grad_b1 = d_z1

        # Gradient clipping
        for g in (grad_w1, grad_b1, grad_w2, grad_b2, grad_w3, grad_b3):
            np.clip(g, -1.0, 1.0, out=g)

        return {
            "gprime_w1": np.ascontiguousarray(grad_w1.astype(np.float32)),
            "gprime_b1": np.ascontiguousarray(grad_b1.astype(np.float32)),
            "gprime_w2": np.ascontiguousarray(grad_w2.astype(np.float32)),
            "gprime_b2": np.ascontiguousarray(grad_b2.astype(np.float32)),
            "gprime_w3": np.ascontiguousarray(grad_w3.astype(np.float32)),
            "gprime_b3": np.ascontiguousarray(grad_b3.astype(np.float32)),
        }

    @staticmethod
    def _compute_confidence(prediction: np.ndarray, target: np.ndarray) -> float:
        """Compute prediction confidence from MSE loss.

        confidence = exp(-mean_MSE)
        where MSE = 0.5 * mean((prediction - target)²)
        - perfect prediction (MSE ≈ 0) → 1.0
        - For GridWorld states {0, 1, 2}, random pred ≈1 gives MSE ≈0.5 → ~0.6
        - poor prediction (MSE > 2) → ~0.14

        Args:
            prediction: Linear output (state_dim,).
            target: Target vector (state_dim,).

        Returns:
            Confidence in [0, 1].
        """
        mse = 0.5 * float(np.mean((prediction - target) ** 2))
        return float(np.exp(-max(mse, 0.0)))

    def _apply_gradient(
        self, grad: Dict[str, np.ndarray], lr: float = 0.01
    ) -> None:
        """Apply gradient via SGD.

        Args:
            grad: Dict with keys matching get_theta().
            lr: Learning rate.
        """
        self.w1 -= lr * grad["gprime_w1"]
        self.b1 -= lr * grad["gprime_b1"]
        self.w2 -= lr * grad["gprime_w2"]
        self.b2 -= lr * grad["gprime_b2"]
        self.w3 -= lr * grad["gprime_w3"]
        self.b3 -= lr * grad["gprime_b3"]

    def set_tspl_bias(self, bias: Optional[np.ndarray]) -> None:
        """Set learned bias from TSPL (AF-002 compatibility).

        Clamps bias norm to ≤1.0 to prevent corruption of MLP output
        when the TSPL delta-rule produces extreme values.
        """
        if bias is not None:
            bias_arr = bias.astype(np.float32) if isinstance(bias, np.ndarray) else np.zeros(self.state_dim, dtype=np.float32)
            bias_norm = float(np.linalg.norm(bias_arr))
            if bias_norm > 1.0:
                bias_arr = bias_arr / bias_norm  # project to unit sphere
            self._tspl_bias = bias_arr

    def get_prediction_accuracy(self, state, action, target) -> float:
        """Get MLP prediction accuracy (G-019 compatibility).
        Uses MSE-based confidence: exp(-0.5 * mean((pred - target)^2)).
        """
        x = np.concatenate([state.astype(np.float32), action.astype(np.float32)])
        _, _, out = self._forward(x)
        mse = 0.5 * float(np.mean((out - target.astype(np.float32)) ** 2))
        return float(np.exp(-max(mse, 0.0)))

    def estimate_empowerment(self, state) -> float:
        """Estimate empowerment (C2 fix compatibility).
        Heuristic: returns 0.3 for n_actions <= 5, else 0.2.
        """
        if hasattr(self, 'action_dim') and self.action_dim <= 5:
            return 0.3
        return 0.2

    def __repr__(self) -> str:
        n_params = (
            (self.state_dim + self.action_dim) * self.hidden_dim
            + self.hidden_dim
            + self.hidden_dim ** 2
            + self.hidden_dim
            + self.hidden_dim * self.state_dim
            + self.state_dim
        )
        return (
            f"WorldModelMLP("
            f"state_dim={self.state_dim}, action_dim={self.action_dim}, "
            f"hidden_dim={self.hidden_dim}, lr={self.lr}, "
            f"params={n_params})"
        )
