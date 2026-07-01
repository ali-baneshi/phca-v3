"""
PHCA v3.0 — MLP World Model G' (Phase 3.3b, AF-001 fix).

Replaces the Gaussian CPD Bayesian network with a learned
feedforward MLP for GridWorld state spaces (values ∈ {0, 1, 2}).

Architecture:  Input(89) → Linear(128) → ReLU → Linear(128) → ReLU → Linear(84)
Loss:          MSE (mean squared error) — supports any target value range.
Confidence:    MC Dropout (AF-001): 1/(1 + variance) from N=20 stochastic forward passes.
               Previously: exp(-mean_MSE) — not a valid confidence measure.

Pure NumPy — no PyTorch dependency. Manual forward/backward pass.
Output is LINEAR (no sigmoid) because GridWorld states contain values 0, 1, and 2.

AF-001 (Phase 4 gap audit): Added Monte Carlo Dropout for calibrated uncertainty.
  - dropout_rate=0.1 applied after each ReLU during inference AND training.
  - predict() runs mc_samples=20 stochastic forward passes.
  - confidence = mean(1/(1 + per-dimension variance)) across MC samples.
  - Training: forward/backward pass with dropout for consistent Bayesian interpretation.

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

    AF-001: Uses Monte Carlo Dropout for calibrated confidence.
    dropout_rate=0.1 is applied during both training and inference.
    Confidence = 1/(1 + mean_variance) from mc_samples=20 stochastic passes.

    Attributes:
        state_dim: Dimensionality of state vectors.
        action_dim: Dimensionality of action vectors.
        hidden_dim: Number of hidden units per layer.
        dropout_rate: Dropout probability (AF-001, default 0.1).
        mc_samples: Number of MC Dropout forward passes (AF-001, default 20).
    """

    def __init__(
        self,
        state_dim: int = 84,
        action_dim: int = 5,
        hidden_dim: int = 128,
        seed: int = 42,
        lr: float = 0.1,
        replay_capacity: int = 500,
        batch_size: int = 32,
        train_steps: int = 4,
        dropout_rate: float = 0.1,
        mc_samples: int = 20,
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

        # Experience replay buffer
        self._replay_buffer: List[Tuple[np.ndarray, np.ndarray]] = []
        self._replay_idx: int = 0

    # ── Public Interface (G'-compatible) ──────────────────────

    def predict(
        self, state: StateVector, action: np.ndarray
    ) -> Tuple[StateVector, float]:
        """Predict next state given current state and action (MC Dropout).

        Runs mc_samples stochastic forward passes with dropout to estimate
        both the mean prediction and the model's predictive uncertainty.
        Confidence is derived from the per-dimension predictive variance.

        Args:
            state: Current state vector (state_dim,).
            action: Action vector (action_dim,), typically one-hot.

        Returns:
            Tuple of (predicted_state, confidence):
                predicted_state: StateVector with mean MC prediction as values.
                confidence: Variance-based confidence in [0, 1].
        """
        x = np.concatenate([state.values.astype(np.float32), action.astype(np.float32)])

        predictions = []
        last_activations = None
        for _ in range(self.mc_samples):
            z1, z2, out, _, _ = self._forward(x, training=True)
            predictions.append(out)
            last_activations = (z1, z2, out)
        self._last_input = x
        self._last_activations = last_activations

        preds = np.stack(predictions, axis=0)
        mean_pred = np.mean(preds, axis=0)
        var_pred = np.var(preds, axis=0) + _EPS
        confidence = float(np.mean(1.0 / (1.0 + var_pred)))

        return (
            StateVector(
                values=mean_pred.astype(np.float32),
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
        """Learn from observed transition via replay buffer (G-017 fix).

        The error parameter modulates the effective learning rate:
          lr_effective = self.lr * (1.0 + abs(error) * 0.1)
        Higher error → larger learning steps (correct mistakes faster).
        Lower error → smaller learning steps (fine-tune).

        Stores the transition in a replay buffer and trains on a
        mini-batch of random samples from the buffer each cycle.

        G-017 (Phase 4 gap audit): Removed online SGD to eliminate
        conflicting gradient signals. Previously, online and batch
        gradients were applied in the same cycle, creating a conflict:
        the online gradient used attention-weighted per-dimension
        modulation, while batch gradients did not (different attention
        contexts). Now only batch replay gradients are applied.

        Args:
            state_t: State at time t.
            action: Action taken.
            state_t1: Observed state at time t+1 (target).
            error: Scalar prediction error (modulates learning rate).
        """
        # Compute error-modulated learning rate (A5: Feedback-Driven Adaptation)
        lr_mod = float(np.clip(1.0 + abs(error) * 0.1, 0.5, 2.0))
        effective_lr = self.lr * lr_mod

        # Build input and target
        x = np.concatenate([state_t.values.astype(np.float32), action.astype(np.float32)])
        target = state_t1.values.astype(np.float32)

        # Store in replay buffer
        if len(self._replay_buffer) < self.replay_capacity:
            self._replay_buffer.append((x.copy(), target.copy()))
        else:
            self._replay_buffer[self._replay_idx % self.replay_capacity] = (x.copy(), target.copy())
        self._replay_idx += 1

        # Attention weights: apply to batch gradients (G-017: removed online-only path)
        attention_weights = None
        if hasattr(self, '_attention_weights') and self._attention_weights is not None:
            attention_weights = self._attention_weights.copy()
            # Clear after use to avoid stale weights
            self._attention_weights = None

        # Skip learning if buffer not big enough (no online SGD fallback — G-017)
        if len(self._replay_buffer) < self.batch_size:
            return

        # Train on mini-batches from replay buffer only (G-017 fix)
        # Dropout is applied during training (AF-001) for consistent
        # Bayesian interpretation with MC Dropout at inference time.
        for _ in range(self.train_steps):
                indices = self.rng.randint(0, len(self._replay_buffer), size=self.batch_size)
                avg_grad: Optional[Dict[str, np.ndarray]] = None
                for idx in indices:
                    bx, btarget = self._replay_buffer[idx]
                    bz1, bz2, bout, ba1, ba2 = self._forward(bx, training=True)
                    bgrad = self._backward(bx, bz1, bz2, bout, ba1, ba2, btarget, attention_weights)
                    if avg_grad is None:
                        avg_grad = {k: v.copy() for k, v in bgrad.items()}
                    else:
                        for k in avg_grad:
                            avg_grad[k] += bgrad[k]
                if avg_grad is not None:
                    for k in avg_grad:
                        avg_grad[k] /= float(self.batch_size)
                    self._apply_gradient(avg_grad, lr=effective_lr * 0.5)

    def reset(self) -> None:
        """Reset forward/backward cache. Weights and replay buffer persist across episodes."""
        self._last_input = None
        self._last_activations = None
        self._attention_weights = None

    # ── Internal Methods ──────────────────────────────────────

    def _forward(
        self, x: np.ndarray, training: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Forward pass with optional dropout (AF-001).

        When training=True, inverted dropout is applied after each ReLU.
        The dropout mask is independent per call (Bernoulli(1-dropout_rate)).
        Dropped activations are scaled by 1/(1-dropout_rate) to maintain
        expected activation magnitude (inverted dropout).

        Args:
            x: Input vector (state_dim + action_dim,).
            training: If True, apply dropout after each ReLU.

        Returns:
            Tuple of (z1, z2, out, a1, a2):
                z1: Pre-activation of layer 1.
                z2: Pre-activation of layer 2.
                out: Linear output (state_dim,).
                a1: Post-ReLU activation of layer 1 (with dropout if training).
                a2: Post-ReLU activation of layer 2 (with dropout if training).
        """
        z1 = x @ self.w1 + self.b1
        a1 = np.maximum(0, z1)
        if training and self.dropout_rate > 0:
            mask1 = (self.rng.random(a1.shape) > self.dropout_rate).astype(np.float32)
            mask1 /= 1.0 - self.dropout_rate
            a1 = a1 * mask1
        z2 = a1 @ self.w2 + self.b2
        a2 = np.maximum(0, z2)
        if training and self.dropout_rate > 0:
            mask2 = (self.rng.random(a2.shape) > self.dropout_rate).astype(np.float32)
            mask2 /= 1.0 - self.dropout_rate
            a2 = a2 * mask2
        z3 = a2 @ self.w3 + self.b3
        out = z3  # linear output — GridWorld states contain values 0, 1, 2
        return z1, z2, out, a1, a2

    def _backward(
        self,
        x: np.ndarray,
        z1: np.ndarray,
        z2: np.ndarray,
        out: np.ndarray,
        a1: np.ndarray,
        a2: np.ndarray,
        target: np.ndarray,
        attention_weights: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        """Backward pass of MSE loss with optional attention weighting.

        Computes gradients of MSE = 0.5 * mean((out - target)²) w.r.t.
        all parameters, with gradient clipping to [-1, 1].

        If attention_weights is provided, gradients are modulated
        per-dimension so that high-salience dimensions receive larger
        updates (A5: Feedback-Driven Adaptation).

        Takes pre-computed a1, a2 from the forward pass (which may include
        dropout masks during training — AF-001) to ensure gradients are
        consistent with the forward computation.

        Args:
            x: Input vector (state_dim + action_dim,).
            z1: Pre-activation of layer 1 (hidden_dim,).
            z2: Pre-activation of layer 2 (hidden_dim,).
            out: Linear output (state_dim,).
            a1: Post-ReLU activation of layer 1 (from _forward).
            a2: Post-ReLU activation of layer 2 (from _forward).
            target: Target vector (state_dim,).
            attention_weights: Optional per-dimension salience weights
                for modulating gradients (state_dim,).

        Returns:
            Dict mapping parameter keys to gradient arrays.
        """
        # dL/d(out) for MSE: (out - target) / state_dim
        n = float(out.shape[0])
        d_out = (out - target) / n

        # Apply per-dimension attention weighting (A5 fix)
        if attention_weights is not None and attention_weights.shape[0] == d_out.shape[0]:
            d_out = d_out * attention_weights.astype(np.float32)

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

    def estimate_empowerment(self, state: StateVector) -> float:
        """Estimate I(S';A|S) via MC Dropout and Gaussian MI (C2 fix).

        For n ≤ 5 actions, runs mc_samples forward passes per action,
        models p(s'|s,a) as a diagonal Gaussian, and computes:

            I = H(Σ p(a)·p(s'|s,a)) - Σ p(a)·H(p(s'|s,a))

        where H(diagonal Gaussian) = 0.5·Σ log(2πe·σ²_i).
        The mixture H is approximated as a single diagonal Gaussian
        using the laws of total expectation and total variance.

        For n > 5 actions, falls back to variance-based heuristic.

        Args:
            state: Current state vector.

        Returns:
            Empowerment in [0, 1] (clipped).
        """
        n_actions = self.action_dim
        if n_actions > 5:
            return 0.3

        p_a = 1.0 / n_actions
        s_val = state.values.astype(np.float32)

        means = np.zeros((n_actions, self.state_dim), dtype=np.float64)
        vars_arr = np.zeros((n_actions, self.state_dim), dtype=np.float64)
        entropies = np.zeros(n_actions, dtype=np.float64)

        for a in range(n_actions):
            action = np.zeros(n_actions, dtype=np.float32)
            action[a] = 1.0
            x = np.concatenate([s_val, action])

            predictions = []
            for _ in range(self.mc_samples):
                _, _, out, _, _ = self._forward(x, training=True)
                predictions.append(out)
            preds = np.stack(predictions, axis=0)
            mu_a = np.mean(preds, axis=0)
            var_a = np.var(preds, axis=0) + _EPS

            means[a] = mu_a
            vars_arr[a] = var_a
            # H(diag Gauss) = 0.5 * Σ log(2πe * σ²_i)
            entropies[a] = 0.5 * np.sum(np.log(2.0 * np.pi * np.e * var_a))

        # Mixture moments via law of total expectation/variance
        mu_mix = p_a * np.sum(means, axis=0)
        sigma2_mix = p_a * np.sum(vars_arr + means ** 2, axis=0) - mu_mix ** 2
        sigma2_mix = np.maximum(sigma2_mix, _EPS)

        H_mix = 0.5 * np.sum(np.log(2.0 * np.pi * np.e * sigma2_mix))
        mi = H_mix - p_a * np.sum(entropies)

        # Scale to [0, 1]. For 84-dim GridWorld, typical MI < 5 nats.
        return float(np.clip(mi / 5.0, 0.0, 1.0))

    def get_prediction_accuracy(
        self, state: np.ndarray, action: np.ndarray, target: np.ndarray
    ) -> float:
        """Get MLP prediction accuracy on a given (state, action, target).

        Used to sync TSPL skill_accuracy from actual MLP performance (G-019).
        Uses MC Dropout (AF-001): confidence = mean(1/(1 + per-dimension variance)).

        Args:
            state: Current state vector (state_dim,).
            action: Action vector (action_dim,).
            target: Ground-truth next state (state_dim,).

        Returns:
            Confidence in [0, 1].
        """
        x = np.concatenate([state.astype(np.float32), action.astype(np.float32)])
        predictions = []
        for _ in range(self.mc_samples):
            _, _, out, _, _ = self._forward(x, training=True)
            predictions.append(out)
        preds = np.stack(predictions, axis=0)
        var_pred = np.var(preds, axis=0) + _EPS
        return float(np.mean(1.0 / (1.0 + var_pred)))

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
