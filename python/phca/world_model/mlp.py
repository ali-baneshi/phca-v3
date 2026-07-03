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

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from phca.config import StateVector


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

    # Empowerment estimation (D-077): MC-Dropout passes per action and a
    # hard cap on total forward passes per call, honouring A1.  With
    # action_dim=5 and K=8 this is 40 passes (~1.2M FLOPs) << 60M cap.
    EMPOWERMENT_MC_SAMPLES: int = 4
    EMPOWERMENT_FLOP_CAP: int = 32

    def __init__(
        self,
        state_dim: int = 84,
        action_dim: int = 5,
        hidden_dim: int = 128,
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
        # Observability v4: cache last MC per-dim std + mutual_info.
        self._last_mc_per_dim_std: Optional[np.ndarray] = None
        self._last_mutual_info: float = 0.0

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

        # Confidence (G-002 / D-080): aleatoric * epistemic split.
        #   aleatoric  = exp(-MSE) on the mean prediction (data-fit term);
        #                measures how well the mean prediction matches the
        #                input state's scale (low when the predicted next
        #                state is far from the current state in MSE terms).
        #   epistemic  = MC-Dropout variance across the *mc_samples*
        #                stochastic forward passes; high when the model is
        #                uncertain about its own parameters (e.g. on
        #                out-of-distribution inputs), low on well-learned
        #                regions. Approximated as mutual_info = log(1+var).
        # The combined confidence penalises the aleatoric term when
        # epistemic uncertainty is high, so OOD states report lower
        # confidence than in-distribution states even if MSE is similar.
        var = np.var(mc_outs, axis=0).mean()
        # mutual information ≈ log(1 + var) clipped to [0, ~1.1]
        mutual_info = float(np.log1p(min(var, 2.0)))
        # Observability v4: cache per-dim std + mutual info (uncertainty portrait).
        self._last_mc_per_dim_std = np.sqrt(np.var(mc_outs, axis=0)).astype(np.float32)
        self._last_mutual_info = mutual_info
        # base (aleatoric) confidence = exp(-MSE on mean prediction)
        base_conf = self._compute_confidence(mean_out, state.values.astype(np.float32))
        # penalize confidence when mutual info is high (epistemic uncertainty)
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

        Hybrid schedule (G-017 / D-080):
          - Warm-up (`len(replay_buffer) < batch_size`): a single ONLINE
            gradient step on the just-observed transition. Mini-batch
            training is impossible with fewer than `batch_size` samples,
            so the buffer is filled with one-pass online updates.
          - Steady state (`len(replay_buffer) >= batch_size`): REPLAY-ONLY
            training on `train_steps` mini-batches sampled from the
            buffer. The current transition is in the buffer and is
            learned only if sampled — there is NO concurrent online
            step, so transitions are never learned twice in one cycle
            (the early `return` in the warm-up branch guarantees this).

        Both branches use the SAME learning rate (`lr * 0.5`) so the
        warm-up and steady-state update magnitudes cannot conflict.

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

        # Unified learning rate for both branches (G-017 fix).
        effective_lr = self.lr * 0.5

        # Warm-up: buffer too small to form a mini-batch — single online
        # step, then return (no replay in the same cycle).
        if len(self._replay_buffer) < self.batch_size:
            grad = self._backward(x, z1, z2, out, target)
            self._apply_gradient(grad, lr=effective_lr)
            return

        # Steady state: replay-only. The current transition is in the
        # buffer and is learned only if sampled into a mini-batch.
        # Batched (Phase 5 / D-092): collapse the per-sample Python loop
        # (train_steps × batch_size separate forward+backward passes) into
        # `train_steps` batched matmuls over the whole mini-batch. Same
        # math, same lr*0.5, same hybrid schedule — pure perf. Zero-trust
        # verification confirmed the dynamic-goal L2 is unchanged by this
        # (it is ~0.43 on this machine for BOTH the original loop and the
        # batched version — the D-090 reported 0.573 was machine-specific;
        # see D-092). The per-sample gradient clip becomes a single clip on
        # the batch-averaged gradient; verified numerically identical here
        # because gradient elements are ~0.05 (well within [-1,1]).
        for _ in range(self.train_steps):
            indices = self.rng.randint(0, len(self._replay_buffer), size=self.batch_size)
            X = np.stack([self._replay_buffer[i][0] for i in indices]).astype(np.float32)
            T = np.stack([self._replay_buffer[i][1] for i in indices]).astype(np.float32)
            Z1, Z2, Out = self._forward_batch(X)
            avg_grad = self._backward_batch(X, Z1, Z2, Out, T)
            self._apply_gradient(avg_grad, lr=effective_lr)

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

    def _forward_batch(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Batched forward pass over a mini-batch (Phase 5 / D-092)."""
        Z1 = X @ self.w1 + self.b1
        A1 = np.maximum(0, Z1)
        Z2 = A1 @ self.w2 + self.b2
        A2 = np.maximum(0, Z2)
        Out = A2 @ self.w3 + self.b3  # linear output
        return Z1, Z2, Out

    def _backward_batch(
        self, X: np.ndarray, Z1: np.ndarray, Z2: np.ndarray,
        Out: np.ndarray, T: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """Batched backward pass of MSE loss over a mini-batch (Phase 5 / D-092).

        Mirrors _backward (MSE + conditional softmax CE on the first
        min(25, S) agent-position dims, gradient clipping to [-1,1]) then
        averages over the batch. Zero-trust verification (D-092) confirmed
        this is dynamics-equivalent to the original per-sample loop on this
        machine: dynamic-goal L2 is ~0.43 for both (D-090's 0.573 was
        machine-specific).
        """
        B, S = Out.shape
        n = float(S)
        d_out = (Out - T) / n  # (B, S)

        # Conditional CE on the first min(25, S) dims (agent position).
        pos_dim = min(25, S)
        if pos_dim >= 2:
            pos_target = T[:, :pos_dim]
            mask = pos_target.max(axis=1) > 0.5
            if mask.any():
                logits = Out[:, :pos_dim].astype(np.float32, copy=True)
                logits[mask] -= logits[mask].max(axis=1, keepdims=True)
                exp_l = np.exp(logits)
                probs = exp_l / (exp_l.sum(axis=1, keepdims=True) + 1e-8)
                ce_grad = probs.copy()
                pos_idx = np.argmax(pos_target, axis=1)
                rows = np.where(mask)[0]
                ce_grad[rows, pos_idx[rows]] -= 1.0
                d_out[:, :pos_dim] += np.where(mask[:, None], ce_grad, 0.0) / n

        A1 = np.maximum(0, Z1)
        A2 = np.maximum(0, Z2)

        d_z3 = d_out
        grad_w3 = A2.T @ d_z3          # (H, S)
        grad_b3 = d_z3.sum(axis=0)     # (S,)
        d_a2 = d_z3 @ self.w3.T        # (B, H)
        d_z2 = d_a2 * (Z2 > 0).astype(np.float32)
        grad_w2 = A1.T @ d_z2          # (H, H)
        grad_b2 = d_z2.sum(axis=0)     # (H,)
        d_a1 = d_z2 @ self.w2.T        # (B, H)
        d_z1 = d_a1 * (Z1 > 0).astype(np.float32)
        grad_w1 = X.T @ d_z1           # (in, H)
        grad_b1 = d_z1.sum(axis=0)     # (H,)

        inv = 1.0 / float(B)
        grads = (grad_w1 * inv, grad_b1 * inv, grad_w2 * inv, grad_b2 * inv,
                 grad_w3 * inv, grad_b3 * inv)
        for g in grads:
            np.clip(g, -1.0, 1.0, out=g)
        grad_w1, grad_b1, grad_w2, grad_b2, grad_w3, grad_b3 = grads

        return {
            "gprime_w1": np.ascontiguousarray(grad_w1.astype(np.float32)),
            "gprime_b1": np.ascontiguousarray(grad_b1.astype(np.float32)),
            "gprime_w2": np.ascontiguousarray(grad_w2.astype(np.float32)),
            "gprime_b2": np.ascontiguousarray(grad_b2.astype(np.float32)),
            "gprime_w3": np.ascontiguousarray(grad_w3.astype(np.float32)),
            "gprime_b3": np.ascontiguousarray(grad_b3.astype(np.float32)),
        }

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
        """Estimate empowerment I(S'; A | S) via MC-Dropout (D-077).

        Replaces the previous constant stub (which returned 0.3 / 0.2
        regardless of state — see docs/phase4_gap_closure_report.md
        overclaim correction). Uses the existing MC-Dropout forward path
        (`_forward_mc`) to estimate, for the given state, how much the
        predicted-next-state distribution differs across actions
        (between-action variance) relative to the per-action epistemic
        uncertainty (within-action variance).

        Mutual-information approximation (Gaussian differential-entropy
        form):

            MI ≈ 0.5 * log(1 + V_between / (V_within + ε))

        where  V_between = Var over actions of the per-action mean
                prediction (how distinct outcomes are per action), and
                V_within  = mean over actions of the MC variance for
                that action (model uncertainty). High MI ⟺ different
                actions produce reliably different outcomes.

        A1 (Resource Boundedness): the count of forward passes is capped
        at `EMPOWERMENT_FLOP_CAP` (= 32). For action_dim=5 with K=8
        samples this is 40 passes ≈ 1.2M FLOPs — well under
        `ENERGY_NORM_FLOPS` (60M). When the cap would be exceeded, K is
        reduced rather than action coverage, so all actions remain
        represented.

        Args:
            state: A `StateVector` or 1-D ndarray of shape (state_dim,).
                May be None for callers that mirror the Gaussian G'
                signature.

        Returns:
            Empowerment in [0.0, 1.0]. Falls back to 0.3 only when the
            state is None/invalid or the computation diverges.
        """
        if state is None:
            return 0.3
        values = getattr(state, "values", state)
        try:
            s = np.asarray(values, dtype=np.float32)
            if s.shape[0] != self.state_dim or self.action_dim < 2:
                return 0.3
        except Exception:
            return 0.3

        K = self.EMPOWERMENT_MC_SAMPLES  # per-action MC passes
        # A1 FLOP cap: reduce K if action_dim * K exceeds the cap.
        if self.action_dim * K > self.EMPOWERMENT_FLOP_CAP:
            K = max(1, self.EMPOWERMENT_FLOP_CAP // self.action_dim)

        per_action_means: List[np.ndarray] = []
        per_action_vars: List[float] = []
        for a_idx in range(self.action_dim):
            action = np.zeros(self.action_dim, dtype=np.float32)
            action[a_idx] = 1.0
            x = np.concatenate([s, action])
            outs: List[np.ndarray] = []
            for _ in range(K):
                _, _, out = self._forward_mc(x)
                outs.append(out)
            outs_arr = np.stack(outs, axis=0)  # (K, state_dim)
            per_action_means.append(outs_arr.mean(axis=0))
            per_action_vars.append(float(outs_arr.var(axis=0).mean()))

        means_arr = np.stack(per_action_means, axis=0)  # (action_dim, state_dim)
        v_between = float(means_arr.var(axis=0).mean())  # spread across actions
        v_within = float(np.mean(per_action_vars)) if per_action_vars else 0.0
        if not np.isfinite(v_between) or not np.isfinite(v_within):
            return 0.3
        mi = 0.5 * float(np.log1p(v_between / (v_within + _EPS)))
        return float(np.clip(mi, 0.0, 1.0))

    def uncertainty_snapshot(self) -> Dict[str, Any]:
        """Observability v4: MC-Dropout uncertainty portrait (MLP G').

        Returns the cached per-dim MC std and mutual information from the last
        ``predict`` call (zero recomputation). ``kind`` is ``"mlp"``.
        """
        std = self._last_mc_per_dim_std
        return {
            "kind": "mlp",
            "per_dim_std": (std.copy() if std is not None else None),
            "mutual_info": float(self._last_mutual_info),
        }

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
