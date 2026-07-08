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

from phca.config import DEFAULT_MODULE_BOUNDS, PHI_MAX, ResourceBounds, StateVector


_EPS = 1e-8


def estimate_mlp_memory_bytes(
    state_dim: int,
    action_dim: int,
    hidden_dim: int = 128,
    replay_capacity: int = 500,
) -> int:
    """Estimate G' MLP footprint: weight matrices + experience replay buffer."""
    params = ((state_dim + action_dim) * hidden_dim + hidden_dim * hidden_dim
              + hidden_dim * state_dim + 2 * hidden_dim + state_dim) * 4
    replay = replay_capacity * ((state_dim + action_dim) + state_dim) * 4
    return max(10_000, int(params + replay))


REF_STATE_DIM = 84  # 5×5 grid reference for RBTA scaling


def estimate_mlp_gprime_time_bound(
    state_dim: int,
    base_time: float = 0.080,
    ref_dim: int = REF_STATE_DIM,
) -> float:
    """Scale G' time bound linearly with state_dim (MLP learn cost grows with sd)."""
    return base_time * max(1.0, state_dim / ref_dim)


def gprime_stress_bounds(cycle: Any, *, b_time: float = 0.080) -> ResourceBounds:
    """Stress/benchmark G' RBTA bounds aligned with ``estimate_mlp_memory_bytes``."""
    g = cycle.gprime
    g_mem = max(
        500_000,
        estimate_mlp_memory_bytes(
            cycle.state_dim, g.action_dim, g.hidden_dim, g.replay_capacity,
        ),
    )
    return ResourceBounds(B_time=b_time, B_mem=g_mem, B_energy=50.0)


def grid_scale(state_dim: int, ref_dim: int = REF_STATE_DIM) -> float:
    """Linear scale factor vs canonical 5×5 GridWorld state dimension."""
    return max(1.0, state_dim / ref_dim)


def estimate_gaussian_gprime_memory_bytes(state_dim: int) -> int:
    """Match Gaussian G' memory estimate in ``CognitiveCycle._collect_runtime_log``."""
    return max(10_000, state_dim * state_dim * 4 * 5)


def _is_mlp_gprime(gprime: Any) -> bool:
    return hasattr(gprime, "replay_capacity") and hasattr(gprime, "hidden_dim")


def estimated_grid_memory_bytes(state_dim: int, *, gprime: Any = None) -> Dict[str, int]:
    """Mirror ``CognitiveCycle._collect_runtime_log`` memory estimates per module."""
    if gprime is not None and _is_mlp_gprime(gprime):
        g_mem = estimate_mlp_memory_bytes(
            state_dim, gprime.action_dim, gprime.hidden_dim, gprime.replay_capacity,
        )
    else:
        g_mem = estimate_gaussian_gprime_memory_bytes(state_dim)
    return {
        "ASI": max(1_000, state_dim * 4 * 2),
        "WM": max(1_000, state_dim * 4 * 7),
        "G'": g_mem,
        "PE": max(1_000, state_dim * 4 * 3),
        "PEU": max(500, state_dim * 4),
        "TSPL-P": max(5_000, state_dim * 4 * 10),
    }


def scaled_time_bound(
    state_dim: int,
    base_time: float,
    *,
    headroom: float = 1.0,
    ref_dim: int = REF_STATE_DIM,
) -> float:
    """Scale a module time bound with state_dim and optional headroom."""
    return estimate_mlp_gprime_time_bound(state_dim, base_time, ref_dim=ref_dim) * headroom


def grid_rbta_bounds(
    cycle: Any = None,
    *,
    state_dim: Optional[int] = None,
    gprime: Any = None,
    b_time: float = 0.080,
    action_b_time: Optional[float] = None,
) -> Dict[str, ResourceBounds]:
    """Scale RBTA bounds with state_dim for large GridWorld runs."""
    sd = state_dim if state_dim is not None else cycle.state_dim
    gp = gprime if gprime is not None else cycle.gprime
    scale = grid_scale(sd)
    bounds: Dict[str, ResourceBounds] = {}
    mem_est = estimated_grid_memory_bytes(sd, gprime=gp) if scale > 1.0 else {}

    if _is_mlp_gprime(gp):
        g_mem = max(500_000, mem_est.get("G'", estimate_mlp_memory_bytes(
            sd, gp.action_dim, gp.hidden_dim, gp.replay_capacity,
        )))
        g_time = estimate_mlp_gprime_time_bound(sd, b_time)
        bounds["G'"] = ResourceBounds(B_time=g_time, B_mem=g_mem, B_energy=50.0)
    elif scale > 1.0:
        g_mem = mem_est["G'"]
        g_base = max(b_time, DEFAULT_MODULE_BOUNDS["G'"].B_time)
        g_time = scaled_time_bound(sd, g_base, headroom=2.0)
        bounds["G'"] = ResourceBounds(B_time=g_time, B_mem=g_mem, B_energy=50.0)

    if scale > 1.0:
        asi = DEFAULT_MODULE_BOUNDS["ASI"]
        action = DEFAULT_MODULE_BOUNDS["ACTION"]
        base_action_time = action_b_time if action_b_time is not None else action.B_time
        bounds["ASI"] = ResourceBounds(
            B_time=scaled_time_bound(sd, asi.B_time),
            B_mem=max(asi.B_mem, mem_est["ASI"]),
            B_energy=asi.B_energy,
            entropy_floor=asi.entropy_floor,
        )
        bounds["ACTION"] = ResourceBounds(
            B_time=scaled_time_bound(sd, base_action_time),
            B_mem=action.B_mem,
            B_energy=action.B_energy,
            entropy_floor=action.entropy_floor,
        )
        for mod in ("WM", "PE", "PEU", "TSPL-P"):
            default = DEFAULT_MODULE_BOUNDS[mod]
            bounds[mod] = ResourceBounds(
                B_time=scaled_time_bound(sd, default.B_time),
                B_mem=max(default.B_mem, mem_est[mod]),
                B_energy=default.B_energy,
                entropy_floor=default.entropy_floor,
            )
        for mod in ("MDIM", "CONSOL"):
            default = DEFAULT_MODULE_BOUNDS[mod]
            headroom = 3.0 if mod == "MDIM" else 1.0
            bounds[mod] = ResourceBounds(
                B_time=scaled_time_bound(sd, default.B_time, headroom=headroom),
                B_mem=default.B_mem,
                B_energy=default.B_energy * scale,
                entropy_floor=default.entropy_floor,
            )
        for mod in ("CR", "ATTN", "HPM"):
            default = DEFAULT_MODULE_BOUNDS[mod]
            bounds[mod] = ResourceBounds(
                B_time=scaled_time_bound(sd, default.B_time, headroom=2.0),
                B_mem=default.B_mem,
                B_energy=default.B_energy * scale,
                entropy_floor=default.entropy_floor,
            )
    return bounds


def apply_grid_rbta_bounds(
    cycle: Any,
    *,
    b_time: float = 0.080,
    action_b_time: Optional[float] = None,
) -> bool:
    """Apply grid-scaled RBTA bounds. Returns True if any bound was updated."""
    updates = grid_rbta_bounds(
        cycle=cycle, b_time=b_time, action_b_time=action_b_time,
    )
    for module_id, bounds in updates.items():
        cycle.rbta.update_bounds(module_id, bounds)
    return bool(updates)


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
        # Input gradient cache for Φ (causal sensitivity) — set by _backward()
        self._last_input_grad: Optional[np.ndarray] = None
        # Output sensitivity cache: gradient of mean(out) w.r.t input — set by _backward()
        self._last_output_sens: Optional[np.ndarray] = None

        # Experience replay buffer
        self._replay_buffer: List[Tuple[np.ndarray, np.ndarray]] = []
        self._replay_idx: int = 0

        # Anti-forgetting: larger replay batches + extra steps when active
        self.replay_boost: bool = False
        self._replay_boost_batch_mult: float = 1.5
        self._replay_boost_extra_steps: int = 4

        # SGD momentum (default 0.9)
        self._momentum: float = 0.9
        self._v: Dict[str, np.ndarray] = {
            "gprime_w1": np.zeros_like(self.w1),
            "gprime_b1": np.zeros_like(self.b1),
            "gprime_w2": np.zeros_like(self.w2),
            "gprime_b2": np.zeros_like(self.b2),
            "gprime_w3": np.zeros_like(self.w3),
            "gprime_b3": np.zeros_like(self.b3),
        }

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

        # Confidence (G-002 / D-080): purely epistemic via MC-Dropout variance.
        #   epistemic  = MC-Dropout variance across the *mc_samples*
        #                stochastic forward passes; high when the model is
        #                uncertain about its own parameters (e.g. on
        #                out-of-distribution inputs), low on well-learned
        #                regions. Approximated as mutual_info = log(1+var).
        #   Note: aleatoric confidence via exp(-MSE) is NOT used here because
        #         the true next state is unavailable at prediction time —
        #         comparing against the current state would reward predicting
        #         "no change" and penalize correct movement predictions.
        var = np.var(mc_outs, axis=0).mean()
        # mutual information ≈ log(1 + var) clipped to [0, ~1.1]
        mutual_info = float(np.log1p(min(var, 2.0)))
        # Observability v4: cache per-dim std + mutual info (uncertainty portrait).
        self._last_mc_per_dim_std = np.sqrt(np.var(mc_outs, axis=0)).astype(np.float32)
        self._last_mutual_info = mutual_info
        # confidence = 1.0 - 0.5 * mutual_info  (pure epistemic)
        confidence = float(np.clip(1.0 - 0.5 * mutual_info, 0.01, 1.0))

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
        batch_size = self.batch_size
        n_steps = self.train_steps
        if self.replay_boost:
            batch_size = min(
                len(self._replay_buffer),
                int(self.batch_size * self._replay_boost_batch_mult),
            )
            n_steps = self.train_steps + self._replay_boost_extra_steps
        for _ in range(n_steps):
            indices = self.rng.randint(0, len(self._replay_buffer), size=batch_size)
            X = np.stack([self._replay_buffer[i][0] for i in indices]).astype(np.float32)
            T = np.stack([self._replay_buffer[i][1] for i in indices]).astype(np.float32)
            Z1, Z2, Out = self._forward_batch(X)
            avg_grad = self._backward_batch(X, Z1, Z2, Out, T)
            self._apply_gradient(avg_grad, lr=effective_lr)

    def learn_m3_episodes(
        self, episodes: list, lr_scale: float = 1.0,
    ) -> Tuple[int, List[Tuple[int, float]]]:
        """Extra gradient steps from M3 episodic transitions (continual replay).

        Also injects transitions into the internal FIFO buffer so later
        mini-batch replay can resample them.

        Computes per-episode current prediction error for PER priority updates.

        Args:
            episodes: List of EpisodeRecord objects.
            lr_scale: Additional LR multiplier (default 1.0; consolidation uses 0.1).

        Returns:
            Tuple of (steps, priority_updates):
                steps: Number of gradient steps taken.
                priority_updates: List of ``(episode_id, current_error)`` tuples
                    for PER priority updates.
        """
        if not episodes:
            return (0, [])
        effective_lr = self.lr * 0.5 * lr_scale
        steps = 0
        priority_updates: List[Tuple[int, float]] = []
        for ep in episodes:
            state_before = getattr(ep, "state_before", None)
            state_after = getattr(ep, "state_after", None)
            action = getattr(ep, "action_taken", None)
            ep_id = getattr(ep, "episode_id", None)
            if state_before is None or state_after is None or action is None:
                continue
            x = np.concatenate([
                state_before.values.astype(np.float32),
                np.asarray(action, dtype=np.float32).reshape(-1),
            ])
            target = state_after.values.astype(np.float32)
            if len(self._replay_buffer) < self.replay_capacity:
                self._replay_buffer.append((x.copy(), target.copy()))
            else:
                self._replay_buffer[self._replay_idx % self.replay_capacity] = (
                    x.copy(),
                    target.copy(),
                )
            self._replay_idx += 1
            z1, z2, out = self._forward(x)
            current_error = float(np.mean((out - target) ** 2))
            if ep_id is not None:
                priority_updates.append((int(ep_id), current_error))
            grad = self._backward(x, z1, z2, out, target)
            self._apply_gradient(grad, lr=effective_lr)
            steps += 1
        return (steps, priority_updates)

    def last_input_sensitivity(self, state_dim: int) -> float:
        """Return Φ = ||d_mean(out)/dx_state|| / sqrt(state_dim).

        Unlike the loss gradient (which vanishes when prediction error → 0),
        this measures how the model's PREDICTION changes with respect to its
        input — the causal sensitivity of the learned world model.  High Φ
        means small input changes cause large prediction changes (critical
        regime).  Low Φ means the model is insensitive (converged / saturated).

        The gradient of mean(out) w.r.t input is computed during ``_backward()``
        with minimal extra cost (~one additional backprop pass per layer).

        Args:
            state_dim: Dimensionality of the state portion of the input.

        Returns:
            Φ in [0.0, PHI_MAX].  Falls back to ``PHI_TARGET`` if no backward
            pass has been run yet.
        """
        from phca.config import PHI_TARGET
        sens = self._last_output_sens if self._last_output_sens is not None else self._last_input_grad
        if sens is None:
            return PHI_TARGET
        grad_state = sens[:state_dim]
        phi = float(np.linalg.norm(grad_state)) / np.sqrt(float(state_dim))
        return float(np.clip(phi, 0.0, PHI_MAX))

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

        # Cache input gradient for Φ (causal sensitivity) — zero extra compute
        self._last_input_grad = (d_z1 @ self.w1.T).astype(np.float32)

        # Cache output sensitivity: gradient of mean(out) w.r.t input.
        # Unlike dL/dx (which vanishes when prediction error → 0), this
        # captures how the prediction itself changes with input, which is
        # meaningful even for a perfectly trained model.
        d_out_sens = np.full_like(self.b3, 1.0 / self.state_dim)
        d_a2_sens = d_out_sens @ self.w3.T
        d_z2_sens = d_a2_sens * (z2 > 0).astype(np.float32)
        d_a1_sens = d_z2_sens @ self.w2.T
        d_z1_sens = d_a1_sens * (z1 > 0).astype(np.float32)
        self._last_output_sens: np.ndarray = (d_z1_sens @ self.w1.T).astype(np.float32)

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
        """Apply gradient via SGD with momentum (default 0.9).

        Args:
            grad: Dict with keys matching get_theta().
            lr: Learning rate.
        """
        self._v["gprime_w1"] = self._momentum * self._v["gprime_w1"] + lr * grad["gprime_w1"]
        self.w1 -= self._v["gprime_w1"]
        self._v["gprime_b1"] = self._momentum * self._v["gprime_b1"] + lr * grad["gprime_b1"]
        self.b1 -= self._v["gprime_b1"]
        self._v["gprime_w2"] = self._momentum * self._v["gprime_w2"] + lr * grad["gprime_w2"]
        self.w2 -= self._v["gprime_w2"]
        self._v["gprime_b2"] = self._momentum * self._v["gprime_b2"] + lr * grad["gprime_b2"]
        self.b2 -= self._v["gprime_b2"]
        self._v["gprime_w3"] = self._momentum * self._v["gprime_w3"] + lr * grad["gprime_w3"]
        self.w3 -= self._v["gprime_w3"]
        self._v["gprime_b3"] = self._momentum * self._v["gprime_b3"] + lr * grad["gprime_b3"]
        self.b3 -= self._v["gprime_b3"]

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
