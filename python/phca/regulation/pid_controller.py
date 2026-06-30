"""
PHCA v3.0 — Criticality Regulator (PID Controller).

Phase 3.2: Full PID controller regulating system criticality phi toward
a target setpoint. Computes temperature (T), exploration noise (eta),
and attention spread (alpha) based on PID error dynamics.

v3.0 Reference: §3.4 Definition 3.7, v3.0 Patch §2.6 (orthogonality constraint)
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


class CriticalityRegulator:
    """Criticality Regulator — PID controller with orthogonality constraint.

    Phase 3.2:
        - PID control: T = T0 + k_p·Δ + k_i·∫Δ + k_d·dΔ/dt
          where Δ = phi_setpoint - phi_current
        - Orthogonality constraint (v3.0 Patch §2.6.1):
          Monitors covariance between T, eta, alpha.
          Freezes slowest parameter if covariance exceeds threshold.
        - Output: (temperature, exploration_noise, attention_spread)

    Phase 3.3+:
        - Adaptive gains (k_p, k_i, k_d tuned online)
        - Meta-parameter optimization

    v3.0 Reference: §3.4 Definition 3.7, §2.6 Orthogonality constraint
    """

    def __init__(
        self,
        setpoint: float = 0.5,
        k_p: float = 1.0,
        k_i: float = 0.1,
        k_d: float = 0.05,
        T_base: float = 1.0,
        eta_base: float = 0.1,
        alpha_base: float = 0.5,
        orthogonality_threshold: float = 0.8,
        integral_limit: float = 10.0,
    ):
        """Initialize PID controller with default gains.

        Args:
            setpoint: Target criticality phi* (default 0.5, edge of chaos).
            k_p: Proportional gain.
            k_i: Integral gain.
            k_d: Derivative gain.
            T_base: Base temperature (output scaling).
            eta_base: Base exploration noise (output scaling).
            alpha_base: Base attention spread (output scaling).
            orthogonality_threshold: Max allowed covariance between parameters.
            integral_limit: Anti-windup limit for integral term.
        """
        self.setpoint = setpoint

        # PID gains
        self.k_p = k_p
        self.k_i = k_i
        self.k_d = k_d

        # Output baselines
        self.T_base = T_base
        self.eta_base = eta_base
        self.alpha_base = alpha_base

        # Orthogonality constraint
        self.orthogonality_threshold = orthogonality_threshold
        self.integral_limit = integral_limit

        # PID state
        self._integral: float = 0.0
        self._prev_error: float = 0.0
        self._prev_output: Tuple[float, float, float] = (T_base, eta_base, alpha_base)
        self._cycle: int = 0

        # Orthogonality tracking (covariance history)
        self._param_history: Dict[str, float] = {
            "T": T_base, "eta": eta_base, "alpha": alpha_base,
        }
        self._param_history_buffer: list[Dict[str, float]] = []
        self._frozen_params: set[str] = set()

    def regulate(self, phi_current: float = 0.5) -> Tuple[float, float, float]:
        """Regulate criticality toward setpoint using PID control.

        Computes:
            Δ = setpoint - phi_current
            ∫Δ = accumulated integral (with anti-windup)
            dΔ/dt = Δ - prev_Δ
            T = T_base + k_p·Δ + k_i·∫Δ + k_d·dΔ/dt
            eta = eta_base * (1 + 0.5 * Δ)
            alpha = alpha_base * (1 - 0.3 * Δ)

        Then applies orthogonality constraint check.

        Args:
            phi_current: Current estimated criticality (approximated as
                -H(module_states) for Phase 3.2).

        Returns:
            Tuple of (temperature, exploration_noise, attention_spread):
                temperature: PID-controlled temperature (T).
                exploration_noise: Eta for TSPL exploration.
                attention_spread: Alpha for attention precision.
        """
        self._cycle += 1

        # Compute PID error
        error = self.setpoint - phi_current

        # Update integral with anti-windup
        self._integral += error
        self._integral = np.clip(self._integral, -self.integral_limit, self.integral_limit)

        # Derivative
        derivative = error - self._prev_error
        self._prev_error = error

        # PID output
        pid_out = self.k_p * error + self.k_i * self._integral + self.k_d * derivative

        # Compute regulated parameters
        T = max(0.1, self.T_base + pid_out)
        eta = max(0.001, self.eta_base * (1.0 + 0.5 * error))
        alpha = np.clip(self.alpha_base * (1.0 - 0.3 * error), 0.01, 1.0)

        # Apply frozen parameters (don't update frozen ones)
        if "T" in self._frozen_params:
            T = self._prev_output[0]
        if "eta" in self._frozen_params:
            eta = self._prev_output[1]
        if "alpha" in self._frozen_params:
            alpha = self._prev_output[2]

        # Store output
        output = (T, eta, alpha)
        self._prev_output = output

        # Update orthogonality tracking
        self._update_orthogonality({"T": T, "eta": eta, "alpha": alpha})

        # Check orthogonality constraint
        self._check_orthogonality()

        return output

    # ── Orthogonality Constraint ─────────────────────────────

    def _update_orthogonality(self, params: Dict[str, float]) -> None:
        """Update the parameter history buffer for covariance tracking.

        Args:
            params: Current (T, eta, alpha) values.
        """
        self._param_history_buffer.append(dict(params))
        if len(self._param_history_buffer) > 100:
            self._param_history_buffer.pop(0)

    def _check_orthogonality(self) -> None:
        """Check orthogonality constraint and freeze parameters if needed.

        Computes pairwise covariance between T, eta, alpha over recent
        history. If any pair exceeds the threshold, freezes the slowest-changing
        parameter (lowest variance) to break the covariance.
        """
        if len(self._param_history_buffer) < 10:
            return

        # Extract arrays
        T_vals = np.array([p["T"] for p in self._param_history_buffer], dtype=np.float64)
        eta_vals = np.array([p["eta"] for p in self._param_history_buffer], dtype=np.float64)
        alpha_vals = np.array([p["alpha"] for p in self._param_history_buffer], dtype=np.float64)

        # Compute correlation matrix
        stack = np.column_stack([T_vals, eta_vals, alpha_vals])
        corr = np.corrcoef(stack.T)
        names = ["T", "eta", "alpha"]

        # Check each pair
        for i in range(3):
            for j in range(i + 1, 3):
                if abs(corr[i, j]) > self.orthogonality_threshold:
                    # Freeze the parameter with lower variance
                    variances = [float(np.var(stack[:, k])) for k in [i, j]]
                    freeze_idx = i if variances[0] < variances[1] else j
                    freeze_name = names[freeze_idx]

                    if freeze_name not in self._frozen_params:
                        self._frozen_params.add(freeze_name)
                        # Unfreeze the other param in the pair
                        other_name = names[j if freeze_idx == i else i]
                        self._frozen_params.discard(other_name)

    def unfreeze_all(self) -> None:
        """Unfreeze all parameters (for new training runs)."""
        self._frozen_params.clear()

    def get_frozen_params(self) -> set[str]:
        """Get the set of currently frozen parameter names.

        Returns:
            Set of parameter names that are frozen ("T", "eta", "alpha").
        """
        return set(self._frozen_params)

    # ── Reset ────────────────────────────────────────────────

    def reset(self) -> None:
        """Reset PID state and orthogonality tracking for a new training run."""
        self._integral = 0.0
        self._prev_error = 0.0
        self._prev_output = (self.T_base, self.eta_base, self.alpha_base)
        self._cycle = 0
        self._param_history_buffer.clear()
        self._frozen_params.clear()
        self._param_history = {
            "T": self.T_base, "eta": self.eta_base, "alpha": self.alpha_base,
        }
