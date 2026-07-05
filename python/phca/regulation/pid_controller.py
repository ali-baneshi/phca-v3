"""
PHCA v3.0 — Adaptive Parameter Controller (PID Controller).

Phase 3.2: Full PID controller regulating system error_volatility toward
a target setpoint. Computes temperature (T), exploration noise (eta),
and attention spread (alpha) based on PID error dynamics.

Note: This was previously called "CriticalityRegulator" but the signal it
regulates is prediction-error volatility, not self-organized criticality.
Renamed in Phase 3.3 gap audit (G-004) to avoid implying SOC.

v3.0 Reference: §3.4 Definition 3.7, v3.0 Patch §2.6 (orthogonality constraint)
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from phca.logging import logger, _log


class AdaptiveParameterController:
    """Adaptive Parameter Controller — PID controller with orthogonality constraint.

    Regulates a scalar signal (prediction-error volatility) toward a target
    setpoint using a PID loop, and outputs three parameters that control
    downstream module behaviour: temperature (T), exploration noise (eta),
    and attention spread (alpha).

    Mechanism:
        - PID control: T = T0 + k_p·Δ + k_i·∫Δ + k_d·dΔ/dt
          where Δ = setpoint - error_volatility
        - Orthogonality constraint (v3.0 Patch §2.6.1):
          Monitors covariance between T, eta, alpha.
          Freezes slowest parameter if covariance exceeds threshold.
        - Output: (temperature, exploration_noise, attention_spread)

    Note: This regulates prediction-error volatility, NOT self-organized
    criticality. The "edge of chaos" language from the v3.0 spec has been
    removed to avoid implying SOC where a simpler PID loop exists.

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
            setpoint: Target setpoint (default 0.5).
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
        self._high_cov_cycles: int = 0  # consecutive cycles with high covariance

        # Orthogonality tracking (covariance history)
        self._param_history: Dict[str, float] = {
            "T": T_base, "eta": eta_base, "alpha": alpha_base,
        }
        self._param_history_buffer: list[Dict[str, float]] = []
        self._frozen_params: set[str] = set()

        # Freeze priority by PID gain timescale
        # T (slowest, 1st to freeze) > alpha (medium, 2nd) > eta (fastest, last)
        self._freeze_priority: List[str] = ["T", "alpha", "eta"]

    def regulate(self, error_volatility: float = 0.5) -> Tuple[float, float, float]:
        """Regulate system toward setpoint using PID control.

        Computes:
            Δ = setpoint - error_volatility
            ∫Δ = accumulated integral (with anti-windup)
            dΔ/dt = Δ - prev_Δ
            T = T_base + k_p·Δ + k_i·∫Δ + k_d·dΔ/dt
            eta = eta_base * (1 + 0.5 * Δ)
            alpha = alpha_base * (1 - 0.3 * Δ)

        Then applies orthogonality constraint check.

        Args:
            error_volatility: Current prediction-error volatility signal
                (formerly called phi_criticality). Ranges ~0.1-0.99.

        Returns:
            Tuple of (temperature, exploration_noise, attention_spread):
                temperature: PID-controlled temperature (T).
                exploration_noise: Eta for TSPL exploration.
                attention_spread: Alpha for attention precision.
        """
        self._cycle += 1

        # G6: NaN gate — clamp error_volatility to previous value if degenerate
        if not np.isfinite(error_volatility):
            _log(logger, "warning", "cr.nan_input",
                 error_volatility=error_volatility, fallback="previous")
            error_volatility = self._prev_output[0] if self._cycle > 1 else self.setpoint

        # Compute PID error
        error = self.setpoint - error_volatility

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
        if len(self._param_history_buffer) >= 100:
            self._param_history_buffer.pop(0)

    def _check_orthogonality(self) -> None:
        """Check orthogonality constraint and freeze parameters if needed.

        Implements v3.0 §2.6.1 Def 2.8a (Orthogonality Constraint):

        1. Compute pairwise covariance between T, eta, alpha over sliding window W=100
        2. If max(Σ) > Σ_max:
            a. Identify the pair (i,j) with highest covariance
            b. Freeze the parameter with the slower timescale (by PID gain priority)
            c. Let the other parameters continue to adjust
            d. If covariance remains high for T_freeze > 100 cycles, unfreeze and freeze next
        """
        W = len(self._param_history_buffer)
        if W < 10:
            return

        # Extract arrays
        T_vals = np.array([p["T"] for p in self._param_history_buffer], dtype=np.float64)
        eta_vals = np.array([p["eta"] for p in self._param_history_buffer], dtype=np.float64)
        alpha_vals = np.array([p["alpha"] for p in self._param_history_buffer], dtype=np.float64)

        # Compute correlation matrix over sliding window (v3.0 §2.6.1)
        # Use np.corrcoef for scale-invariant redundancy detection (C5 fix)
        cols = (T_vals, eta_vals, alpha_vals)
        if any(float(np.std(col, ddof=1)) < 1e-12 for col in cols):
            return

        stack = np.column_stack(cols)
        corr = np.corrcoef(stack.T)
        names = ["T", "eta", "alpha"]

        # Find max absolute correlation
        max_corr = 0.0
        max_pair = (0, 1)
        for i in range(3):
            for j in range(i + 1, 3):
                abs_corr = abs(corr[i, j])
                if abs_corr > max_corr:
                    max_corr = abs_corr
                    max_pair = (i, j)

        if max_corr <= self.orthogonality_threshold:
            self._high_cov_cycles = 0
            return

        self._high_cov_cycles += 1
        i, j = max_pair
        pair_names = (names[i], names[j])

        # Freeze the parameter with the slower timescale (higher freeze priority)
        # Priority list: T (1st) > alpha (2nd) > eta (3rd)
        freeze_name = pair_names[0]
        other_name = pair_names[1]
        for p in self._freeze_priority:
            if p in pair_names:
                freeze_name = p
                other_name = pair_names[0] if pair_names[1] == p else pair_names[1]
                break

        # If high correlation persists for T_freeze > 100 cycles, switch which is frozen.
        # Once swapped, the freeze stays (no reset) to avoid oscillation (AF-004 fix).
        if self._high_cov_cycles > 100:
            freeze_name, other_name = other_name, freeze_name
            _log(logger, "info", "cr.orthogonality.swap",
                 new_frozen=freeze_name, unfrozen=other_name)

        # Apply freeze
        if freeze_name not in self._frozen_params:
            self._frozen_params.add(freeze_name)
            self._frozen_params.discard(other_name)
            _log(logger, "info", "cr.orthogonality.freeze",
                 frozen=freeze_name, unfrozen=other_name, correlation=float(max_corr))


