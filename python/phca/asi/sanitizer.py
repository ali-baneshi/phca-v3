"""
ASI Step 0 Sanitizer — v3.0 Patch B Implementation.

Implements the sensor vector sanitization logic from v3.0 §2.2 Patch B.
Sanitizes incoming sensor vectors before they enter the cognitive cycle.

For each sensor value:
  - If NaN, Inf, or |v| > V_max: replace with last valid value, halve precision.
  - If precision < ε_confidence: raise ASI_SENSOR_FAILURE.
  - If > d/3 sensors failed: global sensor failure recovery.

Cross-ref: v3.0 Patch B (§2.2), Theorem 3.2 (1-cycle propagation bound)
Playbook: §7.1 ASI Implementation Checklist
"""

from __future__ import annotations

import numpy as np
from phca.logging import logger, _log

from ..config import StateVector, ASIStatus, DEFAULT_MODULE_BOUNDS


class ASISanitizer:
    """
    Step 0 sensor vector sanitizer.

    Maintains per-sensor state: last valid value, current precision, failure count.
    Implements the exponential precision decay: p_j^{(t+k)} = p_j^{(t)} * 2^{-k}
    """

    def __init__(self, sensor_dim: int, v_max: float = 1e6, epsilon_confidence: float = 0.01):
        """
        Initialize the sanitizer.

        Args:
            sensor_dim: Dimensionality of the sensor vector (d).
            v_max: Maximum physically plausible sensor value (V_max).
            epsilon_confidence: Minimum confidence threshold (default 0.01).
        """
        self.sensor_dim = sensor_dim
        self.v_max = v_max
        self.epsilon_confidence = epsilon_confidence
        self.asi_failure_limit = sensor_dim // 3  # floor(d / 3)

        # Per-sensor state
        self.last_valid: np.ndarray = np.zeros(sensor_dim, dtype=np.float32)
        self.precision: np.ndarray = np.ones(sensor_dim, dtype=np.float32)
        self.failure_count: np.ndarray = np.zeros(sensor_dim, dtype=np.int32)

        _log(logger, "info", "asi.sanitizer.init",
             sensor_dim=sensor_dim, v_max=v_max, failure_limit=self.asi_failure_limit)

    def sanitize(self, raw: np.ndarray, timestamp: float = 0.0) -> tuple[StateVector, ASIStatus]:
        """
        Sanitize the incoming sensor vector.

        Implements v3.0 Patch B logic:
        1. Check each value for NaN/Inf/|v| > V_max
        2. Replace invalid values with last valid value
        3. Halve precision on consecutive failures
        4. Raise SENSOR_FAILURE if precision < ε_confidence

        Args:
            raw: Raw sensor vector of shape (d,).
            timestamp: Current cycle timestamp.

        Returns:
            (clean_state_vector, status)

        Raises:
            ValueError: If raw has NaN/Inf after sanitization (should never happen).
        """
        assert raw.shape == (self.sensor_dim,), \
            f"Expected shape ({self.sensor_dim},), got {raw.shape}"

        clean = np.copy(raw).astype(np.float32)
        failure_mask = np.zeros(self.sensor_dim, dtype=bool)

        for j in range(self.sensor_dim):
            if np.isnan(raw[j]) or np.isinf(raw[j]) or abs(raw[j]) > self.v_max:
                # Replace with last valid value
                clean[j] = self.last_valid[j]
                # Halve precision
                self.precision[j] *= 0.5
                self.failure_count[j] += 1
                failure_mask[j] = True

                _log(logger, "warning", "asi.sanitize.failure",
                     sensor=j, value=float(raw[j]),
                     precision=float(self.precision[j]),
                     failure_count=int(self.failure_count[j]),
                     timestamp=timestamp)
            else:
                # Valid value: update last valid, reset failure count
                self.last_valid[j] = raw[j]
                self.failure_count[j] = 0
                # Precision unchanged (updated by attention separately)

        # Check global failure limit (before per-sensor SENSOR_FAILURE, so log is reachable)
        total_failed = int(np.sum(self.failure_count > 0))
        if total_failed > self.asi_failure_limit:
            _log(logger, "critical", "asi.sanitizer.global_failure",
                 failed_sensors=total_failed,
                 limit=self.asi_failure_limit,
                 timestamp=timestamp)

        # Check for sensor failures
        if failure_mask.any():
            min_precision = self.precision[failure_mask].min()
            if min_precision < self.epsilon_confidence:
                _log(logger, "critical", "asi.sanitizer.sensor_failure",
                     min_precision=float(min_precision),
                     threshold=self.epsilon_confidence,
                     timestamp=timestamp)
                return StateVector(
                    values=clean,
                    precision=self.precision.copy(),
                    timestamp=timestamp,
                    grounding_level=1,
                ), ASIStatus.SENSOR_FAILURE

        return StateVector(
            values=clean,
            precision=self.precision.copy(),
            timestamp=timestamp,
            grounding_level=1,
        ), ASIStatus.OK

    def reset(self) -> None:
        """Reset all per-sensor state."""
        self.last_valid.fill(0.0)
        self.precision.fill(1.0)
        self.failure_count.fill(0)
        _log(logger, "info", "asi.sanitizer.reset")
