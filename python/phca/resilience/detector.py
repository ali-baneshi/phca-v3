"""Failure detection from per-cycle snapshots (B1, B4, B5, C1, F5, E1)."""

from __future__ import annotations

from typing import List

import numpy as np

from phca.resilience.types import CycleSnapshot, FailureCategory, FailureEvent

M3_CAP = 10_000


class FailureDetector:
    """Pragmatic MVP detector — budget < 1 ms per cycle."""

    def __init__(
        self,
        b1_multiplier: float = 3.0,
        b1_consecutive: int = 3,
        b4_ratio: float = 2.0,
        b5_confidence: float = 0.99,
        b5_min_unique_actions: int = 2,
        b5_window: int = 20,
        c1_consecutive: int = 3,
        f5_fill_ratio: float = 0.9,
        f5_stagnant_cycles: int = 50,
        e1_entropy_threshold: float = 0.85,
    ):
        self.b1_multiplier = b1_multiplier
        self.b1_consecutive = b1_consecutive
        self.b4_ratio = b4_ratio
        self.b5_confidence = b5_confidence
        self.b5_min_unique_actions = b5_min_unique_actions
        self.b5_window = b5_window
        self.c1_consecutive = c1_consecutive
        self.f5_fill_ratio = f5_fill_ratio
        self.f5_stagnant_cycles = f5_stagnant_cycles
        self.e1_entropy_threshold = e1_entropy_threshold
        self._b1_streak: int = 0

    def detect(self, snapshot: CycleSnapshot) -> List[FailureEvent]:
        events: List[FailureEvent] = []
        events.extend(self._detect_b1(snapshot))
        events.extend(self._detect_b4(snapshot))
        events.extend(self._detect_b5(snapshot))
        events.extend(self._detect_c1(snapshot))
        events.extend(self._detect_f5(snapshot))
        events.extend(self._detect_e1(snapshot))
        return events

    def _detect_b1(self, snap: CycleSnapshot) -> List[FailureEvent]:
        errors = snap.recent_prediction_errors
        if len(errors) < self.b1_consecutive:
            return []
        median = float(np.median(errors[:-self.b1_consecutive] or errors))
        if median < 1e-8:
            median = max(errors) * 0.5 or 1e-6
        threshold = self.b1_multiplier * median
        recent = errors[-self.b1_consecutive:]
        if all(e > threshold for e in recent):
            self._b1_streak = self.b1_consecutive
            return [FailureEvent(
                mode_id="B1",
                category=FailureCategory.DISTRIBUTION_SHIFT,
                severity=float(np.mean(recent) / threshold),
                cycle_id=snap.cycle_id,
                measured=float(np.mean(recent)),
                threshold=threshold,
                detail="prediction_error > 3× rolling median",
            )]
        self._b1_streak = 0
        return []

    def _detect_b4(self, snap: CycleSnapshot) -> List[FailureEvent]:
        base = snap.per_task_baseline_goal_rate
        cur = snap.per_task_goal_rate
        if base is None or cur is None or base < 1e-6:
            return []
        ratio = base / max(cur, 1e-8)
        if ratio > self.b4_ratio:
            return [FailureEvent(
                mode_id="B4",
                category=FailureCategory.CATASTROPHIC_FORGETTING,
                severity=ratio,
                cycle_id=snap.cycle_id,
                measured=cur,
                threshold=base / self.b4_ratio,
                detail="per-task goal rate drop > 2× baseline",
            )]
        return []

    def _detect_b5(self, snap: CycleSnapshot) -> List[FailureEvent]:
        # Parking on an extrinsic goal under geometry is not mode collapse —
        # action diversity is intentionally 1 (STAY). Suppress the false positive.
        extra = getattr(snap, "extra", None) or {}
        selector = str(extra.get("selector_mode") or "")
        if bool(extra.get("goal_reached")) and (
            "geometry" in selector
            or selector in (
                "pure_geometry_ablation",
                "adaptive_geometry_fallback",
                "task_lock_planner",
            )
        ):
            return []
        confs = snap.recent_confidences[-self.b5_window:]
        actions = snap.recent_actions[-self.b5_window:]
        if len(confs) < self.b5_window or len(actions) < self.b5_window:
            return []
        mean_conf = float(np.mean(confs))
        unique = len(set(actions))
        if mean_conf > self.b5_confidence and unique < self.b5_min_unique_actions:
            return [FailureEvent(
                mode_id="B5",
                category=FailureCategory.MODE_COLLAPSE,
                severity=mean_conf,
                cycle_id=snap.cycle_id,
                measured=float(unique),
                threshold=float(self.b5_min_unique_actions),
                detail="high confidence + low action diversity",
            )]
        return []

    def _detect_c1(self, snap: CycleSnapshot) -> List[FailureEvent]:
        hist = snap.rbta_action_history[-self.c1_consecutive:]
        if len(hist) < self.c1_consecutive:
            return []
        if all(a == "TERMINATE" for a in hist):
            return [FailureEvent(
                mode_id="C1",
                category=FailureCategory.FEEDBACK_INSTABILITY,
                severity=float(self.c1_consecutive),
                cycle_id=snap.cycle_id,
                measured=float(self.c1_consecutive),
                threshold=float(self.c1_consecutive),
                detail="RBTA TERMINATE for 3+ consecutive cycles",
            )]
        return []

    def _detect_e1(self, snap: CycleSnapshot) -> List[FailureEvent]:
        if snap.wm_entropy_proxy >= self.e1_entropy_threshold and snap.unique_actions_recent <= 1:
            return [FailureEvent(
                mode_id="E1",
                category=FailureCategory.EMERGENCY_ENTROPY,
                severity=snap.wm_entropy_proxy,
                cycle_id=snap.cycle_id,
                measured=snap.wm_entropy_proxy,
                threshold=self.e1_entropy_threshold,
                detail="belief entropy above threshold with low action diversity",
            )]
        return []

    def _detect_f5(self, snap: CycleSnapshot) -> List[FailureEvent]:
        if snap.m3_fill_ratio < self.f5_fill_ratio:
            return []
        if snap.fact_count_stagnant_cycles >= self.f5_stagnant_cycles:
            return [FailureEvent(
                mode_id="F5",
                category=FailureCategory.CONSOLIDATION_FAILURE,
                severity=snap.m3_fill_ratio,
                cycle_id=snap.cycle_id,
                measured=float(snap.fact_count_stagnant_cycles),
                threshold=float(self.f5_stagnant_cycles),
                detail="M3 near cap with stagnant M4 fact count",
            )]
        return []

    def is_recovered(
        self,
        mode_id: str,
        snapshot: CycleSnapshot,
    ) -> bool:
        """Check whether a failure mode's primary KPI is back within threshold."""
        if mode_id == "B1":
            errors = snapshot.recent_prediction_errors
            if len(errors) < 3:
                return False
            median = float(np.median(errors[:-3] or errors))
            threshold = self.b1_multiplier * max(median, 1e-8)
            return errors[-1] <= threshold
        if mode_id == "B4":
            base = snapshot.per_task_baseline_goal_rate
            cur = snapshot.per_task_goal_rate
            if base is None or cur is None:
                return True
            return cur >= base / self.b4_ratio
        if mode_id == "B5":
            actions = snapshot.recent_actions[-self.b5_window:]
            return len(set(actions)) >= self.b5_min_unique_actions
        if mode_id == "C1":
            hist = snapshot.rbta_action_history[-1:]
            return not hist or hist[-1] != "TERMINATE"
        if mode_id == "F5":
            return snapshot.m4_fact_count_delta > 0 or snapshot.m3_fill_ratio < self.f5_fill_ratio
        if mode_id == "E1":
            return snapshot.wm_entropy_proxy < self.e1_entropy_threshold
        return True
