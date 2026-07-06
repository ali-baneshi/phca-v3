"""Recovery protocols for detected cognitive failures."""

from __future__ import annotations

from typing import Any, Dict, List, TYPE_CHECKING

from phca.config import StreamID
from phca.resilience.types import FailureEvent, RecoveryAction, RecoveryResult

if TYPE_CHECKING:
    from phca.core.cycle import CognitiveCycle


class RecoveryManager:
    """Apply bounded recovery actions and track mitigation state."""

    def __init__(self, recovery_window: int = 10, stable_cycles: int = 3):
        self.recovery_window = recovery_window
        self.stable_cycles = stable_cycles
        self._active: Dict[str, int] = {}
        self._stable_count: Dict[str, int] = {}
        self._mitigated: Dict[str, bool] = {}

    def any_active(self) -> bool:
        return bool(self._active)

    def active_modes(self) -> List[str]:
        return list(self._active.keys())

    def apply(self, cycle: "CognitiveCycle", events: List[FailureEvent]) -> RecoveryResult:
        result = RecoveryResult()
        for event in events:
            if event.mode_id not in self._active:
                self._active[event.mode_id] = event.cycle_id
                self._stable_count[event.mode_id] = 0
                self._mitigated[event.mode_id] = False
            result.events_handled.append(event.mode_id)
            actions = self._apply_protocol(cycle, event)
            result.actions_applied.extend(actions)

        self._update_mitigation(cycle)
        result.mitigated = [m for m, ok in self._mitigated.items() if ok]
        result.still_active = [
            m for m in self._active if not self._mitigated.get(m, False)
        ]
        return result

    def _apply_protocol(
        self,
        cycle: "CognitiveCycle",
        event: FailureEvent,
    ) -> List[RecoveryAction]:
        actions: List[RecoveryAction] = []
        mode = event.mode_id

        if mode == "B1":
            cfg = cycle.tspl.configs[StreamID.P_STREAM]
            cfg.eta = min(0.5, cfg.eta * 2.0)
            actions.append(RecoveryAction.BOOST_ETA)

        elif mode == "B4":
            cycle.on_forgetting_detected()
            actions.append(RecoveryAction.REPLAY_BOOST)

        elif mode == "B5":
            cycle.mdim.temperature = min(3.0, cycle.mdim.temperature * 1.5)
            actions.append(RecoveryAction.BOOST_TEMPERATURE)

        elif mode == "C1":
            cycle.adaptive_controller.apply_c1_recovery()
            actions.append(RecoveryAction.PID_C1_RECOVERY)

        elif mode == "F5":
            cycle.consolidation.force_step(cycle.cycle_count)
            actions.append(RecoveryAction.FORCE_CONSOLIDATION)

        return actions

    def _update_mitigation(self, cycle: "CognitiveCycle") -> None:
        from phca.resilience.detector import FailureDetector

        detector = getattr(cycle, "_resilience_detector", None)
        if detector is None:
            detector = FailureDetector()
        snap = cycle._build_resilience_snapshot(cycle.metrics_history[-1] if cycle.metrics_history else None)

        expired = []
        for mode_id, start in list(self._active.items()):
            if cycle.cycle_count - start > self.recovery_window:
                expired.append(mode_id)
                continue
            if detector.is_recovered(mode_id, snap):
                self._stable_count[mode_id] = self._stable_count.get(mode_id, 0) + 1
            else:
                self._stable_count[mode_id] = 0
            if self._stable_count[mode_id] >= self.stable_cycles:
                self._mitigated[mode_id] = True
                expired.append(mode_id)

        for mode_id in expired:
            self._active.pop(mode_id, None)
            self._stable_count.pop(mode_id, None)

    def reset(self) -> None:
        self._active.clear()
        self._stable_count.clear()
        self._mitigated.clear()
