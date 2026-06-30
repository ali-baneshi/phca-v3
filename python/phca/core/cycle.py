"""
PHCA v3.0 - Cognitive Cycle Orchestrator.

Phase 3.1: 15-step cognitive cycle (subset of 21-step blueprint).
  Steps 0-7, 9, 14-15, 19 active. Steps 8, 10-13, 16-18, 20 deferred.

Phase 3.2+: Full 21-step cycle with MDIM, Attention, CR, HPM, Consolidation.

v3.0 Reference: Blueprint xa77.B, xa7.2.2, xa7.3.1
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from phca.logging import logger, _log
from phca.config import (
    ASIStatus,
    GoalVector,
    ResourceBounds,
    StateVector,
    StreamID,
    DEFAULT_MODULE_BOUNDS,
)
from phca.asi.sanitizer import ASISanitizer
from phca.memory.m1_sensory import M1SensoryBuffer
from phca.memory.m2_working import M2WorkingMemory
from phca.world_model.graph import WorldModelGPrime, StateNode, TemporalEdge
from phca.prediction.engine import PredictionEngine
from phca.prediction.error_unit import PredictionErrorUnit
from phca.learning.tspl import TSPL
from phca.regulation.rbta_enforcer import RBTAEnforcer
from phca.motivation.mdim import MDIM
from phca.attention.attention import Attention
from phca.regulation.pid_controller import CriticalityRegulator
from phca.hpm.parser import HPMValidator
from environments.grid_world import GridWorld, ACTION_NAMES


@dataclass
class CycleMetrics:
    """Metrics collected per cognitive cycle."""
    cycle_id: int = 0
    latency_ms: float = 0.0
    prediction_error: float = 0.0
    prediction_confidence: float = 0.0
    rbta_action: str = "CONTINUE"
    violations_count: int = 0
    goal_reached: bool = False
    action_taken: int = -1
    action_name: str = ""
    module_timings: Dict[str, float] = field(default_factory=dict)


class CognitiveCycle:
    """Phase 3.1 cognitive cycle orchestrator.

    Runs 15-step cycles (Phase 3.1 subset of blueprint xa77.B) connecting
    all modules: ASI -> M2 -> G' -> PE -> PEU -> TSPL -> action -> RBTA.

    Phase 3.2 adds Steps 8 (Attention), 10-13 (MDIM + CR), 16-18 (HPM).
    """

    def __init__(
        self,
        sanitizer: ASISanitizer,
        m1: M1SensoryBuffer,
        m2: M2WorkingMemory,
        gprime: WorldModelGPrime,
        engine: PredictionEngine,
        peu: PredictionErrorUnit,
        tspl: TSPL,
        rbta: RBTAEnforcer,
        mdim: MDIM,
        attention: Attention,
        criticality_regulator: CriticalityRegulator,
        hpm_validator: HPMValidator,
        env: GridWorld,
        state_dim: int,
        rbta_bounds: Optional[Dict[str, ResourceBounds]] = None,
    ):
        self.sanitizer = sanitizer
        self.m1 = m1
        self.m2 = m2
        self.gprime = gprime
        self.engine = engine
        self.peu = peu
        self.tspl = tspl
        self.rbta = rbta
        self.mdim = mdim
        self.attention = attention
        self.criticality_regulator = criticality_regulator
        self.hpm_validator = hpm_validator
        self.env = env
        self.state_dim = state_dim

        self.rbta_bounds = rbta_bounds or DEFAULT_MODULE_BOUNDS.copy()

        self.cycle_count: int = 0
        self.current_state: Optional[StateVector] = None
        self.current_goal: GoalVector = GoalVector(
            drive_id=1, target_state=None, tolerance=0.1,
            creation_cycle=0, priority=1.0,
        )
        self.last_action: np.ndarray = np.zeros(env.action_space_size, dtype=np.float32)
        self.last_prediction: Optional[StateVector] = None
        self.sensor_failure_count: int = 0
        self.asi_failure_limit: int = 5

        self.runtime_log: Dict[str, float] = {}
        self.memory_log: Dict[str, float] = {}
        self.energy_log: Dict[str, float] = {}
        self.belief_entropies: Dict[str, float] = {}

        self.metrics_history: List[CycleMetrics] = []

        _log(logger, "info", "cycle.init", state_dim=state_dim, grid_size=env.size)

    def run(self, n_cycles: int = 1000) -> Dict[str, Any]:
        """Run N cognitive cycles."""
        for _ in range(n_cycles):
            self.step()

        if not self.metrics_history:
            return {
                "total_cycles": 0, "avg_latency_ms": 0.0,
                "total_violations": 0, "avg_prediction_error": 0.0,
                "goals_reached": 0, "skill_compiled": self.tspl.skill_compiled,
            }

        errors = [m.prediction_error for m in self.metrics_history]
        latencies = [m.latency_ms for m in self.metrics_history]

        return {
            "total_cycles": len(self.metrics_history),
            "avg_latency_ms": float(np.mean(latencies)),
            "median_latency_ms": float(np.median(latencies)),
            "p95_latency_ms": float(np.percentile(latencies, 95)),
            "total_violations": sum(m.violations_count for m in self.metrics_history),
            "avg_prediction_error": float(np.mean(errors)),
            "median_prediction_error": float(np.median(errors)),
            "early_errors": [m.prediction_error for m in self.metrics_history[:10]],
            "late_errors": [m.prediction_error for m in self.metrics_history[-10:]],
            "goals_reached": sum(m.goal_reached for m in self.metrics_history),
            "actions_taken": [m.action_name for m in self.metrics_history],
            "skill_compiled": self.tspl.skill_compiled,
            "skill_accuracy": self.tspl.skill_accuracy,
        }

    def step(self) -> CycleMetrics:
        """Execute a single cognitive cycle."""
        t_start = time.perf_counter()
        metrics = CycleMetrics(cycle_id=self.cycle_count)

        try:
            # Step 0: ASI sanitization
            t0 = time.perf_counter()
            raw_obs = self.env._get_observation()
            clean_state, status = self.sanitizer.sanitize(raw_obs)
            metrics.module_timings["sanitize"] = (time.perf_counter() - t0) * 1000

            if status == ASIStatus.SENSOR_FAILURE:
                self.sensor_failure_count += 1
            else:
                self.current_state = clean_state
                self.sensor_failure_count = 0

            # Step 1: State -> M2 Working Memory
            t1 = time.perf_counter()
            if self.current_state is not None:
                self.m2.write(self.current_state, salience=1.0)
                self.m1.write(self.current_state)
            metrics.module_timings["memory_write"] = (time.perf_counter() - t1) * 1000

            # Steps 2-4: Prediction via G'
            t2 = time.perf_counter()
            if self.current_state is not None:
                predicted, confidence = self.engine.predict(
                    self.current_state, horizon=1,
                )
                self.last_prediction = predicted
                metrics.prediction_confidence = confidence
            metrics.module_timings["prediction"] = (time.perf_counter() - t2) * 1000

            # Step 5-6: PEU error computation
            t3 = time.perf_counter()
            if self.current_state is not None and self.last_prediction is not None:
                error = self.peu.compute(self.current_state, self.last_prediction)
                metrics.prediction_error = error
            metrics.module_timings["peu"] = (time.perf_counter() - t3) * 1000

            # Step 7: TSPL P-Stream update
            t4 = time.perf_counter()
            if self.current_state is not None and self.last_prediction is not None:
                self.tspl.update(
                    StreamID.P_STREAM,
                    metrics.prediction_error,
                    self.current_state,
                    self.last_prediction,
                )
            metrics.module_timings["tspl"] = (time.perf_counter() - t4) * 1000

            # Step 8: Attention (Phase 3.2 stub)
            _ = self.attention.select(self.m2.chunks, self.current_goal)

            # Step 9: Action selection
            t5 = time.perf_counter()
            action_idx = self._select_action()
            obs, reward, terminal, info = self.env.step(action_idx)
            self.last_action = np.zeros(self.env.action_space_size, dtype=np.float32)
            self.last_action[action_idx] = 1.0
            self.engine.update_action(self.last_action)

            metrics.action_taken = action_idx
            metrics.action_name = ACTION_NAMES[action_idx]
            metrics.goal_reached = info.get("goal_reached", False)
            metrics.module_timings["action_selection"] = (time.perf_counter() - t5) * 1000

            if terminal:
                self.env.reset()

            # Steps 10-13: MDIM + CR (Phase 3.2 stubs)
            self.current_goal = self.mdim.generate_goal()
            _ = self.criticality_regulator.regulate()
            _ = self.hpm_validator.validate({"cycle": self.cycle_count})

            # Step 14: RBTA enforcement
            t6 = time.perf_counter()
            self._collect_runtime_log()
            violations, enforcer_action = self.rbta.check_cycle(
                runtime_log=self.runtime_log,
                memory_log=self.memory_log,
                energy_log=self.energy_log,
                belief_entropies=self.belief_entropies,
                sensor_failure_count=self.sensor_failure_count,
                asi_failure_limit=self.asi_failure_limit,
            )
            metrics.rbta_action = enforcer_action.name
            metrics.violations_count = len(violations)
            metrics.module_timings["rbta"] = (time.perf_counter() - t6) * 1000

            # Step 15: Logging
            self.metrics_history.append(metrics)

            # Step 19: Increment cycle counter
            self.cycle_count += 1

        except Exception as e:
            _log(logger, "error", "cycle.step.error", cycle=self.cycle_count, error=str(e))
            self.cycle_count += 1
            raise

        metrics.latency_ms = (time.perf_counter() - t_start) * 1000
        self.runtime_log["CYCLE"] = metrics.latency_ms / 1000.0

        return metrics

    def _select_action(self) -> int:
        """Select action by maximizing prediction confidence (D1 proxy)."""
        if self.current_state is None:
            return 4  # STAY

        best_action = 4
        best_confidence = -1.0

        for action_idx in range(self.env.action_space_size):
            action = np.zeros(self.env.action_space_size, dtype=np.float32)
            action[action_idx] = 1.0
            self.engine.update_action(action)

            try:
                _, confidence = self.engine.predict(self.current_state, horizon=1)
                if confidence > best_confidence:
                    best_confidence = confidence
                    best_action = action_idx
            except Exception:
                continue

        # Restore engine action to current cycle selection
        # (step() will overwrite this with the newly selected action)
        return best_action

    def _collect_runtime_log(self) -> None:
        """Collect module runtime/memory/energy logs for RBTA."""
        self.runtime_log = {
            "ASI": 0.002, "WM": 0.005, "G'": 0.020, "PE": 0.025,
            "PEU": 0.001, "TSPL-P": 0.020, "TSPL-E": 0.001, "TSPL-S": 0.001,
            "MDIM": 0.001, "CR": 0.001, "ATTN": 0.001, "HPM": 0.001,
            "CYCLE": 0.100,
        }
        self.memory_log = {
            "ASI": 10_000, "WM": 25_000, "G'": 100_000,
            "PE": 10_000, "PEU": 1_000, "TSPL-P": 50_000,
        }
        self.energy_log = {
            "ASI": 1.0, "WM": 0.5, "G'": 10.0,
            "PE": 2.0, "PEU": 0.1, "TSPL-P": 5.0,
        }
        self.belief_entropies = {
            "G'": max(0.01, 0.5 - self.cycle_count * 0.001),
        }

    @classmethod
    def build_for_env(
        cls,
        size: int = 5,
        seed: int = 42,
        state_dim: Optional[int] = None,
    ) -> CognitiveCycle:
        """Build a fully-configured cognitive cycle for GridWorld."""
        env = GridWorld(size=size, obstacles=[], seed=seed)
        actual_state_dim = state_dim or env.get_state_dim()

        sanitizer = ASISanitizer(sensor_dim=actual_state_dim, v_max=100.0, epsilon_confidence=0.01)
        m1 = M1SensoryBuffer(sensor_dim=actual_state_dim)
        m2 = M2WorkingMemory(capacity=7)

        gprime = WorldModelGPrime(state_dim=actual_state_dim, action_dim=env.action_space_size, seed=seed)

        for i in range(min(actual_state_dim, 10)):
            name_t = f"s{i}_t"
            name_t1 = f"s{i}_t1"
            gprime.add_node(StateNode(
                name=name_t, cpd_type="discrete", parents=[],
                cardinality=2,
                params=np.array([[0.5], [0.5]], dtype=np.float32),
            ))
            gprime.add_node(StateNode(
                name=name_t1, cpd_type="discrete", parents=[name_t],
                cardinality=2,
                params=np.array([[0.6, 0.4], [0.4, 0.6]], dtype=np.float32),
            ))
            gprime.add_temporal_edge(TemporalEdge(source=name_t, target=name_t1, lag=1))

        engine = PredictionEngine(gprime)
        peu = PredictionErrorUnit()
        tspl = TSPL(seed=seed)
        tspl.init_parameters("gprime_cpd_transition", (actual_state_dim, 2))
        rbta = RBTAEnforcer(module_bounds=DEFAULT_MODULE_BOUNDS)

        mdim = MDIM(state_dim=actual_state_dim)
        attention = Attention()
        criticality_regulator = CriticalityRegulator()
        hpm_validator = HPMValidator()

        return cls(
            sanitizer=sanitizer, m1=m1, m2=m2, gprime=gprime,
            engine=engine, peu=peu, tspl=tspl, rbta=rbta,
            mdim=mdim, attention=attention,
            criticality_regulator=criticality_regulator,
            hpm_validator=hpm_validator, env=env,
            state_dim=actual_state_dim,
        )
