"""
PHCA v3.0 - Cognitive Cycle Orchestrator.

Phase 3.3: 12-step cognitive cycle (finalized subset of 21-step blueprint).
  Active steps: 0 (ASI), 1 (Memory), 2-4 (Prediction), 8-13 (MDIM/CR/ATTN before action),
  9 (Action), 5-7 (PEU/TSPL/learn post-step), 14 (RBTA), 15 (Logging),
  16-18 (Consolidation), 19 (Increment).

Phase 4: Dynamic composition tree, full Empowerment (D6), M5 procedural memory.

v3.0 Reference: Blueprint xa77.B, xa7.2.2, xa7.3.1
"""

from __future__ import annotations

import copy
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import numpy as np

from phca.logging import logger, _log
from phca.config import (
    ASIStatus,
    ActionResult,
    GoalVector,
    PerceptionFrame,
    ResourceBounds,
    StateVector,
    StreamID,
    STALE_THRESHOLD_MS,
    DEFAULT_MODULE_BOUNDS,
    DiscreteSpace,
    ContinuousSpace,
    PER_ALPHA,
    PER_BETA_INIT,
    PER_BETA_FINAL,
    PER_BETA_ANNEAL_STEPS,
)
from phca.asi.sanitizer import ASISanitizer
from phca.resilience import FailureDetector, RecoveryManager
from phca.resilience.types import CycleSnapshot
from phca.resilience.fallback_controller import FallbackController
from phca.memory.m1_sensory import M1SensoryBuffer
from phca.memory.m2_working import M2WorkingMemory
from phca.memory.m3_episodic import M3EpisodicMemory
from phca.world_model.mlp import (
    WorldModelMLP,
    estimate_mlp_memory_bytes,
    grid_rbta_bounds,
    grid_scale,
    scaled_time_bound,
)

if TYPE_CHECKING:
    from phca.world_model.graph import WorldModelGPrime

from phca.prediction.engine import PredictionEngine
from phca.prediction.error_unit import PredictionErrorUnit
from phca.learning.tspl import TSPL
from phca.regulation.rbta_enforcer import EnforcerAction, RBTAEnforcer
from phca.motivation.mdim import MDIM
from phca.attention.attention import Attention
from phca.regulation.pid_controller import AdaptiveParameterController
from phca.hpm.parser import HPMValidator
from phca.consolidation.scheduler import ConsolidationScheduler
from phca.environments.grid_world import GridWorld
from phca.environments.protocol import EnvironmentProtocol
from phca.asi.noise_injector import NoiseInjector
from phca.evaluation.interventions import DESYNC_STAGE_ORDER, InterventionConfig
from phca.evaluation.trace import TraceCollector

if TYPE_CHECKING:
    from phca.monitoring.metrics_store import MetricsStore
    from phca.monitoring.observability import ObservabilityStore

# AF-005: Normalisation factor for FLOP-based energy signals.
# ~30M FLOPs (MLP forward at h=128, bs=32, ts=4) ≈ 0.5 on energy scale.
ENERGY_NORM_FLOPS = 60_000_000.0

# Task-lock uses geometry-primary greedy only when G' confidence is high (P0-2 aligned).
TASK_LOCK_CONFIDENCE_THRESHOLD = 0.6
STEADY_STATE_GPRIME_ERROR_THRESHOLD = 1.5
STEADY_STATE_GPRIME_SKIP_MOD = 2


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
    # Dashboard-specific fields (populated in step() for live monitoring)
    drive_id: int = 1
    skill_accuracy: float = 0.0
    skill_compiled: bool = False
    fact_count: int = 0
    episode_count: int = 0
    task_id: int = -1
    failure_events: List[str] = field(default_factory=list)
    recovery_active: bool = False
    emergency_active: bool = False
    emergency_entropy: float = 0.0


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
        gprime: "WorldModelGPrime | WorldModelMLP",
        engine: PredictionEngine,
        peu: PredictionErrorUnit,
        tspl: TSPL,
        rbta: RBTAEnforcer,
        mdim: MDIM,
        attention: Attention,
        adaptive_controller: AdaptiveParameterController,
        hpm_validator: HPMValidator,
        consolidation: ConsolidationScheduler,
        env: EnvironmentProtocol,
        state_dim: int,
        rbta_bounds: Optional[Dict[str, ResourceBounds]] = None,
        metrics_store: Optional["MetricsStore"] = None,
        observability_store: Optional["ObservabilityStore"] = None,
        interventions: Optional[InterventionConfig] = None,
        trace_collector: Optional[TraceCollector] = None,
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
        self.adaptive_controller = adaptive_controller
        self.hpm_validator = hpm_validator
        self.consolidation = consolidation
        self.env = env
        self.state_dim = state_dim

        self.rbta_bounds = rbta_bounds or DEFAULT_MODULE_BOUNDS.copy()

        self.cycle_count: int = 0
        self.metrics_store = metrics_store
        self.observability_store = observability_store
        self.interventions = interventions or InterventionConfig()
        self.trace_collector = trace_collector

        self.current_state: Optional[StateVector] = None
        self.current_goal: GoalVector = GoalVector(
            drive_id=1, target_state=None, tolerance=0.1,
            creation_cycle=0, priority=1.0,
        )
        self.last_action: np.ndarray = np.zeros(env.action_space_size, dtype=np.float32)
        self.last_prediction: Optional[StateVector] = None
        # Observability v2: expose signals discarded on the hot path so the
        # dashboard can show RBTA reasons + action-selection rationale.
        self.last_violations: list = []
        self.last_action_rationale: dict = {}
        # Observability v3: per-action / per-candidate scores for the
        # Action-Selection tab of the Qt dashboard.
        self.last_candidate_scores: list = []
        # Observability v4: cognitive-portrait captures (only populated when
        # observability_store is attached — zero-overhead when off).
        self.last_candidate_rollouts: list = []
        self.last_per_dim_peu: Optional[np.ndarray] = None
        self.last_empowerment: float = 0.0
        self.sensor_failure_count: int = 0
        self.asi_failure_limit: int = self.sanitizer.asi_failure_limit

        # Phase 6 / A2: action-space branch. Resolve once; GridWorld and any
        # env without get_action_space() fall back to DiscreteSpace(n).
        space = getattr(env, "get_action_space", None)
        self.action_space = (
            space() if callable(space) else DiscreteSpace(n=env.action_space_size)
        )
        self._is_continuous: bool = isinstance(self.action_space, ContinuousSpace)

        self.runtime_log: Dict[str, float] = {}
        self.memory_log: Dict[str, float] = {}
        self.energy_log: Dict[str, float] = {}
        self.belief_entropies: Dict[str, float] = {}

        self.metrics_history: List[CycleMetrics] = []

        # Cached Φ (gradient-norm criticality) from the most recent backward pass
        self._cached_phi: float = 1.0

        # C3 fix: unified FLOP-based energy signal for D5 and RBTA
        self._cycle_flops: float = 0.0

        # Attention weights for modulating G'.learn() (Issue #4 fix)
        self._attention_weights: np.ndarray = np.ones(self.state_dim, dtype=np.float32)

        # P0: pre-action planning state (MDIM before action, facts → wall map)
        self._last_prediction_error: float = 0.0
        self._last_goal_pos: Optional[tuple] = None
        self._goal_switch_cooldown: int = 0
        self._relevant_facts: list = []
        self._planning_grid: Optional[np.ndarray] = None
        self._task_lock: bool = False

        # Level-4-lite continual learning / anti-forgetting hooks
        self._current_task_id: int = 0
        self._forgetting_mitigation_active: bool = False
        self._replay_boost_duration: int = 200
        self._replay_boost_activated_cycle: int = 0
        self._task_goal_baselines: Dict[int, float] = {}
        self._task_eval_history: Dict[int, List[float]] = {}
        self._last_fact_count: int = 0
        self._fact_stagnant_cycles: int = 0
        self._last_m3_replay_steps: int = 0
        self._m3_replay_total: int = 0
        self._m3_replay_budget: int = 4
        self._env_goal_relocated: bool = False
        self._per_beta: float = PER_BETA_INIT

        # Cognitive resilience (distinct from Observatory session recovery)
        self._resilience_detector = FailureDetector()
        self._resilience_recovery = RecoveryManager()

        # Emergency fallback controller (high-entropy instinctive behavior)
        self._fallback_controller = FallbackController()

        # Noise injector for grounding simulation (None = disabled)
        self._noise_injector: Optional[NoiseInjector] = None

        # RBTA enforcement carry-forward (P1-01): prior cycle action shapes next cycle.
        self._rbta_carry_action: EnforcerAction = EnforcerAction.CONTINUE
        self._rbta_skip_feedback: bool = False
        self._rbta_skip_consolidation: bool = False
        self._rbta_action_candidate_limit: Optional[int] = None
        self._last_step_reward: float = 0.0

        # Async cycle (Feature 1) — off by default
        self._async_mode: bool = False
        self._perception_queue: queue.Queue[PerceptionFrame] = queue.Queue(maxsize=1)
        self._action_result_queue: queue.Queue[ActionResult] = queue.Queue(maxsize=1)
        self._action_thread: threading.Thread | None = None
        self._learning_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._async_reward: float = 0.0
        self._async_terminal: bool = False

        _log(logger, "info", "cycle.init", state_dim=state_dim, grid_size=getattr(env, "size", 0))

    def run(self, n_cycles: int = 1000) -> Dict[str, Any]:
        """Run N cognitive cycles.

        In async mode, starts the action and learning threads and runs
        until ``n_cycles`` learning cycles have completed, then stops.
        In sync mode (default), calls ``step()`` N times sequentially.
        """
        if self._async_mode:
            self.start_async()
            try:
                while len(self.metrics_history) < n_cycles:
                    if self._stop_event.wait(timeout=0.05):
                        break
            finally:
                self.stop_async()
        else:
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
            # Per-cycle RBTA enforcement flags (reset; carry-forward applied below).
            self._rbta_skip_feedback = False
            self._rbta_skip_consolidation = False
            self._rbta_action_candidate_limit = None
            if self._rbta_carry_action == EnforcerAction.TERMINATE:
                self._rbta_skip_feedback = True
            elif self._rbta_carry_action == EnforcerAction.INTERRUPT:
                self._rbta_action_candidate_limit = 1

            # Phase A: Perception + Prediction + Regulation
            hpm_spec = self._run_perception_cycle(metrics)

            # Phase B: Action selection + environment step
            t_action = time.perf_counter()
            if self._rbta_skip_feedback:
                action = self._neutral_action()
                self.last_candidate_scores = []
                self.last_candidate_rollouts = []
                self.last_action_rationale = self._finalize_action_rationale({
                    "explored": False,
                    "eps": 0.0,
                    "goal_id": int(self.current_goal.drive_id) if self.current_goal else 1,
                    "continuous": bool(self._is_continuous),
                    "best_score": None,
                    "k_candidates": 1,
                    "chosen_idx": (
                        None if self._is_continuous else int(action)
                    ),
                    "task_lock": bool(self._task_lock),
                    "rbta_safe_mode": True,
                    "selector_mode": (
                        "continuous_safe_mode" if self._is_continuous else "discrete_safe_mode"
                    ),
                    "relevant_fact_ids": [f.fact_id for f in self._relevant_facts],
                }, decision_reason="rbta_safe")
            else:
                action = self._select_action()
            metrics.module_timings["action_selection"] = (
                time.perf_counter() - t_action
            ) * 1000

            step_reward = 0.0
            t_env = time.perf_counter()
            if self._is_continuous:
                action_vec_step = np.asarray(action, dtype=np.float32)
                obs, step_reward, terminal, info = self.env.step(action_vec_step)
            else:
                obs, step_reward, terminal, info = self.env.step(action)
            self._last_step_reward = float(step_reward)
            metrics.module_timings["env_step"] = (time.perf_counter() - t_env) * 1000

            # Phase C: Learning (PEU + TSPL + G'.learn + M3 + action tracking)
            self._run_learning_phase(metrics, action, obs, info,
                                     action_vec_step if self._is_continuous else None)

            if terminal:
                self.env.reset()
                self.gprime.reset()
                self.sanitizer.reset()  # clear stale precision across episode boundaries (N6)
                self._last_goal_pos = None
                self._goal_switch_cooldown = 0

            # Step 14: RBTA enforcement
            t6 = time.perf_counter()
            # Compute total cycle latency BEFORE check_cycle so composition tree's CYCLE
            # leaf reads the correct total time (was being set AFTER check_cycle — P0-B fix)
            metrics.latency_ms = (time.perf_counter() - t_start) * 1000
            self._collect_runtime_log(metrics)
            self.runtime_log["CYCLE"] = metrics.latency_ms / 1000.0
            # Compute composite bounds from HPM validator using actual module timings
            # Replaces hardcoded 200ms/500ms with structure-aware computed bounds
            hpm_bounds = self.hpm_validator.compute_bounds(hpm_spec, self.runtime_log)
            reg_b_time, reg_b_energy = self._scaled_hpm_composite_bounds(hpm_bounds)
            # Build composition tree reflecting the 21-step cycle's structure
            # with full three-dimensional bounds (B_time, B_energy) per A1 fix.
            composition_tree = {
                "type": "SEQUENCE", "id": "cognitive_cycle",
                "children": [
                    "ASI",
                    "WM",
                    "PE",
                    {
                        "type": "PARALLEL", "id": "regulation_block",
                        "children": ["MDIM", "CR", "ATTN", "HPM"],
                        "bounds": {"B_time": reg_b_time, "B_energy": reg_b_energy},
                    },
                    "ACTION",
                    "PEU",
                    "TSPL-P",
                    "CYCLE",
                ],
                "bounds": {
                    "B_time": reg_b_time * 2 + 0.050,  # 2× regulator + safety margin
                    "B_energy": reg_b_energy * 2 + 0.010,  # 2× regulator + energy overhead
                },
            }
            violations, enforcer_action = self.rbta.check_cycle(
                runtime_log=self.runtime_log,
                memory_log=self.memory_log,
                energy_log=self.energy_log,
                belief_entropies=self.belief_entropies,
                sensor_failure_count=self.sensor_failure_count,
                asi_failure_limit=self.asi_failure_limit,
                composition_tree=composition_tree,
            )
            metrics.rbta_action = enforcer_action.name
            metrics.violations_count = len(violations)
            self.last_violations = violations  # Observability v2: RBTA reasons
            metrics.module_timings["rbta"] = (time.perf_counter() - t6) * 1000
            self._rbta_carry_action = enforcer_action
            if enforcer_action in (EnforcerAction.INTERRUPT, EnforcerAction.TERMINATE):
                self._rbta_skip_consolidation = True

            # Cognitive resilience: detect failures and apply recovery protocols
            snapshot = self._build_resilience_snapshot(metrics)
            events = self._resilience_detector.detect(snapshot)
            if events:
                self._resilience_recovery.apply(self, events)
                for e in events:
                    self._fallback_controller.notify_failure(
                        e.mode_id, e.severity, self.cycle_count,
                    )
            metrics.failure_events = [e.mode_id for e in events]
            metrics.recovery_active = self._resilience_recovery.any_active()
            metrics.emergency_active = self._fallback_controller.active()
            metrics.emergency_entropy = self._fallback_controller.smoothed_entropy

            # Step 15: Logging — also populate dashboard fields
            consol_stats = self.consolidation.get_stats()
            metrics.drive_id = self.current_goal.drive_id if self.current_goal else 1
            metrics.skill_accuracy = self.tspl.skill_accuracy
            metrics.skill_compiled = self.tspl.skill_compiled
            metrics.fact_count = consol_stats.get("total_facts_stored", 0)
            metrics.episode_count = self.consolidation.m3.count() if hasattr(self, 'consolidation') else 0
            metrics.task_id = self._current_task_id
            if metrics.fact_count == self._last_fact_count:
                self._fact_stagnant_cycles += 1
            else:
                self._fact_stagnant_cycles = 0
            self._last_fact_count = metrics.fact_count
            self.metrics_history.append(metrics)
            # Deep-copy before pushing to prevent dashboard thread from
            # observing a mutating object (G-009: thread-safe monitoring)
            if self.metrics_store is not None:
                self.metrics_store.push(copy.deepcopy(metrics))
            # Visual Observability Layer (Phase 7 ext): opt-in, zero-overhead
            # when None. Frame build + ring push only; rendering/recording run
            # off the hot path in the visualiser thread.
            if self.observability_store is not None:
                from phca.monitoring.observability import ObservabilityFrame
                self.observability_store.push(ObservabilityFrame.from_cycle(self))
            if self.trace_collector is not None:
                self.trace_collector.record(
                    cycle_id=self.cycle_count,
                    action=metrics.action_taken,
                    reward=self._last_step_reward,
                    prediction_error=metrics.prediction_error,
                    prediction_confidence=metrics.prediction_confidence,
                    goal_drive=metrics.drive_id,
                    rbta_action=metrics.rbta_action,
                    violations=metrics.violations_count,
                    latency_ms=metrics.latency_ms,
                    goal_reached=metrics.goal_reached,
                    module_timings=dict(metrics.module_timings),
                    attention_weights=(
                        self._attention_weights.tolist()
                        if self.interventions.enable_attention else None
                    ),
                    env_goal_relocated=self._env_goal_relocated,
                )
                self._env_goal_relocated = False
            if len(self.metrics_history) > 5000:
                self.metrics_history = self.metrics_history[-5000:]

            # Steps 16-18: Consolidation (periodic E→S transfer)
            if self._rbta_skip_consolidation or not self.interventions.enable_consolidation:
                consol_report = type("_Skip", (), {
                    "success": False,
                    "episodes_processed": 0,
                    "facts_generated": 0,
                    "duration_ms": 0.0,
                })()
                metrics.module_timings["consolidation"] = 0.0
            else:
                t_consol = time.perf_counter()
                consol_report = self.consolidation.step(
                    self.cycle_count,
                    gprime=self.gprime if self.interventions.enable_gprime_learn else None,
                )
                metrics.module_timings["consolidation"] = (
                    time.perf_counter() - t_consol
                ) * 1000
            if consol_report.success and consol_report.episodes_processed > 0:
                # Log consolidated fact types for transparency
                recent_facts = self.consolidation.get_semantic_facts(
                    min_confidence=0.3, max_results=10,
                )
                fact_types = {}
                for f in recent_facts:
                    fact_types[f.fact_type] = fact_types.get(f.fact_type, 0) + 1
                _log(logger, "info", "cycle.consolidation",
                     episodes=consol_report.episodes_processed,
                     facts=consol_report.facts_generated,
                     fact_types=fact_types,
                     total_facts=self.consolidation.get_stats()["total_facts_stored"],
                     ms=f"{consol_report.duration_ms:.1f}")
            self.runtime_log["CONSOL"] = metrics.module_timings.get("consolidation", 0.0) / 1000.0
            consol_energy = max(0.1, min(10.0, self.runtime_log["CONSOL"] * 50.0))
            self.energy_log["CONSOL"] = consol_energy

            # Step 19: Increment cycle counter
            self.cycle_count += 1

            # Anneal PER beta from init toward final (unbiased IS weights)
            anneal_progress = min(1.0, self.cycle_count / PER_BETA_ANNEAL_STEPS)
            self._per_beta = PER_BETA_INIT + (PER_BETA_FINAL - PER_BETA_INIT) * anneal_progress

            # Step 20: (removed) Sleep-cycle check — A-008 fix. Consolidation
            # handled by Steps 16-18 every consolidation_interval=10 cycles.

        except Exception as e:
            _log(logger, "error", "cycle.step.error", cycle=self.cycle_count, error=str(e))
            self.cycle_count += 1
            raise

        return metrics

    def _run_perception_cycle(self, metrics: CycleMetrics) -> dict:
        """Steps 0-1 (ASI → M1/M2) + Steps 2-4 (prediction) + Steps 8-13 (regulation).

        Returns the ``hpm_spec`` dict for downstream RBTA enforcement.
        """
        # Step 0: ASI sanitization (with optional noise injection for grounding sim)
        t0 = time.perf_counter()
        raw_obs = self.env.get_observation()
        if self._noise_injector is not None:
            raw_obs = self._noise_injector.inject(raw_obs, self.cycle_count)
        clean_state, status = self.sanitizer.sanitize(raw_obs)
        metrics.module_timings["sanitize"] = (time.perf_counter() - t0) * 1000

        if status == ASIStatus.SENSOR_FAILURE:
            self.sensor_failure_count += 1
        else:
            self.current_state = clean_state
            self.sensor_failure_count = 0

        # Step 1: State → M2 Working Memory
        t1 = time.perf_counter()
        if self.current_state is not None and self.interventions.enable_m2:
            self.m2.write(self.current_state, salience=1.0)
            self.m1.write(self.current_state)
        metrics.module_timings["memory_write"] = (time.perf_counter() - t1) * 1000

        desync = (
            self.interventions.stage_order is not None
            and list(self.interventions.stage_order) == DESYNC_STAGE_ORDER
        )

        # ── Steps 2-4: Prediction phase ────────────────────────
        def _run_prediction_phase() -> None:
            t2 = time.perf_counter()
            if self.current_state is not None and self.interventions.enable_prediction:
                predicted, confidence = self.engine.predict(
                    self.current_state, horizon=1,
                )
                if not np.all(np.isfinite(predicted.values)):
                    _log(logger, "warning", "cycle.prediction.nan_detected",
                         fallback="identity")
                    predicted = StateVector(
                        values=self.current_state.values.copy(),
                        precision=np.ones_like(self.current_state.precision) * 0.01,
                        timestamp=predicted.timestamp,
                    )
                    confidence = 0.0
                if self.interventions.prediction_mode == "zero":
                    predicted = StateVector(
                        values=np.zeros_like(self.current_state.values),
                        precision=np.ones_like(self.current_state.precision) * 0.01,
                        timestamp=predicted.timestamp,
                    )
                    confidence = 0.0
                elif self.interventions.prediction_mode == "scramble":
                    perm = np.random.permutation(len(predicted.values))
                    predicted = StateVector(
                        values=predicted.values[perm],
                        precision=predicted.precision,
                        timestamp=predicted.timestamp,
                    )
                    confidence *= 0.5
                self.last_prediction = predicted
                metrics.prediction_confidence = confidence
            elif self.current_state is not None:
                self.last_prediction = StateVector(
                    values=self.current_state.values.copy(),
                    precision=self.current_state.precision.copy(),
                    timestamp=float(self.cycle_count),
                )
                metrics.prediction_confidence = 0.0
            metrics.module_timings["prediction"] = (time.perf_counter() - t2) * 1000

        # ── Steps 8-13: Regulation phase ───────────────────────
        hpm_spec: Dict[str, Any] = {}

        def _run_regulation_phase() -> None:
            nonlocal hpm_spec
            if self.interventions.minimal_cycle:
                self.current_goal = GoalVector(
                    drive_id=1, target_state=None, tolerance=0.1,
                    creation_cycle=self.cycle_count, priority=1.0,
                )
                self._attention_weights = np.ones(self.state_dim, dtype=np.float32)
                self._relevant_facts = []
                self._planning_grid = None
                metrics.module_timings["mdim"] = 0.0
                metrics.module_timings["cr"] = 0.0
                metrics.module_timings["attn"] = 0.0
                hpm_spec = {"type": "SEQUENCE", "id": "minimal_cycle", "children": ["ACTION"]}
                metrics.module_timings["hpm"] = 0.0
                return

            consol_stats = self.consolidation.get_stats()
            t_mdim = time.perf_counter()
            error_volatility = self._compute_phi_criticality()
            empowerment = self._estimate_empowerment()
            if self.observability_store is not None:
                self.last_empowerment = float(empowerment)

            if self.current_state is not None and self.interventions.enable_consolidation:
                self._relevant_facts = self.consolidation.get_relevant_facts(
                    self.current_state, n=5, min_confidence=0.3,
                )
            else:
                self._relevant_facts = []
            fact_confidence_mean = float(
                np.mean([f.confidence for f in self._relevant_facts])
            ) if self._relevant_facts else 0.0
            fact_count = len(self._relevant_facts)

            if (
                self.interventions.enable_consolidation
                and not self.interventions.disable_planning_grid
            ):
                self._planning_grid = self._build_planning_wall_grid()
            else:
                self._planning_grid = None

            env_goal_pos = self.env.get_goal_position()
            goal_switch_boost = False
            if env_goal_pos is not None:
                if (
                    self._last_goal_pos is not None
                    and tuple(env_goal_pos) != tuple(self._last_goal_pos)
                ):
                    goal_switch_boost = True
                    self._goal_switch_cooldown = 15
                self._last_goal_pos = tuple(env_goal_pos)
            if self._goal_switch_cooldown > 0:
                goal_switch_boost = True
                self._goal_switch_cooldown -= 1

            self._task_lock = env_goal_pos is not None
            if self.interventions.disable_task_lock:
                self._task_lock = False
            self._cycle_flops = self._compute_cycle_flops()
            if self._cycle_flops > 0:
                energy_cost = max(0.01, min(1.0, self._cycle_flops / ENERGY_NORM_FLOPS))
            else:
                energy_cost = max(0.01, min(1.0, (time.perf_counter() - t_mdim) * 2.0))
            model_entropy = self._epistemic_entropy()

            if self.interventions.enable_mdim:
                mdim_context = {
                    "prediction_error": self._last_prediction_error,
                    "error_volatility": error_volatility,
                    "skill_accuracy": self.tspl.skill_accuracy,
                    "model_entropy": model_entropy,
                    "energy_cost": energy_cost,
                    "cycle": self.cycle_count,
                    "prediction_confidence": metrics.prediction_confidence,
                    "empowerment": empowerment,
                    "consolidation_facts": consol_stats.get("total_facts_stored", 0),
                    "fact_confidence_mean": fact_confidence_mean,
                    "fact_count": fact_count,
                    "env_goal_pos": env_goal_pos,
                    "size": getattr(self.env, "size", 0),
                    "task_lock": self._task_lock,
                    "goal_switch_boost": goal_switch_boost,
                }
                self.current_goal = self.mdim.generate_goal(mdim_context)
            else:
                self.current_goal = GoalVector(
                    drive_id=1, target_state=None, tolerance=0.1,
                    creation_cycle=self.cycle_count, priority=1.0,
                )
            metrics.module_timings["mdim"] = (time.perf_counter() - t_mdim) * 1000

            t_cr = time.perf_counter()
            if self.interventions.enable_apc:
                T, eta, alpha = self.adaptive_controller.regulate(error_volatility)
                self.mdim.temperature = T
                if not (self._resilience_recovery.any_active() and "B1" in self._resilience_recovery.active_modes()):
                    self.tspl.configs[StreamID.P_STREAM].eta = eta
                self.attention.gumbel_temperature = alpha * 0.5
            metrics.module_timings["cr"] = (time.perf_counter() - t_cr) * 1000

            t_attn = time.perf_counter()
            if self.interventions.enable_attention:
                attention_chunks = self.attention.select(
                    self.m2.chunks, self.current_goal,
                    prediction=self.last_prediction,
                )
                for chunk in attention_chunks:
                    self.attention.update_precision(
                        chunk.chunk_id, self._last_prediction_error,
                    )
                if attention_chunks:
                    weights = np.array([c.salience for c in attention_chunks])
                    w_sum = weights.sum()
                    if w_sum > 1e-8:
                        weights = weights / w_sum
                    else:
                        weights = np.ones_like(weights) / max(len(weights), 1)
                    if len(weights) >= self.state_dim:
                        self._attention_weights = weights[:self.state_dim]
                    else:
                        reps = int(np.ceil(self.state_dim / max(len(weights), 1)))
                        self._attention_weights = np.tile(weights, reps)[:self.state_dim]
                else:
                    self._attention_weights = np.ones(self.state_dim, dtype=np.float32)
            else:
                self._attention_weights = np.ones(self.state_dim, dtype=np.float32)
            metrics.module_timings["attn"] = (time.perf_counter() - t_attn) * 1000

            t_hpm = time.perf_counter()
            hpm_spec = {
                "type": "SEQUENCE", "id": "cognitive_cycle",
                "children": [
                    {"type": "ASI_Input", "id": "ASI", "dim": self.state_dim},
                    {"type": "SEQUENCE", "id": "prediction_block",
                     "children": [
                        "WM",
                        {"type": "Predict", "id": "G'_PE", "horizon": 1},
                    ]},
                    {"type": "PARALLEL", "id": "regulation_block",
                     "children": ["MDIM", "CR", "ATTN", "HPM"]},
                    "ACTION",
                    {"type": "SEQUENCE", "id": "feedback_block",
                     "children": ["PEU", "TSPL-P"]},
                    "CYCLE",
                ],
            }
            metrics.module_timings["hpm"] = (time.perf_counter() - t_hpm) * 1000

        if desync:
            _run_regulation_phase()
            _run_prediction_phase()
        else:
            _run_prediction_phase()
            _run_regulation_phase()

        # RBTA preflight: same-cycle TERMINATE/INTERRUPT before expensive action/feedback.
        preflight_action = self._rbta_preflight_check(metrics, hpm_spec)
        if preflight_action == EnforcerAction.TERMINATE:
            self._rbta_skip_feedback = True
            self._rbta_skip_consolidation = True
        elif preflight_action == EnforcerAction.INTERRUPT:
            self._rbta_action_candidate_limit = 1
            self._rbta_skip_consolidation = True

        return hpm_spec

    def _run_learning_phase(
        self,
        metrics: CycleMetrics,
        action: int | np.ndarray,
        obs: np.ndarray,
        info: dict,
        action_vec_step: np.ndarray | None,
    ) -> None:
        """Steps 5-7: PEU + TSPL + G'.learn + M3 episode storage + action tracking.

        Called after ``env.step()`` returns.  ``action_vec_step`` is the
        raw action vector for continuous spaces (``None`` for discrete).
        """
        # ── Action tracking (runs every cycle, even when feedback is skipped) ──
        if self._is_continuous and action_vec_step is not None:
            self.last_action = action_vec_step.copy()
            self.engine.update_action(self.last_action)
            metrics.action_taken = -1
            metrics.action_name = "continuous"
        else:
            self.last_action = np.zeros(self.env.action_space_size, dtype=np.float32)
            self.last_action[action] = 1.0
            self.engine.update_action(self.last_action)
            metrics.action_taken = int(action) if isinstance(action, (int, np.integer)) else -1
            action_names = self.env.get_action_names()
            metrics.action_name = action_names[int(action)] if 0 <= int(action) < len(action_names) else f"#{int(action)}"
        metrics.goal_reached = info.get("goal_reached", False)

        if self.current_state is None or self._rbta_skip_feedback:
            return

        # Build action vector for learning
        if self._is_continuous:
            action_vec = action_vec_step
        else:
            action_vec = np.zeros(self.env.action_space_size, dtype=np.float32)
            action_vec[action] = 1.0

        next_state = StateVector(
            values=obs.astype(np.float32),
            precision=np.ones(self.state_dim, dtype=np.float32),
            timestamp=float(self.cycle_count),
        )

        # Step 5-6: PEU — actual next_state vs prediction conditioned on action_vec
        corrected_prediction, corrected_conf = self.gprime.predict(
            self.current_state, action_vec
        )
        t3 = time.perf_counter()
        error = self.peu.compute_precision_weighted(
            next_state, corrected_prediction, next_state.precision
        )
        metrics.prediction_error = error
        if self.observability_store is not None:
            try:
                diff = (next_state.values - corrected_prediction.values)
                self.last_per_dim_peu = (
                    next_state.precision * (diff ** 2)
                ).astype(np.float32)
            except Exception:
                self.last_per_dim_peu = None
        if corrected_conf > metrics.prediction_confidence:
            metrics.prediction_confidence = corrected_conf
        metrics.module_timings["peu"] = (time.perf_counter() - t3) * 1000

        # Step 7: TSPL P-Stream update
        mlp_accuracy = None
        if isinstance(self.gprime, WorldModelMLP):
            mlp_accuracy = self.gprime.get_prediction_accuracy(
                self.current_state.values, action_vec, next_state.values
            )
        t4 = time.perf_counter()
        if self.interventions.enable_tspl:
            tspl_gradient = None
            theta_new, _ = self.tspl.update(
                StreamID.P_STREAM,
                metrics.prediction_error,
                self.current_state,
                self.last_prediction,
                gradient=tspl_gradient,
                accuracy_override=mlp_accuracy,
            )
            if isinstance(self.gprime, WorldModelMLP):
                bias = theta_new.get("gprime")
                if bias is not None:
                    self.gprime.set_tspl_bias(bias)
        metrics.module_timings["tspl"] = (time.perf_counter() - t4) * 1000

        # LEARN: update G' with observed transition, weighted by attention
        t_glearn = time.perf_counter()
        if self.interventions.enable_gprime_learn:
            if isinstance(self.gprime, WorldModelMLP):
                boost_still_valid = (
                    self.cycle_count - self._replay_boost_activated_cycle
                ) < self._replay_boost_duration
                self.gprime.replay_boost = (
                    self._forgetting_mitigation_active and boost_still_valid
                )
            if self._should_skip_gprime_learn(metrics):
                self._last_m3_replay_steps = 0
            else:
                attn_weighted_error = metrics.prediction_error * float(np.mean(self._attention_weights))
                if hasattr(self.gprime, '_attention_weights'):
                    self.gprime._attention_weights = self._attention_weights.copy()
                self.gprime.learn(
                    self.current_state, action_vec, next_state,
                    error=attn_weighted_error,
                )
                self._update_phi_from_gradient()
                self._last_m3_replay_steps = self._replay_m3_prior_tasks()
                self._m3_replay_total += self._last_m3_replay_steps
        metrics.module_timings["gprime_learn"] = (time.perf_counter() - t_glearn) * 1000

        # Store episode in M3 episodic memory (Task B fix)
        if self.interventions.enable_m3_write:
            m3_drive_id = self.current_goal.drive_id if self.current_goal else None
            self.consolidation.m3.store_episode(
                state_before=self.current_state,
                action_taken=action_vec,
                state_after=next_state,
                prediction_error=metrics.prediction_error,
                confidence=metrics.prediction_confidence,
                drive_id=m3_drive_id,
                task_id=self._current_task_id,
                timestamp=self.cycle_count,
            )
        self._last_prediction_error = metrics.prediction_error

    # ── Async Cycle (Feature 1) ──────────────────────────────────────────

    def start_async(self) -> None:
        """Start the async action and learning threads.

        In async mode, Thread A continuously runs the perception-action
        cycle and pushes ``ActionResult`` to a queue.  Thread B pulls
        results and runs the learning phase (PEU, TSPL, G'.learn, M3)
        with RBTA enforcement, logging, and consolidation.

        This is safe to call multiple times — subsequent calls are no-ops
        when threads are already running.
        """
        if self._action_thread is not None and self._action_thread.is_alive():
            return
        self._async_mode = True
        self._stop_event.clear()
        self._action_thread = threading.Thread(
            target=self._action_loop, name="phca-action", daemon=True,
        )
        self._learning_thread = threading.Thread(
            target=self._learning_loop, name="phca-learning", daemon=True,
        )
        self._action_thread.start()
        self._learning_thread.start()
        _log(logger, "info", "cycle.async.start")

    def stop_async(self, timeout: float = 5.0) -> None:
        """Signal stop and join both threads."""
        self._stop_event.set()
        if self._action_thread is not None and self._action_thread.is_alive():
            self._action_thread.join(timeout=timeout)
        if self._learning_thread is not None and self._learning_thread.is_alive():
            self._learning_thread.join(timeout=timeout)
        self._async_mode = False
        _log(logger, "info", "cycle.async.stop")

    def _build_async_perception_frame(self) -> PerceptionFrame:
        """Build a fresh PerceptionFrame from the current environment state."""
        raw_obs = self.env.get_observation()
        return PerceptionFrame(
            observation=raw_obs,
            action=None,
            timestamp=time.time(),
            age=0,
        )

    def _action_loop(self) -> None:
        """Thread A: continuous perception-action loop.

        Each iteration:
          1. Snapshot environment observation into a PerceptionFrame
          2. Run perception cycle (ASI → M2 → prediction → regulation)
          3. Select and execute action
          4. Push ActionResult to ``_action_result_queue``
          5. If terminal, reset environment
        """
        while not self._stop_event.is_set():
            try:
                # Phase A: Perception + Prediction + Regulation
                hpm_spec = self._run_perception_cycle(CycleMetrics(cycle_id=self.cycle_count))

                # Phase B: Action selection + env step
                if self._rbta_skip_feedback:
                    action = self._neutral_action()
                else:
                    action = self._select_action()

                if self._is_continuous:
                    obs, step_reward, terminal, info = self.env.step(
                        np.asarray(action, dtype=np.float32)
                    )
                else:
                    obs, step_reward, terminal, info = self.env.step(action)
                self._last_step_reward = float(step_reward)

                # Push result to learning thread
                result = ActionResult(
                    reward=step_reward,
                    terminal=terminal,
                    state=obs,
                    age=0,
                    action=action,
                    info=info,
                )
                self._action_result_queue.put(result)

                if terminal:
                    self.env.reset()
                    self.gprime.reset()
                    self.sanitizer.reset()
                    self._last_goal_pos = None
                    self._goal_switch_cooldown = 0

            except Exception as e:
                _log(logger, "error", "cycle.action_loop.error", error=str(e))
                if self._stop_event.wait(timeout=0.1):
                    break

    def _learning_loop(self) -> None:
        """Thread B: continuous learning loop.

        Each iteration:
          1. Pull an ActionResult from the queue (blocking with timeout)
          2. If stale (age > STALE_THRESHOLD_MS), skip learning
          3. Run the learning phase (PEU + TSPL + G'.learn + M3)
          4. RBTA enforcement + resilience + logging + consolidation
        """
        while not self._stop_event.is_set():
            try:
                # Block for up to STALE_THRESHOLD_MS for a result
                result: ActionResult | None = None
                try:
                    result = self._action_result_queue.get(
                        timeout=STALE_THRESHOLD_MS / 1000.0
                    )
                except queue.Empty:
                    pass

                t_start = time.perf_counter()
                metrics = CycleMetrics(cycle_id=self.cycle_count)

                # Reset RBTA flags each learning cycle
                self._rbta_skip_feedback = False
                self._rbta_skip_consolidation = False
                self._rbta_action_candidate_limit = None
                if self._rbta_carry_action == EnforcerAction.TERMINATE:
                    self._rbta_skip_feedback = True
                elif self._rbta_carry_action == EnforcerAction.INTERRUPT:
                    self._rbta_action_candidate_limit = 1

                if result is None or result.age > STALE_THRESHOLD_MS:
                    # Stale frame — skip learning, still increment cycle
                    metrics.latency_ms = (time.perf_counter() - t_start) * 1000
                    self._finalize_learning_cycle(metrics, None)
                    continue

                # Phase C: Learning
                action_vec_step = (
                    np.asarray(result.action, dtype=np.float32)
                    if isinstance(result.action, (list, np.ndarray))
                    else None
                )
                self._run_learning_phase(
                    metrics, result.action, result.state,
                    result.info or {"goal_reached": False},
                    action_vec_step,
                )

                metrics.latency_ms = (time.perf_counter() - t_start) * 1000
                self._finalize_learning_cycle(metrics, result)

                if result.terminal:
                    # Log terminal reset info
                    _log(logger, "info", "cycle.async.terminal",
                         cycle=self.cycle_count,
                         reward=f"{result.reward:.3f}")

            except Exception as e:
                _log(logger, "error", "cycle.learning_loop.error",
                     cycle=self.cycle_count, error=str(e))
                if self._stop_event.wait(timeout=0.1):
                    break

    def _finalize_learning_cycle(
        self,
        metrics: CycleMetrics,
        result: ActionResult | None,
    ) -> None:
        """RBTA enforcement, resilience, logging, consolidation, cycle count.

        Shared by ``_learning_loop`` and (eventually) ``step()``.
        """
        # RBTA enforcement
        self._collect_runtime_log(metrics)
        self.runtime_log["CYCLE"] = metrics.latency_ms / 1000.0
        # Build a minimal hpm spec for RBTA (use last one from perception cycle)
        hpm_spec: Dict[str, Any] = {
            "type": "SEQUENCE", "id": "cognitive_cycle",
            "children": [
                {"type": "ASI_Input", "id": "ASI", "dim": self.state_dim},
                {"type": "SEQUENCE", "id": "prediction_block",
                 "children": ["WM", {"type": "Predict", "id": "G'_PE", "horizon": 1}]},
                {"type": "PARALLEL", "id": "regulation_block",
                 "children": ["MDIM", "CR", "ATTN", "HPM"]},
                "ACTION",
                {"type": "SEQUENCE", "id": "feedback_block",
                 "children": ["PEU", "TSPL-P"]},
                "CYCLE",
            ],
        }
        hpm_bounds = self.hpm_validator.compute_bounds(hpm_spec, self.runtime_log)
        reg_b_time, reg_b_energy = self._scaled_hpm_composite_bounds(hpm_bounds)
        composition_tree = {
            "type": "SEQUENCE", "id": "cognitive_cycle",
            "children": [
                "ASI", "WM", "PE",
                {"type": "PARALLEL", "id": "regulation_block",
                 "children": ["MDIM", "CR", "ATTN", "HPM"],
                 "bounds": {"B_time": reg_b_time, "B_energy": reg_b_energy}},
                "ACTION", "PEU", "TSPL-P", "CYCLE",
            ],
            "bounds": {
                "B_time": reg_b_time * 2 + 0.050,
                "B_energy": reg_b_energy * 2 + 0.010,
            },
        }
        violations, enforcer_action = self.rbta.check_cycle(
            runtime_log=self.runtime_log,
            memory_log=self.memory_log,
            energy_log=self.energy_log,
            belief_entropies=self.belief_entropies,
            sensor_failure_count=self.sensor_failure_count,
            asi_failure_limit=self.asi_failure_limit,
            composition_tree=composition_tree,
        )
        metrics.rbta_action = enforcer_action.name
        metrics.violations_count = len(violations)
        self.last_violations = violations
        self._rbta_carry_action = enforcer_action
        if enforcer_action in (EnforcerAction.INTERRUPT, EnforcerAction.TERMINATE):
            self._rbta_skip_consolidation = True

        # Cognitive resilience
        snapshot = self._build_resilience_snapshot(metrics)
        events = self._resilience_detector.detect(snapshot)
        if events:
            self._resilience_recovery.apply(self, events)
            for e in events:
                self._fallback_controller.notify_failure(
                    e.mode_id, e.severity, self.cycle_count,
                )
        metrics.failure_events = [e.mode_id for e in events]
        metrics.recovery_active = self._resilience_recovery.any_active()
        metrics.emergency_active = self._fallback_controller.active()
        metrics.emergency_entropy = self._fallback_controller.smoothed_entropy

        # Logging — populate dashboard fields
        consol_stats = self.consolidation.get_stats()
        metrics.drive_id = self.current_goal.drive_id if self.current_goal else 1
        metrics.skill_accuracy = self.tspl.skill_accuracy
        metrics.skill_compiled = self.tspl.skill_compiled
        metrics.fact_count = consol_stats.get("total_facts_stored", 0)
        metrics.episode_count = self.consolidation.m3.count() if hasattr(self, 'consolidation') else 0
        metrics.task_id = self._current_task_id
        if metrics.fact_count == self._last_fact_count:
            self._fact_stagnant_cycles += 1
        else:
            self._fact_stagnant_cycles = 0
        self._last_fact_count = metrics.fact_count
        self.metrics_history.append(metrics)

        if self.metrics_store is not None:
            self.metrics_store.push(copy.deepcopy(metrics))
        if self.observability_store is not None:
            from phca.monitoring.observability import ObservabilityFrame
            self.observability_store.push(ObservabilityFrame.from_cycle(self))
        if self.trace_collector is not None:
            self.trace_collector.record(
                cycle_id=self.cycle_count,
                action=metrics.action_taken,
                reward=self._last_step_reward,
                prediction_error=metrics.prediction_error,
                prediction_confidence=metrics.prediction_confidence,
                goal_drive=metrics.drive_id,
                rbta_action=metrics.rbta_action,
                violations=metrics.violations_count,
                latency_ms=metrics.latency_ms,
                goal_reached=metrics.goal_reached,
                module_timings=dict(metrics.module_timings),
                attention_weights=(
                    self._attention_weights.tolist()
                    if self.interventions.enable_attention else None
                ),
                env_goal_relocated=self._env_goal_relocated,
            )
            self._env_goal_relocated = False
        if len(self.metrics_history) > 5000:
            self.metrics_history = self.metrics_history[-5000:]

        # Consolidation
        if self._rbta_skip_consolidation or not self.interventions.enable_consolidation:
            metrics.module_timings["consolidation"] = 0.0
        else:
            t_consol = time.perf_counter()
            consol_report = self.consolidation.step(
                self.cycle_count,
                gprime=self.gprime if self.interventions.enable_gprime_learn else None,
            )
            metrics.module_timings["consolidation"] = (
                time.perf_counter() - t_consol
            ) * 1000
            if consol_report.success and consol_report.episodes_processed > 0:
                recent_facts = self.consolidation.get_semantic_facts(
                    min_confidence=0.3, max_results=10,
                )
                fact_types = {}
                for f in recent_facts:
                    fact_types[f.fact_type] = fact_types.get(f.fact_type, 0) + 1
                _log(logger, "info", "cycle.consolidation",
                     episodes=consol_report.episodes_processed,
                     facts=consol_report.facts_generated,
                     fact_types=fact_types,
                     total_facts=self.consolidation.get_stats()["total_facts_stored"],
                     ms=f"{consol_report.duration_ms:.1f}")
        self.runtime_log["CONSOL"] = metrics.module_timings.get("consolidation", 0.0) / 1000.0
        consol_energy = max(0.1, min(10.0, self.runtime_log["CONSOL"] * 50.0))
        self.energy_log["CONSOL"] = consol_energy

        # Increment cycle counter + anneal PER beta
        self.cycle_count += 1
        anneal_progress = min(1.0, self.cycle_count / PER_BETA_ANNEAL_STEPS)
        self._per_beta = PER_BETA_INIT + (PER_BETA_FINAL - PER_BETA_INIT) * anneal_progress

    def on_task_boundary(self, task_id: int) -> None:
        """Called by Level-4 runner on task switch — enable anti-forgetting hooks."""
        self._current_task_id = task_id
        self.tspl.protect_parameters(lambda_boost=0.05)
        self._forgetting_mitigation_active = True
        self._replay_boost_activated_cycle = self.cycle_count
        self._last_goal_pos = None
        if hasattr(self.gprime, "reset"):
            self.gprime.reset()

    def set_m3_replay_budget(self, budget: int) -> None:
        """Cap M3 episodic replay steps injected into G′ per learn cycle."""
        self._m3_replay_budget = max(0, int(budget))

    def on_forgetting_detected(self) -> None:
        """B4 recovery: replay boost + halve P-Stream learning rate."""
        self._forgetting_mitigation_active = True
        self._replay_boost_activated_cycle = self.cycle_count
        if isinstance(self.gprime, WorldModelMLP):
            self.gprime.replay_boost = True
        cfg = self.tspl.configs[StreamID.P_STREAM]
        cfg.alpha = max(0.01, cfg.alpha * 0.5)

    def record_task_baseline(self, task_id: int, goal_rate: float) -> None:
        """Store end-of-task accuracy baseline for B4 / forgetting metrics."""
        self._task_goal_baselines[task_id] = goal_rate

    def record_task_eval(self, task_id: int, goal_reached: bool) -> None:
        """Append eval-cycle goal outcome for per-task tracking."""
        history = self._task_eval_history.setdefault(task_id, [])
        history.append(float(goal_reached))
        if len(history) > 100:
            history[:] = history[-100:]

    def _replay_m3_prior_tasks(self) -> int:
        """Sample prior-task M3 episodes into G′ when forgetting mitigation is active.

        Uses PER (prioritized experience replay) sampling when the M3 instance
        supports it.  Falls back to stratified random sampling for backward
        compatibility.

        Returns:
            Number of gradient steps taken.
        """
        if not self._forgetting_mitigation_active:
            return 0
        if not isinstance(self.gprime, WorldModelMLP):
            return 0
        prior_id = self._current_task_id
        if prior_id <= 0:
            return 0
        if not self.interventions.enable_m3_write:
            return 0
        m3 = self.consolidation.m3
        boost_mult = 2 if self.gprime.replay_boost else 1
        n_budget = self._m3_replay_budget * boost_mult

        if hasattr(m3, "sample_episodes_per"):
            episodes = m3.sample_episodes_per(
                n_budget, task_id=self._current_task_id,
                alpha=PER_ALPHA, beta=self._per_beta,
            )
        else:
            episodes = m3.sample_prior_task_episodes(n_budget, self._current_task_id)

        if not episodes:
            return 0
        steps, priority_updates = self.gprime.learn_m3_episodes(episodes)
        if priority_updates and hasattr(m3, "batch_update_priorities"):
            m3.batch_update_priorities(priority_updates)
        return steps

    def _current_task_goal_rate(self) -> Optional[float]:
        tid = self._current_task_id
        hist = self._task_eval_history.get(tid, [])
        if not hist:
            return None
        window = hist[-20:]
        return float(np.mean(window))

    def _should_skip_gprime_learn(self, metrics: CycleMetrics) -> bool:
        """Throttle replay-only G' learning once the model is in steady state."""
        if not isinstance(self.gprime, WorldModelMLP):
            return False
        if self._forgetting_mitigation_active:
            return False
        if self.cycle_count < 64:
            return False
        if metrics.prediction_error > STEADY_STATE_GPRIME_ERROR_THRESHOLD:
            return False
        if self._is_continuous and metrics.prediction_error > 2.0:
            return False
        if len(getattr(self.gprime, "_replay_buffer", [])) < self.gprime.batch_size:
            return False
        return (self.cycle_count % STEADY_STATE_GPRIME_SKIP_MOD) != 0

    def _build_resilience_snapshot(
        self,
        metrics: Optional[CycleMetrics] = None,
    ) -> CycleSnapshot:
        """Build detector input from current cycle signals."""
        m = metrics or (self.metrics_history[-1] if self.metrics_history else CycleMetrics())
        recent = self.metrics_history[-20:]
        rbta_hist = [h.rbta_action for h in self.metrics_history[-5:]]
        actions = [h.action_taken for h in recent if h.action_taken >= 0]
        errors = [h.prediction_error for h in recent]
        confs = [h.prediction_confidence for h in recent]
        episode_count = m.episode_count
        m3_cap = getattr(self.consolidation.m3, "_max_episodes", 10_000)
        deficits = {}
        if hasattr(self.mdim, "drives"):
            deficits = {
                name: float(d.deficit)
                for name, d in self.mdim.drives.items()
            }
        fact_delta = 0
        if len(self.metrics_history) >= 2:
            fact_delta = m.fact_count - self.metrics_history[-2].fact_count
        return CycleSnapshot(
            cycle_id=self.cycle_count,
            prediction_error=m.prediction_error,
            prediction_confidence=m.prediction_confidence,
            wm_entropy_proxy=float(
                np.mean(list(self.belief_entropies.values()))
                if self.belief_entropies else 0.5
            ),
            mdim_deficits=deficits,
            rbta_action_history=rbta_hist,
            m3_fill_ratio=episode_count / max(m3_cap, 1),
            m4_fact_count=m.fact_count,
            m4_fact_count_delta=fact_delta,
            module_timings=dict(m.module_timings),
            unique_actions_recent=len(set(actions)) if actions else 0,
            per_task_goal_rate=self._current_task_goal_rate(),
            per_task_baseline_goal_rate=self._task_goal_baselines.get(self._current_task_id),
            recent_prediction_errors=errors,
            recent_actions=actions,
            recent_confidences=confs,
            fact_count_stagnant_cycles=self._fact_stagnant_cycles,
        )

    def _facts_ids(self) -> list:
        return [f.fact_id for f in self._relevant_facts]

    def _facts_summary(self) -> list:
        return [
            {
                "fact_id": f.fact_id,
                "confidence": round(float(f.confidence), 3),
            }
            for f in self._relevant_facts[:5]
        ]

    def _mechanism_for_reason(self, decision_reason: str) -> str:
        return {
            "rbta_safe": "rbta_safe",
            "explore": "explore",
            "continuous_explore": "explore",
            "d5_stay": "stay",
            "greedy_fallback": "greedy_fallback",
            "prediction": "prediction",
            "continuous_mpc": "continuous",
        }.get(decision_reason, "other")

    def _chosen_label_from_rationale(self, rationale: dict) -> str:
        is_cont = bool(rationale.get("continuous"))
        if is_cont:
            ci = rationale.get("chosen_idx")
            if isinstance(ci, (int, float)) and int(ci) >= 0:
                return f"τ#{int(ci)}"
            return "τ"
        ci = rationale.get("chosen_idx")
        if isinstance(ci, (int, float)):
            idx = int(ci)
            names = (
                self.env.get_action_names()
                if hasattr(self.env, "get_action_names") else []
            )
            if 0 <= idx < len(names):
                return f"{names[idx]} #{idx}"
            return f"#{idx}"
        return "—"

    def _finalize_action_rationale(
        self,
        rationale: dict,
        *,
        decision_reason: str,
    ) -> dict:
        """Attach Phase 14 explain fields from cycle-computed selection state."""
        r = dict(rationale)
        r["decision_reason"] = decision_reason
        goal = self.current_goal
        gid = r.get("goal_id")
        if gid is None and goal is not None:
            gid = goal.drive_id
        if gid is not None:
            r["goal_id"] = int(gid)
            r["drive_id"] = int(gid)
        if "relevant_fact_ids" not in r:
            r["relevant_fact_ids"] = self._facts_ids()
        if "relevant_facts_summary" not in r:
            r["relevant_facts_summary"] = self._facts_summary()
        r["mechanism"] = self._mechanism_for_reason(decision_reason)
        r["chosen_label"] = self._chosen_label_from_rationale(r)
        return r

    def _neutral_action(self):
        """Safe-mode action: discrete stay index, or continuous zero vector.

        Continuous MuJoCo envs (Reacher/Pendulum) expose ``stay_action`` as an
        int for discrete-map compatibility; stepping with that int yields a
        0-d array and gymnasium raises ``Action dimension mismatch``.
        """
        if self._is_continuous:
            getter = getattr(self.env, "neutral_action", None)
            if callable(getter):
                return np.asarray(getter(), dtype=np.float32)
            dim = int(getattr(self.action_space, "dim", 0) or 0)
            return np.zeros(dim, dtype=np.float32)
        return self.env.stay_action

    def _epistemic_entropy(self) -> float:
        """Epistemic uncertainty for D4 and A3 (MC-dropout mutual information).

        Uses additive floor (0.01 + mi) so the signal stays responsive above
        the RBTA entropy_floor without max-clip flattening small variations.
        """
        mi = float(getattr(self.gprime, "_last_mutual_info", 0.5))
        return min(1.0, 0.01 + mi)

    def _select_action(self):
        """Select action using goal-directed planning with MDIM goal awareness.

        Returns an int (discrete space) or an np.ndarray (continuous space,
        Phase 6 / A2). The continuous branch is MPC-style: sample K candidate
        actions, predict each via G', pick the one whose predicted next state
        best matches the goal reference. Prediction/goal-driven, NOT RL.

        Emergency fallback is checked first: if belief entropy is critically
        high, the agent defaults to instinctive stop behavior (STAY in
        GridWorld, zero torque in MuJoCo).
        """
        # Emergency fallback: instinctive stop under critical entropy
        if not self._is_continuous:
            entropy = float(
                np.mean(list(self.belief_entropies.values()))
                if self.belief_entropies else 0.5
            )
            event = self._fallback_controller.check(entropy, self.cycle_count)
            if event is not None:
                action = self._neutral_action()
                self.last_action_rationale = self._finalize_action_rationale({
                    "explored": False,
                    "eps": 0.0,
                    "goal_id": int(self.current_goal.drive_id) if self.current_goal else 1,
                    "continuous": False,
                    "best_score": None,
                    "k_candidates": 1,
                    "chosen_idx": int(action),
                    "task_lock": bool(self._task_lock),
                    "emergency_mode": True,
                    "emergency_entropy": float(entropy),
                    "selector_mode": "emergency_stop",
                    "relevant_fact_ids": [f.fact_id for f in self._relevant_facts],
                }, decision_reason="emergency")
                return action

        if self._is_continuous:
            return self._select_continuous_action()

        if self.current_state is None:
            return self.env.stay_action

        goal = self.current_goal
        goal_id = goal.drive_id if goal else 1
        target = goal.target_state if goal else None

        # Adaptive ε-greedy: explore less as model converges; task-lock reduces ε (P0-3)
        eps = max(0.02, 0.10 * (1.0 - self.cycle_count / 500.0))
        if self._task_lock:
            eps = 0.0
        rng = np.random.RandomState(self.cycle_count)
        if eps > 0.0 and rng.random() < eps:
            pick = int(rng.randint(0, self.env.action_space_size))
            self.last_action_rationale = self._finalize_action_rationale({
                "explored": True, "eps": float(eps),
                "goal_id": int(goal_id), "continuous": False,
                "best_score": None,
                "k_candidates": int(self.env.action_space_size),
                "chosen_idx": pick,
                "task_lock": bool(self._task_lock),
                "selector_mode": "discrete_explore",
                "relevant_fact_ids": [f.fact_id for f in self._relevant_facts],
            }, decision_reason="explore")
            self.last_candidate_scores = []
            self.last_candidate_rollouts = []
            return pick

        # D5 (Energy Efficiency): prefer STAY for grid envs — skip under task-lock (P0-3)
        # Non-grid envs (bandit) have no meaningful stay action; fall through
        if goal_id == 5 and not self._task_lock and hasattr(self.env, 'grid'):
            self.last_action_rationale = self._finalize_action_rationale({
                "explored": False, "eps": float(eps),
                "goal_id": 5, "continuous": False,
                "best_score": None, "k_candidates": None,
                "note": "D5 energy: STAY",
                "chosen_idx": int(self.env.stay_action),
                "task_lock": bool(self._task_lock),
                "selector_mode": "energy_stay",
                "relevant_fact_ids": [f.fact_id for f in self._relevant_facts],
            }, decision_reason="d5_stay")
            self.last_candidate_scores = []
            self.last_candidate_rollouts = []
            return self.env.stay_action

        # Task-lock: geometry-primary when G' is confident; else fall through to
        # blended scorer (per-candidate predict) for prediction-primary selection.
        _cycle_conf = (
            float(self.last_prediction.precision[0])
            if self.last_prediction is not None
            else 0.0
        )
        if (
            self._task_lock
            and hasattr(self.env, "agent_pos")
            and _cycle_conf >= TASK_LOCK_CONFIDENCE_THRESHOLD
        ):
            spec = getattr(self.env, "spec", None)
            long_horizon = bool(getattr(spec, "dynamic_goals_every", 0))
            explore_ties = self._goal_switch_cooldown > 0 or bool(
                getattr(self.env, "switch_cycles", [])
            )
            goal_pos = self.env.get_goal_position()
            on_goal = (
                goal_pos is not None
                and tuple(self.env.agent_pos) == tuple(goal_pos)
            )
            sparse_probe = long_horizon and on_goal and (
                self.cycle_count % 50 == 0
                or self._goal_switch_cooldown >= 14
            )
            greedy_action, greedy_score, greedy_comp = self._select_greedy_grid_action(
                explore_ties=explore_ties,
                at_goal_explore=sparse_probe,
            )
            self.last_candidate_scores = []
            self.last_candidate_rollouts = []
            self.last_action_rationale = self._finalize_action_rationale({
                "explored": False, "eps": 0.0,
                "goal_id": int(goal_id), "continuous": False,
                "best_score": float(greedy_score),
                "k_candidates": int(self.env.action_space_size),
                "chosen_idx": int(greedy_action),
                "task_lock": True,
                "greedy_fallback": True,
                "selector_mode": "task_lock_planner",
                "score_components": greedy_comp,
                "relevant_fact_ids": [f.fact_id for f in self._relevant_facts],
            }, decision_reason="greedy_fallback")
            return greedy_action

        best_action = self.env.stay_action
        best_score = -float("inf")
        action_confidences = []
        scores_per_action = [0.0] * self.env.action_space_size
        best_components: dict = {}
        _obs = self.observability_store is not None
        _rollouts: list = []

        candidate_indices = range(self.env.action_space_size)
        if self._rbta_action_candidate_limit is not None:
            candidate_indices = list(candidate_indices)[
                : self._rbta_action_candidate_limit
            ]

        for action_idx in candidate_indices:
            action = np.zeros(self.env.action_space_size, dtype=np.float32)
            action[action_idx] = 1.0
            self.engine.update_action(action)

            try:
                predicted, confidence = self.engine.predict(
                    self.current_state, horizon=1,
                )
                action_confidences.append(confidence)
                if _obs:
                    _rollouts.append({
                        "action": action.copy(),
                        "action_idx": action_idx,
                        "predicted": np.asarray(predicted.values,
                                                dtype=np.float32).copy(),
                        "score": 0.0,
                        "chosen": False,
                    })

                distance_gain = self._compute_distance_gain(action_idx)

                ramp = float(np.clip((self.cycle_count - 50) / 100.0, 0.0, 1.0))
                pga = self._predicted_goal_alignment(predicted)
                conf = min(confidence, 1.0)
                alignment = 0.0

                # P0-2: prediction-primary under low confidence; task-lock stays geometry-primary
                mean_conf_preview = float(np.mean(action_confidences)) if action_confidences else conf
                if self._task_lock:
                    prediction_primary = False
                else:
                    prediction_primary = mean_conf_preview < 0.4

                if goal_id in (1, 3):
                    if target is not None:
                        alignment = self._state_space_alignment(predicted, target)
                    if prediction_primary:
                        base_score = 0.45 * pga + 0.35 * conf + 0.20 * (1.0 - distance_gain)
                        score = base_score + alignment * 0.15
                    else:
                        base_score = (1.0 - distance_gain) * 0.6 + conf * 0.2
                        base_score = base_score * (1.0 - ramp * 0.4) + pga * (ramp * 0.4)
                        score = base_score + alignment * 0.2 if target is not None else base_score
                elif goal_id in (2, 4):
                    if target is not None:
                        alignment = self._state_space_alignment(predicted, target)
                    if prediction_primary:
                        base_score = 0.35 * (1.0 - conf) + 0.25 * pga + 0.20 * (1.0 - distance_gain)
                        score = base_score + alignment * 0.2
                    else:
                        base_score = (1.0 - conf) * 0.5 + (1.0 - distance_gain) * 0.2
                        score = base_score + alignment * 0.3 if target is not None else base_score
                else:
                    state_align = self._state_space_alignment(predicted, target)
                    score = state_align
                    alignment = state_align

                scores_per_action[action_idx] = float(score)
                if score > best_score:
                    best_score = score
                    best_action = action_idx
                    best_components = {
                        "distance_gain": float(distance_gain),
                        "pga": float(pga),
                        "confidence": float(conf),
                        "alignment": float(alignment),
                        "prediction_primary": bool(prediction_primary),
                        "fact_boost": float(len(self._relevant_facts) > 0),
                    }
                if _obs and _rollouts and _rollouts[-1].get("action_idx") == action_idx:
                    _rollouts[-1]["score"] = float(score)
            except Exception as e:
                action_confidences.append(0.0)
                _log(logger, "warning", "action_selection.predict_failed",
                     error=str(e))
                continue

        # Cache confidences for _estimate_empowerment (avoids duplicate 5× predict)
        self._cached_confidences = action_confidences
        self.last_candidate_scores = scores_per_action
        if _obs:
            for r in _rollouts:
                r["chosen"] = (r["action_idx"] == best_action)
            _rollouts.sort(key=lambda r: r["score"], reverse=True)
            self.last_candidate_rollouts = _rollouts[:8]
        else:
            self.last_candidate_rollouts = []

        _gated_fallback = (
            self._task_lock
            and _cycle_conf < TASK_LOCK_CONFIDENCE_THRESHOLD
        )
        self.last_action_rationale = self._finalize_action_rationale({
            "explored": False, "eps": float(eps),
            "goal_id": int(goal_id), "continuous": False,
            "best_score": float(best_score),
            "k_candidates": int(self.env.action_space_size),
            "chosen_idx": int(best_action),
            "task_lock": bool(self._task_lock),
            "prediction_gated_fallback": bool(_gated_fallback),
            "selector_mode": "prediction_scored",
            "score_components": best_components,
            "relevant_fact_ids": [f.fact_id for f in self._relevant_facts],
        }, decision_reason="prediction")
        return best_action

    def _select_continuous_action(self) -> np.ndarray:
        """MPC-style continuous action selection (Phase 6 / A2).

        Sample K candidate actions ~ U(low, high); for each, predict the next
        state via G'; score by prediction confidence + goal-reference alignment
        + predicted-goal-alignment (PGA, neutral for non-grid envs). Return the
        argmax candidate as an np.ndarray within [low, high]. ε-greedy returns a
        random in-bounds action. Prediction/goal-driven — no reward, no value
        function, no policy gradient (A4/A5 preserved).

        A1 (Resource Boundedness): K·dim ≤ 16 forward passes per call.

        Emergency fallback check: if belief entropy is critically high, agent
        defaults to instinctive zero-torque (neutral) behavior.
        """
        entropy = float(
            np.mean(list(self.belief_entropies.values()))
            if self.belief_entropies else 0.5
        )
        event = self._fallback_controller.check(entropy, self.cycle_count)
        if event is not None:
            action = self._neutral_action()
            self.last_action_rationale = self._finalize_action_rationale({
                "explored": False,
                "eps": 0.0,
                "goal_id": int(self.current_goal.drive_id) if self.current_goal else 1,
                "continuous": True,
                "best_score": None,
                "k_candidates": 1,
                "chosen_idx": None,
                "task_lock": bool(self._task_lock),
                "emergency_mode": True,
                "emergency_entropy": float(entropy),
                "selector_mode": "emergency_stop",
                "relevant_fact_ids": [f.fact_id for f in self._relevant_facts],
            }, decision_reason="emergency")
            return action

        space = self.action_space
        low, high = space.low, space.high
        K = 8
        if K * space.dim > 16:
            K = max(2, 16 // space.dim)

        rng = np.random.RandomState(self.cycle_count)
        eps = max(0.02, 0.10 * (1.0 - self.cycle_count / 500.0))
        goal = self.current_goal
        goal_id = goal.drive_id if goal else None
        if rng.random() < eps:
            self.last_action_rationale = self._finalize_action_rationale({
                "explored": True, "eps": float(eps),
                "goal_id": int(goal_id) if goal_id is not None else None,
                "continuous": True,
                "best_score": None, "k_candidates": int(K),
                "selector_mode": "continuous_explore",
            }, decision_reason="continuous_explore")
            self.last_candidate_scores = []
            self.last_candidate_rollouts = []
            return (low + (high - low) * rng.uniform(size=space.dim)).astype(np.float32)

        ref = getattr(self.env, "get_goal_reference", lambda: None)()
        ref = np.asarray(ref, dtype=np.float32) if ref is not None else None

        best_a, best_score, best_idx = None, -float("inf"), -1
        best_components: dict = {}
        cand_scores = []
        _obs = self.observability_store is not None
        _rollouts: list = []
        for _ in range(K):
            a = (low + (high - low) * rng.uniform(size=space.dim)).astype(np.float32)
            self.engine.update_action(a)
            try:
                predicted, confidence = self.engine.predict(self.current_state, horizon=1)
            except Exception:
                cand_scores.append(0.0)
                continue
            pga = self._predicted_goal_alignment(predicted)
            if ref is not None:
                min_d = min(predicted.values.shape[0], ref.shape[0])
                ref_align = float(np.clip(
                    1.0 - np.linalg.norm(predicted.values[:min_d] - ref[:min_d])
                    / max(np.linalg.norm(ref[:min_d]), 1e-6), 0.0, 1.0))
            else:
                ref_align = 0.5
            score = 0.4 * float(np.clip(confidence, 0.0, 1.0)) + 0.5 * ref_align + 0.1 * pga
            cand_scores.append(float(score))
            ci = len(cand_scores) - 1
            if _obs:
                _rollouts.append({
                    "action": a.copy(),
                    "predicted": np.asarray(predicted.values,
                                            dtype=np.float32).copy(),
                    "score": float(score),
                    "chosen": False,
                })
            if score > best_score:
                best_score, best_a, best_idx = score, a, ci
                best_components = {
                    "confidence": float(np.clip(confidence, 0.0, 1.0)),
                    "ref_align": float(ref_align),
                    "pga": float(pga),
                }
        self.last_candidate_scores = cand_scores
        if _obs:
            for r in _rollouts:
                if best_a is not None and np.array_equal(r["action"], best_a):
                    r["chosen"] = True
            self.last_candidate_rollouts = _rollouts[:8]
        else:
            self.last_candidate_rollouts = []
        self.last_action_rationale = self._finalize_action_rationale({
            "explored": False, "eps": float(eps),
            "goal_id": int(goal_id) if goal_id is not None else None,
            "continuous": True,
            "best_score": float(best_score) if best_a is not None else None,
            "k_candidates": int(K),
            "chosen_idx": int(best_idx),
            "selector_mode": "continuous_mpc",
            "score_components": best_components,
            "relevant_fact_ids": [f.fact_id for f in self._relevant_facts],
        }, decision_reason="continuous_mpc")
        return best_a if best_a is not None else (
            low + (high - low) * 0.5).astype(np.float32)

    def _state_space_alignment(
        self,
        predicted: StateVector,
        target: Optional[StateVector],
    ) -> float:
        """Compute how well the predicted state matches the MDIM goal target.

        Uses weighted cosine similarity between predicted and target state
        vectors, where target precision weights each dimension's importance.

        When target is None or has no meaningful values, returns 0.5
        (neutral — no preference).

        Args:
            predicted: Predicted next state from G'.
            target: MDIM goal target_state (may be None).

        Returns:
            0.0-1.0 alignment score (1.0 = exact match).
        """
        if target is None:
            return 0.5

        # Use minimum dimension to avoid shape mismatch
        min_dim = min(
            predicted.values.shape[0],
            target.values.shape[0],
        )
        p = predicted.values[:min_dim].astype(np.float64)
        t = target.values[:min_dim].astype(np.float64)
        w = target.precision[:min_dim].astype(np.float64)

        # Weighted cosine similarity
        p_norm = np.linalg.norm(p * w)
        t_norm = np.linalg.norm(t * w)
        if p_norm < 1e-8 or t_norm < 1e-8:
            return 0.5

        cos_sim = float(np.dot(p * w, t * w) / (p_norm * t_norm))
        # Map [-1, 1] → [0, 1]
        return float(np.clip((cos_sim + 1.0) / 2.0, 0.0, 1.0))

    def _predicted_goal_alignment(self, predicted: StateVector) -> float:
        """Score predicted state by proximity of predicted agent pos to goal.

        Extracts the predicted agent position from the first size² dims
        and measures Manhattan distance from the predicted peak to goal.
        Returns [0, 1] where 1.0 = peak predicted at goal position.
        Lower-bounded at 0.1 so it never fully blocks an action.
        """
        env = self.env
        goal_pos = env.get_goal_position()
        if goal_pos is None or not hasattr(env, "agent_pos") or not hasattr(env, "size"):
            return 0.5
        n_pos = env.size * env.size
        pred_agent = predicted.values[:n_pos]
        peak = int(np.argmax(pred_agent))
        max_val = float(pred_agent[peak])
        if max_val < 0.05:
            return 0.5
        g_row, g_col = goal_pos
        p_row, p_col = peak // env.size, peak % env.size
        dist = abs(p_row - g_row) + abs(p_col - g_col)
        max_dist = 2 * (env.size - 1)
        return float(np.clip(1.0 - dist / max_dist, 0.1, 1.0))

    def _build_planning_wall_grid(self) -> Optional[np.ndarray]:
        """Merge env wall map with consolidation fact wall hints (P0-5)."""
        env = self.env
        if not (hasattr(env, "grid") and hasattr(env, "WALL") and hasattr(env, "size")):
            return None
        n = env.size * env.size
        grid = np.array(env.grid, copy=True)
        if self.current_state is not None and len(self.current_state.values) >= 3 * n:
            obs_walls = self.current_state.values[2 * n:3 * n].reshape(env.size, env.size)
            grid = np.where(obs_walls > 0.5, env.WALL, grid)
        for fact in self._relevant_facts:
            if fact.state_pattern is None:
                continue
            pat = fact.state_pattern.values
            if len(pat) >= 3 * n:
                fact_walls = pat[2 * n:3 * n].reshape(env.size, env.size)
                grid = np.where(fact_walls > 0.5, env.WALL, grid)
        return grid

    def _select_greedy_grid_action(
        self,
        *,
        explore_ties: bool = False,
        at_goal_explore: bool = False,
    ) -> tuple:
        """One-step Manhattan controller using observed goal/walls (fair vs greedy_observed).

        On large plain GridWorld benchmarks (size >= 10), one-step greedy traps in maze
        local minima behind barrier walls; use BFS for the first path step instead.
        ScenarioGridWorld wrappers (causal eval partial-map) keep one-step greedy.
        """
        env = self.env
        goal_pos = env.get_goal_position()
        if goal_pos is None:
            return env.stay_action, 0.0, {}
        agent_pos = env.agent_pos
        grid_size = getattr(env, "size", 0)
        use_bfs_planner = (
            grid_size >= 10
            and not hasattr(env, "base")
            and hasattr(env, "grid")
            and not at_goal_explore
        )
        if use_bfs_planner and tuple(agent_pos) != tuple(goal_pos):
            from phca.evaluation.baselines.search import bfs_action

            best_action = bfs_action(env)
            best_distance = abs(agent_pos[0] - goal_pos[0]) + abs(agent_pos[1] - goal_pos[1])
            score = 1.0 / max(best_distance, 1)
            return best_action, score, {
                "distance_gain": 0.0,
                "pga": 0.0,
                "confidence": 1.0,
                "alignment": 0.0,
                "prediction_primary": False,
                "fact_boost": float(len(self._relevant_facts) > 0),
                "greedy_fallback": True,
                "bfs_planner": True,
            }
        grid = env.grid
        best_action = env.stay_action
        best_distance = abs(agent_pos[0] - goal_pos[0]) + abs(agent_pos[1] - goal_pos[1])
        best_unvisited = False
        action_names = env.get_action_names()
        deltas_fn = getattr(env, "get_action_deltas", None)
        deltas = deltas_fn() if callable(deltas_fn) else {}
        visited = getattr(env, "visited", None)

        def _dest_unvisited(action_idx: int) -> bool:
            if visited is None:
                return False
            dr, dc = deltas[action_names[action_idx]]
            return (agent_pos[0] + dr, agent_pos[1] + dc) not in visited

        for action_idx, name in enumerate(action_names):
            dr, dc = deltas[name]
            row = agent_pos[0] + dr
            col = agent_pos[1] + dc
            if not (0 <= row < env.size and 0 <= col < env.size):
                continue
            if grid[row, col] == env.WALL:
                continue
            dist = abs(row - goal_pos[0]) + abs(col - goal_pos[1])
            cand_unvisited = visited is not None and (row, col) not in visited
            if dist < best_distance:
                best_distance = dist
                best_action = action_idx
                best_unvisited = cand_unvisited
            elif (
                explore_ties
                and dist == best_distance
                and cand_unvisited
                and not best_unvisited
            ):
                best_action = action_idx
                best_unvisited = True

        if (
            at_goal_explore
            and tuple(agent_pos) == tuple(goal_pos)
            and visited is not None
        ):
            for action_idx, name in enumerate(action_names):
                if name == "STAY":
                    continue
                dr, dc = deltas[name]
                row = agent_pos[0] + dr
                col = agent_pos[1] + dc
                if not (0 <= row < env.size and 0 <= col < env.size):
                    continue
                if grid[row, col] == env.WALL:
                    continue
                if (row, col) not in visited:
                    best_action = action_idx
                    best_distance = abs(row - goal_pos[0]) + abs(col - goal_pos[1])
                    break

        score = 1.0 / max(best_distance, 1)
        return best_action, score, {
            "distance_gain": 0.0,
            "pga": 0.0,
            "confidence": 1.0,
            "alignment": 0.0,
            "prediction_primary": False,
            "fact_boost": float(len(self._relevant_facts) > 0),
            "greedy_fallback": True,
        }

    def _compute_distance_gain(self, action_idx: int) -> float:
        """Compute normalized distance gain for an action.

        Returns 0.0 if action moves directly toward goal, 1.0 if away,
        0.5 if same distance or environment lacks position information.
        Continuous metric: (current_dist - new_dist) / current_dist mapped
        to [0, 1] where lower = better.

        STAY (idx=4) penalized when not at goal (0.7 vs 0.5) to discourage
        lingering. For environments without position, falls back to 0.5 (neutral).
        """
        env = self.env
        goal_pos = env.get_goal_position()
        if goal_pos is not None and hasattr(env, "agent_pos"):
            if hasattr(env, "grid") and hasattr(env, "WALL"):
                grid = self._planning_grid if self._planning_grid is not None else env.grid
                deltas_fn = getattr(env, "get_action_deltas", None)
                if not callable(deltas_fn):
                    return 0.5
                action_names = env.get_action_names()
                dr, dc = deltas_fn()[action_names[action_idx]]
                new_row = env.agent_pos[0] + dr
                new_col = env.agent_pos[1] + dc

                if not (0 <= new_row < env.size and 0 <= new_col < env.size):
                    return 1.0
                if grid[new_row, new_col] == env.WALL:
                    return 1.0

                g_row, g_col = goal_pos
                current_dist = abs(env.agent_pos[0] - g_row) + abs(env.agent_pos[1] - g_col)
                new_dist = abs(new_row - g_row) + abs(new_col - g_col)

                if current_dist == 0:
                    # At goal: STAY (gain=1.0 → 0.0 best), leaving goal (gain=-1.0 → 1.0 worst)
                    gain = 1.0 if new_dist == current_dist else -1.0
                    return float(np.clip((1.0 - gain) / 2.0, 0.0, 1.0))

                if action_idx == self.env.stay_action and new_dist == current_dist:
                    return 0.7

                gain = (current_dist - new_dist) / current_dist
                return float(np.clip((1.0 - gain) / 2.0, 0.0, 1.0))
        return 0.5

    def _compute_cycle_flops(self) -> float:
        """Compute total FLOPs for current cycle (C3 fix: unified energy signal).

        Returns FLOP count for the dominant compute module (G').
        Used by both D5 (mdim_context) and RBTA (energy_log) so both
        derive energy from the same underlying computation.
        """
        if hasattr(self.gprime, 'hidden_dim'):
            h = self.gprime.hidden_dim
            s = self.state_dim
            a = self.gprime.action_dim
            fwd = 2.0 * ((s + a) * h + h * h + h * s)
            bs = getattr(self.gprime, 'batch_size', 32)
            ts = getattr(self.gprime, 'train_steps', 4)
            passes = bs * ts
            return fwd * passes * 3.0
        if hasattr(self.gprime, 'state_dim'):
            s = getattr(self.gprime, 'state_dim', self.state_dim)
            return 2.0 * s ** 3 + 4.0 * s ** 2
        return 0.0

    def _estimate_empowerment(self) -> float:
        """Estimate empowerment I(S';A|S) from the world model (C2 fix).

        Uses model-specific methods:
          - Gaussian G' (WorldModelGPrime): Closed-form Gaussian MI via
            conditional covariance Schur complements (AF-002).
          - MLP G' (WorldModelMLP): MC Dropout-based Gaussian MI via
            estimate_empowerment() (C2 fix — replaces std(confidences)).
          - Discrete G' (WorldModelGPrime): Variance-based heuristic.

        Falls back to 0.3 if no model-specific method is available.

        Returns:
            Float in [0.0, 1.0] estimating empowerment.
        """
        if self.current_state is None:
            return 0.3

        # Both MLP and Gaussian G' now implement estimate_empowerment()
        if hasattr(self.gprime, 'estimate_empowerment'):
            empowerment = self.gprime.estimate_empowerment(self.current_state)
            return float(np.clip(empowerment, 0.0, 1.0))

        # Fallback
        return 0.3

    def _compute_phi_criticality(self) -> float:
        """Return the current cached Φ (gradient-norm criticality).

        Φ is updated in ``_update_phi_from_gradient()`` which runs after
        each G' backward pass.  Between backward passes the cached value
        persists unchanged.
        """
        return self._cached_phi

    def _update_phi_from_gradient(self) -> None:
        """Read the fresh input gradient from G' backward pass and update cached Φ.

        Φ = (2/π) * arctan(||dL/dx_state|| / sqrt(state_dim)), EMA-filtered.
        The arctan maps [0, ∞) → [0, 1) so PID setpoints and MDIM drive
        targets need no adjustment from the old error-volatility heuristic.
        """
        if isinstance(self.gprime, WorldModelMLP):
            try:
                raw = self.gprime.last_input_sensitivity(self.state_dim)
            except Exception:
                return
            normalized = float(np.arctan(raw) * 2.0 / np.pi)
            self._cached_phi = self._cached_phi * 0.7 + normalized * 0.3

    def _scaled_hpm_composite_bounds(
        self, hpm_bounds: Optional[Dict[str, float]],
    ) -> tuple[float, float]:
        """Scale HPM composite bounds for large GridWorld state dimensions."""
        reg_b_time = hpm_bounds["B_time"] if hpm_bounds else 0.200
        reg_b_energy = hpm_bounds.get("B_energy", 10.0) if hpm_bounds else 10.0
        if grid_scale(self.state_dim) > 1.0:
            reg_b_time = scaled_time_bound(self.state_dim, reg_b_time)
            reg_b_energy = reg_b_energy * grid_scale(self.state_dim)
        return reg_b_time, reg_b_energy

    def _rbta_preflight_check(
        self,
        metrics: CycleMetrics,
        hpm_spec: Dict[str, Any],
    ) -> EnforcerAction:
        """Partial RBTA check after regulation, before action selection.

        Uses measured module timings collected so far. Same-cycle TERMINATE
        skips feedback (STAY); INTERRUPT limits prediction rollouts.
        """
        self._collect_runtime_log(metrics)
        hpm_bounds = self.hpm_validator.compute_bounds(hpm_spec, self.runtime_log)
        reg_b_time, reg_b_energy = self._scaled_hpm_composite_bounds(hpm_bounds)
        composition_tree = {
            "type": "SEQUENCE",
            "id": "cognitive_cycle",
            "children": [
                "ASI",
                "WM",
                "PE",
                {
                    "type": "PARALLEL",
                    "id": "regulation_block",
                    "children": ["MDIM", "CR", "ATTN", "HPM"],
                    "bounds": {"B_time": reg_b_time, "B_energy": reg_b_energy},
                },
            ],
            "bounds": {
                "B_time": reg_b_time + 0.050,
                "B_energy": reg_b_energy + 0.010,
            },
        }
        _, action = self.rbta.check_cycle(
            runtime_log=self.runtime_log,
            memory_log=self.memory_log,
            energy_log=self.energy_log,
            belief_entropies=self.belief_entropies,
            sensor_failure_count=self.sensor_failure_count,
            asi_failure_limit=self.asi_failure_limit,
            composition_tree=composition_tree,
        )
        return action

    def _collect_runtime_log(self, metrics: Optional[CycleMetrics] = None) -> None:
        """Collect module runtime/memory/energy logs for RBTA from actual measurements.

        Uses time.perf_counter() timings from metrics.module_timings.
        Memory/energy/entropy are still estimated (need instrumentation in Phase 3.3+).

        Args:
            metrics: Current cycle's metrics (with module_timings populated).
                If None (e.g., during testing), uses hardcoded fallback.
        """
        # Map metric keys → RBTA module IDs
        # Map metric keys → RBTA module IDs.
        # NOTE: action_selection uses "ACTION" (not "WM") to avoid overwriting
        # the WM memory_write timing in runtime_log. Previously both mapped to
        # "WM", causing WM's runtime_log entry to show action_selection time
        # (~10ms) instead of memory_write time (~0.02ms), which exceeded WM's
        # B_time=0.005 bound every cycle (P2-G fix: timing_map collision).
        timing_map = {
            "sanitize": "ASI",
            "memory_write": "WM",
            "prediction": "PE",
            "peu": "PEU",
            "tspl": "TSPL-P",
            "action_selection": "ACTION",
            "env_step": "ENV",
            "gprime_learn": "G'",
            "mdim": "MDIM",
            "cr": "CR",
            "attn": "ATTN",
            "hpm": "HPM",
            "consolidation": "CONSOL",
            "rbta": "CYCLE",
        }
        if metrics is not None:
            self.runtime_log = {
                module_id: (metrics.module_timings.get(metric, 0.0) / 1000.0)
                for metric, module_id in timing_map.items()
            }
        else:
            # Fallback for tests that call _collect_runtime_log() without metrics
            self.runtime_log = {
                module_id: 0.001 for module_id in set(timing_map.values())
            }
            self.runtime_log["G'"] = 0.001  # G' not in timing_map but expected by tests
        # Memory estimates from state dimensionality
        sd = self.state_dim
        if isinstance(self.gprime, WorldModelMLP):
            g_mem = estimate_mlp_memory_bytes(
                sd, self.gprime.action_dim, self.gprime.hidden_dim, self.gprime.replay_capacity,
            )
        else:
            g_mem = max(10_000, sd * sd * 4 * 5)
        self.memory_log = {
            "ASI": max(1_000, sd * 4 * 2),
            "WM": max(1_000, sd * 4 * 7),
            "G'": g_mem,
            "PE": max(1_000, sd * 4 * 3),
            "PEU": max(500, sd * 4),
            "TSPL-P": max(5_000, sd * 4 * 10),
            "MDIM": 20_000, "CR": 10_000, "ATTN": 5_000,
            "HPM": 10_000, "CONSOL": 2_000,
        }
        # Energy estimates — FLOP-based primary signal for G' (AF-005 fix).
        # MLP FLOPs/cycle: replay-buffer only (G-017), ~30M FLOPs at h=128, bs=32, ts=4.
        # Gaussian FLOPs/cycle: O(s^3) for covariance + O(s^2) for posterior → ~600K FLOPs.
        # For other modules, runtime*50.0 fallback provides ordinal plausibility.
        self.energy_log = {}
        for mod, runtime_s in self.runtime_log.items():
            self.energy_log[mod] = max(0.1, min(10.0, runtime_s * 50.0))
        # Fill any missing standard modules at realistic baseline
        baseline = {"ASI": 2.0, "WM": 1.0, "G'": 5.0, "CONSOL": 0.5}
        for mod, val in baseline.items():
            if mod not in self.energy_log:
                self.energy_log[mod] = val
        # FLOP-based G' energy (C3 fix: unified with MDIM energy_cost via self._cycle_flops)
        if self._cycle_flops > 0:
            gprime_energy = self._cycle_flops / ENERGY_NORM_FLOPS
            self.energy_log["G'"] = max(0.1, min(10.0, gprime_energy))
        # Belief entropy from MC-dropout epistemic uncertainty (A3: Incomplete Knowledge)
        self.belief_entropies = {"G'": self._epistemic_entropy()}

    # ── Unified Builder (Phase 4) ─────────────────────────────

    @classmethod
    def build(
        cls,
        env: EnvironmentProtocol,
        seed: int = 42,
        use_mlp: bool = False,
        use_continuous: bool = False,
        mlp_lr: float = 0.2,
        mlp_hidden_dim: int = 128,
        gprime_b_time: float = 0.020,
        action_b_time: float = 0.020,
        action_b_energy: float = 2.0,
        metrics_store: Optional["MetricsStore"] = None,
        observability_store: Optional["ObservabilityStore"] = None,
        interventions: Optional[InterventionConfig] = None,
        trace_collector: Optional[TraceCollector] = None,
        noise_profile: Optional[str] = None,
        noise_intensity: float = 0.1,
    ) -> CognitiveCycle:
        """Build a fully-configured cognitive cycle for any EnvironmentProtocol.

        This is the canonical builder — accepts any environment implementing
        EnvironmentProtocol and wires up all PHCA modules with appropriate
        dimensions based on env.get_state_dim() and env.action_space_size.

        Args:
            env: An environment implementing EnvironmentProtocol
                (e.g., GridWorld, MuJoCoSimpleEnv).
            seed: Random seed for reproducibility.
            use_mlp: If True, use the pure-NumPy MLP world model.
            use_continuous: If True (and use_mlp=False), use Gaussian
                CPDs with analytic inference. If both False, uses discrete
                binary G' with pgmpy exact inference.
            mlp_lr: Learning rate for MLP (default 0.1; use 0.05 for
                smooth continuous targets like MuJoCo).
            mlp_hidden_dim: MLP hidden layer size (default 128).
            gprime_b_time: RBTA time bound for G' module in seconds
                (default 0.020; use 0.080 for MuJoCo with physics sim overhead).
            action_b_time: RBTA time bound for ACTION module in seconds
            action_b_energy: RBTA energy bound for ACTION module
                (default 0.020; use 0.050 for MuJoCo).
            metrics_store: Optional MetricsStore for live monitoring.
            noise_profile: Sensor noise profile ("gaussian", "dropout",
                "drift", "salt_pepper"). None = disabled.
            noise_intensity: Noise level 0.0-1.0 (default 0.1).

        Returns:
            Configured CognitiveCycle instance.
        """
        state_dim = env.get_state_dim()
        action_dim = env.action_space_size

        sanitizer = ASISanitizer(
            sensor_dim=state_dim, v_max=100.0, epsilon_confidence=0.01,
        )
        m1 = M1SensoryBuffer(sensor_dim=state_dim)
        m2 = M2WorkingMemory(capacity=7)

        if use_mlp:
            gprime = WorldModelMLP(
                state_dim=state_dim,
                action_dim=action_dim,
                hidden_dim=mlp_hidden_dim,
                seed=seed,
                lr=mlp_lr,
            )
        elif use_continuous:
            from phca.world_model.graph import WorldModelGPrime

            gprime = WorldModelGPrime.build_gaussian_grid(
                state_dim=state_dim,
                action_dim=action_dim,
                transition_std=0.5,
                seed=seed,
            )
        else:
            # Phase 3.1: Discrete binary G' with pgmpy exact inference
            from phca.world_model.graph import StateNode, TemporalEdge, WorldModelGPrime

            gprime = WorldModelGPrime(
                state_dim=state_dim,
                action_dim=action_dim,
                seed=seed,
            )
            for i in range(min(state_dim, 10)):
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
                gprime.add_temporal_edge(TemporalEdge(
                    source=name_t, target=name_t1, lag=1,
                ))

        engine = PredictionEngine(gprime)
        peu = PredictionErrorUnit()
        tspl = TSPL(seed=seed)
        tspl.init_parameters("gprime", (state_dim,))
        iv = interventions or InterventionConfig()
        if not iv.enable_tspl:
            tspl.configs[StreamID.P_STREAM].enabled = False
        rbta = RBTAEnforcer(module_bounds=DEFAULT_MODULE_BOUNDS)

        rbta.update_bounds(
            "ACTION",
            ResourceBounds(B_time=action_b_time, B_mem=10_000, B_energy=action_b_energy),
        )
        for module_id, bounds in grid_rbta_bounds(
            state_dim=state_dim,
            gprime=gprime,
            b_time=gprime_b_time,
            action_b_time=action_b_time,
        ).items():
            rbta.update_bounds(module_id, bounds)

        mdim = MDIM(state_dim=state_dim)
        attention = Attention()
        adaptive_controller = AdaptiveParameterController()
        hpm_validator = HPMValidator()
        m3 = M3EpisodicMemory(
            state_dim=state_dim, action_dim=action_dim,
        )
        consolidation = ConsolidationScheduler(
            m3=m3, state_dim=state_dim,
            consolidation_interval=10, max_facts_per_cycle=50,
        )

        noise_injector: Optional[NoiseInjector] = None
        if noise_profile is not None:
            noise_injector = NoiseInjector(
                sensor_dim=state_dim,
                initial_profile=noise_profile,
                initial_intensity=noise_intensity,
                seed=seed + 100,
            )

        cycle = cls(
            sanitizer=sanitizer, m1=m1, m2=m2, gprime=gprime,
            engine=engine, peu=peu, tspl=tspl, rbta=rbta,
            mdim=mdim, attention=attention,
            adaptive_controller=adaptive_controller,
            hpm_validator=hpm_validator,
            consolidation=consolidation, env=env,
            state_dim=state_dim,
            metrics_store=metrics_store,
            observability_store=observability_store,
            interventions=iv,
            trace_collector=trace_collector,
        )
        if noise_injector is not None:
            cycle._noise_injector = noise_injector
        return cycle

    # ── Backward-Compatible Builders ───────────────────────

    @classmethod
    def build_for_mujoco(
        cls,
        env_name: str = "InvertedPendulum-v5",
        seed: int = 42,
        use_mlp: bool = True,
        use_continuous: bool = True,
        render_mode: Optional[str] = None,
        enable_camera: bool = False,
        metrics_store: Optional["MetricsStore"] = None,
        observability_store: Optional["ObservabilityStore"] = None,
        interventions: Optional[InterventionConfig] = None,
        trace_collector: Optional[TraceCollector] = None,
        noise_profile: Optional[str] = None,
        noise_intensity: float = 0.1,
    ) -> CognitiveCycle:
        """Build a cognitive cycle for a MuJoCo physics environment.

        Thin wrapper around build() that creates a MuJoCoSimpleEnv.
        Uses MuJoCo-appropriate defaults: MLP with LR=0.05, higher RBTA
        bounds for G' (0.120s), ACTION (0.080s, MPC only), and ENV (0.250s,
        physics step) on variable CI runners (D-128, D-131).

        Args:
            env_name: gymnasium MuJoCo environment ID.
            seed: Random seed.
            use_mlp: If True, use the pure-NumPy MLP world model.
            use_continuous: If True (and use_mlp=False), use Gaussian CPDs.
            render_mode: Legacy gym render mode (human only). Live dashboard
                camera uses ``enable_camera`` / ``mujoco.Renderer``.
            enable_camera: Use offscreen ``mujoco.Renderer`` for dashboard RGB.
            metrics_store: Optional MetricsStore for live monitoring.

        Returns:
            Configured CognitiveCycle instance.

        Raises:
            ImportError: If gymnasium is not installed.
        """
        from phca.environments.mujoco_env import MuJoCoSimpleEnv

        use_camera = bool(enable_camera or render_mode == "rgb_array")
        env = MuJoCoSimpleEnv(env_name=env_name, seed=seed,
                              render_mode=render_mode,
                              enable_camera=use_camera)

        if not use_mlp and not use_continuous:
            import warnings
            warnings.warn(
                "Discrete G' with continuous MuJoCo observations will "
                "quantise all values to 0/1. Use use_mlp=True or "
                "use_continuous=True for MuJoCo environments."
            )

        cycle = cls.build(
            env=env, seed=seed,
            use_mlp=use_mlp, use_continuous=use_continuous,
            mlp_lr=0.05,              # lower LR for smooth continuous targets
            gprime_b_time=0.120,      # MuJoCo + MLP learn headroom (D-128)
            action_b_time=0.080,      # MPC only (D-131; env.step → ENV)
            action_b_energy=4.0,      # Runtime-derived ACTION energy (time×50)
            metrics_store=metrics_store,
            observability_store=observability_store,
            interventions=interventions,
            trace_collector=trace_collector,
            noise_profile=noise_profile,
            noise_intensity=noise_intensity,
        )
        cycle.rbta.update_bounds(
            "ENV",
            ResourceBounds(B_time=0.250, B_mem=10_000, B_energy=12.5),
        )
        # MuJoCo live/replay runs add small but repeatable regulator overhead on
        # shared runners, especially around Reacher camera/continuous-control
        # sessions. Widen only the lightweight regulator modules here rather than
        # masking the main ACTION/ENV/G' bounds globally.
        cycle.rbta.update_bounds(
            "CR",
            ResourceBounds(B_time=0.010, B_mem=50_000, B_energy=5.0),
        )
        cycle.rbta.update_bounds(
            "ATTN",
            ResourceBounds(B_time=0.008, B_mem=20_000, B_energy=2.0),
        )
        cycle.rbta.update_bounds(
            "HPM",
            ResourceBounds(B_time=0.008, B_mem=50_000, B_energy=5.0),
        )
        return cycle

    @classmethod
    def build_for_env(
        cls,
        size: int = 5,
        seed: int = 42,
        state_dim: Optional[int] = None,
        use_continuous: bool = False,
        use_mlp: bool = False,
        obstacles: Optional[List[tuple]] = None,
        action_slip: float = 0.0,
        maze: bool = False,
        metrics_store: Optional["MetricsStore"] = None,
        observability_store: Optional["ObservabilityStore"] = None,
        interventions: Optional[InterventionConfig] = None,
        trace_collector: Optional[TraceCollector] = None,
        noise_profile: Optional[str] = None,
        noise_intensity: float = 0.1,
    ) -> CognitiveCycle:
        """Build a cognitive cycle for GridWorld.

        Thin wrapper around build() that creates a GridWorld environment.
        Uses GridWorld-appropriate defaults: discrete G' (default),
        standard RBTA bounds.

        Args:
            size: GridWorld size (5, 10, or 20).
            seed: Random seed.
            state_dim: Override state dimensionality (default: auto from env).
            use_continuous: If True, use Gaussian CPDs.
            use_mlp: If True, use the pure-NumPy MLP world model.
            obstacles: Wall positions. If None, generates random walls.
            metrics_store: Optional MetricsStore for live monitoring.

        Returns:
            Configured CognitiveCycle instance.
        """
        env = GridWorld(
            size=size,
            obstacles=obstacles if obstacles is not None else [],
            seed=seed,
            action_slip=action_slip,
            maze=maze,
        )
        actual_state_dim = state_dim or env.get_state_dim()
        # Note: build() uses env.get_state_dim() internally, so if state_dim
        # override is provided, we need to adjust. Pass use_mlp to trigger
        # MLP bounds update if needed.
        cycle = cls.build(
            env=env, seed=seed,
            use_mlp=use_mlp, use_continuous=use_continuous,
            gprime_b_time=0.050 if use_mlp else 0.020,
            metrics_store=metrics_store,
            observability_store=observability_store,
            interventions=interventions,
            trace_collector=trace_collector,
            noise_profile=noise_profile,
            noise_intensity=noise_intensity,
        )
        # If state_dim was overridden, update the cycle's state_dim
        if state_dim is not None and state_dim != actual_state_dim:
            cycle.state_dim = state_dim
        return cycle
