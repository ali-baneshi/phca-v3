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
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import numpy as np

from phca.logging import logger, _log
from phca.config import (
    ASIStatus,
    GoalVector,
    ResourceBounds,
    StateVector,
    StreamID,
    DEFAULT_MODULE_BOUNDS,
    DiscreteSpace,
    ContinuousSpace,
)
from phca.asi.sanitizer import ASISanitizer
from phca.memory.m1_sensory import M1SensoryBuffer
from phca.memory.m2_working import M2WorkingMemory
from phca.memory.m3_episodic import M3EpisodicMemory
from phca.world_model.mlp import WorldModelMLP

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

if TYPE_CHECKING:
    from phca.monitoring.metrics_store import MetricsStore
    from phca.monitoring.observability import ObservabilityStore

# AF-005: Normalisation factor for FLOP-based energy signals.
# ~30M FLOPs (MLP forward at h=128, bs=32, ts=4) ≈ 0.5 on energy scale.
ENERGY_NORM_FLOPS = 60_000_000.0


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

        # Rolling window of prediction errors for temporal Φ approximation (P1-E fix)
        self._error_vol_window: List[float] = []
        self._error_vol_window_size: int = 20

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

        # RBTA enforcement carry-forward (P1-01): prior cycle action shapes next cycle.
        self._rbta_carry_action: EnforcerAction = EnforcerAction.CONTINUE
        self._rbta_skip_feedback: bool = False
        self._rbta_skip_consolidation: bool = False
        self._rbta_action_candidate_limit: Optional[int] = None

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
            # Per-cycle RBTA enforcement flags (reset; carry-forward applied below).
            self._rbta_skip_feedback = False
            self._rbta_skip_consolidation = False
            self._rbta_action_candidate_limit = None
            if self._rbta_carry_action == EnforcerAction.TERMINATE:
                self._rbta_skip_feedback = True
            elif self._rbta_carry_action == EnforcerAction.INTERRUPT:
                self._rbta_action_candidate_limit = 1

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
                # G1: NaN gate — clamp prediction to identity if degenerate
                if not np.all(np.isfinite(predicted.values)):
                    _log(logger, "warning", "cycle.prediction.nan_detected",
                         fallback="identity")
                    predicted = StateVector(
                        values=self.current_state.values.copy(),
                        precision=np.ones_like(self.current_state.precision) * 0.01,
                        timestamp=predicted.timestamp,
                    )
                    confidence = 0.0
                self.last_prediction = predicted
                metrics.prediction_confidence = confidence
            metrics.module_timings["prediction"] = (time.perf_counter() - t2) * 1000

            # Steps 8-13: MDIM + CR + ATTN + facts (before action — P0-1).
            # Uses prior-cycle prediction error; current obs + confidence.
            consol_stats = self.consolidation.get_stats()
            t_mdim = time.perf_counter()
            error_volatility = self._approximate_error_volatility()
            empowerment = self._estimate_empowerment()
            if self.observability_store is not None:
                self.last_empowerment = float(empowerment)

            if self.current_state is not None:
                self._relevant_facts = self.consolidation.get_relevant_facts(
                    self.current_state, n=5, min_confidence=0.3,
                )
                fact_confidence_mean = float(
                    np.mean([f.confidence for f in self._relevant_facts])
                ) if self._relevant_facts else 0.0
                fact_count = len(self._relevant_facts)
            else:
                self._relevant_facts = []
                fact_confidence_mean = 0.0
                fact_count = 0

            self._planning_grid = self._build_planning_wall_grid()

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
            self._cycle_flops = self._compute_cycle_flops()
            if self._cycle_flops > 0:
                energy_cost = max(0.01, min(1.0, self._cycle_flops / ENERGY_NORM_FLOPS))
            else:
                energy_cost = max(0.01, min(1.0, (time.perf_counter() - t_start) * 2.0))
            model_entropy = max(0.01, 1.0 - metrics.prediction_confidence)

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
                "size": self.env.size if hasattr(self.env, "size") else 0,
                "task_lock": self._task_lock,
                "goal_switch_boost": goal_switch_boost,
            }
            self.current_goal = self.mdim.generate_goal(mdim_context)
            metrics.module_timings["mdim"] = (time.perf_counter() - t_mdim) * 1000

            t_cr = time.perf_counter()
            T, eta, alpha = self.adaptive_controller.regulate(error_volatility)
            self.mdim.temperature = T
            self.tspl.configs[StreamID.P_STREAM].eta = eta
            self.attention.gumbel_temperature = alpha * 0.5
            metrics.module_timings["cr"] = (time.perf_counter() - t_cr) * 1000

            t_attn = time.perf_counter()
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

            # RBTA preflight: same-cycle TERMINATE/INTERRUPT before expensive action/feedback.
            preflight_action = self._rbta_preflight_check(metrics, hpm_spec)
            if preflight_action == EnforcerAction.TERMINATE:
                self._rbta_skip_feedback = True
                self._rbta_skip_consolidation = True
            elif preflight_action == EnforcerAction.INTERRUPT:
                self._rbta_action_candidate_limit = 1
                self._rbta_skip_consolidation = True

            # Step 9: Action selection + environment step
            t5 = time.perf_counter()
            if self._rbta_skip_feedback:
                action = self.env.stay_action
                self.last_candidate_scores = []
                self.last_candidate_rollouts = []
                self.last_action_rationale = {
                    "explored": False,
                    "eps": 0.0,
                    "goal_id": int(self.current_goal.drive_id) if self.current_goal else 1,
                    "continuous": False,
                    "best_score": None,
                    "k_candidates": 1,
                    "chosen_idx": int(action),
                    "task_lock": bool(self._task_lock),
                    "rbta_safe_mode": True,
                    "relevant_fact_ids": [f.fact_id for f in self._relevant_facts],
                }
            else:
                action = self._select_action()
            if self._is_continuous:
                action_vec_step = np.asarray(action, dtype=np.float32)
                obs, reward, terminal, info = self.env.step(action_vec_step)
            else:
                obs, reward, terminal, info = self.env.step(action)
            metrics.module_timings["action_selection"] = (time.perf_counter() - t5) * 1000

            # Step 5-7: PEU + TSPL + LEARN with correct action context.
            # PEU now compares the actual next_state against a prediction
            # conditioned on the action that was taken (not last_action
            # from the previous cycle). This makes the error meaningful for
            # learned world models like the MLP.
            if self.current_state is not None and not self._rbta_skip_feedback:
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
                # Observability v4: per-dim PEU breakdown (cheap, guarded).
                if self.observability_store is not None:
                    try:
                        diff = (next_state.values - corrected_prediction.values)
                        self.last_per_dim_peu = (
                            next_state.precision * (diff ** 2)
                        ).astype(np.float32)
                    except Exception:
                        self.last_per_dim_peu = None
                # Update confidence to reflect the corrected prediction
                if corrected_conf > metrics.prediction_confidence:
                    metrics.prediction_confidence = corrected_conf
                # Push error onto rolling Φ window (P1-E fix: temporal variance → Φ)
                self._error_vol_window.append(metrics.prediction_error)
                if len(self._error_vol_window) >= self._error_vol_window_size:
                    self._error_vol_window.pop(0)
                metrics.module_timings["peu"] = (time.perf_counter() - t3) * 1000

                # Step 7: TSPL P-Stream update
                # Compute MLP accuracy for TSPL sync (G-019) before TSPL update
                # so skill compilation uses actual MLP performance
                mlp_accuracy = None
                if isinstance(self.gprime, WorldModelMLP):
                    mlp_accuracy = self.gprime.get_prediction_accuracy(
                        self.current_state.values, action_vec, next_state.values
                    )
                t4 = time.perf_counter()
                tspl_gradient = None
                theta_new, _ = self.tspl.update(
                    StreamID.P_STREAM,
                    metrics.prediction_error,
                    self.current_state,
                    self.last_prediction,
                    gradient=tspl_gradient,
                    accuracy_override=mlp_accuracy,
                )
                # AF-002: Apply TSPL learned bias to MLP output (safe mode).
                # MLP.set_tspl_bias() clamps bias norm to ≤1.0 to prevent corruption.
                if isinstance(self.gprime, WorldModelMLP):
                    bias = theta_new.get("gprime")
                    if bias is not None:
                        self.gprime.set_tspl_bias(bias)
                metrics.module_timings["tspl"] = (time.perf_counter() - t4) * 1000

                # LEARN: update G' with observed transition, weighted by attention
                t_glearn = time.perf_counter()
                attn_weighted_error = metrics.prediction_error * float(np.mean(self._attention_weights))
                # Pass attention weights to MLP for per-dimension gradient modulation (A5 fix)
                if hasattr(self.gprime, '_attention_weights'):
                    self.gprime._attention_weights = self._attention_weights.copy()
                self.gprime.learn(
                    self.current_state, action_vec, next_state,
                    error=attn_weighted_error,
                )
                metrics.module_timings["gprime_learn"] = (time.perf_counter() - t_glearn) * 1000

                # Store episode in M3 episodic memory (Task B fix)
                m3_drive_id = self.current_goal.drive_id if self.current_goal else None
                self.consolidation.m3.store_episode(
                    state_before=self.current_state,
                    action_taken=action_vec,
                    state_after=next_state,
                    prediction_error=metrics.prediction_error,
                    confidence=metrics.prediction_confidence,
                    drive_id=m3_drive_id,
                    timestamp=self.cycle_count,
                )
                self._last_prediction_error = metrics.prediction_error

            if self._is_continuous:
                self.last_action = action_vec_step.copy()
                self.engine.update_action(self.last_action)
                metrics.action_taken = -1
                metrics.action_name = "continuous"
            else:
                self.last_action = np.zeros(self.env.action_space_size, dtype=np.float32)
                self.last_action[action] = 1.0
                self.engine.update_action(self.last_action)
                metrics.action_taken = action
                action_names = self.env.get_action_names()
                metrics.action_name = action_names[action]
            metrics.goal_reached = info.get("goal_reached", False)

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
            reg_b_time = hpm_bounds["B_time"] if hpm_bounds else 0.200
            reg_b_energy = hpm_bounds.get("B_energy", 10.0) if hpm_bounds else 10.0
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

            # Step 15: Logging — also populate dashboard fields
            metrics.drive_id = self.current_goal.drive_id if self.current_goal else 1
            metrics.skill_accuracy = self.tspl.skill_accuracy
            metrics.skill_compiled = self.tspl.skill_compiled
            metrics.fact_count = consol_stats.get("total_facts_stored", 0)
            metrics.episode_count = self.consolidation.m3.count() if hasattr(self, 'consolidation') else 0
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
            if len(self.metrics_history) > 5000:
                self.metrics_history = self.metrics_history[-5000:]

            # Steps 16-18: Consolidation (periodic E→S transfer)
            if self._rbta_skip_consolidation:
                consol_report = type("_Skip", (), {
                    "success": False,
                    "episodes_processed": 0,
                    "facts_generated": 0,
                    "duration_ms": 0.0,
                })()
                metrics.module_timings["consolidation"] = 0.0
            else:
                t_consol = time.perf_counter()
                consol_report = self.consolidation.step(self.cycle_count)
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

            # Step 20: (removed) Sleep-cycle check — A-008 fix. Consolidation
            # handled by Steps 16-18 every consolidation_interval=10 cycles.

        except Exception as e:
            _log(logger, "error", "cycle.step.error", cycle=self.cycle_count, error=str(e))
            self.cycle_count += 1
            raise

        return metrics

    def _select_action(self):
        """Select action using goal-directed planning with MDIM goal awareness.

        Returns an int (discrete space) or an np.ndarray (continuous space,
        Phase 6 / A2). The continuous branch is MPC-style: sample K candidate
        actions, predict each via G', pick the one whose predicted next state
        best matches the goal reference. Prediction/goal-driven, NOT RL.
        """
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
            self.last_action_rationale = {
                "explored": True, "eps": float(eps),
                "goal_id": int(goal_id), "continuous": False,
                "best_score": None,
                "k_candidates": int(self.env.action_space_size),
                "chosen_idx": pick,
                "task_lock": bool(self._task_lock),
                "relevant_fact_ids": [f.fact_id for f in self._relevant_facts],
            }
            self.last_candidate_scores = []
            self.last_candidate_rollouts = []
            return pick

        # D5 (Energy Efficiency): prefer STAY — skip under task-lock (P0-3)
        if goal_id == 5 and not self._task_lock:
            self.last_action_rationale = {
                "explored": False, "eps": float(eps),
                "goal_id": 5, "continuous": False,
                "best_score": None, "k_candidates": None,
                "note": "D5 energy: STAY",
                "chosen_idx": int(self.env.stay_action),
                "task_lock": bool(self._task_lock),
                "relevant_fact_ids": [f.fact_id for f in self._relevant_facts],
            }
            self.last_candidate_scores = []
            self.last_candidate_rollouts = []
            return self.env.stay_action

        # Task-lock: pure observed-greedy navigation (beats blended scorer on L3)
        if self._task_lock and hasattr(self.env, "agent_pos"):
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
            sparse_probe = long_horizon and on_goal and (self.cycle_count % 50 == 0)
            greedy_action, greedy_score, greedy_comp = self._select_greedy_grid_action(
                explore_ties=explore_ties,
                at_goal_explore=sparse_probe,
            )
            self.last_candidate_scores = []
            self.last_candidate_rollouts = []
            self.last_action_rationale = {
                "explored": False, "eps": 0.0,
                "goal_id": int(goal_id), "continuous": False,
                "best_score": float(greedy_score),
                "k_candidates": int(self.env.action_space_size),
                "chosen_idx": int(greedy_action),
                "task_lock": True,
                "greedy_fallback": True,
                "score_components": greedy_comp,
                "relevant_fact_ids": [f.fact_id for f in self._relevant_facts],
            }
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

        self.last_action_rationale = {
            "explored": False, "eps": float(eps),
            "goal_id": int(goal_id), "continuous": False,
            "best_score": float(best_score),
            "k_candidates": int(self.env.action_space_size),
            "chosen_idx": int(best_action),
            "task_lock": bool(self._task_lock),
            "score_components": best_components,
            "relevant_fact_ids": [f.fact_id for f in self._relevant_facts],
        }
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
        """
        space = self.action_space
        low, high = space.low, space.high
        K = 8
        if K * space.dim > 16:
            K = max(2, 16 // space.dim)

        rng = np.random.RandomState(self.cycle_count)
        eps = max(0.02, 0.10 * (1.0 - self.cycle_count / 500.0))
        if rng.random() < eps:
            self.last_action_rationale = {"explored": True, "eps": float(eps),
                                          "goal_id": None, "continuous": True,
                                          "best_score": None, "k_candidates": int(K)}
            self.last_candidate_scores = []
            self.last_candidate_rollouts = []
            return (low + (high - low) * rng.uniform(size=space.dim)).astype(np.float32)

        ref = getattr(self.env, "get_goal_reference", lambda: None)()
        ref = np.asarray(ref, dtype=np.float32) if ref is not None else None

        best_a, best_score, best_idx = None, -float("inf"), -1
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
        self.last_candidate_scores = cand_scores
        if _obs:
            for r in _rollouts:
                if best_a is not None and np.array_equal(r["action"], best_a):
                    r["chosen"] = True
            self.last_candidate_rollouts = _rollouts[:8]
        else:
            self.last_candidate_rollouts = []
        self.last_action_rationale = {"explored": False, "eps": float(eps),
                                      "goal_id": None, "continuous": True,
                                      "best_score": float(best_score) if best_a is not None else None,
                                      "k_candidates": int(K),
                                      "chosen_idx": int(best_idx)}
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
        if goal_pos is None or not hasattr(env, "agent_pos"):
            return 0.5
        n_pos = env.size * env.size if hasattr(env, 'size') else 25
        pred_agent = predicted.values[:n_pos]
        peak = int(np.argmax(pred_agent))
        max_val = float(pred_agent[peak])
        if max_val < 0.05:
            return 0.5
        g_row, g_col = goal_pos
        p_row, p_col = peak // env.size, peak % env.size
        dist = abs(p_row - g_row) + abs(p_col - g_col)
        max_dist = 2 * (env.size - 1) if hasattr(env, 'size') else 8
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
        """One-step Manhattan controller using observed goal/walls (fair vs greedy_observed)."""
        env = self.env
        goal_pos = env.get_goal_position()
        if goal_pos is None:
            return env.stay_action, 0.0, {}
        agent_pos = env.agent_pos
        grid = env.grid
        best_action = env.stay_action
        best_distance = abs(agent_pos[0] - goal_pos[0]) + abs(agent_pos[1] - goal_pos[1])
        best_unvisited = False
        action_names = env.get_action_names()
        deltas = {
            "MOVE_N": (-1, 0), "MOVE_S": (1, 0),
            "MOVE_E": (0, 1), "MOVE_W": (0, -1), "STAY": (0, 0),
        }
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
                action_names = env.get_action_names()
                dr, dc = {
                    "MOVE_N": (-1, 0), "MOVE_S": (1, 0),
                    "MOVE_E": (0, 1), "MOVE_W": (0, -1), "STAY": (0, 0),
                }[action_names[action_idx]]
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

    def _approximate_error_volatility(self) -> float:
        """Compute prediction-error volatility from recent error history.

        Uses a coefficient-of-variance heuristic: vol = min(1.0, std(window) / (mean(window) + ε)).
        This measures how much the prediction error fluctuates over recent cycles.
        High volatility = error varies significantly (system exploring/learning).
        Low volatility = error is stable (system converged or stuck).

        Note: This was previously called "_approximate_phi" and claimed to measure
        integrated information (IIT Φ). It does NOT — it measures prediction-error
        volatility. Renamed in Phase 3.3 gap audit (G-001) to accurately reflect
        what it measures.

        Returns:
            Float in (0.0, 1.0] representing prediction-error volatility.
        """
        if len(self._error_vol_window) < 3:
            return 0.5  # not enough samples for meaningful variance

        errors = self._error_vol_window[-10:]  # use last 10 for responsiveness
        mean_err = float(np.mean(errors))
        std_err = float(np.std(errors))

        if mean_err < 1e-8:
            return 0.1  # near-zero error → stable

        cv = std_err / mean_err
        vol = min(1.0, cv)
        return float(np.clip(vol, 0.1, 0.99))

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
        reg_b_time = hpm_bounds["B_time"] if hpm_bounds else 0.200
        reg_b_energy = hpm_bounds.get("B_energy", 10.0) if hpm_bounds else 10.0
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
        self.memory_log = {
            "ASI": max(1_000, sd * 4 * 2),
            "WM": max(1_000, sd * 4 * 7),
            "G'": max(10_000, sd * sd * 4 * 5),
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
        # Belief entropy from prediction error variance (A3: Incomplete Knowledge)
        if len(self.metrics_history) >= 5:
            recent_errs = [m.prediction_error for m in self.metrics_history[-10:]]
            err_var = float(np.var(recent_errs)) if len(recent_errs) > 1 else 0.5
            entropy_val = min(1.0, max(0.01, err_var * 10.0))
        else:
            entropy_val = 0.5
        self.belief_entropies = {"G'": entropy_val}

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
        metrics_store: Optional["MetricsStore"] = None,
        observability_store: Optional["ObservabilityStore"] = None,
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
                (default 0.020; use 0.050 for MuJoCo).
            metrics_store: Optional MetricsStore for live monitoring.

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
        rbta = RBTAEnforcer(module_bounds=DEFAULT_MODULE_BOUNDS)

        if use_mlp:
            rbta.update_bounds(
                "G'", ResourceBounds(B_time=gprime_b_time, B_mem=500_000, B_energy=50.0),
            )
        rbta.update_bounds(
            "ACTION", ResourceBounds(B_time=action_b_time, B_mem=10_000, B_energy=2.0),
        )

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

        return cls(
            sanitizer=sanitizer, m1=m1, m2=m2, gprime=gprime,
            engine=engine, peu=peu, tspl=tspl, rbta=rbta,
            mdim=mdim, attention=attention,
            adaptive_controller=adaptive_controller,
            hpm_validator=hpm_validator,
            consolidation=consolidation, env=env,
            state_dim=state_dim,
            metrics_store=metrics_store,
            observability_store=observability_store,
        )

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
    ) -> CognitiveCycle:
        """Build a cognitive cycle for a MuJoCo physics environment.

        Thin wrapper around build() that creates a MuJoCoSimpleEnv.
        Uses MuJoCo-appropriate defaults: MLP with LR=0.05, higher RBTA
        bounds for G' (0.080s) and ACTION (0.050s) to account for physics
        simulation overhead.

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

        return cls.build(
            env=env, seed=seed,
            use_mlp=use_mlp, use_continuous=use_continuous,
            mlp_lr=0.05,              # lower LR for smooth continuous targets
            gprime_b_time=0.080,      # MuJoCo physics sim overhead
            action_b_time=0.050,      # MuJoCo step() overhead
            metrics_store=metrics_store,
            observability_store=observability_store,
        )

    @classmethod
    def build_for_env(
        cls,
        size: int = 5,
        seed: int = 42,
        state_dim: Optional[int] = None,
        use_continuous: bool = False,
        use_mlp: bool = False,
        obstacles: Optional[List[tuple]] = None,
        metrics_store: Optional["MetricsStore"] = None,
        observability_store: Optional["ObservabilityStore"] = None,
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
        )
        actual_state_dim = state_dim or env.get_state_dim()
        # Note: build() uses env.get_state_dim() internally, so if state_dim
        # override is provided, we need to adjust. Pass use_mlp to trigger
        # MLP bounds update if needed.
        cycle = cls.build(
            env=env, seed=seed,
            use_mlp=use_mlp, use_continuous=use_continuous,
            gprime_b_time=0.050 if use_mlp else 0.020,  # MLP needs wider G' bound
            metrics_store=metrics_store,
            observability_store=observability_store,
        )
        # If state_dim was overridden, update the cycle's state_dim
        if state_dim is not None and state_dim != actual_state_dim:
            cycle.state_dim = state_dim
        return cycle
