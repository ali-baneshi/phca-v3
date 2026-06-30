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
from phca.memory.m3_episodic import M3EpisodicMemory
from phca.world_model.graph import WorldModelGPrime, StateNode, TemporalEdge
from phca.world_model.mlp import WorldModelMLP

from phca.prediction.engine import PredictionEngine
from phca.prediction.error_unit import PredictionErrorUnit
from phca.learning.tspl import TSPL
from phca.regulation.rbta_enforcer import RBTAEnforcer
from phca.motivation.mdim import MDIM
from phca.attention.attention import Attention
from phca.regulation.pid_controller import CriticalityRegulator
from phca.hpm.parser import HPMValidator
from phca.consolidation.scheduler import ConsolidationScheduler
from environments.grid_world import GridWorld
from environments.protocol import EnvironmentProtocol


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
        gprime: WorldModelGPrime,
        engine: PredictionEngine,
        peu: PredictionErrorUnit,
        tspl: TSPL,
        rbta: RBTAEnforcer,
        mdim: MDIM,
        attention: Attention,
        criticality_regulator: CriticalityRegulator,
        hpm_validator: HPMValidator,
        consolidation: ConsolidationScheduler,
        env: EnvironmentProtocol,
        state_dim: int,
        rbta_bounds: Optional[Dict[str, ResourceBounds]] = None,
        metrics_store: Optional["MetricsStore"] = None,
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
        self.consolidation = consolidation
        self.env = env
        self.state_dim = state_dim

        self.rbta_bounds = rbta_bounds or DEFAULT_MODULE_BOUNDS.copy()

        self.cycle_count: int = 0
        self.metrics_store = metrics_store

        self.current_state: Optional[StateVector] = None
        self.current_goal: GoalVector = GoalVector(
            drive_id=1, target_state=None, tolerance=0.1,
            creation_cycle=0, priority=1.0,
        )
        self.last_action: np.ndarray = np.zeros(env.action_space_size, dtype=np.float32)
        self.last_prediction: Optional[StateVector] = None
        self.sensor_failure_count: int = 0
        self.asi_failure_limit: int = self.sanitizer.asi_failure_limit

        self.runtime_log: Dict[str, float] = {}
        self.memory_log: Dict[str, float] = {}
        self.energy_log: Dict[str, float] = {}
        self.belief_entropies: Dict[str, float] = {}

        self.metrics_history: List[CycleMetrics] = []

        # Rolling window of prediction errors for temporal Φ approximation (P1-E fix)
        self._phi_error_window: List[float] = []
        self._phi_window_size: int = 20

        # Attention weights for modulating G'.learn() (Issue #4 fix)
        self._attention_weights: np.ndarray = np.ones(self.state_dim, dtype=np.float32)

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
                # G1: NaN gate — clamp prediction to identity if degenerate
                if not np.all(np.isfinite(predicted.values)):
                    _log(logger, "warning", "cycle.prediction.nan_detected",
                         fallback="identity")
                    predicted = StateVector(
                        values=self.current_state.values.copy(),
                        precision=np.ones_like(self.current_state.precision) * 0.01,
                        timestamp=predicted.timestamp,
                        grounding_level=predicted.grounding_level,
                    )
                    confidence = 0.0
                self.last_prediction = predicted
                metrics.prediction_confidence = confidence
            metrics.module_timings["prediction"] = (time.perf_counter() - t2) * 1000

            # (Step 5-6: PEU — deferred after env.step for correct action context)

            # (Step 7: TSPL — deferred after PEU)

            # (Step 8: Attention moved after MDIM/CR to use CR-controlled parameters)

            # Step 9a: Action selection + environment step
            t5 = time.perf_counter()
            action_idx = self._select_action()
            obs, reward, terminal, info = self.env.step(action_idx)
            metrics.module_timings["action_selection"] = (time.perf_counter() - t5) * 1000

            # Step 5-7: PEU + TSPL + LEARN with correct action context.
            # PEU now compares the actual next_state against a prediction
            # conditioned on the action that was taken (not last_action
            # from the previous cycle). This makes the error meaningful for
            # learned world models like the MLP.
            if self.current_state is not None:
                action_vec = np.zeros(self.env.action_space_size, dtype=np.float32)
                action_vec[action_idx] = 1.0
                next_state = StateVector(
                    values=obs.astype(np.float32),
                    precision=np.ones(self.state_dim, dtype=np.float32),
                    timestamp=float(self.cycle_count),
                    grounding_level=1,
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
                # Update confidence to reflect the corrected prediction
                if corrected_conf > metrics.prediction_confidence:
                    metrics.prediction_confidence = corrected_conf
                # Push error onto rolling Φ window (P1-E fix: temporal variance → Φ)
                self._phi_error_window.append(metrics.prediction_error)
                if len(self._phi_error_window) >= self._phi_window_size:
                    self._phi_error_window.pop(0)
                metrics.module_timings["peu"] = (time.perf_counter() - t3) * 1000

                # Step 7: TSPL P-Stream update
                t4 = time.perf_counter()
                self.tspl.update(
                    StreamID.P_STREAM,
                    metrics.prediction_error,
                    self.current_state,
                    self.last_prediction,
                )
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

            self.last_action = np.zeros(self.env.action_space_size, dtype=np.float32)
            self.last_action[action_idx] = 1.0
            self.engine.update_action(self.last_action)

            metrics.action_taken = action_idx
            action_names = self.env.get_action_names()
            metrics.action_name = action_names[action_idx]
            metrics.goal_reached = info.get("goal_reached", False)

            if terminal:
                self.env.reset()
                self.gprime.reset()
                self.sanitizer.reset()  # clear stale precision across episode boundaries (N6)

            # Steps 10-13: MDIM + CR + ATTN + HPM (Phase 3.2)
            # Compute Φ approximation from module states (replaces synthetic decay)
            t_mdim = time.perf_counter()
            phi_current = self._approximate_phi()
            # Estimate empowerment from prediction confidence spread across actions
            empowerment = self._estimate_empowerment()
            # Gather consolidation facts from prev cycle's E→S transfer (P1-D fix)
            consol_stats = self.consolidation.get_stats()
            total_facts = consol_stats.get("total_facts_stored", 0)
            # Get relevant facts for the current state to bias goals (A3 fix)
            if self.current_state is not None:
                relevant_facts = self.consolidation.get_relevant_facts(
                    self.current_state, n=5, min_confidence=0.3,
                )
                fact_confidence_mean = float(np.mean([f.confidence for f in relevant_facts])) if relevant_facts else 0.0
                fact_count = len(relevant_facts)
            else:
                relevant_facts = []
                fact_confidence_mean = 0.0
                fact_count = 0
            mdim_context = {
                "prediction_error": metrics.prediction_error,
                "phi_criticality": phi_current,
                "skill_accuracy": self.tspl.skill_accuracy,
                "model_entropy": 0.5 - self.cycle_count * 0.001,
                "energy_cost": max(0.01, min(1.0, (time.perf_counter() - t_start) * 2.0)),
                "cycle": self.cycle_count,
                "prediction_confidence": metrics.prediction_confidence,
                "empowerment": empowerment,

                "consolidation_facts": total_facts,
                "fact_confidence_mean": fact_confidence_mean,
                "fact_count": fact_count,
            }
            self.current_goal = self.mdim.generate_goal(mdim_context)
            metrics.module_timings["mdim"] = (time.perf_counter() - t_mdim) * 1000

            # CR regulates and returns (T, eta, alpha) — forward to downstream modules
            t_cr = time.perf_counter()
            T, eta, alpha = self.criticality_regulator.regulate(phi_current)
            self.mdim.temperature = T
            self.tspl.configs[StreamID.P_STREAM].eta = eta
            self.attention.gumbel_temperature = alpha * 0.5
            metrics.module_timings["cr"] = (time.perf_counter() - t_cr) * 1000

            t_attn = time.perf_counter()
            attention_chunks = self.attention.select(self.m2.chunks, self.current_goal)
            for chunk in attention_chunks:
                self.attention.update_precision(
                    chunk.chunk_id, metrics.prediction_error,
                )
            # Wire attention weights into learning: high-salience chunks get more weight
            if attention_chunks:
                weights = np.array([c.salience for c in attention_chunks])
                w_sum = weights.sum()
                if w_sum > 1e-8:
                    weights = weights / w_sum
                else:
                    weights = np.ones_like(weights) / max(len(weights), 1)
                # Pad/truncate to state_dim for per-dimension weight
                if len(weights) >= self.state_dim:
                    self._attention_weights = weights[:self.state_dim]
                else:
                    # Repeat weights to match state_dim
                    reps = int(np.ceil(self.state_dim / max(len(weights), 1)))
                    self._attention_weights = np.tile(weights, reps)[:self.state_dim]
            else:
                self._attention_weights = np.ones(self.state_dim, dtype=np.float32)
            metrics.module_timings["attn"] = (time.perf_counter() - t_attn) * 1000

            t_hpm = time.perf_counter()
            hpm_spec = {
                "type": "SEQUENCE", "id": "cognitive_cycle",
                "children": [
                    {"type": "ASI_Input", "id": "ASI", "dim": self.state_dim, "grounding_level": 1},
                    {"type": "SEQUENCE", "id": "prediction_block",
                     "children": [
                         "WM",
                         {"type": "Predict", "id": "G'_PE", "horizon": 1},
                         "PEU",
                         "TSPL-P",
                     ]},
                    {"type": "PARALLEL", "id": "regulation_block",
                     "children": ["MDIM", "CR", "ATTN", "HPM"]},
                    "CYCLE",
                ],
            }
            metrics.module_timings["hpm"] = (time.perf_counter() - t_hpm) * 1000

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
                    "ASI",       # Step 0: sanitization
                    "WM",        # Step 1: memory write
                    "PE",        # Steps 2-4: prediction
                    "PEU",       # Steps 5-6: error computation
                    "TSPL-P",    # Step 7: P-Stream update
                    {
                        "type": "PARALLEL", "id": "regulation_block",
                        "children": ["MDIM", "CR", "ATTN", "HPM"],
                        "bounds": {"B_time": reg_b_time, "B_energy": reg_b_energy},
                    },
                    "CYCLE",     # Step 19: increment + logging
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
            metrics.module_timings["rbta"] = (time.perf_counter() - t6) * 1000

            # Step 15: Logging — also populate dashboard fields
            metrics.drive_id = self.current_goal.drive_id if self.current_goal else 1
            metrics.skill_accuracy = self.tspl.skill_accuracy
            metrics.skill_compiled = self.tspl.skill_compiled
            metrics.fact_count = consol_stats.get("total_facts_stored", 0)
            metrics.episode_count = self.consolidation.m3.count() if hasattr(self, 'consolidation') else 0
            self.metrics_history.append(metrics)
            if self.metrics_store is not None:
                self.metrics_store.push(metrics)
            if len(self.metrics_history) > 10000:
                self.metrics_history = self.metrics_history[-5000:]

            # Steps 16-18: Consolidation (periodic E→S transfer)
            t_consol = time.perf_counter()
            consol_report = self.consolidation.step(self.cycle_count)
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
            metrics.module_timings["consolidation"] = (
                time.perf_counter() - t_consol
            ) * 1000
            self.runtime_log["CONSOL"] = metrics.module_timings["consolidation"] / 1000.0
            # Recompute energy_log to reflect actual CONSOL runtime
            consol_energy = max(0.1, min(10.0, self.runtime_log["CONSOL"] * 50.0))
            self.energy_log["CONSOL"] = consol_energy

            # Step 19: Increment cycle counter
            self.cycle_count += 1

            # Step 20: (removed) Sleep-cycle check — not needed. Consolidation
            # is already handled by Steps 16-18 every consolidation_interval=10
            # cycles. The 50-cycle sleep cycle duplicated this work. (A-008 fix)

        except Exception as e:
            _log(logger, "error", "cycle.step.error", cycle=self.cycle_count, error=str(e))
            self.cycle_count += 1
            raise

        return metrics

    def _select_action(self) -> int:
        """Select action using goal-directed planning with MDIM goal awareness.

        Uses continuous distance gain rather than discrete goal alignment,
        with ε-greedy exploration and drive-appropriate scoring.

        The blend depends on the MDIM goal's drive_id:
          D1/D3 (error/competence): distance gain + confidence
          D2/D4 (exploration): uncertainty + distance gain
          D5 (energy): STAY
          D6 (empowerment): state-space alignment
        """
        if self.current_state is None:
            return self.env.stay_action

        goal = self.current_goal
        goal_id = goal.drive_id if goal else 1
        target = goal.target_state if goal else None

        # ε-greedy exploration (5% random action)
        rng = np.random.RandomState(self.cycle_count)
        if rng.random() < 0.05:
            return int(rng.randint(0, self.env.action_space_size))

        # D5 (Energy Efficiency): prefer STAY
        if goal_id == 5:
            return self.env.stay_action

        best_action = self.env.stay_action
        best_score = -float("inf")
        action_confidences = []

        for action_idx in range(self.env.action_space_size):
            action = np.zeros(self.env.action_space_size, dtype=np.float32)
            action[action_idx] = 1.0
            self.engine.update_action(action)

            try:
                predicted, confidence = self.engine.predict(
                    self.current_state, horizon=1,
                )
                action_confidences.append(confidence)

                # Continuous distance gain (0 = toward goal, 1 = away)
                distance_gain = self._compute_distance_gain(action_idx)

                # D1/D3: strongly prioritize reducing distance, confidence secondary.
                # MDIM target_state alignment adds drive-specific bias (G5 fix).
                if goal_id in (1, 3):
                    base_score = (1.0 - distance_gain) * 0.6 + min(confidence, 1.0) * 0.2
                    if target is not None:
                        alignment = self._state_space_alignment(predicted, target)
                        score = base_score + alignment * 0.2
                    else:
                        score = base_score
                elif goal_id in (2, 4):
                    # D2/D4: seek uncertainty, penalize moving away.
                    # MDIM target_state alignment adds drive-specific bias (G5 fix).
                    base_score = (1.0 - min(confidence, 1.0)) * 0.5 + (1.0 - distance_gain) * 0.2
                    if target is not None:
                        alignment = self._state_space_alignment(predicted, target)
                        score = base_score + alignment * 0.3
                    else:
                        score = base_score
                else:
                    # D6 (Empowerment): state-space alignment only
                    state_align = self._state_space_alignment(predicted, target)
                    score = state_align

                if score > best_score:
                    best_score = score
                    best_action = action_idx
            except Exception as e:
                action_confidences.append(0.0)
                _log(logger, "warning", "action_selection.predict_failed",
                     error=str(e))
                continue

        # Cache confidences for _estimate_empowerment (avoids duplicate 5× predict)
        self._cached_confidences = action_confidences

        return best_action

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
            # GridWorld-specific: compute Manhattan distance gain
            if hasattr(env, "grid") and hasattr(env, "WALL"):
                action_names = env.get_action_names()
                dr, dc = {
                    "MOVE_N": (-1, 0), "MOVE_S": (1, 0),
                    "MOVE_E": (0, 1), "MOVE_W": (0, -1), "STAY": (0, 0),
                }[action_names[action_idx]]
                new_row = env.agent_pos[0] + dr
                new_col = env.agent_pos[1] + dc

                if not (0 <= new_row < env.size and 0 <= new_col < env.size):
                    return 1.0
                if env.grid[new_row, new_col] == env.WALL:
                    return 1.0

                g_row, g_col = goal_pos
                current_dist = abs(env.agent_pos[0] - g_row) + abs(env.agent_pos[1] - g_col)
                new_dist = abs(new_row - g_row) + abs(new_col - g_col)

                if current_dist == 0:
                    return 0.0

                if action_idx == 4 and new_dist == current_dist:
                    return 0.7

                gain = (current_dist - new_dist) / current_dist
                return float(np.clip((1.0 - gain) / 2.0, 0.0, 1.0))
        return 0.5

    def _estimate_empowerment(self) -> float:
        """Estimate empowerment from cached action confidences.

        Uses confidences stored by _select_action to compute variance
        across actions — higher spread = more discriminative actions.
        No duplicate predictions needed (cache avoids 5× predict loop).

        Phase 3.3+: Full I(state_{t+1}; a_t | state_t) computation.

        Returns:
            Float in [0.0, 1.0] estimating empowerment.
        """
        confidences = getattr(self, '_cached_confidences', None)
        if not confidences or self.current_state is None:
            return 0.3

        # Restore engine action after action selection's predict loop
        if self.last_action is not None:
            self.engine.update_action(self.last_action)

        empowerment = float(np.std(confidences))
        return float(np.clip(empowerment, 0.0, 1.0))

    def _approximate_phi(self) -> float:
        """Approximate Φ (integrated information) from prediction error temporal variance.

        Uses a practical heuristic: Φ = min(1.0, std(window) / (mean(window) + ε)).
        This is the coefficient of variation of prediction error over the last N cycles.

        When the system is ordered (stable prediction error), the CV is low → low Φ.
        When the system is at criticality (error fluctuates between low and high),
        the CV is high → high Φ.

        This replaces the Phase 3.1 approach that computed correlation between
        WM and PE vectors (always near 1.0 with only 2 samples — meaningless).

        Phase 3.3+: Full IIT Φ computation over bipartitions.

        Returns:
            Float in (0.0, 1.0] approximating integrated information.
        """
        if len(self._phi_error_window) < 3:
            return 0.5  # not enough samples for meaningful variance

        errors = self._phi_error_window[-10:]  # use last 10 for responsiveness
        mean_err = float(np.mean(errors))
        std_err = float(np.std(errors))

        if mean_err < 1e-8:
            return 0.1  # near-zero error → ordered system

        # Coefficient of variation: high spread = critical, low spread = ordered
        cv = std_err / mean_err
        phi = min(1.0, cv)
        return float(np.clip(phi, 0.1, 0.99))

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
        # Energy estimates from actual module runtimes (scaled to match original magnitude)
        # TODO (Phase 4): Replace runtime_s * 50.0 with actual FLOP-based estimate:
        #   - MLP: 3 * hidden_dim^2 + 2 * hidden_dim * state_dim FLOPs/cycle
        #   - Gaussian G': O(n^3) for n = state_dim
        #   - SQLite M3: ~1000 * rows_written
        # For now, scale factor 50.0 keeps values in 0.1-10.0 range (A-007 fix).
        self.energy_log = {}
        for mod, runtime_s in self.runtime_log.items():
            self.energy_log[mod] = max(0.1, min(10.0, runtime_s * 50.0))
        # Fill any missing standard modules at realistic baseline
        baseline = {"ASI": 2.0, "WM": 1.0, "G'": 5.0, "CONSOL": 0.5}
        for mod, val in baseline.items():
            if mod not in self.energy_log:
                self.energy_log[mod] = val
        # Belief entropy from prediction error variance (A3: Incomplete Knowledge)
        if len(self.metrics_history) >= 5:
            recent_errs = [m.prediction_error for m in self.metrics_history[-10:]]
            err_var = float(np.var(recent_errs)) if len(recent_errs) > 1 else 0.5
            entropy_val = min(1.0, max(0.01, err_var * 10.0))
        else:
            entropy_val = 0.5
        self.belief_entropies = {"G'": entropy_val}

    @classmethod
    def build_for_mujoco(
        cls,
        env_name: str = "InvertedPendulum-v5",
        seed: int = 42,
        use_mlp: bool = True,
        use_continuous: bool = True,
        metrics_store: Optional["MetricsStore"] = None,
    ) -> CognitiveCycle:
        """Build a cognitive cycle for a MuJoCo physics environment.

        Creates a MuJoCoSimpleEnv wrapper for the given gymnasium MuJoCo
        environment ID and wires up all PHCA modules with appropriate
        dimensions and RBTA bounds.

        Args:
            env_name: gymnasium MuJoCo environment ID (e.g.
                'InvertedPendulum-v5', 'Pendulum-v1', 'Reacher-v5').
            seed: Random seed.
            use_mlp: If True, use the pure-NumPy MLP world model.
                Recommended for continuous MuJoCo observations.
            use_continuous: If True (and use_mlp=False), use Gaussian
                CPDs with analytic inference. If both use_mlp and
                use_continuous are False, a discrete binary G' is used,
                which is NOT recommended for continuous MuJoCo observations.

        Returns:
            Configured CognitiveCycle instance.

        Raises:
            ImportError: If gymnasium is not installed.
        """
        from environments.mujoco_env import MuJoCoSimpleEnv

        env = MuJoCoSimpleEnv(env_name=env_name, seed=seed)
        state_dim = env.get_state_dim()

        sanitizer = ASISanitizer(
            sensor_dim=state_dim, v_max=100.0, epsilon_confidence=0.01,
        )
        m1 = M1SensoryBuffer(sensor_dim=state_dim)
        m2 = M2WorkingMemory(capacity=7)

        if use_mlp:
            # MLP with slightly lower LR for smooth continuous targets
            gprime = WorldModelMLP(
                state_dim=state_dim,
                action_dim=env.action_space_size,
                seed=seed,
                lr=0.05,
            )
        elif use_continuous:
            gprime = WorldModelGPrime.build_gaussian_grid(
                state_dim=state_dim,
                action_dim=env.action_space_size,
                transition_std=0.5,
                seed=seed,
            )
        else:
            # Discrete G' — NOT recommended for continuous MuJoCo obs.
            # Included for compatibility. Observations will be quantised.
            import warnings
            warnings.warn(
                "Discrete G' with continuous MuJoCo observations will "
                "quantise all values to 0/1. Use use_mlp=True or "
                "use_continuous=True for MuJoCo environments."
            )
            gprime = WorldModelGPrime(
                state_dim=state_dim,
                action_dim=env.action_space_size,
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

        # Adjust G' and ACTION bounds for MuJoCo simulation overhead
        rbta.update_bounds(
            "G'", ResourceBounds(B_time=0.080, B_mem=500_000, B_energy=50.0),
        )
        rbta.update_bounds(
            "ACTION", ResourceBounds(B_time=0.050, B_mem=10_000, B_energy=2.0),
        )

        mdim = MDIM(state_dim=state_dim)
        attention = Attention()
        criticality_regulator = CriticalityRegulator()
        hpm_validator = HPMValidator()
        m3 = M3EpisodicMemory(
            state_dim=state_dim, action_dim=env.action_space_size,
        )
        consolidation = ConsolidationScheduler(
            m3=m3, state_dim=state_dim,
            consolidation_interval=10, max_facts_per_cycle=50,
        )

        return cls(
            sanitizer=sanitizer, m1=m1, m2=m2, gprime=gprime,
            engine=engine, peu=peu, tspl=tspl, rbta=rbta,
            mdim=mdim, attention=attention,
            criticality_regulator=criticality_regulator,
            hpm_validator=hpm_validator,
            consolidation=consolidation, env=env,
            state_dim=state_dim,
            metrics_store=metrics_store,
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
    ) -> CognitiveCycle:
        """Build a fully-configured cognitive cycle for GridWorld.

        Args:
            size: GridWorld size (5, 10, or 20).
            seed: Random seed.
            state_dim: Override state dimensionality (default: auto from env).
            use_continuous: If True, use Gaussian CPDs with analytic inference
                (Phase 3.2). If False (default), use discrete binary CPDs with
                pgmpy exact inference (Phase 3.1 compatible).
            use_mlp: If True, use the pure-NumPy MLP world model instead of
                the Bayesian graph G'. Overrides use_continuous.
                (Phase 3.3b — feature gate controlled by USE_MLP_GPRIME flag.)
            obstacles: Wall positions for GridWorld. If None, random walls generated.
                If empty list, no walls.

        Returns:
            Configured CognitiveCycle instance.
        """
        # obstacles=None → use empty list (backward compatible: original code passed [])
        env = GridWorld(size=size, obstacles=obstacles if obstacles is not None else [], seed=seed)
        actual_state_dim = state_dim or env.get_state_dim()

        sanitizer = ASISanitizer(sensor_dim=actual_state_dim, v_max=100.0, epsilon_confidence=0.01)
        m1 = M1SensoryBuffer(sensor_dim=actual_state_dim)
        m2 = M2WorkingMemory(capacity=7)

        if use_mlp:
            gprime = WorldModelMLP(
                state_dim=actual_state_dim,
                action_dim=env.action_space_size,
                seed=seed,
            )
        elif use_continuous:
            # Phase 3.2: Continuous Gaussian G' with analytic inference
            gprime = WorldModelGPrime.build_gaussian_grid(
                state_dim=actual_state_dim,
                action_dim=env.action_space_size,
                transition_std=0.5,
                seed=seed,
            )
        else:
            # Phase 3.1: Discrete binary G' with pgmpy exact inference
            gprime = WorldModelGPrime(
                state_dim=actual_state_dim,
                action_dim=env.action_space_size,
                seed=seed,
            )

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
        # Initialize TSPL theta with G' parameter shape so TSPL.update() reaches
        # accuracy computation and skill compilation. Theta is a bookkeeping mirror
        # of G' parameters — Phase 3.3 will wire theta→G' when MLP replaces G'.
        tspl.init_parameters("gprime", (actual_state_dim,))
        rbta = RBTAEnforcer(module_bounds=DEFAULT_MODULE_BOUNDS)
        # MLP learn() does 8×64 forward+backward passes per cycle (~31ms)
        if use_mlp:
            rbta.update_bounds(
                "G'", ResourceBounds(B_time=0.050, B_mem=500_000, B_energy=50.0),
            )

        mdim = MDIM(state_dim=actual_state_dim)
        attention = Attention()
        criticality_regulator = CriticalityRegulator()
        hpm_validator = HPMValidator()
        m3 = M3EpisodicMemory(state_dim=actual_state_dim, action_dim=env.action_space_size)
        consolidation = ConsolidationScheduler(
            m3=m3, state_dim=actual_state_dim,
            consolidation_interval=10, max_facts_per_cycle=50,
        )

        return cls(
            sanitizer=sanitizer, m1=m1, m2=m2, gprime=gprime,
            engine=engine, peu=peu, tspl=tspl, rbta=rbta,
            mdim=mdim, attention=attention,
            criticality_regulator=criticality_regulator,
            hpm_validator=hpm_validator,
            consolidation=consolidation, env=env,
            state_dim=actual_state_dim,
            metrics_store=metrics_store,
        )
