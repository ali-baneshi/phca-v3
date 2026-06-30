"""
PHCA v3.0 — RBTA Constraint Enforcer (Python).

Implements v3.0 §2.1 Definition 2.2 (Constraint Enforcer),
Definition 2.3 (Enforcement Status), and Theorem 2.1 / Theorem 3.1 (Composition).

Pure Python implementation for Phase 3.1. Rust FFI version deferred to Phase 3.2.

Audit Decision O-1: Use Python for Phase 3.1 to eliminate FFI overhead.
"""

from __future__ import annotations

from enum import Enum, auto
from typing import Dict, List, Optional, Tuple

import numpy as np

from phca.config import ConstraintViolation, ResourceBounds
from phca.logging import logger, _log


class BoundType(Enum):
    """Types of resource bounds that can be violated (v3.0 §2.1 Definition 2.2).

    Values match the string convention used in ConstraintViolation.bound_type.
    """
    TIME = "TIME"
    MEMORY = "MEM"
    ENERGY = "ENERGY"
    ENTROPY_FLOOR = "ENTROPY"
    SENSOR_FAILURE = "SENSOR"


class EnforcerAction(Enum):
    """Aggregate enforcement status for a cycle (v3.0 §2.1 Definition 2.3).

    - CONTINUE: All bounds satisfied — continue normal operation.
    - INTERRUPT: Warning-level violation — interrupt current module, continue cycle.
    - TERMINATE: Critical violation — terminate the current cognitive cycle.
    """
    CONTINUE = auto()
    INTERRUPT = auto()
    TERMINATE = auto()


class RBTAEnforcer:
    """RBTA constraint enforcer for a complete cognitive cycle.

    Checks per-module resource bounds (time, memory, energy, entropy floor)
    and aggregate sensor failure count. Phase 3.1: flat per-module checks.
    Phase 3.2+: composition tree for SEQUENCE/PARALLEL verification.

    v3.0 References:
        - §2.1 Definition 2.2: Resource bounds B_X
        - §2.1 Definition 2.3: Enforcement status (Continue/Interrupt/Terminate)
        - §2.1 Theorem 2.1: Corrected SEQUENCE time = sum(time_i) + tau_comp
        - §2.1 Theorem 3.1: Corrected PARALLEL time = max(time_i) + tau_sync
    """

    def __init__(self, module_bounds: Dict[str, ResourceBounds]):
        """Initialize enforcer with per-module resource bounds.

        Args:
            module_bounds: Mapping from module_id (e.g., "ASI", "G'") to
                ResourceBounds for that module.
        """
        self._bounds = dict(module_bounds)
        self._asi_failure_limit: int = 5  # default from v3.0 Patch B

    @property
    def asi_failure_limit(self) -> int:
        return self._asi_failure_limit

    @asi_failure_limit.setter
    def asi_failure_limit(self, value: int) -> None:
        assert value > 0, f"asi_failure_limit must be positive, got {value}"
        self._asi_failure_limit = value

    def check_cycle(
        self,
        runtime_log: Dict[str, float],
        memory_log: Dict[str, float],
        energy_log: Dict[str, float],
        belief_entropies: Dict[str, float],
        sensor_failure_count: int,
        asi_failure_limit: Optional[int] = None,
        composition_tree: Optional[Dict] = None,
    ) -> Tuple[List[ConstraintViolation], EnforcerAction]:
        """Check all resource bounds for the current cognitive cycle.

        Args:
            runtime_log: Measured runtime (seconds) per module_id.
                Missing module_id means 0.0 (not a violation).
            memory_log: Measured memory (bytes) per module_id.
            energy_log: Measured energy (estimated Joules) per module_id.
            belief_entropies: Current belief entropy H(beliefs) per module_id.
            sensor_failure_count: Number of consecutive ASI sensor failures.
            asi_failure_limit: ASI failure limit override (default: self.asi_failure_limit).
            composition_tree: HPM composition tree for SEQUENCE/PARALLEL checks.
                Phase 3.1: None (deferred to Phase 3.2).

        Returns:
            Tuple of (violations, action):
                violations: List of ConstraintViolation detected.
                action: EnforcerAction (Continue/Interrupt/Terminate).
        """
        violations: List[ConstraintViolation] = []
        limit = asi_failure_limit if asi_failure_limit is not None else self._asi_failure_limit

        for module_id, bounds in self._bounds.items():
            # Check time bound: runtime > B_time → violation
            runtime = runtime_log.get(module_id, 0.0)
            # G7: NaN gate — skip NaN values that would bypass detection
            if not np.isfinite(runtime):
                _log(logger, "warning", "rbta.nan_log_value",
                     module=module_id, bound="TIME", value=runtime)
                continue
            if runtime > bounds.B_time:
                violations.append(ConstraintViolation(
                    module_id=module_id,
                    bound_type=BoundType.TIME.value,
                    measured=runtime,
                    allowed=bounds.B_time,
                ))

            # Check memory bound: memory > B_mem → violation
            memory = memory_log.get(module_id, 0.0)
            if not np.isfinite(memory):
                _log(logger, "warning", "rbta.nan_log_value",
                     module=module_id, bound="MEM", value=memory)
                continue
            if memory > bounds.B_mem:
                violations.append(ConstraintViolation(
                    module_id=module_id,
                    bound_type=BoundType.MEMORY.value,
                    measured=memory,
                    allowed=bounds.B_mem,
                ))

            # Check energy bound: energy > B_energy → violation
            energy = energy_log.get(module_id, 0.0)
            if not np.isfinite(energy):
                _log(logger, "warning", "rbta.nan_log_value",
                     module=module_id, bound="ENERGY", value=energy)
                continue
            if energy > bounds.B_energy:
                violations.append(ConstraintViolation(
                    module_id=module_id,
                    bound_type=BoundType.ENERGY.value,
                    measured=energy,
                    allowed=bounds.B_energy,
                ))

            # Check entropy floor (A3: Incomplete Knowledge): H < entropy_floor → violation
            entropy = belief_entropies.get(module_id, None)
            if entropy is not None:
                if not np.isfinite(entropy):
                    _log(logger, "warning", "rbta.nan_log_value",
                         module=module_id, bound="ENTROPY", value=entropy)
                    continue
                if entropy < bounds.entropy_floor:
                    violations.append(ConstraintViolation(
                        module_id=module_id,
                        bound_type=BoundType.ENTROPY_FLOOR.value,
                        measured=entropy,
                        allowed=bounds.entropy_floor,
                    ))

        # Check ASI sensor failure limit (v3.0 Patch B)
        if sensor_failure_count > limit:
            violations.append(ConstraintViolation(
                module_id="ASI",
                bound_type=BoundType.SENSOR_FAILURE.value,
                measured=float(sensor_failure_count),
                allowed=float(limit),
            ))

        # Composition tree check (Phase 3.2+)
        if composition_tree is not None:
            self._check_composition_tree(composition_tree, runtime_log, violations)

        # Determine action based on violation severity
        action = self._classify_action(violations)
        return (violations, action)

    # ── Internal Methods ─────────────────────────────────────

    def _classify_action(self, violations: List[ConstraintViolation]) -> EnforcerAction:
        """Classify aggregate enforcement action (v3.0 §2.1 Definition 2.3).

        0 violations → CONTINUE
        1-2 violations → INTERRUPT
        3+ violations → TERMINATE
        """
        count = len(violations)
        if count == 0:
            return EnforcerAction.CONTINUE
        elif count <= 2:
            return EnforcerAction.INTERRUPT
        else:
            return EnforcerAction.TERMINATE

    def _check_composition_tree(
        self,
        tree: Dict,
        runtime_log: Dict[str, float],
        violations: List[ConstraintViolation],
    ) -> None:
        """Check composite resource bounds via HPM composition tree.

        Implements v3.0 Theorem 2.1 (corrected SEQUENCE) and Theorem 3.1 (PARALLEL).

        Recipe:
            SEQUENCE:  B_time = sum(child_time) + τ_comp (τ_comp = 1ms)
                       B_mem  = max(child_mem) + δ_shared (δ_shared = 1KB)
            PARALLEL:  B_time = max(child_time) + τ_sync (τ_sync = 2ms)
                       B_mem  = sum(child_mem) + δ_comm  (δ_comm = 2KB)

        Args:
            tree: Composition tree node with format:
                {"type": "SEQUENCE" | "PARALLEL",
                 "children": [module_id_str | nested_tree_dict, ...],
                 "bounds": ResourceBounds (composite bound)}
            runtime_log: Measured runtimes per module.
            violations: Shared violation list (appended in-place).
        """
        TAU_COMP = 0.001   # 1ms composition overhead
        TAU_SYNC = 0.002   # 2ms synchronization overhead
        DELTA_SHARED = 1024.0   # 1KB shared memory
        DELTA_COMM = 2048.0     # 2KB communication memory

        op = tree.get("type", "SEQUENCE")
        children = tree.get("children", [])
        tree_bounds = tree.get("bounds")

        if not children:
            return

        # Collect child runtimes and energies
        child_times = []
        child_energies = []
        for child in children:
            if isinstance(child, dict):
                # Nested composition tree — recurse
                self._check_composition_tree(child, runtime_log, violations)
                # For the parent check, compute the child's composite time and energy
                child_times.append(self._compute_subtree_runtime(child, runtime_log))
                child_energies.append(self._compute_subtree_energy(child, runtime_log))
            elif isinstance(child, str):
                # Leaf module — look up runtime and energy
                child_times.append(runtime_log.get(child, 0.0))
                child_energies.append(runtime_log.get(child + "_energy", 1.0))

        if not child_times:
            return

        # Compute composite time and energy based on composition type
        if op == "SEQUENCE":
            total_time = sum(child_times) + TAU_COMP
            total_energy = sum(child_energies) + 0.001  # ε_overhead = 1mJ
        elif op == "PARALLEL":
            total_time = max(child_times) + TAU_SYNC
            total_energy = sum(child_energies) + 0.002  # ε_comm = 2mJ
        else:
            # Unknown operator — skip check
            return

        # Check against composite bounds
        if tree_bounds is not None:
            # Time check
            bound = tree_bounds.get("B_time", float("inf"))
            if isinstance(bound, ResourceBounds):
                bound = bound.B_time
            if total_time > bound:
                violations.append(ConstraintViolation(
                    module_id=f"composite:{tree.get('id', op)}",
                    bound_type=BoundType.TIME.value,
                    measured=total_time,
                    allowed=float(bound),
                ))
            # Energy check (v3.0 §2.1.1 Def 3.6)
            energy_bound = tree_bounds.get("B_energy", float("inf"))
            if isinstance(energy_bound, ResourceBounds):
                energy_bound = energy_bound.B_energy
            if total_energy > energy_bound:
                violations.append(ConstraintViolation(
                    module_id=f"composite:{tree.get('id', op)}",
                    bound_type=BoundType.ENERGY.value,
                    measured=total_energy,
                    allowed=float(energy_bound),
                ))

    def _compute_subtree_runtime(
        self, tree: Dict, runtime_log: Dict[str, float]
    ) -> float:
        """Compute the composite runtime of a composition subtree.

        Recursive helper: mirrors the math in _check_composition_tree
        without generating violations.
        """
        TAU_COMP = 0.001
        TAU_SYNC = 0.002

        op = tree.get("type", "SEQUENCE")
        children = tree.get("children", [])

        if not children:
            return 0.0

        child_times = []
        for child in children:
            if isinstance(child, dict):
                child_times.append(self._compute_subtree_runtime(child, runtime_log))
            elif isinstance(child, str):
                child_times.append(runtime_log.get(child, 0.0))

        if not child_times:
            return 0.0

        if op == "SEQUENCE":
            return sum(child_times) + TAU_COMP
        elif op == "PARALLEL":
            return max(child_times) + TAU_SYNC
        else:
            return sum(child_times)  # fallback

    def _compute_subtree_energy(
        self, tree: Dict, runtime_log: Dict[str, float]
    ) -> float:
        """Compute the composite energy of a composition subtree.

        Energy composition rules (v3.0 §2.1.1 Def 3.6):
          SEQUENCE: B_energy = sum(child_energy) + ε_overhead
          PARALLEL: B_energy = sum(child_energy) + ε_comm
        """
        EPSILON_OVERHEAD = 0.001  # 1mJ
        EPSILON_COMM = 0.002      # 2mJ

        op = tree.get("type", "SEQUENCE")
        children = tree.get("children", [])

        if not children:
            return 0.0

        child_energies = []
        for child in children:
            if isinstance(child, dict):
                child_energies.append(self._compute_subtree_energy(child, runtime_log))
            elif isinstance(child, str):
                child_energies.append(runtime_log.get(child + "_energy", 1.0))

        if not child_energies:
            return 0.0

        if op == "SEQUENCE":
            return sum(child_energies) + EPSILON_OVERHEAD
        elif op == "PARALLEL":
            return sum(child_energies) + EPSILON_COMM
        else:
            return sum(child_energies)  # fallback

    def update_bounds(self, module_id: str, bounds: ResourceBounds) -> None:
        """Update bounds for a module (e.g., when new modules are registered).

        Args:
            module_id: Module identifier (e.g., "ASI", "G'").
            bounds: New resource bounds for this module.
        """
        self._bounds[module_id] = bounds

    def get_bounds(self, module_id: str) -> Optional[ResourceBounds]:
        """Get bounds for a module, or None if not registered."""
        return self._bounds.get(module_id)
