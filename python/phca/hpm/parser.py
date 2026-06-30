"""
PHCA v3.0 - HPM Composite Resource Bound Computation.

Computes composite resource bounds for SEQUENCE/PARALLEL module
composition using v3.0 Theorem 2.1/Theorem 3.1 additivity rules.

Phase 3.2: Pure bound computation (validation stripped — the cycle
builds a single hardcoded spec that never changes).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class CompositionOp(Enum):
    """HPM composition operators (v3.0 §3.2 Definition 3.4)."""
    SEQUENCE = "SEQUENCE"
    PARALLEL = "PARALLEL"
    CONDITIONAL = "CONDITIONAL"
    HIERARCHY = "HIERARCHY"
    RECURSE = "RECURSE"
    INTERLEAVE = "INTERLEAVE"
    TEMPORAL_INVARIANT = "TEMPORAL_INVARIANT"
    REACTIVE = "REACTIVE"


@dataclass
class HPMNode:
    """A node in the HPM composition tree."""
    operator: str
    children: List[HPMNode] = field(default_factory=list)
    module_id: Optional[str] = None
    bounds: Optional[Dict[str, float]] = None


# ── Composite Bound Constants ─────────────────────────────────

TAU_COMP = 0.001    # 1ms composition overhead (SEQUENCE)
TAU_SYNC = 0.002    # 2ms synchronization overhead (PARALLEL)
DELTA_SHARED = 1024.0  # 1KB shared memory
DELTA_COMM = 2048.0     # 2KB communication memory
EPSILON_OVERHEAD = 0.001  # 1mJ energy overhead (SEQUENCE)
EPSILON_COMM = 0.002      # 2mJ communication energy (PARALLEL)


class HPMValidator:
    """HPM resource bound computer.

    Uses v3.0 corrected resource additivity rules:
      SEQUENCE:  B_time = sum(child_time) + TAU_COMP
                 B_mem  = max(child_mem) + DELTA_SHARED
      PARALLEL:  B_time = max(child_time) + TAU_SYNC
                 B_mem  = sum(child_mem) + DELTA_COMM
    """

    def compute_bounds(
        self,
        tree: Dict[str, Any],
        runtime_log: Dict[str, float],
    ) -> Optional[Dict[str, float]]:
        """Compute composite resource bounds from a tree with runtimes.

        Args:
            tree: Composition tree.
            runtime_log: Measured runtimes (module_id → seconds).

        Returns:
            Dict with "B_time" and "B_mem" or None if invalid.
        """
        return self._compute_bounds(tree, runtime_log)

    def compute_composite_bounds(
        self, tree: Dict[str, Any]
    ) -> Optional[Dict[str, float]]:
        """Compute composite resource bounds without runtime log (pure structure)."""
        return self._compute_bounds(tree, {})

    def _compute_bounds(
        self,
        node: Dict[str, Any],
        runtime_log: Dict[str, float],
    ) -> Optional[Dict[str, float]]:
        """Recursively compute bounds for a tree node."""
        op = node.get("type", "")
        children = node.get("children", [])
        node_id = node.get("id", op)

        # Check for stored bounds override
        if "bounds" in node:
            b = node["bounds"]
            if isinstance(b, dict):
                return {
                    "B_time": b.get("B_time", 0.0),
                    "B_mem": b.get("B_mem", 0.0),
                }

        # Leaf node: look up runtime from log, use stored bounds
        if not children:
            actual_time = runtime_log.get(node_id, 0.0)
            actual_energy = runtime_log.get(node_id, 0.0) * 50.0
            return {
                "B_time": node.get("B_time", actual_time + 0.001),
                "B_mem": node.get("B_mem", 1024.0),
                "B_energy": node.get("B_energy", max(0.1, min(10.0, actual_energy))),
            }

        # Composition node: compute child bounds first
        child_bounds = []
        for child in children:
            if isinstance(child, dict):
                cb = self._compute_bounds(child, runtime_log)
                if cb is not None:
                    child_bounds.append(cb)
            elif isinstance(child, str):
                child_bounds.append({
                    "B_time": runtime_log.get(child, 0.0),
                    "B_mem": 1024.0,
                })

        if not child_bounds:
            return None

        # Apply composition rules (v3.0 §2.1.1 Def 3.6)
        if op == CompositionOp.SEQUENCE.value:
            return {
                "B_time": sum(c["B_time"] for c in child_bounds) + TAU_COMP,
                "B_mem": max(c["B_mem"] for c in child_bounds) + DELTA_SHARED,
                "B_energy": sum(c.get("B_energy", 1.0) for c in child_bounds) + EPSILON_OVERHEAD,
            }
        elif op == CompositionOp.PARALLEL.value:
            return {
                "B_time": max(c["B_time"] for c in child_bounds) + TAU_SYNC,
                "B_mem": sum(c["B_mem"] for c in child_bounds) + DELTA_COMM,
                "B_energy": sum(c.get("B_energy", 1.0) for c in child_bounds) + EPSILON_COMM,
            }
        elif op == CompositionOp.CONDITIONAL.value:
            branch_times = child_bounds[1:]
            return {
                "B_time": max(c["B_time"] for c in branch_times) + child_bounds[0]["B_time"],
                "B_mem": max(c["B_mem"] for c in branch_times),
                "B_energy": max(c.get("B_energy", 1.0) for c in branch_times) + child_bounds[0].get("B_energy", 1.0),
            }
        elif op == CompositionOp.HIERARCHY.value:
            return {
                "B_time": child_bounds[0]["B_time"] + child_bounds[1]["B_time"] + TAU_COMP,
                "B_mem": max(c["B_mem"] for c in child_bounds),
                "B_energy": child_bounds[0].get("B_energy", 1.0) + child_bounds[1].get("B_energy", 1.0) + EPSILON_OVERHEAD,
            }
        elif op == CompositionOp.RECURSE.value:
            n = node.get("n", 1)
            return {
                "B_time": child_bounds[0]["B_time"] * n + TAU_COMP * (n - 1),
                "B_mem": child_bounds[0]["B_mem"],
                "B_energy": child_bounds[0].get("B_energy", 1.0) * n + EPSILON_OVERHEAD * (n - 1),
            }
        elif op == CompositionOp.REACTIVE.value:
            return {
                "B_time": max(c["B_time"] for c in child_bounds),
                "B_mem": child_bounds[0]["B_mem"] + child_bounds[1]["B_mem"],
                "B_energy": child_bounds[0].get("B_energy", 1.0) + child_bounds[1].get("B_energy", 1.0),
            }
        else:
            return {
                "B_time": max(c["B_time"] for c in child_bounds),
                "B_mem": max(c["B_mem"] for c in child_bounds),
                "B_energy": max(c.get("B_energy", 1.0) for c in child_bounds),
            }
