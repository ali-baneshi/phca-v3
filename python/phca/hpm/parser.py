"""
PHCA v3.0 - HPM (Hierarchical Predictive Module) Grammar.

Phase 3.2: Full typed grammar for module composition with 8 operators,
type safety checking, and resource bound computation.

Composition operators (v3.0 §3.2 Definition 3.4):
    SEQUENCE:           M1 → M2 (serial, time = sum + τ_comp)
    PARALLEL:           M1 ∥ M2 (concurrent, time = max + τ_sync)
    CONDITIONAL:        IF predictor THEN M1 ELSE M2
    HIERARCHY:          Predictor decomposes into sub-module
    RECURSE:            Repeat module n times
    INTERLEAVE:         Interleaved execution sharing variable set
    TEMPORAL_INVARIANT: Repeat module, maintaining state across window
    REACTIVE:           Direct sensor → controller bypass

v3.0 Reference: §3.2 Definition 3.4-3.7
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple


class ModuleType(Enum):
    """Typed module categories for type safety (v3.0 §3.2 Definition 3.5)."""
    SENSOR_STREAM = auto()
    STATE_VECTOR = auto()
    GOAL_VECTOR = auto()
    COMMAND_BUFFER = auto()
    ERROR_SIGNAL = auto()
    CONFIDENCE_SCORE = auto()


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


class LeafOp(Enum):
    """Leaf module operators (v3.0 §3.2 Definition 3.4)."""
    ASI_INPUT = "ASI_Input"
    PREDICT = "Predict"
    CONTROL = "Control"


# ── Type Safety Tables ─────────────────────────────────────────

# Valid input/output type pairs for leaf modules
LEAF_TYPE_SIGNATURES: Dict[LeafOp, Tuple[Optional[ModuleType], ModuleType]] = {
    LeafOp.ASI_INPUT: (None, ModuleType.SENSOR_STREAM),       # no input → sensor stream
    LeafOp.PREDICT: (ModuleType.STATE_VECTOR, ModuleType.STATE_VECTOR),  # state → state
    LeafOp.CONTROL: (ModuleType.COMMAND_BUFFER, ModuleType.COMMAND_BUFFER),
}

# Composition operator type compatibility rules
COMPOSITION_TYPE_RULES: Dict[CompositionOp, str] = {
    CompositionOp.SEQUENCE: "τ_out(M1) == τ_in(M2)",
    CompositionOp.PARALLEL: "τ_out(M1) == τ_out(M2) (merged output)",
    CompositionOp.CONDITIONAL: "τ_out(M1) == τ_in(M2) == τ_in(M3)",
    CompositionOp.HIERARCHY: "τ_out(predictor) compatible with τ_in(sub)",
    CompositionOp.RECURSE: "τ_in == τ_out (fixed point)",
    CompositionOp.INTERLEAVE: "module types compatible with shared var set",
    CompositionOp.TEMPORAL_INVARIANT: "τ_in == τ_out",
    CompositionOp.REACTIVE: "τ_in(sensor) compatible with τ_in(controller)",
}


@dataclass
class HPMNode:
    """A node in the HPM composition tree.

    Attributes:
        operator: Composition or leaf operator.
        children: Child nodes (empty for leaf nodes).
        module_id: Optional module identifier.
        input_type: Expected input type (None for root).
        output_type: Output type of this module.
        bounds: Resource bounds (computed for composites).
    """
    operator: str
    children: List[HPMNode] = field(default_factory=list)
    module_id: Optional[str] = None
    input_type: Optional[ModuleType] = None
    output_type: Optional[ModuleType] = None
    bounds: Optional[Dict[str, float]] = None


@dataclass
class ValidationResult:
    """Result of HPM composition validation.

    Attributes:
        valid: Whether the composition is valid.
        errors: List of validation error messages.
        warnings: List of non-blocking warnings.
        computed_bounds: Computed resource bounds (if valid).
    """
    valid: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    computed_bounds: Optional[Dict[str, float]] = None


# ── Composite Bound Constants ─────────────────────────────────

TAU_COMP = 0.001    # 1ms composition overhead (SEQUENCE)
TAU_SYNC = 0.002    # 2ms synchronization overhead (PARALLEL)
DELTA_SHARED = 1024.0  # 1KB shared memory
DELTA_COMM = 2048.0     # 2KB communication memory


class HPMValidator:
    """HPM Grammar Validator — Phase 3.2 full implementation.

    Validates typed composition specifications against the HPM grammar,
    checks type compatibility, and computes composite resource bounds.
    """

    def __init__(self) -> None:
        self._composition_count: int = 0
        self._last_result: Optional[ValidationResult] = None

    def validate(self, spec: Dict[str, Any]) -> bool:
        """Validate a composition specification.

        Accepts either:
        - A full composition tree (dict with "type", "children", etc.)
        - A simple dict with "composition" key (backward compat with cycle)
        - Any truthy dict (Phase 3.1 fallback)

        Args:
            spec: Composition specification dict.

        Returns:
            True if valid, False otherwise.
        """
        self._composition_count += 1

        # Phase 3.1 backward compatibility: non-dict or simple dict passes
        if not isinstance(spec, dict):
            return True

        # If spec has a composition tree, validate it properly
        if "type" in spec and "children" in spec:
            result = self._validate_tree(spec)
            self._last_result = result
            return result.valid

        # If spec has a "composition" key, validate that structure
        if "composition" in spec and isinstance(spec["composition"], dict):
            result = self._validate_tree(spec["composition"])
            self._last_result = result
            return result.valid

        # Simple passthrough (e.g., {"cycle": n} from cycle.py)
        return True

    def validate_structured(
        self, tree: Dict[str, Any]
    ) -> ValidationResult:
        """Validate a structured composition tree and return detailed results.

        Args:
            tree: Composition tree dict with format:
                {"type": "SEQUENCE" | ..., "children": [...], ...}

        Returns:
            ValidationResult with errors, warnings, and computed bounds.
        """
        result = self._validate_tree(tree)
        self._last_result = result
        return result

    def get_valid_compositions(self) -> List[str]:
        """Get list of valid composition types."""
        return [op.value for op in CompositionOp]

    def get_leaf_module_types(self) -> List[str]:
        """Get list of valid leaf module types."""
        return [op.value for op in LeafOp]

    def compute_composite_bounds(
        self, tree: Dict[str, Any]
    ) -> Optional[Dict[str, float]]:
        """Compute composite resource bounds for a validated tree.

        Uses v3.0 corrected resource additivity rules.

        Args:
            tree: Validated composition tree.

        Returns:
            Dict with "B_time", "B_mem" keys, or None if tree is invalid.
        """
        result = self._validate_tree(tree)
        if not result.valid:
            return None
        return self._compute_bounds(tree, {})

    def get_last_result(self) -> Optional[ValidationResult]:
        """Get the result of the last validation."""
        return self._last_result

    # ── Internal Validation ──────────────────────────────────

    def _validate_tree(
        self, node: Dict[str, Any]
    ) -> ValidationResult:
        """Recursively validate a composition tree node.

        Args:
            node: Tree node dict.

        Returns:
            ValidationResult for this node.
        """
        errors: List[str] = []
        warnings: List[str] = []

        op = node.get("type", "")
        children = node.get("children", [])

        # Check operator validity
        if not op:
            errors.append("Node missing 'type' field")
            return ValidationResult(valid=False, errors=errors)

        # Check if it's a valid composition or leaf operator
        is_composition = any(op == c.value for c in CompositionOp)
        is_leaf = any(op == c.value for c in LeafOp)

        if not is_composition and not is_leaf:
            errors.append(f"Unknown operator '{op}'. Valid: {self.get_valid_compositions()}")
            return ValidationResult(valid=False, errors=errors)

        # Leaf node validation
        if is_leaf:
            return self._validate_leaf(node, op, errors, warnings)

        # Composition node validation
        return self._validate_composition(node, op, children, errors, warnings)

    def _validate_leaf(
        self,
        node: Dict[str, Any],
        op: str,
        errors: List[str],
        warnings: List[str],
    ) -> ValidationResult:
        """Validate a leaf module node."""
        module_id = node.get("id", f"{op}_anon")

        # Check required fields per leaf type
        if op == LeafOp.ASI_INPUT.value:
            if "dim" not in node:
                warnings.append(f"{module_id}: ASI_Input missing 'dim', defaulting to 84")
            if "grounding_level" not in node:
                warnings.append(f"{module_id}: ASI_Input missing 'grounding_level', defaulting to 1")
        elif op == LeafOp.PREDICT.value:
            if "horizon" not in node:
                warnings.append(f"{module_id}: Predict missing 'horizon', defaulting to 1")
        elif op == LeafOp.CONTROL.value:
            if "horizon" not in node:
                warnings.append(f"{module_id}: Control missing 'horizon', defaulting to 1")

        return ValidationResult(valid=len(errors) == 0, errors=errors, warnings=warnings)

    def _validate_composition(
        self,
        node: Dict[str, Any],
        op: str,
        children: List[Any],
        errors: List[str],
        warnings: List[str],
    ) -> ValidationResult:
        """Validate a composition node with type checking."""
        module_id = node.get("id", op)

        # Check children exist
        if not children or len(children) == 0:
            errors.append(f"{module_id}: {op} requires at least one child")
            return ValidationResult(valid=False, errors=errors)

        # Check child count per operator type
        if op == CompositionOp.CONDITIONAL.value and len(children) != 3:
            errors.append(f"{module_id}: CONDITIONAL requires exactly 3 children (predictor, true_branch, false_branch), got {len(children)}")
        elif op == CompositionOp.HIERARCHY.value and len(children) != 2:
            errors.append(f"{module_id}: HIERARCHY requires exactly 2 children (predictor, sub_module), got {len(children)}")
        elif op == CompositionOp.REACTIVE.value and len(children) != 2:
            errors.append(f"{module_id}: REACTIVE requires exactly 2 children (sensor, controller), got {len(children)}")

        # Validate each child recursively
        child_results = []
        for child in children:
            if isinstance(child, dict):
                child_results.append(self._validate_tree(child))
            elif isinstance(child, str):
                # String child — assume valid leaf reference
                child_results.append(ValidationResult())
            else:
                child_results.append(ValidationResult(
                    valid=False, errors=[f"Invalid child type: {type(child).__name__}"]
                ))

        # Aggregate child errors
        for cr in child_results:
            if not cr.valid:
                errors.extend(cr.errors)

        # Type compatibility checking for SEQUENCE
        if op == CompositionOp.SEQUENCE.value and len(children) >= 2:
            self._check_sequence_types(children, module_id, warnings)

        # RECURSE validation
        if op == CompositionOp.RECURSE.value:
            n = node.get("n", 1)
            if not isinstance(n, int) or n < 1:
                errors.append(f"{module_id}: RECURSE 'n' must be a positive integer, got {n}")
            if n > 50:
                warnings.append(f"{module_id}: RECURSE n={n} exceeds recommended maximum of 50")

        # INTERLEAVE validation
        if op == CompositionOp.INTERLEAVE.value:
            var_set = node.get("var_set", [])
            if not var_set:
                warnings.append(f"{module_id}: INTERLEAVE with empty var_set (no shared variables)")

        # TEMPORAL_INVARIANT validation
        if op == CompositionOp.TEMPORAL_INVARIANT.value:
            window = node.get("window", 1)
            if not isinstance(window, int) or window < 1:
                warnings.append(f"{module_id}: TEMPORAL_INVARIANT window should be positive, got {window}")

        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
        )

    def _check_sequence_types(
        self,
        children: List[Any],
        module_id: str,
        warnings: List[str],
    ) -> None:
        """Check type compatibility between sequential modules.

        For SEQUENCE(M1, M2), τ_out(M1) should match τ_in(M2).
        Phase 3.2: heuristic check based on type hints in node dict.
        """
        for i in range(len(children) - 1):
            left = children[i]
            right = children[i + 1]

            if isinstance(left, dict) and isinstance(right, dict):
                left_out = left.get("output_type")
                right_in = right.get("input_type")
                if left_out is not None and right_in is not None:
                    if left_out != right_in:
                        warnings.append(
                            f"{module_id}: Type mismatch at position {i}: "
                            f"{left.get('id', '?')} outputs {left_out} "
                            f"but {right.get('id', '?')} expects {right_in}"
                        )

    # ── Resource Bound Computation ───────────────────────────

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
            return {
                "B_time": node.get("B_time", actual_time + 0.001),
                "B_mem": node.get("B_mem", 1024.0),
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

        # Apply composition rules
        if op == CompositionOp.SEQUENCE.value:
            return {
                "B_time": sum(c["B_time"] for c in child_bounds) + TAU_COMP,
                "B_mem": max(c["B_mem"] for c in child_bounds) + DELTA_SHARED,
            }
        elif op == CompositionOp.PARALLEL.value:
            return {
                "B_time": max(c["B_time"] for c in child_bounds) + TAU_SYNC,
                "B_mem": sum(c["B_mem"] for c in child_bounds) + DELTA_COMM,
            }
        elif op == CompositionOp.CONDITIONAL.value:
            # Time = max over branches (only one executes)
            branch_times = child_bounds[1:]  # skip predictor
            branch_mems = child_bounds[1:]
            return {
                "B_time": max(c["B_time"] for c in branch_times) + child_bounds[0]["B_time"],
                "B_mem": max(c["B_mem"] for c in branch_mems),
            }
        elif op == CompositionOp.HIERARCHY.value:
            # Time = predictor + sub_module (sequential)
            return {
                "B_time": child_bounds[0]["B_time"] + child_bounds[1]["B_time"] + TAU_COMP,
                "B_mem": max(c["B_mem"] for c in child_bounds),
            }
        elif op == CompositionOp.RECURSE.value:
            n = node.get("n", 1)
            return {
                "B_time": child_bounds[0]["B_time"] * n + TAU_COMP * (n - 1),
                "B_mem": child_bounds[0]["B_mem"],
            }
        elif op == CompositionOp.REACTIVE.value:
            # Fast bypass: time = max(sensor, controller)
            return {
                "B_time": max(c["B_time"] for c in child_bounds),
                "B_mem": child_bounds[0]["B_mem"] + child_bounds[1]["B_mem"],
            }
        else:
            # Fallback: use max
            return {
                "B_time": max(c["B_time"] for c in child_bounds),
                "B_mem": max(c["B_mem"] for c in child_bounds),
            }

    def reset(self) -> None:
        """Reset validator state."""
        self._composition_count = 0
        self._last_result = None
