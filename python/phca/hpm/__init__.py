"""Hierarchical Predictive Module Grammar — typed composition of cognitive modules.

Phase 3.2: Full typed grammar with 8 composition operators, type safety,
and resource bound computation.

v3.0 Reference: §3.2 Definition 3.4-3.7
"""

from phca.hpm.parser import (
    HPMValidator,
    HPMNode,
    ValidationResult,
    ModuleType,
    CompositionOp,
    LeafOp,
    LEAF_TYPE_SIGNATURES,
    COMPOSITION_TYPE_RULES,
)

__all__ = [
    "HPMValidator",
    "HPMNode",
    "ValidationResult",
    "ModuleType",
    "CompositionOp",
    "LeafOp",
]
