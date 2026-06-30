"""
PHCA v3.0 - HPM (Hierarchical Predictive Module) Grammar Stub.

Phase 3.1: Stub - validates all compositions as correct.
Phase 3.2+: Typed grammar for module composition with SEQUENCE/PARALLEL combinators.

v3.0 Reference: xa7.2.1 Definition 2.1 (composition tree), xa7.4
"""

from __future__ import annotations

from typing import Any, Dict, List


class HPMValidator:
    """HPM Grammar Validator - Phase 3.1 stub."""

    def validate(self, spec: Dict[str, Any]) -> bool:
        """Validate a composition specification (Phase 3.1: always valid)."""
        return True

    def get_valid_compositions(self) -> List[str]:
        """Get list of valid composition types."""
        return ["SEQUENCE", "PARALLEL", "PIPELINE"]
