"""Tests for PHCA-3.2-007: HPM Grammar — typed module composition."""

from __future__ import annotations

import pytest

from phca.hpm.parser import (
    HPMValidator,
    HPMNode,
    ValidationResult,
    ModuleType,
    CompositionOp,
    LeafOp,
    TAU_COMP,
    TAU_SYNC,
    DELTA_SHARED,
    DELTA_COMM,
)


class TestHPMValidatorInit:
    """Tests for HPMValidator initialization."""

    def test_init_defaults(self):
        """Default init should create a valid validator."""
        v = HPMValidator()
        assert v.get_last_result() is None

    def test_get_valid_compositions(self):
        """Should return all 8 composition operators."""
        v = HPMValidator()
        ops = v.get_valid_compositions()
        assert len(ops) == 8
        assert "SEQUENCE" in ops
        assert "PARALLEL" in ops
        assert "CONDITIONAL" in ops
        assert "HIERARCHY" in ops
        assert "RECURSE" in ops
        assert "INTERLEAVE" in ops
        assert "TEMPORAL_INVARIANT" in ops
        assert "REACTIVE" in ops

    def test_get_leaf_module_types(self):
        """Should return all 3 leaf module types."""
        v = HPMValidator()
        leaves = v.get_leaf_module_types()
        assert len(leaves) == 3
        assert "ASI_Input" in leaves
        assert "Predict" in leaves
        assert "Control" in leaves


class TestHPMValidateSimple:
    """Tests for basic validation."""

    def test_trivially_true_for_simple_dict(self):
        """Simple dict like {'cycle': n} should pass (Phase 3.1 compat)."""
        v = HPMValidator()
        assert v.validate({"cycle": 42}) is True

    def test_trivially_true_for_non_dict(self):
        """Non-dict input passes trivially."""
        v = HPMValidator()
        assert v.validate("anything") is True
        assert v.validate(42) is True
        assert v.validate(None) is True

    def test_valid_sequence(self):
        """Valid SEQUENCE composition should pass."""
        v = HPMValidator()
        spec = {
            "type": "SEQUENCE",
            "children": [
                {"type": "ASI_Input", "id": "ASI", "dim": 84, "grounding_level": 1},
                {"type": "Predict", "id": "PE", "horizon": 1},
            ],
        }
        assert v.validate(spec) is True

    def test_valid_parallel(self):
        """Valid PARALLEL composition should pass."""
        v = HPMValidator()
        spec = {
            "type": "PARALLEL",
            "children": [
                {"type": "ASI_Input", "id": "ASI"},
                {"type": "ASI_Input", "id": "ASI2"},
            ],
        }
        assert v.validate(spec) is True

    def test_valid_hierarchy(self):
        """Valid HIERARCHY composition should pass."""
        v = HPMValidator()
        spec = {
            "type": "HIERARCHY",
            "children": [
                {"type": "Predict", "id": "predictor", "horizon": 1},
                {"type": "Control", "id": "controller", "horizon": 5},
            ],
        }
        assert v.validate(spec) is True

    def test_valid_recurse(self):
        """Valid RECURSE composition should pass."""
        v = HPMValidator()
        spec = {
            "type": "RECURSE",
            "n": 3,
            "children": [
                {"type": "Predict", "id": "PE", "horizon": 1},
            ],
        }
        assert v.validate(spec) is True

    def test_valid_interleave(self):
        """Valid INTERLEAVE composition should pass."""
        v = HPMValidator()
        spec = {
            "type": "INTERLEAVE",
            "var_set": ["x", "y"],
            "children": [
                {"type": "Predict", "id": "P1"},
                {"type": "Predict", "id": "P2"},
            ],
        }
        assert v.validate(spec) is True

    def test_valid_temporal_invariant(self):
        """Valid TEMPORAL_INVARIANT composition should pass."""
        v = HPMValidator()
        spec = {
            "type": "TEMPORAL_INVARIANT",
            "window": 10,
            "children": [
                {"type": "Predict", "id": "PE"},
            ],
        }
        assert v.validate(spec) is True

    def test_valid_reactive(self):
        """Valid REACTIVE composition should pass."""
        v = HPMValidator()
        spec = {
            "type": "REACTIVE",
            "children": [
                {"type": "ASI_Input", "id": "sensor"},
                {"type": "Control", "id": "controller"},
            ],
        }
        assert v.validate(spec) is True

    def test_valid_nested_composition(self):
        """Deeply nested composition should pass."""
        v = HPMValidator()
        spec = {
            "type": "SEQUENCE", "id": "root",
            "children": [
                {"type": "ASI_Input", "id": "ASI"},
                {
                    "type": "PARALLEL", "id": "mid",
                    "children": [
                        {"type": "Predict", "id": "P1"},
                        {"type": "Predict", "id": "P2"},
                    ],
                },
                {"type": "Control", "id": "CTRL"},
            ],
        }
        assert v.validate(spec) is True


class TestHPMValidateErrors:
    """Tests for validation error detection."""

    def test_missing_type_field(self):
        """Node without 'type' should fail."""
        v = HPMValidator()
        result = v.validate_structured({"children": []})
        assert result.valid is False
        assert any("missing 'type'" in e for e in result.errors)

    def test_unknown_operator(self):
        """Unknown operator should fail."""
        v = HPMValidator()
        spec = {"type": "INVALID_OP", "children": []}
        assert v.validate(spec) is False

    def test_conditional_wrong_child_count(self):
        """CONDITIONAL requires exactly 3 children."""
        v = HPMValidator()
        spec = {"type": "CONDITIONAL", "children": [{"type": "Predict"}]}
        result = v.validate_structured(spec)
        assert result.valid is False
        assert any("CONDITIONAL" in e and "3 children" in e for e in result.errors)

    def test_hierarchy_wrong_child_count(self):
        """HIERARCHY requires exactly 2 children."""
        v = HPMValidator()
        spec = {"type": "HIERARCHY", "children": [
            {"type": "Predict"}, {"type": "Predict"}, {"type": "Predict"},
        ]}
        result = v.validate_structured(spec)
        assert result.valid is False
        assert any("HIERARCHY" in e and "2 children" in e for e in result.errors)

    def test_empty_children(self):
        """Composition with empty children should fail."""
        v = HPMValidator()
        spec = {"type": "SEQUENCE", "children": []}
        result = v.validate_structured(spec)
        assert result.valid is False
        assert any("at least one child" in e for e in result.errors)

    def test_recurse_invalid_n(self):
        """RECURSE with non-positive n should fail."""
        v = HPMValidator()
        spec = {"type": "RECURSE", "n": 0, "children": [{"type": "Predict"}]}
        result = v.validate_structured(spec)
        assert result.valid is False


class TestHPMValidateWarnings:
    """Tests for non-blocking warnings."""

    def test_asi_input_missing_dim(self):
        """ASI_Input without dim should warn."""
        v = HPMValidator()
        spec = {"type": "ASI_Input", "id": "ASI_no_dim"}
        result = v.validate_structured(spec)
        assert result.valid is True
        assert any("missing 'dim'" in w for w in result.warnings)

    def test_predict_missing_horizon(self):
        """Predict without horizon should warn."""
        v = HPMValidator()
        spec = {"type": "Predict", "id": "P_no_horizon"}
        result = v.validate_structured(spec)
        assert result.valid is True
        assert any("missing 'horizon'" in w for w in result.warnings)

    def test_recurse_large_n_warns(self):
        """RECURSE with n > 50 should warn."""
        v = HPMValidator()
        spec = {"type": "RECURSE", "n": 100, "children": [{"type": "Predict"}]}
        result = v.validate_structured(spec)
        assert result.valid is True
        assert any("exceeds recommended maximum" in w for w in result.warnings)

    def test_interleave_empty_var_set_warns(self):
        """INTERLEAVE with empty var_set should warn."""
        v = HPMValidator()
        spec = {"type": "INTERLEAVE", "var_set": [], "children": [
            {"type": "Predict"}, {"type": "Predict"},
        ]}
        result = v.validate_structured(spec)
        assert result.valid is True
        assert any("empty var_set" in w for w in result.warnings)


class TestHPMResourceBounds:
    """Tests for resource bound computation."""

    def test_sequence_bounds(self):
        """SEQUENCE: time = sum + τ_comp, mem = max + δ_shared."""
        v = HPMValidator()
        tree = {
            "type": "SEQUENCE",
            "children": [
                {"type": "ASI_Input", "B_time": 0.002, "B_mem": 10_000},
                {"type": "Predict", "B_time": 0.025, "B_mem": 50_000},
            ],
        }
        bounds = v.compute_composite_bounds(tree)
        assert bounds is not None
        assert abs(bounds["B_time"] - (0.002 + 0.025 + TAU_COMP)) < 0.0001
        assert abs(bounds["B_mem"] - (max(10_000, 50_000) + DELTA_SHARED)) < 1

    def test_parallel_bounds(self):
        """PARALLEL: time = max + τ_sync, mem = sum + δ_comm."""
        v = HPMValidator()
        tree = {
            "type": "PARALLEL",
            "children": [
                {"type": "Predict", "B_time": 0.010, "B_mem": 20_000},
                {"type": "Predict", "B_time": 0.050, "B_mem": 30_000},
            ],
        }
        bounds = v.compute_composite_bounds(tree)
        assert bounds is not None
        assert abs(bounds["B_time"] - (0.050 + TAU_SYNC)) < 0.0001
        assert abs(bounds["B_mem"] - (20_000 + 30_000 + DELTA_COMM)) < 1

    def test_conditional_bounds(self):
        """CONDITIONAL: time = predictor_time + max(branch_times)."""
        v = HPMValidator()
        tree = {
            "type": "CONDITIONAL",
            "children": [
                {"type": "Predict", "B_time": 0.005, "B_mem": 5_000},  # predictor
                {"type": "Control", "B_time": 0.020, "B_mem": 10_000},  # true branch
                {"type": "Control", "B_time": 0.010, "B_mem": 8_000},   # false branch
            ],
        }
        bounds = v.compute_composite_bounds(tree)
        assert bounds is not None
        assert abs(bounds["B_time"] - (0.005 + max(0.020, 0.010))) < 0.0001

    def test_recurse_bounds(self):
        """RECURSE(n): time = n * child_time + (n-1) * τ_comp."""
        v = HPMValidator()
        tree = {
            "type": "RECURSE", "n": 5,
            "children": [
                {"type": "Predict", "B_time": 0.010, "B_mem": 10_000},
            ],
        }
        bounds = v.compute_composite_bounds(tree)
        assert bounds is not None
        expected_time = 5 * 0.010 + 4 * TAU_COMP
        assert abs(bounds["B_time"] - expected_time) < 0.0001

    def test_nested_bounds(self):
        """Nested composition should compute bounds recursively."""
        v = HPMValidator()
        tree = {
            "type": "SEQUENCE",
            "children": [
                {"type": "ASI_Input", "B_time": 0.002, "B_mem": 10_000},
                {
                    "type": "PARALLEL",
                    "children": [
                        {"type": "Predict", "B_time": 0.015, "B_mem": 20_000},
                        {"type": "Predict", "B_time": 0.025, "B_mem": 30_000},
                    ],
                },
            ],
        }
        bounds = v.compute_composite_bounds(tree)
        assert bounds is not None
        parallel_time = 0.025 + TAU_SYNC
        expected_time = 0.002 + parallel_time + TAU_COMP
        assert abs(bounds["B_time"] - expected_time) < 0.0001

    def test_bounds_from_runtime_log(self):
        """compute_bounds() with runtime_log should use measured times."""
        v = HPMValidator()
        tree = {
            "type": "SEQUENCE",
            "children": ["ASI", "PE"],
        }
        runtime_log = {"ASI": 0.003, "PE": 0.020}
        bounds = v.compute_bounds(tree, runtime_log)
        assert bounds is not None
        assert abs(bounds["B_time"] - (0.003 + 0.020 + TAU_COMP)) < 0.0001

    def test_bounds_with_override(self):
        """Node with explicit bounds should use those instead of computing."""
        v = HPMValidator()
        tree = {
            "type": "SEQUENCE",
            "bounds": {"B_time": 0.100, "B_mem": 50_000},
            "children": [{"type": "Predict"}, {"type": "Predict"}],
        }
        bounds = v.compute_composite_bounds(tree)
        assert bounds is not None
        assert abs(bounds["B_time"] - 0.100) < 0.0001
        assert abs(bounds["B_mem"] - 50_000) < 1

    def test_invalid_tree_returns_none(self):
        """Invalid tree should return None bounds."""
        v = HPMValidator()
        bounds = v.compute_composite_bounds({"not_a_valid_tree": True})
        assert bounds is None


class TestHPMValidateStructured:
    """Tests for validate_structured() returning ValidationResult."""

    def test_returns_validation_result(self):
        """validate_structured should return ValidationResult."""
        v = HPMValidator()
        result = v.validate_structured({"type": "SEQUENCE", "children": []})
        assert isinstance(result, ValidationResult)

    def test_stores_last_result(self):
        """Last validation result should be stored."""
        v = HPMValidator()
        v.validate_structured({"type": "SEQUENCE", "children": [
            {"type": "Predict", "id": "P1"},
        ]})
        last = v.get_last_result()
        assert last is not None
        assert last.valid is True

    def test_warns_on_missing_leaf_fields(self):
        """validate_structured should collect warnings for missing optional fields."""
        v = HPMValidator()
        result = v.validate_structured({
            "type": "ASI_Input", "id": "bad_sensor",
            # missing dim, grounding_level
        })
        assert result.valid is True  # non-blocking
        assert len(result.warnings) >= 1

    def test_validation_result_errors(self):
        """Invalid tree should produce error messages."""
        v = HPMValidator()
        result = v.validate_structured({"type": "CONDITIONAL", "children": [
            {"type": "Predict"},
        ]})
        assert result.valid is False
        assert len(result.errors) > 0


class TestHPMReset:
    """Tests for reset()."""

    def test_reset_clears_state(self):
        """reset() should clear composition count and last result."""
        v = HPMValidator()
        v.validate({"type": "SEQUENCE", "children": [{"type": "Predict"}]})
        assert v.get_last_result() is not None
        v.reset()
        assert v.get_last_result() is None


class TestHPMNodeDataclass:
    """Tests for the HPMNode dataclass."""

    def test_create_leaf_node(self):
        """HPMNode with operator and module_id."""
        node = HPMNode(operator="ASI_Input", module_id="ASI")
        assert node.operator == "ASI_Input"
        assert node.module_id == "ASI"
        assert node.children == []

    def test_create_composite_node(self):
        """HPMNode with children."""
        leaf = HPMNode(operator="Predict", module_id="P1")
        comp = HPMNode(operator="SEQUENCE", children=[leaf])
        assert comp.operator == "SEQUENCE"
        assert len(comp.children) == 1
        assert comp.children[0].module_id == "P1"
