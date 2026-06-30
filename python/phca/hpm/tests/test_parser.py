"""Tests for HPM composite resource bound computation."""

from __future__ import annotations

import pytest

from phca.hpm.parser import (
    HPMValidator,
    CompositionOp,
    TAU_COMP,
    TAU_SYNC,
    DELTA_SHARED,
    DELTA_COMM,
)


def test_leaf_bound() -> None:
    v = HPMValidator()
    tree = {"type": "ASI_Input", "id": "ASI"}
    bounds = v.compute_composite_bounds(tree)
    assert bounds is not None
    assert "B_time" in bounds
    assert "B_mem" in bounds


def test_sequence_bounds() -> None:
    v = HPMValidator()
    tree = {
        "type": "SEQUENCE", "id": "seq",
        "children": [
            {"type": "ASI_Input", "id": "ASI"},
            {"type": "Predict", "id": "G'_PE"},
        ],
    }
    bounds = v.compute_composite_bounds(tree)
    assert bounds is not None
    assert bounds["B_time"] > 0
    assert bounds["B_mem"] > 0


def test_parallel_bounds() -> None:
    v = HPMValidator()
    tree = {
        "type": "PARALLEL", "id": "par",
        "children": [
            {"type": "Predict", "id": "gprime"},
            {"type": "Predict", "id": "mlp"},
        ],
    }
    bounds = v.compute_composite_bounds(tree)
    assert bounds is not None
    assert bounds["B_time"] > 0
    assert bounds["B_mem"] > 0


def test_sequence_time_additivity() -> None:
    """SEQUENCE time = sum(child_time) + TAU_COMP."""
    v = HPMValidator()
    log = {"A": 0.1, "B": 0.2}
    tree = {
        "type": "SEQUENCE", "id": "test",
        "children": ["A", "B"],
    }
    bounds = v.compute_bounds(tree, log)
    assert bounds is not None
    assert abs(bounds["B_time"] - (0.1 + 0.2 + TAU_COMP)) < 1e-6


def test_parallel_time_additivity() -> None:
    """PARALLEL time = max(child_time) + TAU_SYNC."""
    v = HPMValidator()
    log = {"A": 0.1, "B": 0.2}
    tree = {
        "type": "PARALLEL", "id": "test",
        "children": ["A", "B"],
    }
    bounds = v.compute_bounds(tree, log)
    assert bounds is not None
    assert abs(bounds["B_time"] - (0.2 + TAU_SYNC)) < 1e-6


def test_sequence_memory_additivity() -> None:
    """SEQUENCE mem = max(child_mem) + DELTA_SHARED."""
    v = HPMValidator()
    tree = {
        "type": "SEQUENCE", "id": "test",
        "children": ["A", "B"],
    }
    bounds = v.compute_bounds(tree, {})
    assert bounds is not None
    assert abs(bounds["B_mem"] - (1024.0 + DELTA_SHARED)) < 1e-6


def test_parallel_memory_additivity() -> None:
    """PARALLEL mem = sum(child_mem) + DELTA_COMM."""
    v = HPMValidator()
    tree = {
        "type": "PARALLEL", "id": "test",
        "children": ["A", "B"],
    }
    bounds = v.compute_bounds(tree, {})
    assert bounds is not None
    assert abs(bounds["B_mem"] - (1024.0 + 1024.0 + DELTA_COMM)) < 1e-6


def test_nested_composition() -> None:
    v = HPMValidator()
    tree = {
        "type": "SEQUENCE", "id": "cognitive_cycle",
        "children": [
            "ASI",
            {
                "type": "SEQUENCE", "id": "prediction_block",
                "children": ["G'_PE", "PEU", "TSPL-P"],
            },
            {
                "type": "PARALLEL", "id": "regulation_block",
                "children": ["MDIM", "CR", "ATTN"],
            },
            "CYCLE",
        ],
    }
    bounds = v.compute_composite_bounds(tree)
    assert bounds is not None
    assert bounds["B_time"] > 0
    assert bounds["B_mem"] > 0


def test_empty_runtime_log_fallback() -> None:
    """Empty log should not crash."""
    v = HPMValidator()
    tree = {"type": "Predict", "id": "gprime"}
    bounds = v.compute_bounds(tree, {})
    assert bounds is not None


def test_recursive_bounds() -> None:
    v = HPMValidator()
    tree = {
        "type": "RECURSE", "id": "recurse", "n": 3,
        "children": [{"type": "Predict", "id": "gprime"}],
    }
    bounds = v.compute_composite_bounds(tree)
    assert bounds is not None
    assert bounds["B_time"] > 0


def test_conditional_bounds() -> None:
    v = HPMValidator()
    tree = {
        "type": "CONDITIONAL", "id": "if",
        "children": [
            {"type": "Predict", "id": "predictor"},
            {"type": "Predict", "id": "true_branch"},
            {"type": "Predict", "id": "false_branch"},
        ],
    }
    bounds = v.compute_composite_bounds(tree)
    assert bounds is not None
    assert bounds["B_time"] > 0


def test_hierarchy_bounds() -> None:
    v = HPMValidator()
    tree = {
        "type": "HIERARCHY", "id": "hier",
        "children": [
            {"type": "Predict", "id": "predictor"},
            {"type": "Predict", "id": "sub_module"},
        ],
    }
    bounds = v.compute_composite_bounds(tree)
    assert bounds is not None
    assert bounds["B_time"] > 0


def test_energy_computation() -> None:
    v = HPMValidator()
    log = {"A": 0.1, "B": 0.2}
    tree = {
        "type": "SEQUENCE", "id": "seq",
        "children": ["A", "B"],
    }
    bounds = v.compute_bounds(tree, log)
    assert bounds is not None
    assert "B_energy" in bounds


def test_nested_time_propagation() -> None:
    """Runtime log propagation through nested trees."""
    v = HPMValidator()
    log = {"inner": 0.05, "outer": 0.03}
    tree = {
        "type": "SEQUENCE", "id": "root",
        "children": [
            {
                "type": "PARALLEL", "id": "inner_group",
                "children": ["inner", {"type": "Predict", "id": "nested"}],
            },
            "outer",
        ],
    }
    bounds = v.compute_bounds(tree, log)
    assert bounds is not None
    assert bounds["B_time"] == pytest.approx(
        # inner=0.05 (string child, no leaf padding), nested=0.001 (dict leaf), outer=0.03 (string)
        max(0.05, 0.001) + TAU_SYNC + 0.03 + TAU_COMP, abs=1e-6)
