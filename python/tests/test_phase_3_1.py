"""
Phase 3.1 Integration Tests (IT-3.1-1 through IT-3.1-5).

These tests validate end-to-end operation of the Phase 3.1 core engine.
They are skippable if dependencies are not yet fully implemented.

Cross-ref: Playbook §8.1 Integration Test Plan
"""

import numpy as np
import pytest

from phca.config import StateVector, ASIStatus
from phca.asi.sanitizer import ASISanitizer
from phca.memory.m1_sensory import M1SensoryBuffer
from phca.memory.m2_working import M2WorkingMemory
from environments.grid_world import GridWorld


class TestIT31_ASIToWM:
    """IT-3.1-1: ASI→M2 Pipeline — sanitizer output flows to WM correctly."""

    def test_sanitized_state_flows_to_wm(self, asi_sanitizer, m2_memory):
        """After sanitization, the state should be writable to M2."""
        raw = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        state, status = asi_sanitizer.sanitize(raw)
        assert status == ASIStatus.OK

        chunk = m2_memory.write(state)
        assert chunk is not None
        assert len(m2_memory.chunks) == 1
        np.testing.assert_array_almost_equal(chunk.state.values, raw)

    def test_nan_does_not_corrupt_wm(self, asi_sanitizer, m2_memory):
        """NaN in sensor should be sanitized before reaching M2."""
        # First establish baseline
        valid_raw = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        state, _ = asi_sanitizer.sanitize(valid_raw)
        m2_memory.write(state)

        # Now send NaN
        nan_raw = np.array([np.nan, 2.0, 3.0, 4.0], dtype=np.float32)
        clean_state, status = asi_sanitizer.sanitize(nan_raw)
        assert status == ASIStatus.OK
        assert not np.any(np.isnan(clean_state.values)), "NaN should not reach WM"
        chunk = m2_memory.write(clean_state)
        assert chunk.state.values[0] == 1.0  # should be last valid value


class TestIT31_PredictionErrorLoop:
    """IT-3.1-2: G'→PE→PEU→TSPL loop will be validated in later tickets."""

    def test_prediction_error_computation_placeholder(self):
        """Placeholder: will test prediction error computation when G' is available."""
        pytest.skip("G' not yet implemented — will test in PHCA-3.1-008")


class TestIT31_FullCognitiveCycle:
    """IT-3.1-3: Full cognitive cycle latency will be validated in PHCA-3.1-011."""

    def test_cycle_latency_placeholder(self):
        """Placeholder: will test full cycle latency in PHCA-3.1-011."""
        pytest.skip("Cognitive cycle orchestrator not yet implemented — will test in PHCA-3.1-011")


class TestIT31_RBTAEnforcement:
    """IT-3.1-4: RBTA violation detection (via FFI to Rust crate)."""

    def test_rbta_violation_detection_placeholder(self):
        """Placeholder: will test RBTA enforcer via Python FFI when available."""
        pytest.skip("RBTA Rust crate FFI not yet connected — will test via cargo test")


class TestIT31_SkillCompilation:
    """IT-3.1-5: P-Stream skill compilation will be validated in PHCA-3.1-010."""

    def test_skill_compilation_placeholder(self):
        """Placeholder: will test P-Stream skill compilation in PHCA-3.1-010."""
        pytest.skip("P-Stream not yet implemented — will test in PHCA-3.1-010")
