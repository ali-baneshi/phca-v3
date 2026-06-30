"""Tests for PHCA-3.1-010: P-Stream TSPL + Skill Compilation."""

from __future__ import annotations

import numpy as np
import pytest

from phca.config import StateVector, StreamID
from phca.learning.tspl import TSPL
from phca.learning.skill_compilation import SkillLibrary


# ── Fixtures ──────────────────────────────────────────────────


@pytest.fixture
def tspl() -> TSPL:
    model = TSPL(seed=42)
    model.init_parameters("gprime_cpd_transition", (4, 4))
    model.init_parameters("gprime_cpd_bias", (4,))
    return model


@pytest.fixture
def sample_state() -> StateVector:
    return StateVector(
        values=np.array([0.5, -0.3], dtype=np.float32),
        precision=np.array([0.9, 0.9], dtype=np.float32),
    )


@pytest.fixture
def sample_prediction() -> StateVector:
    return StateVector(
        values=np.array([0.48, -0.28], dtype=np.float32),
        precision=np.array([0.9, 0.9], dtype=np.float32),
    )


# ── TSPL Tests ───────────────────────────────────────────────


class TestTSPLInit:
    """Tests for TSPL initialization."""

    def test_default_configs(self):
        """Default stream configs should have correct ordering."""
        tspl = TSPL()
        # P-Stream: highest alpha, lowest lambda, highest eta
        assert tspl.configs[StreamID.P_STREAM].alpha == 0.05
        assert tspl.configs[StreamID.P_STREAM].lambda_ == 0.01
        assert tspl.configs[StreamID.P_STREAM].eta == 0.1
        assert tspl.configs[StreamID.P_STREAM].accuracy_threshold == 0.95

        # E-Stream: medium alpha, medium lambda, medium eta
        assert tspl.configs[StreamID.E_STREAM].alpha == 0.005
        assert tspl.configs[StreamID.E_STREAM].lambda_ == 0.1
        assert tspl.configs[StreamID.E_STREAM].eta == 0.01

        # S-Stream: lowest alpha, highest lambda, lowest eta
        assert tspl.configs[StreamID.S_STREAM].alpha == 0.0005
        assert tspl.configs[StreamID.S_STREAM].lambda_ == 1.0
        assert tspl.configs[StreamID.S_STREAM].eta == 0.001

    def test_initial_state(self):
        """Fresh TSPL should have no parameters."""
        tspl = TSPL()
        assert tspl.theta == {}
        assert not tspl.skill_compiled
        assert tspl.skill_accuracy == 0.0
        assert tspl.compiled_skill_ids == []

    def test_init_parameters_creates_array(self, tspl):
        """init_parameters should create float32 arrays."""
        assert "gprime_cpd_transition" in tspl.theta
        assert tspl.theta["gprime_cpd_transition"].shape == (4, 4)
        assert tspl.theta["gprime_cpd_transition"].dtype == np.float32

    def test_reset(self, tspl):
        """reset should clear all state."""
        tspl.update(StreamID.P_STREAM, 0.01, StateVector(values=np.zeros(2, dtype=np.float32), precision=np.ones(2, dtype=np.float32)),
                    StateVector(values=np.zeros(2, dtype=np.float32), precision=np.ones(2, dtype=np.float32)))
        assert tspl.skill_accuracy > 0
        tspl.reset()
        assert tspl.theta == {}
        assert not tspl.skill_compiled
        assert tspl.skill_accuracy == 0.0
        assert tspl.compiled_skill_ids == []


class TestTSPLUpdate:
    """Tests for TSPL.update()."""

    def test_p_stream_update_changes_theta(self, tspl, sample_state, sample_prediction):
        """P-Stream update should modify theta."""
        original = {k: v.copy() for k, v in tspl.theta.items()}
        theta_new, compiled = tspl.update(
            StreamID.P_STREAM, 0.01, sample_state, sample_prediction,
        )
        # Theta should be updated
        for key in original:
            assert not np.allclose(original[key], theta_new[key]), \
                f"{key} should change after P-Stream update"

    def test_p_stream_update_returns_theta(self, tspl, sample_state, sample_prediction):
        """P-Stream update should return theta dict."""
        theta_new, compiled = tspl.update(
            StreamID.P_STREAM, 0.01, sample_state, sample_prediction,
        )
        assert isinstance(theta_new, dict)
        assert "gprime_cpd_transition" in theta_new
        assert theta_new["gprime_cpd_transition"].shape == (4, 4)

    def test_e_stream_updates_theta(self, tspl, sample_state, sample_prediction):
        """E-Stream update should modify theta (Phase 3.2: GEM active)."""
        original = {k: v.copy() for k, v in tspl.theta.items()}
        theta_new, compiled = tspl.update(
            StreamID.E_STREAM, 0.01, sample_state, sample_prediction,
        )
        # Theta should be updated (no longer a stub)
        for key in original:
            assert not np.allclose(original[key], theta_new[key]), \
                f"{key} should change after E-Stream update"

    def test_e_stream_stores_references(self, tspl, sample_state, sample_prediction):
        """E-Stream updates should accumulate GEM reference gradients."""
        tspl.update(StreamID.E_STREAM, 0.01, sample_state, sample_prediction)
        assert tspl._gem_tasks_seen == 1
        assert len(tspl._gem_reference_grads) == 1

        tspl.update(StreamID.E_STREAM, 0.02, sample_state, sample_prediction)
        assert tspl._gem_tasks_seen == 2
        assert len(tspl._gem_reference_grads) == 2

    def test_e_stream_gem_projection(self, tspl, sample_state, sample_prediction):
        """GEM projection should not crash and return valid gradient."""
        # First update: establish reference
        tspl.update(StreamID.E_STREAM, 0.01, sample_state, sample_prediction)
        assert len(tspl._gem_reference_grads) == 1

        # Second update: should project against existing reference
        theta_new, compiled = tspl.update(
            StreamID.E_STREAM, 0.5, sample_state, sample_prediction,
        )
        assert isinstance(theta_new, dict)
        assert "gprime_cpd_transition" in theta_new

    def test_s_stream_updates_theta(self, tspl, sample_state, sample_prediction):
        """S-Stream update should modify theta (Phase 3.2: EWC active)."""
        original = {k: v.copy() for k, v in tspl.theta.items()}
        theta_new, compiled = tspl.update(
            StreamID.S_STREAM, 0.01, sample_state, sample_prediction,
        )
        # Theta should be updated (no longer a stub)
        for key in original:
            assert not np.allclose(original[key], theta_new[key]), \
                f"{key} should change after S-Stream update"

    def test_s_stream_ewc_fisher_update(self, tspl, sample_state, sample_prediction):
        """S-Stream should update Fisher information matrix."""
        tspl.update(StreamID.S_STREAM, 0.01, sample_state, sample_prediction)
        assert len(tspl._ewc_fisher) > 0
        assert "gprime_cpd_transition" in tspl._ewc_fisher

    def test_s_stream_ewc_fisher_accumulates(self, tspl):
        """Fisher information should accumulate over multiple updates.

        Uses different state/prediction inputs per update so the gradient
        differs, which causes the EMA Fisher to update. With identical
        inputs the gradient would be identical and the EMA would converge
        instantly (correct behavior — the test uses different inputs).
        """
        state1 = StateVector(
            values=np.array([0.5, -0.3], dtype=np.float32),
            precision=np.array([0.9, 0.9], dtype=np.float32),
        )
        pred1 = StateVector(
            values=np.array([0.48, -0.28], dtype=np.float32),
            precision=np.array([0.9, 0.9], dtype=np.float32),
        )
        state2 = StateVector(
            values=np.array([0.7, -0.1], dtype=np.float32),
            precision=np.array([0.8, 0.8], dtype=np.float32),
        )
        pred2 = StateVector(
            values=np.array([0.6, 0.0], dtype=np.float32),
            precision=np.array([0.8, 0.8], dtype=np.float32),
        )

        tspl.update(StreamID.S_STREAM, 0.01, state1, pred1)
        f1 = tspl._ewc_fisher["gprime_cpd_transition"].copy()

        tspl.update(StreamID.S_STREAM, 0.05, state2, pred2)
        f2 = tspl._ewc_fisher["gprime_cpd_transition"]
        # Fisher should be updated (running average with different gradient)
        assert not np.allclose(f1, f2)

    def test_reset_clears_gem_and_ewc(self, tspl, sample_state, sample_prediction):
        """Reset clears GEM and EWC state."""
        tspl.update(StreamID.E_STREAM, 0.01, sample_state, sample_prediction)
        tspl.update(StreamID.S_STREAM, 0.01, sample_state, sample_prediction)
        assert tspl._gem_tasks_seen > 0
        assert len(tspl._ewc_fisher) > 0

        tspl.reset()
        assert tspl._gem_tasks_seen == 0
        assert len(tspl._gem_reference_grads) == 0
        assert len(tspl._ewc_fisher) == 0
        assert len(tspl._ewc_theta_star) == 0

    def test_update_with_no_theta_returns_empty(self, sample_state, sample_prediction):
        """Update with no initialized parameters should return empty."""
        tspl = TSPL()
        theta_new, compiled = tspl.update(
            StreamID.P_STREAM, 0.01, sample_state, sample_prediction,
        )
        assert theta_new == {}
        assert not compiled

    def test_update_with_gradient(self, tspl, sample_state, sample_prediction):
        """Providing explicit gradient should be used instead of auto-computation."""
        gradient = {
            "gprime_cpd_transition": np.zeros((4, 4), dtype=np.float32),
            "gprime_cpd_bias": np.zeros((4,), dtype=np.float32),
        }
        theta_new, compiled = tspl.update(
            StreamID.P_STREAM, 0.01, sample_state, sample_prediction,
            gradient=gradient,
        )
        assert isinstance(theta_new, dict)

    def test_skill_not_compiled_on_large_error(self, tspl):
        """Large prediction error should not trigger skill compilation."""
        state = StateVector(
            values=np.array([0.5, -0.3], dtype=np.float32),
            precision=np.array([0.9, 0.9], dtype=np.float32),
        )
        prediction = StateVector(
            values=np.array([-0.5, 0.3], dtype=np.float32),  # large diff
            precision=np.array([0.9, 0.9], dtype=np.float32),
        )
        theta_new, compiled = tspl.update(
            StreamID.P_STREAM, 5.0, state, prediction,
        )
        assert not compiled
        assert tspl.skill_accuracy < 0.5

    def test_accuracy_estimation(self, tspl):
        """_estimate_accuracy should return 0 for large error, 1 for zero error."""
        state = StateVector(
            values=np.array([0.5, -0.3], dtype=np.float32),
            precision=np.array([0.9, 0.9], dtype=np.float32),
        )

        # Zero error → accuracy = 1.0
        acc1 = tspl._estimate_accuracy(state, state, 0.0)
        assert acc1 == pytest.approx(1.0)

        # Large error → accuracy near 0
        large_error = 10.0  # sqrt(10/2) ≈ 2.236 → max(0, 1-2.236) = 0
        acc2 = tspl._estimate_accuracy(state, state, large_error)
        assert acc2 == 0.0


class TestSkillCompilation:
    """Tests for skill compilation."""

    def test_freeze_skill(self, tspl):
        """freeze_skill should snapshot theta."""
        tspl.freeze_skill("test_skill_v1")
        assert "test_skill_v1" in tspl.compiled_skill_ids
        assert "gprime_cpd_transition" in tspl.theta_protected
        np.testing.assert_array_equal(
            tspl.theta_protected["gprime_cpd_transition"],
            tspl.theta["gprime_cpd_transition"],
        )

    def test_freeze_skill_is_snapshot_not_reference(self, tspl):
        """Protected parameters should be a copy, not a reference."""
        tspl.freeze_skill("test_skill_v1")
        old_val = tspl.theta_protected["gprime_cpd_transition"][0, 0]
        tspl.theta["gprime_cpd_transition"][0, 0] = 99.0
        assert tspl.theta_protected["gprime_cpd_transition"][0, 0] == old_val


class TestSkillLibrary:
    """Tests for SkillLibrary class."""

    def test_store_and_retrieve(self):
        """Store and retrieve a skill."""
        lib = SkillLibrary()
        params = {
            "transition": np.array([[0.8, 0.2], [0.2, 0.8]], dtype=np.float32),
        }
        lib.store("nav_v1", params, {"accuracy": 0.97})
        retrieved = lib.retrieve("nav_v1")
        assert retrieved is not None
        np.testing.assert_array_equal(retrieved["transition"], params["transition"])

    def test_retrieve_missing(self):
        """Retrieving non-existent skill should return None."""
        lib = SkillLibrary()
        assert lib.retrieve("nonexistent") is None

    def test_store_is_snapshot(self):
        """Stored params should be a copy."""
        lib = SkillLibrary()
        params = {"w": np.array([1.0, 2.0], dtype=np.float32)}
        lib.store("test", params)
        params["w"][0] = 99.0
        retrieved = lib.retrieve("test")
        assert retrieved["w"][0] == 1.0

    def test_list_skills(self):
        """List should return all stored skill IDs."""
        lib = SkillLibrary()
        lib.store("a", {"w": np.zeros(1)})
        lib.store("b", {"w": np.zeros(1)})
        skills = lib.list_skills()
        assert set(skills) == {"a", "b"}

    def test_get_metadata(self):
        """get_metadata should return stored metadata."""
        lib = SkillLibrary()
        lib.store("test", {"w": np.zeros(1)}, {"accuracy": 0.95, "cycles": 100})
        meta = lib.get_metadata("test")
        assert meta["accuracy"] == 0.95
        assert meta["cycles"] == 100

    def test_get_metadata_missing(self):
        """get_metadata for missing skill should return empty dict."""
        lib = SkillLibrary()
        assert lib.get_metadata("nonexistent") == {}
