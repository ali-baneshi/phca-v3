"""Tests for ASI NoiseInjector — grounding simulation module."""

import numpy as np

from phca.asi.noise_injector import NoiseInjector, NoiseProfile


class TestNoiseInjector:
    SENSOR_DIM = 84  # 5x5 GridWorld

    def test_init_defaults(self):
        inj = NoiseInjector(self.SENSOR_DIM)
        assert inj.sensor_dim == self.SENSOR_DIM
        assert inj.profile == NoiseProfile.GAUSSIAN
        assert inj.intensity == 0.1
        assert inj.get_stats()["total_noise_cycles"] == 0

    def test_zero_intensity_no_change(self):
        inj = NoiseInjector(self.SENSOR_DIM, initial_intensity=0.0)
        raw = np.random.randn(self.SENSOR_DIM).astype(np.float32)
        result = inj.inject(raw)
        assert np.allclose(raw, result)

    def test_gaussian_noise_shape(self):
        inj = NoiseInjector(self.SENSOR_DIM, initial_intensity=0.5)
        raw = np.ones(self.SENSOR_DIM, dtype=np.float32)
        result = inj.inject(raw)
        assert result.shape == raw.shape
        assert result.dtype == np.float32

    def test_gaussian_noise_clipped(self):
        inj = NoiseInjector(self.SENSOR_DIM, initial_intensity=10.0)
        raw = np.ones(self.SENSOR_DIM, dtype=np.float32) * 1.0
        result = inj.inject(raw)
        assert result.min() >= 0.0
        assert result.max() <= 2.0

    def test_dropout_noise_zeros_some(self):
        inj = NoiseInjector(self.SENSOR_DIM, initial_intensity=0.5)
        raw = np.ones(self.SENSOR_DIM, dtype=np.float32)
        result = inj.inject(raw)
        n_zero = (result == 0.0).sum()
        assert n_zero > 0, "dropout should zero some elements"

    def test_dropout_noise_keeps_others(self):
        inj = NoiseInjector(self.SENSOR_DIM, initial_intensity=0.9)
        raw = np.ones(self.SENSOR_DIM, dtype=np.float32)
        result = inj.inject(raw)
        assert result.shape == raw.shape
        assert result.dtype == np.float32

    def test_drift_noise_accumulates(self):
        inj = NoiseInjector(self.SENSOR_DIM, initial_intensity=0.3)
        raw = np.zeros(self.SENSOR_DIM, dtype=np.float32)
        r1 = inj.inject(raw, cycle=1)
        r2 = inj.inject(raw, cycle=2)
        r3 = inj.inject(raw, cycle=3)
        assert not np.allclose(r1, r2)
        assert not np.allclose(r2, r3)

    def test_salt_pepper_changes_values(self):
        inj = NoiseInjector(self.SENSOR_DIM, initial_intensity=0.5)
        inj.set_profile("salt_pepper")
        raw = np.ones(self.SENSOR_DIM, dtype=np.float32) * 1.0
        result = inj.inject(raw)
        assert not np.allclose(raw, result)
        assert set(np.unique(result)).issubset({0.0, 1.0, 2.0})

    def test_set_profile_and_intensity(self):
        inj = NoiseInjector(self.SENSOR_DIM)
        inj.set_profile("dropout")
        assert inj.profile == NoiseProfile.DROPOUT
        inj.set_intensity(0.9)
        assert inj.intensity == 0.9
        inj.set_intensity(1.5)
        assert inj.intensity == 1.0

    def test_reset_clears_drift(self):
        inj = NoiseInjector(self.SENSOR_DIM, initial_intensity=0.5)
        raw = np.zeros(self.SENSOR_DIM, dtype=np.float32)
        inj.set_profile("drift")
        inj.inject(raw, cycle=1)
        inj.inject(raw, cycle=2)
        assert np.abs(inj._drift_state).sum() > 0
        inj.reset()
        assert inj._drift_state.sum() == 0.0
        assert inj.get_stats()["total_noise_cycles"] == 0

    def test_inject_assert_shape_mismatch(self):
        inj = NoiseInjector(84)
        try:
            inj.inject(np.zeros(10, dtype=np.float32))
            assert False, "should have raised AssertionError"
        except AssertionError:
            pass
