"""
Shared test fixtures for the PHCA v3.0 project.

Provides deterministic fixtures for environment, memory, ASI, etc.
All stochastic components use fixed seed=42.
"""

import numpy as np
import pytest

from phca.config import StateVector, ResourceBounds
from phca.asi.sanitizer import ASISanitizer
from phca.memory.m1_sensory import M1SensoryBuffer
from phca.memory.m2_working import M2WorkingMemory
from environments.grid_world import GridWorld


# ── Random seed fixture ──────────────────────────────────────

@pytest.fixture(autouse=True)
def set_random_seed():
    """All tests must be deterministic — fix the random seed."""
    np.random.seed(42)


# ── ASI fixtures ─────────────────────────────────────────────

@pytest.fixture
def asi_sanitizer():
    """A 4-element ASI sanitizer for testing."""
    return ASISanitizer(sensor_dim=4, v_max=100.0, epsilon_confidence=0.01)


@pytest.fixture
def clean_state():
    """A clean 4-element state vector for testing."""
    return StateVector(
        values=np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32),
        precision=np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
        timestamp=0.0,
        grounding_level=1,
    )


@pytest.fixture
def faulty_state():
    """A 4-element state vector with NaN in the first element."""
    return StateVector(
        values=np.array([np.nan, 0.2, 0.3, 0.4], dtype=np.float32),
        precision=np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
        timestamp=0.0,
        grounding_level=1,
    )


# ── Memory fixtures ──────────────────────────────────────────

@pytest.fixture
def m1_buffer():
    """A sensory buffer with sensor_dim=4."""
    return M1SensoryBuffer(sensor_dim=4)


@pytest.fixture
def m2_memory():
    """A working memory with default capacity (7)."""
    return M2WorkingMemory(capacity=7)


# ── Environment fixtures ─────────────────────────────────────

@pytest.fixture(params=[5, 10, 20])
def grid_world(request):
    """A grid-world fixture for all standard sizes (5, 10, 20)."""
    return GridWorld(size=request.param, seed=42)


@pytest.fixture
def small_grid_world():
    """A 5×5 grid-world with no random walls for deterministic testing."""
    grid = GridWorld(size=5, obstacles=[], seed=42)
    return grid


# ── Resource bound fixtures ──────────────────────────────────

@pytest.fixture
def default_resource_bounds():
    """Default resource bounds for all Phase 3.1 modules."""
    return {
        "ASI": ResourceBounds(B_time=0.002, B_mem=100_000, B_energy=10.0),
        "WM": ResourceBounds(B_time=0.005, B_mem=50_000, B_energy=5.0),
        "G'": ResourceBounds(B_time=0.020, B_mem=500_000, B_energy=50.0),
        "PE": ResourceBounds(B_time=0.025, B_mem=200_000, B_energy=20.0),
    }
