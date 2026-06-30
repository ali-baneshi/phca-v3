# PHCA v3.0 — Test Coverage Gap Report

**Date:** 2026-06-30  
**Before:** 373 tests (existing)  
**After:** 398 tests (+25 new)  

## 1. Before/After Summary

| Category | Before | After | Delta |
|----------|-------:|------:|------:|
| Unit tests | 373 | 398 | **+25** |
| Edge case tests | 0 | 13 | +13 |
| Stress tests (slow) | 0 | 5 | +5 |
| Chaos/resilience tests | 0 | 7 | +7 |

| Area | Before | After | Delta |
|------|-------:|------:|------:|
| ASI sanitizer edge cases | 11 | 17 | +6 |
| M2 working memory | 0 | 2 | +2 |
| M3 episodic memory capacity | 0 | 3 | +3 |
| RBTA classification boundaries | 2 | 6 | +4 |
| Cognitive cycle boundaries | 0 | 2 | +2 |
| Multi-cycle stress (≥100 cycles) | 0 | 5 | +5 |
| Sensor chaos / recovery | 0 | 4 | +4 |
| Total test files | 8 | 11 | +3 |

## 2. New Test Files

| File | Tests | Coverage | Runtime |
|------|-------|----------|---------|
| `tests/test_edge_cases.py` | 13 | ASI empty/wrong-size, M2 eviction, M3 empty+capacity, RBTA boundaries, cycle zero-cycle/wrong-dim | <2s |
| `tests/test_stress.py` | 5 | 100-cycle run, latency stability, memory bound, learning convergence, Φ stability | ~10s (marked `slow`) |
| `tests/test_chaos.py` | 7 | Sensor dropout, full death+recovery, global failure, slow module, memory exhaustion | <2s |
| `pytest.ini` | — | Registers `slow` marker | — |

## 3. Gap Analysis

### Critical gaps closed

| Gap | Before | After | Test |
|-----|--------|-------|------|
| ASI with `sensor_dim=0` | Untested | `test_asi_empty_vector` | Verifies empty vector accepted |
| ASI with wrong-sized input | Untested | `test_asi_wrong_size_vector` | Verifies `AssertionError` raised |
| M2 full-buffer eviction | Untested | `test_m2_evicts_lowest_salience` | Verifies lowest-salience chunk popped |
| M3 consolidation with empty store | Untested | `test_m3_empty_consolidation` | Verifies no-crash, success=True |
| M3 fill-to-capacity eviction | Untested | `test_m3_fill_to_capacity_then_evict` | Verifies FIFO eviction at boundary |
| RBTA 0-violation classification | Untested | `test_rbta_no_violations_continue` | Boundary: 0→CONTINUE |
| RBTA 1-violation classification | Untested | `test_rbta_single_violation_interrupt` | Boundary: 1→INTERRUPT |
| RBTA 3-violation classification | Untested | `test_rbta_three_violations_terminate` | Boundary: 3→TERMINATE |
| RBTA unknown module handling | Untested | `test_rbta_missing_module_id_ignored` | Verifies no crash |
| Cycle with wrong sanitizer dim | Untested | `test_cycle_wrong_sanitizer_dim` | Verifies early crash (safety) |
| Cycle run(n_cycles=0) defaults | Untested | `test_cycle_zero_cycles` | Verifies safe defaults |

### Stress gaps closed

| Gap | Before | After | Test |
|-----|--------|-------|------|
| Sustained 100-cycle operation | 10-cycle max | `test_100_cycle_run` | 100 cycles, all complete |
| Latency stability over 100 cycles | 10-cycle check | `test_latency_stable` | p50<1000ms, p95<2000ms |
| Memory growth over cycles | None | `test_no_memory_leak` | `metrics_history` capped at 10k |
| Learning convergence | 50-cycle check | `test_prediction_error_decreases` | Late error <150% early error |
| Φ-IQ criticality collapse | None | `test_phi_iq_stable` | Mean Φ > 0.1 over last 20 cycles |

### Resilience gaps closed

| Gap | Before | After | Test |
|-----|--------|-------|------|
| Random sensor dropout | None | `test_asi_random_sensor_dropout` | 50% NaN for 50 cycles |
| Full sensor death + recovery | None | `test_asi_all_dead_sensors_recovers` | 7×NaN → SENSOR_FAILURE → valid → recovery |
| Global sensor failure | Partially | `test_asi_global_failure_recovery` | > d/3 sensors fail then recover |
| Slow module RBTA detection | Artificial | `test_slow_module_triggers_rbta` | Tight bound → violations detected |
| M3 memory exhaustion | Pruning only | `test_m3_evicts_oldest_when_over_capacity` | FIFO eviction verified |

## 4. Remaining Gaps (Phase 3.3)

| Gap | Priority | Reason deferred |
|-----|----------|----------------|
| Concurrent write contention (M2/M4) | Medium | Requires threading test harness |
| Network partition / timeout | Low | No networking in Phase 3.1 |
| GPU OOM (MLP mode) | Low | MLP mode is Phase 3.3b |
| Corrupted SQLite database | Low | Requires pre-seeded corrupted DB |
| Power-loss mid-cycle | Low | Requires simulation infrastructure |
| 1M+ cycle run | Low | 398 tests×14s ≈ 93min; marked `slow` already |

## 5. Test Infrastructure Changes

| Change | File | Purpose |
|--------|------|---------|
| `[pytest] markers = slow: ...` | `pytest.ini` | Register `slow` marker for stress tests |
| `pytestmark = pytest.mark.slow` | `test_stress.py` | Stress tests excluded from default runs |
| `pytestmark = pytest.mark.timeout(60)` | `test_edge_cases.py` | Timeout guard for edge case tests |
| `pytestmark = pytest.mark.timeout(120)` | `test_stress.py` | Timeout guard for 100-cycle stress tests |
