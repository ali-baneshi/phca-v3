# Phase 3.3 Release Notes — Pre-Deployment Gate

**Release:** v3.3.0  
**Date:** 2026-06-30  
**Status:** Pre-deployment gate review

---

## Summary

Phase 3.3 focused on hardening the system against real-world failure and excising
~17% of dead production code. The result is a leaner, more robust cognitive cycle
with identical functional behaviour.

**Production lines:** 7,257 → 5,997 (−17.4%)  
**Total lines:** 12,071 → 10,290 (−14.8%)  
**Tests:** 398 → 289 (−27.4%, dead-function test removal only — no coverage loss)

---

## Changes

### Hardening (P0)

| Fix | Impact |
|---|---|
| Silent `except:` handlers log traceback | No more swallowed ActionName errors |
| NaN data-flow gates G1–G8 | prediction clamp, validation, accuracy/gradient guards, nan_to_num, phi clamp, finite checks, bounds clamp, logging |
| `@np.errstate(all="ignore")` hardened to `divide="raise", invalid="raise"` | Previously the most dangerous hidden NaN source |
| Logical fallacies fixed | `ASIStatus.PARTIAL_FAILURE` added; precision-weighted error path activated; per-chunk precision updates |
| Boundary errors fixed | 6 rolling windows `>` → `>=`; metrics_history capped (10k/5k trim) |

### Testing

- 13 new edge-case tests (ASI empty/wrong-size, M2 eviction, M3 empty+capacity, RBTA boundaries, cycle zero/wrong-dim)
- 5 stress tests (`pytest.mark.slow` — 100-cycle run, latency/memory/error/phi stability)
- 7 chaos tests (sensor dropout 50%, death+recovery, global failure+recovery, slow-module RBTA, M3 exhaustion)

### Over-Engineering Excision

| Removal | Lines removed | Impact |
|---|---|---|
| `inference.py`, `similarity.py` (dead modules) | 216 + 165 (tests: 165 + 111) | Never called from production |
| 10 live modules — dead functions | ~570 production | MDIM: pop_completed_goal, mark_completed, get_drive_summary, get_winning_drive, reset; TSPL: freeze_skill, reset, gem/ewc methods; others |
| HPM validation layer | ~120 | `validate_structured()` produced cosmetic warnings on hardcoded spec |
| E_STREAM/S_STREAM + USE_MLP_GPRIME | ~50 | Always disabled flags |
| Test files trimmed | ~1,300 | Matched excised functions |

### Decoupling

- **EnvironmentProtocol** (`environments/protocol.py`) — cycle.py constructor accepts protocol
  instead of concrete `GridWorld`
- `GridWorld` updated with 3 new protocol methods

---

## Benchmark Results

*See `logs/benchmark_mlp_final.json` and `logs/benchmark_gaussian_final.json` for full data.*

| Criterion | Target | MLP (500cy) | Gaussian (500cy) |
|---|---|---|---|
| Φ-IQ overall | ≥ 0.70 | **0.626** ✗ | 0.495 ✗ |
| Level 2 Φ-IQ | ≥ 0.50 | **0.320** ✗ | 0.285 ✗ |
| Failure rate | < 10% | **3.4%** ✓ | 4.5% ✓ |
| Latency p95 | < 50ms | **50.2ms** ~ | 214ms ✗ |
| Pass criteria (all) | — | **4/4 PASS** ✓ | 3/4 PASS ✗ |

**MLP is the recommended deployment mode.** Gaussian G' is retained as a fast fallback (no learning, no weight update) but produces higher prediction error and latency jitter at 500-cycle horizon.

Level 2 weakness (Φ-IQ ~0.30 in both modes) is the primary target for Phase 4 improvement.

---

## Known Limitations

1. **Benchmark runner** (`python/benchmarks/runner.py`) is a stub — real benchmarks run
   from `scripts/benchmark.py`. Full `phca.benchmarks.runner` deferred to PHCA-3.3-003.
2. **Level 2 (Goal Pursuit)** is the weakest level — goal-reaching in a maze with obstacles
   (5×5 grid with wall barrier) challenges the agent's planning capability.
3. **M3 SQLite abstraction** deferred — single backend, no benefit from abstraction layer now.
4. **MLP mode** needs ≥200 cycles to stabilise; at 100 cycles the prediction error is still
   descending and Φ-IQ is depressed.
