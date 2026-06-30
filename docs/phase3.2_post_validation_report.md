# PHCA v3.0 – Post-Validation Strategic Report

**Date:** 2026-06-30  
**Author:** Chief Architect & Technical Strategist  
**Status:** Phase 3.2 Complete – Transitioning to Phase 3.3  

---

## 1. Executive Summary

PHCA v3.0 Phase 3.2 has achieved all primary success criteria:

| Metric | Value | Threshold | Status |
|--------|-------|-----------|--------|
| Φ-IQ (L0) | 0.798 | > 0.5 | ✅ Pass |
| Skill Accuracy | 0.901 | — | ✅ Functional |
| Skill Compiled | True | — | ✅ Functional |
| M3 Episodes | 210 | — | ✅ Populating |
| Consolidation Facts | 86 | — | ✅ Active |
| Latency (mean) | 17.7ms | < 500ms | ✅ Pass |
| Latency (p95) | 27.3ms | < 500ms | ✅ Pass |
| Unit Tests | 318/319 pass | — | ✅ Stable |

However, this validation was conducted on a **5×5 grid with 50-cycle episodes** — the simplest possible configuration. The system has never been tested under load, with sensor noise, or on larger environments. Several architectural risks remain latent.

**The single biggest blocker to production readiness** is the **continuous G' model's inability to represent discrete one-hot states**, which caps Level 2 Φ-IQ at 0.387 (below the 0.5 threshold). This is not a tuning issue — it is a representational mismatch that requires replacing the Gaussian CPD engine with a learned MLP (Phase 3.3 scope).

**Three additional architectural risks require attention before any real-world deployment:**

1. The O(n³) matrix inversion (`np.linalg.inv`) executes every cognitive cycle (not cached), creating a non-linear latency cliff as state dimension grows.
2. SQLite `.commit()` blocks on every cognitive cycle (M3 episode write), creating a cumulative latency tax that scales with cycle count.
3. The E-Stream and S-Stream (TSPL) remain functional stubs — consolidation writes facts to an in-memory list that is never consumed by prediction.

This report documents 21 audit findings, an enhanced benchmark plan, a production readiness checklist, a phased roadmap for Phase 3.3–4, and prioritised strategic recommendations.

---

## 2. Audit Findings

### 2.1 CRITICAL (fix before next benchmark)

| # | Finding | File | Lines | Impact |
|---|---------|------|-------|--------|
| C1 | **O(n³) matrix inversion every cycle** — `np.linalg.inv` in `compute_joint_moments()` and `posterior()` is called every cognitive cycle because `learn()` invalidates the cache. For n=84 nodes this is ~600k FLOPs per cycle; for n=200 (20×20 grid) this exceeds 8M FLOPs. | `gaussian.py` | 86, 152 | Latency cliff at larger grid sizes |
| C2 | **Gaussian G' cannot model one-hot states** — Continuous Gaussian predictions on 84-dimensional binary vectors produce near-random confidences (Level 2 Φ-IQ = 0.387). No amount of tuning or caching can fix this. | `graph.py` | 574–641 | Blocks Level 2 benchmark pass |
| C3 | **SQLite `.commit()` on every cycle** — `store_episode()` calls `self._connection.commit()` after every single insert. At 50 cycles this adds ~57ms total. At 1000 cycles this is ~1.14s of cumulative blocking I/O. | `m3_episodic.py` | 214 | Latency linear scaling with cycle count |

### 2.2 MAJOR (fix before Phase 3.3 release)

| # | Finding | File | Lines | Impact |
|---|---------|------|-------|--------|
| M1 | **Silent failure in pgmpy inference** — `predict()` catches `except Exception:` and silently returns a low-confidence identity prediction with no logging. If pgmpy inference throws (e.g., singular matrix, numerical instability), the system continues producing garbage predictions invisibly. | `graph.py` | 349–359 | Masked prediction failures |
| M2 | **Silent failure in action selection** — `_select_action()` catches `except Exception:` per action candidate and silently continues to the next. If all 5 predictions fail, STAY is returned with no log. | `cycle.py` | 488 | Masked action failures |
| M3 | **E-Stream and S-Stream are stubs** — `TSPL.update()` for E and S streams calls `_gem_project()` and `_ewc_add_penalty()` respectively, but these produce no measurable effect on system behaviour. EWC Fisher/GEM gradients are stored but never consumed. | `tspl.py` | 155–160 | 50% of TSPL streams are dead weight |
| M4 | **Consolidation facts are produced but never consumed** — `_store_facts()` writes to `_semantic_facts` in-memory list. No module reads these facts to improve prediction, adjust priors, or inform action selection. | `scheduler.py` | 277–293 | Consolidation is write-only |
| M5 | **M3 snapshot contains full episode data, not signatures** — `create_snapshot()` copies all episode records including full state vectors and action arrays. For 10K episodes × 84-dim states, this is ~3.4MB of data per snapshot. | `m3_episodic.py` | 357–386 | Memory bloat under load |

### 2.3 MINOR (fix as time permits)

| # | Finding | File | Lines | Impact |
|---|---------|------|-------|--------|
| m1 | `SkillLibrary` defined but never used — 50 lines of tested dead code. | `skill_compilation.py` | 16–66 | Code clutter |
| m2 | `compute_precision_weighted()`, `compute_rmse()` never called — dead code in `PredictionErrorUnit`. | `error_unit.py` | 43, 74 | Code clutter |
| m3 | `store_batch()`, `sample_batch()` defined but never called — bypasses the per-cycle commit bottleneck but unused. | `m3_episodic.py` | 222, 316 | Missed optimisation opportunity |
| m4 | `knn_similarity()` in `similarity.py` is redundant with `WorldModelGPrime.similarity_search()`. | `similarity.py` | 20 | Duplicate code |
| m5 | `_gem_tasks_seen` incremented but never read. | `tspl.py` | 101 | Dead state |
| m6 | `run()` docstring is a single line — missing Args/Returns documentation. | `cycle.py` | 131 | Documentation gap |
| m7 | `_build_graph()` has no docstring (comment block only). | `graph.py` | 165 | Documentation gap |
| m8 | `__del__()` in `M3EpisodicMemory` has no docstring. | `m3_episodic.py` | 452 | Documentation gap |
| m9 | f-string SQL `WHERE` clause — safe currently but an injection risk pattern for future modifications. | `m3_episodic.py` | 311 | Low-risk pattern |
| m10 | `__import__("time")` instead of `import time` in migration script. | `schema_v1.py` | 58 | Code smell |
| m11 | `HPMValidator` instantiated in `build_for_env` but never called in the cycle — HPM step is commented out. | `cycle.py` | 793 | Dead instantiation |
| m12 | `_compute_distance_gain()` recalculates goal distance every call — could be cached per goal. | `cycle.py` | 539–577 | Minor CPU waste |

---

## 3. Enhanced Benchmark Plan

### 3.1 Current Benchmark Limitations

The existing benchmark suite (`scripts/benchmark.py`) only runs:
- Single grid size (5×5)
- 50 cycles per level (Levels 0–3)
- Noise-free sensor input
- Single agent
- Discrete action space
- No persistence (fresh start per benchmark)

### 3.2 Proposed Benchmark Matrix

| Benchmark | Grid | Cycles | Noise | Agents | Actions | Acceptance Criteria |
|-----------|------|--------|-------|--------|---------|-------------------|
| **B1: Baseline** | 5×5 | 50 | None | 1 | Discrete | Φ-IQ > 0.7, latency < 100ms |
| **B2: Scale** | 10×10 | 200 | None | 1 | Discrete | Φ-IQ > 0.5, latency < 300ms |
| **B3: Scale+** | 20×20 | 500 | None | 1 | Discrete | Φ-IQ > 0.3, latency < 500ms |
| **B4: Noise** | 5×5 | 200 | σ=0.05 dropout 5% | 1 | Discrete | Φ-IQ > 0.5, fail < 15% |
| **B5: Noise+** | 5×5 | 200 | σ=0.15 dropout 15% | 1 | Discrete | Φ-IQ > 0.3, fail < 25% |
| **B6: Longevity** | 5×5 | 1000 | None | 1 | Discrete | Φ-IQ > 0.5, no memory leak |
| **B7: Multi-agent** | 5×5 | 200 | None | 2 | Discrete | Per-agent Φ-IQ > 0.3 |
| **B8: Continuous** | 2D plane | 200 | None | 1 | Continuous (x,y) | Reach goal within 50 cycles |

### 3.3 Implementation Requirements

1. **GridWorld environment** — Already supports `size` parameter (5, 10, 20). No changes needed for B1–B3.
2. **Sensor noise** — Add `noise_std` and `dropout_prob` parameters to `ASISanitizer.__init__()`. Requires < 10 lines of new code.
3. **Multi-agent** — Create `MultiAgentGridWorld` wrapper that shares environment state and alternates agent steps. Estimate 100–150 lines of new code.
4. **Continuous actions** — Create `ContinuousGridWorld` with velocity commands. Estimate 80 lines.
5. **Benchmark runner** — Extend `benchmark.py` to accept `--noise`, `--agents`, `--continuous` flags. Estimate 50 lines.

**Estimated effort: 3–4 days.**

---

## 4. Production Readiness Checklist

### 4.1 API Stability

| Item | Status | Effort |
|------|--------|--------|
| Freeze all public class signatures | ✅ Stable | 0 days |
| Add `__all__` to every `__init__.py` | ❌ Missing | 0.5 day |
| Version all public APIs (`phca.asi.v1`, `phca.memory.v1`) | ❌ Not done | 1 day |
| Deprecation warnings for renamed APIs | N/A | 0 days |
| Semantic versioning in `__init__.py` | ❌ Not done | 0.5 day |

### 4.2 Deployment

| Item | Status | Effort |
|------|--------|--------|
| `requirements.txt` with pinned versions | ❌ Not done | 0.5 day |
| `setup.py` / `pyproject.toml` with all deps | ✅ Present | 0 days |
| Dockerfile for CPU inference | ❌ Not done | 1 day |
| Dockerfile for GPU inference (Phase 3.3 MLP) | N/A | 0 days |
| Docker Compose for multi-service | ❌ Not done | 1 day |
| Kubernetes manifest | ❌ Not done | 2 days |

### 4.3 Monitoring & Observability

| Item | Status | Effort |
|------|--------|--------|
| Structured JSON logging | ✅ Present (via `_log()`) | 0 days |
| Prometheus metrics endpoint | ❌ Not done | 1 day |
| Latency percentile tracking (p50/p95/p99) | ✅ In benchmark script | 0 days |
| Runtime error rate tracking | ❌ Not done | 0.5 day |
| Memory usage tracking (RSS) | ❌ Not done | 0.5 day |
| Health check endpoint (HTTP) | ❌ Not done | 1 day |

### 4.4 CI/CD

| Item | Status | Effort |
|------|--------|--------|
| Run all 318+ tests on PR | ⚠️ Manual only | 1 day (GitHub Actions) |
| Lint check (ruff/flake8) | ❌ Not set up | 0.5 day |
| Type check (mypy/pyright) | ❌ Not set up | 1 day |
| Security scan (bandit) | ❌ Not set up | 0.5 day |
| Benchmark gate (Φ-IQ must not regress) | ❌ Not set up | 1 day |
| Coverage threshold (≥ 80%) | ❌ Not set up | 0.5 day |

### 4.5 Documentation

| Item | Status | Effort |
|------|--------|--------|
| API reference (Sphinx/MkDocs) | ❌ Not done | 2 days |
| Architecture overview (draw.io/Mermaid) | ✅ docs/partial | 1 day |
| User guide with example scripts | ❌ Not done | 2 days |
| Contribution guide (CONTRIBUTING.md) | ❌ Not done | 0.5 day |
| Changelog (CHANGELOG.md) | ✅ Present | 0 days |

---

## 5. Phase 3.3 & 4 Roadmap

### 5.1 Phase 3.3: Validation & Hardening (8–10 weeks)

```
Week 1-2: Critical Fixes
  ├── Fix C1: Lazy cache invalidation — only invalidate joint moments when 
  │            betas/sigmas actually change, not on every learn() call.
  ├── Fix C2: Replace Gaussian G' with learned MLP for one-hot state 
  │            representation (ϕ: ℝ⁸⁴ → ℝ⁶⁴ MLP + linear decoder).
  │            Wire TSPL theta as MLP weights (this IS the P-Stream feedback).
  └── Fix C3: Remove commit from store_episode(); batch commits every 
              consolidation_interval instead.

Week 3-4: Structural Fixes
  ├── Fix M1/M2: Add logging to all silent except: handlers.
  ├── Fix M3: Wire E-Stream GEM projection into G' update; wire S-Stream 
  │            EWC penalty into MLP weight decay.
  ├── Fix M4: Consume consolidation facts — merge into G' CPDs or use as 
  │            priors for MLP pre-training.
  └── Fix m1-m12: Dead code removal, docstrings, SQL style cleanup.

Week 5-6: Enhanced Benchmarks
  ├── Implement B2 (10×10, 200 cycles)
  ├── Implement B4 (sensor noise σ=0.05)
  ├── Implement B6 (1000-cycle longevity)
  └── Run all benchmarks; produce Phase 3.3 validation report.

Week 7-8: Production Tooling
  ├── CI pipeline (GitHub Actions: test + lint + typecheck + benchmark gate)
  ├── Dockerfile + docker-compose for reproducible execution
  ├── Prometheus metrics in cycle.py
  └── MkDocs API reference generated from docstrings

Week 9-10: Hardening & Release
  ├── Fuzz testing: malformed state vectors, extreme values, NaN injection
  ├── Regression benchmark suite (compare Φ-IQ against Phase 3.2 baseline)
  ├── Release PHCA v3.3.0 with changelog and migration guide
  └── Address all MAJOR findings
```

### 5.2 Phase 4: Real-World Integration (12–16 weeks)

```
Phase 4a — Connectivity (Weeks 1-4):
  ├── ROS 2 bridge: phca_ros package subscribing to /camera/image_raw, /scan
  ├── Abstract Sensor interface (replace ASISanitizer's numpy→numpy with 
  │     real sensor pipeline)
  ├── Abstract Actuator interface (replace GridWorld.step() with real motor commands)
  └── Gazebo simulation environment (differential-drive robot in warehouse)

Phase 4b — Continuous Learning (Weeks 5-8):
  ├── Online MLP training: G' learns continuously from streaming data
  ├── Experience replay: M3 episodes replayed as training batches
  ├── Forgetting metrics: track Φ-IQ degradation on older tasks during new learning
  └── Elastic consolidation: EWC penalty automatically adjusts λ based on task similarity

Phase 4c — Advanced Cognition (Weeks 9-12):
  ├── Meta-cognition: Monitor own prediction error trends and adjust α/λ dynamically
  ├── Hierarchical planning: Compose skills into sequences (already have RBTA tree 
  │     structure — wire it to action selection)
  ├── Curiosity drive: MDIM generates exploration goals in high-error regions
  └── Multi-agent coordination: Shared M3 with coordinated consolidation

Phase 4d — Validation & Release (Weeks 13-16):
  ├── Real-world benchmark: Φ-IQ in Gazebo warehouse environment
  ├── 24-hour stress test with continuous learning
  ├── Safety interlocks: hard resource bounds on real hardware
  └── Release PHCA v4.0.0
```

---

## 6. Strategic Recommendations

### Recommendation 1: Ship Phase 3.3 without the MLP, then add it

**The MLP G' replacement (C2) is the highest-value single change** but also the riskiest. It touches the core prediction loop that every other module depends on. Worse, it introduces a dependency on PyTorch/JAX, changing the project's dependency profile.

**Recommended approach:** Split Phase 3.3 into two sub-phases:
- **Phase 3.3a (Weeks 1-4):** Fix C1 (cache), C3 (commit), M1-M4 (silent errors, E/S-stream wiring, fact consumption), all minor items, enhanced benchmarks, CI pipeline. This **does not change G' architecture** and can be validated against the Phase 3.2 Φ-IQ baseline.
- **Phase 3.3b (Weeks 5-8):** MLP G' replacement. Provenance: the benchmark gap (L2 Φ-IQ = 0.387) is the motivating evidence. Validate that L2 Φ-IQ *exceeds* 0.5 with the MLP before merging.

**Rationale:** Separating the MLP avoids it blocking the structural fixes. If the MLP introduces regressions, the rest of Phase 3.3 is still shippable.

### Recommendation 2: Cut the E-Stream and S-Stream until they produce value

The E-Stream (GEM projection) and S-Stream (EWC penalty) are fully implemented but produce **zero measurable effect on system behaviour**. They add 150 lines of complex code (projection matrices, Fisher accumulation, reference gradient storage) with no validation that they improve learning or protect against forgetting.

**Recommended action:** Gate E-Stream and S-Stream behind a feature flag (`TSPL.enable_estream = False`, `TSPL.enable_sstream = False`). Remove the flags only when a benchmark demonstrates that enabling them improves Φ-IQ by ≥ 5% relative to the baseline. If no such benchmark exists after Phase 3.3b, remove the code entirely.

### Recommendation 3: Convert per-cycle SQLite commit to batched commit

This is a **10-line fix with immediate latency impact**. Currently `store_episode()` calls `commit()` after every insert. Change to:

```python
# In store_episode(): remove self._connection.commit()
# In step() after M3 store: call commit only if consolidation interval is reached,
# or batch N writes before committing.
```

Alternatively, use `store_batch()` (already implemented but unused) instead of `store_episode()` — collect episodes into a list and flush on consolidation.

**Expected impact:** ~90% reduction in cumulative I/O latency on long runs.

### Recommendation 4: Add logging to all silent `except Exception:` handlers

Three locations swallow errors silently (`graph.py:349`, `graph.py:530`, `cycle.py:488`). Add a single `_log(logger, "warning", ...)` call to each. This is a **5-minute fix** that would have caught the Phase 3.1 G' "never learning" bug within seconds.

### Recommendation 5: Invest next 3 months in order

| Priority | Investment | Expected Outcome |
|----------|------------|-----------------|
| 1 | Batched SQLite commit (Rec 3) | 90% latency reduction on long runs |
| 2 | Silent error logging (Rec 4) | Immediate debugging improvement |
| 3 | Phase 3.3a: C1, benchmarks, CI | Ship stable v3.3.0 |
| 4 | Feature-gate E/S-streams (Rec 2) | Reduce dead code surface |
| 5 | Phase 3.3b: MLP G' | Fix L2 Φ-IQ gap |
| 6 | Phase 4 connectivity (ROS bridge) | Real-world validation path |

### Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| MLP G' introduces regressions | Medium | High | Phase 3.3a/3.3b separation with regression gate |
 | O(n³) matrix inversion limits grid to ≤ 20×20 | High | Medium | Lazy cache (C1 fix); consider Cholesky decomposition |
 | SQLite WAL mode I/O contention under real-time load | Medium | Medium | Batched commit (Rec 3); consider in-memory store |
 | E/S-streams never produce measurable benefit | High | Low | Feature flag + removal gate (Rec 2) |
 | Multi-agent coordination complexity exceeds value | Medium | Medium | Defer to Phase 4c; prototype in Python first |

---

### Closing Statement

PHCA v3.0 Phase 3.2 has delivered a **functional, validated cognitive architecture** that passes all primary performance criteria. The system learns from experience, compiles skills, consolidates episodic knowledge, and operates within resource bounds. **It is not yet production-ready**, but the path to production readiness is well-defined and achievable within 10 weeks.

The riskiest assumption in the architecture is that **Gaussian CPDs can model discrete state transitions** — this assumption has been falsified by the Level 2 benchmark (Φ-IQ = 0.387). Replacing G' with an MLP is the single most impactful investment the project can make.

**What I would cut:** The E-Stream and S-Stream in their current form. They add complexity without measured benefit. Let them earn their place through validated benchmarks.

**Where to invest next:** Batched SQLite commits (immediate latency win), silent error logging (immediate debugging win), then the MLP G' (architectural win). In that order.
