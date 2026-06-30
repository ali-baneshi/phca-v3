# PHCA v3.0 — Decision Log

This file records all significant design decisions made during implementation.
Every entry must reference the v3.0 specification section it affects.

---

## Decision D-001: G' Inference Engine — pgmpy exact junction tree for Phase 3.1

- **Date:** 2026-06-29
- **Author:** Lead Implementation Engineer
- **Category:** Tier 2 (implementation-dependent)
- **Option chosen:** pgmpy exact junction tree for Phase 3.1 (|V| ≤ 100)
- **Alternatives:** Pyro variational inference (slower at small scale), custom loopy BP (more complex)
- **Rationale:** pgmpy provides exact inference for small graphs covering all Phase 3.1 tasks. Pyro reserved for Phase 3.2+ larger graphs with learned CPDs.
- **v3.0 trace:** §2.2 Def 2.4b

## Decision D-002: Rust workspace with 3 crates (common, rpta, hpm-runtime)

- **Date:** 2026-06-29
- **Author:** Lead Implementation Engineer
- **Category:** Tier 3 (open for optimization)
- **Option chosen:** Separate crates for each component with shared common types
- **Alternatives:** Single crate, Python-only implementation
- **Rationale:** Rust components are performance-critical (RBTA, HPM) and need separate compilation units. Shared types in `phca-common` prevents duplication.
- **v3.0 trace:** §2.1 Def 2.2, §3.2 Def 3.4

## Decision D-003: GridWorld state vector = 3×size² + 9 dimensions

- **Date:** 2026-06-29
- **Author:** Lead Implementation Engineer
- **Category:** Tier 2 (implementation-dependent)
- **Option chosen:** Agent one-hot map + goal one-hot map + wall map (3×size²) + 3×3 local neighborhood (9)
- **Alternatives:** Full grid encoding, compact coordinates
- **Rationale:** One-hot agent/goal/wall maps provide unambiguous position encoding. Local 3×3 view enables local navigation without full grid visibility.
- **v3.0 trace:** Blueprint §D.1

## Decision D-004: structlog fallback to stdlib logging

- **Date:** 2026-06-29
- **Author:** Lead Implementation Engineer
- **Category:** Tier 3
- **Option chosen:** try/except ImportError with `_log()` helper function
- **Alternatives:** Make structlog a hard dependency, use only stdlib logging
- **Rationale:** structlog provides structured logging for production use. stdlib fallback ensures code works without it during initial setup. The `_log()` wrapper handles both APIs transparently.
- **v3.0 trace:** None (tooling)

## Decision D-005: Uniform entropy floor for Phase 3.1

- **Date:** 2026-06-29
- **Author:** Lead Implementation Engineer
- **Category:** Tier 2
- **Option chosen:** All 13 modules use entropy_floor=0.01 in DEFAULT_MODULE_BOUNDS
- **Alternatives:** Differentiate by module type (sensory: 0.001, model: 0.01, meta: 0.05)
- **Rationale:** Uniform 0.01 works for Phase 3.1 grid-world tasks. Differentiating requires formal v3.0 spec interpretation. Will be updated when M3/M4 added in Phase 3.2.
- **v3.0 trace:** §2.3 Definition 2.2; Audit Finding V-3 (`12-sprint-0-audit.md`)

## Decision D-006: Split requirements into phase-specific files

- **Date:** 2026-06-29
- **Author:** Lead Implementation Engineer
- **Category:** Tier 3
- **Option chosen:** Three files: `requirements-phase-3.1.txt`, `requirements-phase-3.2.txt`, `requirements-dev.txt`
- **Alternatives:** Single monolithic requirements.txt, use of `pip extras`
- **Rationale:** Phase 3.1 drops 1.1GB of unused dependencies (torch, torchvision, pyro, sklearn). CI installs in 30s vs 5min. Dev deps separate to avoid production bloat.
- **v3.0 trace:** Audit Finding O-3 (`12-sprint-0-audit.md`)

## Decision D-007: Python-only RBTA for Phase 3.1

- **Date:** 2026-06-29
- **Author:** Lead Implementation Engineer
- **Category:** Tier 1 (must match v3.0 spec)
- **Option chosen:** Pure Python RBTA enforcer for Phase 3.1; Rust version tagged as experimental
- **Alternatives:** Rust RBTA with JSON FFI (original plan), Python mock with Rust production
- **Rationale:** Eliminates FFI overhead, CI Python 3.12+ compatibility issues, and simplifies Week 2-3 integration. Rust RBTA still present and tested via cargo test. Python RBTA is ~50 lines vs ~220 Rust lines.
- **v3.0 trace:** §2.1 Def 2.2; Audit Finding O-1 (`12-sprint-0-audit.md`)

## Decision D-008: orjson with stdlib json fallback

- **Date:** 2026-06-29
- **Author:** Lead Implementation Engineer
- **Category:** Tier 3
- **Option chosen:** try/except ImportError with stdlib json fallback for serialization
- **Alternatives:** Hard dependency on orjson, replace entirely with stdlib json
- **Rationale:** orjson is faster (3-4x) for Phase 3.2 SQLite BLOBs. stdlib fallback ensures no crash if not installed. The fallback pattern is consistent with the structlog approach in D-004.
- **v3.0 trace:** Audit Finding V-2 (`12-sprint-0-audit.md`)

## Decision D-009: G' graph structure for Phase 3.1 — 10 binary nodes

- **Date:** 2026-06-30
- **Author:** Lead Implementation Engineer
- **Category:** Tier 2 (implementation-dependent)
- **Option chosen:** G' model has 10 binary discrete nodes in `build_for_env()`, each representing one state dimension with a 60/40 temporal CPD
- **Alternatives:** Full 84-dim continuous model with Gaussian CPDs, hierarchical model with per-cell nodes
- **Rationale:** Binary nodes with simple CPDs keep pgmpy inference under 1ms for Phase 3.1. Full continuous modeling deferred to Phase 3.2 when VSA or differentiable G' is available. The simplified graph means prediction errors are near-zero and skill compilation triggers immediately — acceptable for infrastructure validation.
- **v3.0 trace:** §2.2 Def 2.4b (simplified, Phase 3.1)

## Decision D-010: Action selection — confidence maximization as D1 proxy

- **Date:** 2026-06-30
- **Author:** Lead Implementation Engineer
- **Category:** Tier 2
- **Option chosen:** Phase 3.1 action selection picks the action with highest G' prediction confidence (minimizing prediction uncertainty)
- **Alternatives:** Random action, heuristic rule-based, full MDIM (Phase 3.2+)
- **Rationale:** Without MDIM (Phase 3.2), we need a simple action policy. Confidence maximization is equivalent to D1 (prediction error minimization) since higher confidence = lower expected error. This works for simple grid-world tasks where staying in familiar regions is beneficial.
- **v3.0 trace:** §3.3 Def 3.5 (D1), §3.1 Def 3.2 (P-Stream action selection note)

## Decision D-011: Skill compilation triggers at 0 error — Phase 3.1 limitation

- **Date:** 2026-06-30
- **Author:** Lead Implementation Engineer
- **Category:** Tier 3
- **Option chosen:** Skill compiles immediately when prediction error is 0 (accuracy = 1.0)
- **Alternatives:** Require sustained accuracy over N cycles, require non-zero error threshold
- **Rationale:** With the simplified G' graph (D-009), prediction errors are near-zero, causing accuracy = 1.0. This is a Phase 3.1 artifact. In Phase 3.2 with proper continuous CPDs, error will be non-zero and the 95% threshold will gate compilation meaningfully.
- **v3.0 trace:** §3.1 Def 3.3.3 (skill compilation)

## Decision D-012: E/S-Stream feature gate via single `enabled` field (Phase 3.3a)

- **Date:** 2026-06-30
- **Author:** Implementation Engineer
- **Category:** Tier 2 (implementation-dependent)
- **Option chosen:** Added `enabled: bool = True` to `StreamConfig` dataclass; E-Stream and S-Stream set to `False` by default; wrapped E/S stream logic in `if config.enabled:` gates
- **Alternatives:** Separate `enabled_e/enabled_s` flags, removing E/S code entirely, passing env vars
- **Rationale:** Single `enabled` field is cleaner and general (can extend to future streams). Disabled by default to match Phase 3.3a scope (P-Stream only). Wrapping in-condition prevents dead code without removing it. No env vars or config files needed — pure Python toggle for CI determinism.
- **v3.0 trace:** §3.1 Def 3.2, §3.2 Def 3.2a/b

## Decision D-013: TSPL configs deep-copied in __init__ to prevent test pollution (Phase 3.3a)

- **Date:** 2026-06-30
- **Author:** Implementation Engineer
- **Category:** Tier 3
- **Option chosen:** `TSPL.__init__()` deep-copies `DEFAULT_STREAM_CONFIGS` via `dataclasses.asdict()` + re-construction instead of storing references
- **Alternatives:** `copy.deepcopy()`, requiring callsites to pass their own copies, defensive copying in each stream handler
- **Rationale:** The global `DEFAULT_STREAM_CONFIGS` dict was shared across all TSPL instances, causing test pollution when tests modified configs. `dataclasses.asdict()` + re-construction is faster than `copy.deepcopy()` for simple dataclasses and avoids accidental reference sharing.
- **v3.0 trace:** §3.1 Def 3.2

## Decision D-014: Batch SQLite commits with periodic flush (Phase 3.3a)

- **Date:** 2026-06-30
- **Author:** Implementation Engineer
- **Category:** Tier 3
- **Option chosen:** Replaced per-episode `commit()` with `flush()` (non-durable write + counter); call `m3.flush()` from consolidation scheduler every 10 cycles before `mark_consolidated`
- **Alternatives:** Single commit after all episodes, WAL mode, no commits until consolidation
- **Rationale:** Per-episode commit caused ~90% of cycle time in fsync. Batch commit (interval=10) reduces fsync overhead 10× while ensuring episodes written to disk before consolidation reads them. `flush()` with counter is simpler than tracking dirty pages. WAL mode deferred to Phase 3.3+ for potential speedup.
- **v3.0 trace:** §2.2 Def 2.7, §4.1 Def 4.1a

## Decision D-015: Pure NumPy MLP G' (Phase 3.3b)

- **Date:** 2026-06-30
- **Author:** Implementation Engineer
- **Category:** Tier 1 (must match v3.0 spec)
- **Option chosen:** 3-layer MLP (89→128→128→84) with ReLU, Sigmoid output, BCE loss, manual backward pass, gradient clipping to [-1, 1], confidence = exp(-mean_BCE)
- **Alternatives:** PyTorch MLP (fails CI — 1.1GB dep dropped in D-006), Gaussian CPD continuation, sklearn MLPRegressor, custom CUDA
- **Rationale:** No PyTorch dependency — Phase 3.3b requirement (confirmed `import torch` fails). Pure NumPy backward pass is ~40 lines and verified via finite-difference gradient checking (rel_error < 1% on all params). He initialization matches standard practice. 38,868 parameters (~3× Gaussian betas) provides enough capacity for GridWorld transitions.
- **v3.0 trace:** §2.2 Def 2.4b

## Decision D-016: MLP learn() redoes forward pass with correct action (Phase 3.3b)

- **Date:** 2026-06-30
- **Author:** Implementation Engineer
- **Category:** Tier 3
- **Option chosen:** `WorldModelMLP.learn()` calls `predict()` internally with the actual `action` parameter before computing gradients, overwriting the cached activations from the cycle's prediction step (which used `last_action` from the previous cycle)
- **Alternatives:** Change cycle order to compute gradient before action selection, pass `last_action` to compute_gradient, store per-action caches
- **Rationale:** The forward pass during `step()` (via `engine.predict()`) uses `last_action` from the previous cycle, but `learn()` needs the gradient w.r.t. the actual action taken. Redoing the forward pass in `learn()` is the minimal change — no cycle restructuring, no TSPL changes, one extra ~1ms forward pass per cycle.
- **v3.0 trace:** §2.2 Def 2.4b, Phase 3.3b Strategic Report §3.2

## Decision D-017: MLP gradient NOT passed to TSPL (Phase 3.3b)

- **Date:** 2026-06-30
- **Author:** Implementation Engineer
- **Category:** Tier 2
- **Option chosen:** `WorldModelMLP.learn()` applies SGD gradient directly to internal weights. TSPL.update() continues to use its own delta-rule gradient for theta bookkeeping. No gradient kwarg passed from MLP to TSPL.
- **Alternatives:** Pass MLP gradient as TSPL `gradient=` kwarg, requiring TSPL theta key mapping. Sync TSPL theta from MLP after each learn step.
- **Rationale:** MLP gradient keys (`gprime_w1`, `gprime_b1`, ...) don't match TSPL theta keys (`gprime`). Mapping would require TSPL changes (forbidden by Phase 3.3b constraint: zero TSPL internals changes). MLP internal weights are the authoritative copy; TSPL theta is a bookkeeping mirror for skill compilation. Decoupling simplifies integration and avoids accidental gradient double-counting.
- **v3.0 trace:** §2.2 Def 2.4b, §3.1 Def 3.2

## Decision D-018: HPM validation layer stripped (Phase 3.3 final)

- **Date:** 2026-06-30
- **Author:** Implementation Engineer
- **Category:** Tier 2
- **Option chosen:** Remove `ModuleType`, `LeafOp`, `ValidationResult`, `validate()`, `validate_structured()`, and three `_validate_*` methods from `phca/hpm/parser.py`. Keep `CompositionOp`, `HPMNode`, and `compute_bounds()`.
- **Alternatives:** Keep validation with a flag; rewrite validation to be correct.
- **Rationale:** `validate_structured()` was a stub that printed cosmetic warnings against a hardcoded spec. Only `compute_bounds()` (v3.0 Theorem 2.1/3.1 resource additivity) is genuinely used by RBTA. The validation produced no runtime action and created false confidence.
- **v3.0 trace:** §2.1 Def 2.2, §4.2

## Decision D-019: Dead module removal — inference.py, similarity.py (Phase 3.3 final)

- **Date:** 2026-06-30
- **Author:** Implementation Engineer
- **Category:** Tier 2
- **Option chosen:** Delete `world_model/inference.py` (216 prod + 165 test lines) and `world_model/similarity.py` (82 prod + 111 test lines) entirely.
- **Alternatives:** Keep as "future use" stubs; mark deprecated.
- **Rationale:** `forward_inference()` and `knn_similarity()` were never imported or called from any production code path. Dead code increases maintenance burden and confuses readers.
- **v3.0 trace:** N/A

## Decision D-020: E_STREAM / S_STREAM removed (Phase 3.3 final)

- **Date:** 2026-06-30
- **Author:** Implementation Engineer
- **Category:** Tier 3
- **Option chosen:** Remove `E_STREAM` and `S_STREAM` from `StreamID` enum, `config.py` defaults, and all TSPL update paths. `cycle.py` only calls `tspl.update(P_STREAM, ...)`.
- **Alternatives:** Keep flags disabled; keep as no-ops for future use.
- **Rationale:** E/S streams were always disabled (`False` default) and had no production code path. Consolidation runs on a fixed 50-cycle timer instead, obviating TSPL-mediated stream consolidation.
- **v3.0 trace:** §3.1 Def 3.2

## Decision D-021: Dead function excision across 10 modules (Phase 3.3 final)

- **Date:** 2026-06-30
- **Author:** Implementation Engineer
- **Category:** Tier 3
- **Option chosen:** Remove ~570 lines of dead functions across MDIM, TSPL, PEU, PID, RBTA, M2, M3, MLP, graph, attention.
- **Alternatives:** Keep all public methods for "API completeness."
- **Rationale:** Every removed function was either never called, only called from tests, or duplicated functionality. Examples: `MDIM.get_drive_summary()` replaced by `drives` dict access; `MDIM.reset()` never used; `TSPL.freeze_skill()` never used; `GEM`/`EWC` methods never used.
- **v3.0 trace:** N/A

## Decision D-022: EnvironmentProtocol decoupling (Phase 3.3 final)

- **Date:** 2026-06-30
- **Author:** Implementation Engineer
- **Category:** Tier 2
- **Option chosen:** Create `environments/protocol.py` with `EnvironmentProtocol`. `CognitiveCycle.build_for_env()` accepts the protocol instead of a concrete `GridWorld`. `GridWorld` adds `get_action_names()`, `get_possible_actions()`, `get_goal_position()`.
- **Alternatives:** Keep GridWorld dependency; abstract via ABC in a shared base class.
- **Rationale:** Protocol typing (structural subtyping) avoids ABC inheritance and allows any duck-typed environment to drive the cycle. Zero runtime overhead.
- **v3.0 trace:** §1.2

## Decision D-023: M3 SQLite abstraction deferred (Phase 3.3 final)

- **Date:** 2026-06-30
- **Author:** Implementation Engineer
- **Category:** Tier 3
- **Option chosen:** Keep M3 SQLite-backed with no storage backend abstraction.
- **Alternatives:** Add abstract BaseStorage class with SQLite and in-memory implementations.
- **Rationale:** Single backend, zero benefit from an abstraction layer now. Deferred to Phase 3.4 if a second backend is ever needed.
- **v3.0 trace:** §2.3

## Decision D-024: MLP abstraction skipped (Phase 3.3 final)

- **Date:** 2026-06-30
- **Author:** Implementation Engineer
- **Category:** Tier 3
- **Option chosen:** Keep MLP as a single concrete class with no abstract base.
- **Alternatives:** AbstractBaseMLP with WorldModelMLP and (future) TorchMLP implementations.
- **Rationale:** Single implementation would add ~80 abstraction lines for zero immediate benefit.
- **v3.0 trace:** §2.2 Def 2.4b

## Decision D-025: NaN hardening with @np.errstate raise (Phase 3.3 final)

- **Date:** 2026-06-30
- **Author:** Implementation Engineer
- **Category:** Tier 1
- **Option chosen:** Change `@np.errstate(all="ignore")` to `@np.errstate(divide="raise", invalid="raise", over="ignore")` in all modules.
- **Alternatives:** Add manual np.isnan/np.isfinite checks everywhere; keep floating-point exceptions suppressed.
- **Rationale:** `all="ignore"` silently converts all FP exceptions to NaN, which then propagates silently. Raising on divide-by-zero and invalid operations forces NaN to surface immediately at the source. `over="ignore"` is retained because overflow to ±inf is recoverable.
- **v3.0 trace:** §2.1 Def 2.1a

## Decision D-026: Benchmark runner implementation — Phase 3.3 (Issue #3 fix)

- **Date:** 2026-06-30
- **Author:** Chief Architect
- **Category:** Tier 1
- **Option chosen:** Rewrote `python/benchmarks/runner.py` with full `run_level_0()` through `run_level_3()` implementations, porting logic from `scripts/benchmark.py`. Added `run_all_levels()` for composite reports with pass criteria.
- **Alternatives:** Keep as deferred stub per PHCA-3.3-003; move `scripts/benchmark.py` into `phca` package.
- **Rationale:** The benchmark stub violated the Phase 3.3 gate condition — Φ-IQ scores could not be independently verified from the package itself. Porting `scripts/benchmark.py` logic was the minimal, faithful implementation.
- **v3.0 trace:** §1.3 success criteria, §5 Φ-IQ composite metric

## Decision D-027: Attention weights wired into cognitive cycle — Phase 3.3 (Issue #4 fix)

- **Date:** 2026-06-30
- **Author:** Chief Architect
- **Category:** Tier 2
- **Option chosen:** After `attention.select()`, chose to normalize chunk saliences and tile/truncate them to `state_dim` dimensions, storing as `self._attention_weights`. The mean weight is multiplied with prediction error and passed to `gprime.learn()`. Also fixed an indentation bug where `gprime.learn()` was outside the `if self.current_state is not None:` guard.
- **Alternatives:** Store raw saliences; pass weights as separate learn() parameter; modify world model learn() interface to consume per-dimension weights.
- **Rationale:** Storing weights on the cycle object enables future wiring into the world model's `learn()` method (Phase 4). The indentation fix was needed to prevent calling `learn()` when `self.current_state` is None. The `error` parameter is accepted by learn() but not yet consumed (noted in comment).
- **v3.0 trace:** §3.2 Def 3.4, §2.2 Def 2.4b
