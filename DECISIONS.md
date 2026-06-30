# PHCA v3.0 — Decision Log

This file records all significant design decisions made during implementation.
Every entry must reference the v3.0 specification section it affects.

> **Note on phase labels:** Entries below reference the phase in which each decision
> was made (e.g., "Phase 3.1", "Phase 3.2", "Phase 3.3"). All referenced phases
> are now complete. The system is at Phase 3.3 final, ready for Phase 4.

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

## Decision D-028: MLP hidden_dim 32→128 — Phase 3.3 completion (Issue #1 fix)

- **Date:** 2026-06-30
- **Author:** Chief Architect
- **Category:** Tier 1 (configuration error)
- **Option chosen:** Changed `WorldModelMLP` default `hidden_dim` from 32 to 128 (matching documented 38,868 params). Reduced `train_steps` from 8 to 4 and `batch_size` from 64 to 32 to compensate for 4× larger per-pass compute.
- **Alternatives:** hidden_dim=64 (conservative), keep 32 and accept capacity limitation
- **Rationale:** The architecture documentation consistently describes a 38,868-parameter MLP (hidden_dim=128). The 32-hidden-unit default was never updated from the initial prototype. With hidden_dim=128, each forward/backward pass is ~4× more expensive, so we halved both batch_size and train_steps to keep total compute similar. This restores A4 (Prediction as Primary) capacity.
- **v3.0 trace:** §2.2 Def 2.4b, A4

## Decision D-029: Error-modulated learning rate — Phase 3.3 completion (Issue #3 fix)

- **Date:** 2026-06-30
- **Author:** Chief Architect
- **Category:** Tier 2
- **Option chosen:** Both MLP and Gaussian `learn()` now use `lr_effective = lr * clip(1.0 + abs(error) * 0.1, 0.5, 2.0)` where error is the attention-weighted prediction error from PEU. MLP also receives per-dimension attention_weights passed via dynamic attribute from cycle to modulate output gradients. Batch-replay gradients do NOT receive attention modulation (replayed samples are from different states).
- **Alternatives:** Store attention_weights in replay buffer, pass through learn() interface formally
- **Rationale:** Error-modulated lr and per-dimension attention weighting are the minimal changes to make A5 (Feedback-Driven Adaptation) functional. Storing weights in the replay buffer would require buffer schema changes (Phase 4 scope). The dynamic attribute pattern avoids interface changes while preserving correctness.
- **v3.0 trace:** §2.2 Def 2.4b, A5

## Decision D-030: MDIM target_state into action scoring — Phase 3.3 completion (Issue #4 fix)

- **Date:** 2026-06-30
- **Author:** Chief Architect
- **Category:** Tier 2
- **Option chosen:** D1/D3 action scoring now blends distance_gain (0.6), confidence (0.2), and target_state alignment (0.2). D2/D4 scoring blends uncertainty (0.5), distance_gain (0.2), and alignment (0.3). Previously target_state was only used for D6.
- **Alternatives:** Separate scoring for each drive, use alignment as main signal
- **Rationale:** The original formula used only drive_id (integer) to distinguish drive behaviour. Incorporating target_state alignment makes each drive's generated goal actually influence which actions are preferred, completing the MDIM→action pipeline.
- **v3.0 trace:** §3.3 Def 3.5, G5

## Decision D-031: Consolidation facts wired into MDIM context — Phase 3.3 completion (Issue #2 fix)

- **Date:** 2026-06-30
- **Author:** Chief Architect
- **Category:** Tier 2
- **Option chosen:** Added `get_relevant_facts()` to `ConsolidationScheduler` that returns top-N facts by relevant score (cosine similarity × confidence). In cycle.py, facts are queried for the current state and their mean confidence and count are injected into `mdim_context`.
- **Alternatives:** Wire facts directly into prediction engine bias, use facts as MLP pre-training data
- **Rationale:** Minimal wiring to make the S-Stream pipeline functional. Facts now influence goal generation via MDIM context. Direct prediction biasing (Phase 4 scope) would require `PredictionEngine` changes.
- **v3.0 trace:** §2.3 (S-Stream), A3

## Decision D-032: Energy bounds wired into composition tree — Phase 3.3 completion (Issue #5 fix)

- **Date:** 2026-06-30
- **Author:** Chief Architect
- **Category:** Tier 1
- **Option chosen:** Extracted `B_energy` from `hpm_bounds` in cycle.py and added it to the composition tree's `bounds` dict for both the regulation_block and root SEQUENCE node. The RBTA's `_check_composition_tree()` already checked `B_energy` — the missing piece was wiring it through.
- **Alternatives:** Add energy-specific log, change `_compute_subtree_energy` to read from `energy_log`
- **Rationale:** The HPM correctly computes `B_energy` for all composition operators. The RBTA already checks energy bounds in composition trees. The only gap was that cycle.py's composition tree only included `B_time`. Adding `B_energy` required 2 lines. This completes A1 enforcement for all 3 dimensions.
- **v3.0 trace:** §2.1 Def 2.1, Def 3.6, A1

## Decision D-033: Thread-safe MetricsStore for live monitoring (Phase 3.3 monitoring)

- **Date:** 2026-06-30
- **Author:** Chief Architect
- **Category:** Tier 3
- **Option chosen:** Created `python/phca/monitoring/metrics_store.py` with a lock-guarded `deque` ring buffer. `MetricsStore` is injected as optional parameter into `CognitiveCycle.__init__()`, `build_for_env()`, and `build_for_mujoco()`. When present, `step()` pushes `CycleMetrics` to the store after each cycle.
- **Alternatives:** Direct `cycle.metrics_history` access (thread-unsafe), multiprocessing shared memory (over-engineered), log-file parsing (delayed)
- **Rationale:** Lock-guarded deque is the simplest non-blocking pattern. < 5 μs per push. Zero new dependencies. The forward reference `Optional["MetricsStore"]` with `from __future__ import annotations` avoids circular imports.
- **v3.0 trace:** A1 (resource boundedness)

## Decision D-034: Curses terminal dashboard for live metrics (Phase 3.3 monitoring)

- **Date:** 2026-06-30
- **Author:** Chief Architect
- **Category:** Tier 3
- **Option chosen:** Created `scripts/phca-monitor.py` using stdlib `curses` for a live terminal dashboard. Runs as a daemon thread reading from `MetricsStore` every 500ms. Main thread runs the cognitive cycle. CycleMetrics extended with `drive_id`, `skill_accuracy`, `skill_compiled`, `fact_count`, `episode_count` fields for richer dashboard rendering.
- **Alternatives:** Web dashboard (needs HTTP server), TUI framework (external dep), ASCII art in terminal (no live update)
- **Rationale:** Curses is stdlib, daemon thread avoids blocking the cycle. Push cost < 5 μs per cycle (< 0.02% of 25ms cycle). New CycleMetrics fields add ~40 bytes per entry (negligible).
- **v3.0 trace:** A1 (resource boundedness)

## Decision D-035: File logging with RotatingFileHandler (Phase 3.3 monitoring)

- **Date:** 2026-06-30
- **Author:** Chief Architect
- **Category:** Tier 3
- **Option chosen:** Added `setup_file_logging()` to `logging.py` that writes structured logs to `logs/phca.log` with 10MB rotation and 3 backups. Uses global `_FILE_LOGGING_CONFIGURED` flag for idempotency. Works with both structlog (JSON lines) and stdlib fallback (formatted key=value).
- **Alternatives:** External log shipper (Loki, ELK), syslog, no file logging
- **Rationale:** 10MB × 4 backups = 40MB max disk, acceptable for development. Log viewer (`scripts/phca-logs.py`) uses subprocess `tail -f` for zero-overhead following.
- **v3.0 trace:** A1 (resource boundedness)

## Decision D-036: RBTA composition tree reads from energy_log (Phase 3.3 gap closure — A-001/A-004)

- **Date:** 2026-06-30
- **Author:** Chief Architect
- **Category:** Tier 1 (bug fix — energy violations undetected)
- **Option chosen:** `_check_composition_tree()` and `_compute_subtree_energy()` now accept a separate `energy_log: Dict[str, float]` parameter and read energy values from it. Previously both methods read energy from `runtime_log` with the key convention `child + "_energy"` or bare `child` — neither matched `_collect_runtime_log()`'s key convention, causing energy violations to go undetected since the composition tree was added in Phase 3.2.
- **Alternatives:** Keep reading from `runtime_log` with bare keys (gets seconds not Joules); merge energy into runtime_log
- **Rationale:** `check_cycle()` already receives both logs. Passing `energy_log` (estimated Joules) through the composition tree is semantically correct and requires no data restructuring. Falls back to `0.0` for missing modules.
- **v3.0 trace:** §2.1 Def 2.2, Def 3.6, A1

## Decision D-037: Monitoring wired into benchmark.py (Phase 3.3 gap closure — A-002)

- **Date:** 2026-06-30
- **Author:** Chief Architect
- **Category:** Tier 3
- **Option chosen:** Added `ensure_logging()` call to `scripts/benchmark.py main()`. No MetricsStore integration — benchmarks build their own cycles without monitoring to avoid performance interference.
- **Alternatives:** Wire MetricsStore into every benchmark run
- **Rationale:** `ensure_logging()` ensures `logs/phca.log` is written during benchmark runs for post-hoc analysis. MetricsStore is appropriate for interactive monitoring only.
- **v3.0 trace:** A1

## Decision D-038: Dead compute_pareto_front() call commented out (Phase 3.3 gap closure — A-005)

- **Date:** 2026-06-30
- **Author:** Chief Architect
- **Category:** Tier 2
- **Option chosen:** Commented out `pareto = self.compute_pareto_front()` in `mdim.py` `generate_goal()` and added a TODO for Phase 4 wiring into `_is_deeply_meta_stable()`.
- **Alternatives:** Remove the Pareto front computation entirely; wire it into meta-stability now
- **Rationale:** The Pareto front computation is not harmful (fast) but its output was unused. Preserving the code behind a TODO allows Phase 4 to complete the wiring without reimplementing from scratch.
- **v3.0 trace:** §3.3 Def 3.6, v3.0 Patch §2.4

## Decision D-039: SkillLibrary removed (Phase 3.3 gap closure — A-006)

- **Date:** 2026-06-30
- **Author:** Chief Architect
- **Category:** Tier 3
- **Option chosen:** Deleted `python/phca/learning/skill_compilation.py`, removed its import from `__init__.py`, removed `TestSkillLibrary` class from tests.
- **Alternatives:** Keep as dead code; convert to utility functions
- **Rationale:** `SkillLibrary` was never instantiated in production code. TSPL uses its own inlined compilation logic. 50 lines of dead code + 6 tests removed.
- **v3.0 trace:** N/A

## Decision D-040: Sleep cycle (Step 20) removed (Phase 3.3 gap closure — A-008)

- **Date:** 2026-06-30
- **Author:** Chief Architect
- **Category:** Tier 2
- **Option chosen:** Removed the Step 20 sleep-cycle block and `_get_sleep_interval()` static method from `cycle.py`. Consolidation is already handled by Steps 16-18 every `consolidation_interval=10` cycles. The 50-cycle sleep cycle duplicated this work.
- **Alternatives:** Keep sleep cycle for future energy conservation; reduce interval
- **Rationale:** Single consolidation path simplifies the cycle. Every 50 cycles, consolidation ran twice (once at 10-cycle interval, once at 50-cycle sleep).
- **v3.0 trace:** A1

## Decision D-041: Dynamic energy_cost from elapsed cycle time (Phase 3.3 gap closure — A-009)

- **Date:** 2026-06-30
- **Author:** Chief Architect
- **Category:** Tier 2
- **Option chosen:** Changed `mdim_context["energy_cost"]` from hardcoded `0.1` to `max(0.01, min(1.0, elapsed_time * 2.0))` where elapsed_time is `time.perf_counter() - t_start` at MDIM context time.
- **Alternatives:** Use `self.energy_log.get("CYCLE", 0.0) / 10.0` (but energy_log isn't populated until after MDIM context); use a fixed scaling
- **Rationale:** `elapsed * 2.0` gives ~0.05 for a 25ms cycle (D5 deficit ≈ 0.15) and ~0.1 for a 50ms cycle (D5 deficit ≈ 0.10). Scaling factor 2.0 approximates the original plan's `energy_log / 10.0 ≈ 0.125` convention without depending on energy_log being populated.
- **v3.0 trace:** §3.3 Def 3.5

## Decision D-042: TSPL-E/S removed from logs and bounds (Phase 3.3 gap closure — B-001)

- **Date:** 2026-06-30
- **Author:** Chief Architect
- **Category:** Tier 3
- **Option chosen:** Removed `TSPL-E` and `TSPL-S` entries from `memory_log` dict, `energy_log` baseline, and `DEFAULT_MODULE_BOUNDS` in `config.py`. These streams were removed in Phase 3.3 (D-020) but their stale entries persisted in logs and bounds.
- **Alternatives:** Keep entries but zero them out; add B_time=0 bounds
- **Rationale:** Since the streams don't exist anymore, they shouldn't consume bounds or log space. RBTA was checking their bounds every cycle unnecessarily.
- **v3.0 trace:** §2.1 Def 2.2

## Decision D-043: attention_focus removed from MDIM context (Phase 3.3 gap closure — B-002)

- **Date:** 2026-06-30
- **Author:** Chief Architect
- **Category:** Tier 3
- **Option chosen:** Removed `"attention_focus"` from `mdim_context` dict and its computation block in `cycle.py`. No drive reads this signal.
- **Alternatives:** Wire into D2 criticality seeking; keep for future use
- **Rationale:** Dead signal in the context dict. If D2 needs attention data in Phase 4, it can be added then with a proper neural interface.
- **v3.0 trace:** §3.2 Def 3.4

## Decision D-044: D6 empowerment blend reverted — old logic was correct (Phase 3.3 gap closure — B-003)

---

## Decision D-045: Rename `phi` → `error_volatility` to stop claiming IIT (Phase 4 gap audit — G-001)

- **Date:** 2026-06-30
- **Author:** Chief Architect
- **Category:** Tier 2 (architectural honesty)
- **Option chosen:** Renamed `_approximate_phi()` → `_approximate_error_volatility()`, `phi_current` → `error_volatility`, `_phi_error_window` → `_error_vol_window`, `"phi_criticality"` context key → `"error_volatility"` throughout `cycle.py`, `mdim.py`, `pid_controller.py`, and all test files.
- **Rationale:** The original name claimed to approximate Φ (integrated information from IIT 3.0). The actual computation is `cv(prediction_error)` — the coefficient of variation of recent prediction errors, which measures prediction-error volatility, not integration. Keeping the name "phi" misleads readers into thinking the system implements actual IIT, which it does not. The rename is purely cosmetic — no behavioural change.
- **v3.0 trace:** §2.2 Def 2.4b (prediction engine), Phase 4 gap report finding G-001

## Decision D-046: Rename `CriticalityRegulator` → `AdaptiveParameterController` (Phase 4 gap audit — G-004)

- **Date:** 2026-06-30
- **Author:** Chief Architect
- **Category:** Tier 2 (architectural honesty)
- **Option chosen:** Renamed `CriticalityRegulator` class to `AdaptiveParameterController`, removed all "edge of chaos" and "self-organized criticality" language from docstrings, updated all imports and test references in `cycle.py`, `regulation/__init__.py`, and `test_pid_controller.py`.
- **Rationale:** The class is a textbook PID controller that regulates prediction-error volatility toward a fixed setpoint (0.5). Calling it a "Criticality Regulator" implies it implements self-organized criticality (SOC) à la Langton/Packard, which it does not. A PID loop with fixed gains does not model phase transitions, bifurcations, or self-organization. The rename accurately reflects what the component does: adaptively adjust three scalar parameters (T, eta, alpha) based on a PID error signal.
- **v3.0 trace:** §3.4 Def 3.7 (parameter modulation), Phase 4 gap report finding G-004

## Decision D-047: Unify `build_for_env`/`build_for_mujoco` → `CognitiveCycle.build(env)` (Phase 4 gap audit — G-008)

- **Date:** 2026-06-30
- **Author:** Chief Architect
- **Category:** Tier 2 (architectural debt)
- **Option chosen:** Created `CognitiveCycle.build(env, ...)` as the canonical builder that accepts any `EnvironmentProtocol`. `build_for_env` and `build_for_mujoco` are now thin wrappers that create their respective environments and delegate to `build()` with environment-appropriate defaults (e.g., MLP LR=0.05, G' B_time=0.080, ACTION B_time=0.050 for MuJoCo; G' B_time=0.050 for MLP with GridWorld).
- **Alternatives:** Keep two separate builders with ~50 lines of duplicated module wiring; create a factory pattern with registry.
- **Rationale:** The original `build_for_env` and `build_for_mujoco` each independently wired all ~13 PHCA modules with identical logic but slightly different parameters (LR, bounds, state_dim). This duplication caused maintenance issues (fixes in one builder didn't propagate to the other) and made the module wiring a hidden architecture concern. The unified builder eliminates duplication, makes the module wiring visible as a single canonical reference, and allows any EnvironmentProtocol to drive the system without adding a new builder method. All 47 existing call sites continue to work unchanged.
- **v3.0 trace:** §2.2 Def 2.4b, G1 (embodiment), Phase 4 gap report finding G-008

## Decision D-048: Wire Pareto front into meta-stable drive suppression (Phase 4 gap audit — G-006/G-012)

- **Date:** 2026-06-30
- **Author:** Chief Architect
- **Category:** Tier 2 (zombie feature activated)
- **Option chosen:** Uncommented `self.compute_pareto_front()` in `mdim.py:generate_goal()`, stored result in `self._pareto_front_ids`, and changed the meta-stable suppression block to only suppress drives NOT on the Pareto front (rather than suppressing all D1/D3/D5 unconditionally). Added empty-Pareto fallback (suppress all) for edge cases.
- **Rationale:** The Pareto front computation (60+ lines) existed as a zombie function — called in a commented-out line, with output discarded. The meta-stable state was suppressing ALL D1/D3/D5 drives, ignoring the Pareto front's purpose of identifying optimal trade-offs between conflicting objectives. Drives on the Pareto front represent configurations where no single drive can be improved without worsening another — these should NOT be suppressed, as they represent the system's best compromise. Non-Pareto drives (dominated by others) are still suppressed, as their suppression frees the system to focus on drives that are at their optimal frontier.
- **v3.0 trace:** §2.4.1 Def 3.10 (Drive Pareto Front), §3.3 Def 3.6, Phase 4 gap report findings G-006/G-012

## Decision D-049: Goal-driven attention biasing with drive-dependent blend (Phase 4 gap audit — G-005)

- **Date:** 2026-06-30
- **Author:** Chief Architect
- **Category:** Tier 2 (over-simplification corrected)
- **Option chosen:** Updated `Attention.select()` to compute drive-dependent alpha/beta blends when a goal with target_state is present. D1/D3 get bottom-up heavy (0.7/0.3), D2/D4 get top-down heavy (0.3/0.7), D5 gets top-down heavy (0.2/0.8), D6 balanced (0.5/0.5). Beta is further scaled by goal priority (0.2-2.0x), then re-normalized. Replaced `_cosine_similarity()` with `_precision_weighted_similarity()` that weights dimensions by goal.target_state.precision. Updated cycle.py to pass `self.last_prediction` to `attention.select()` for accurate bottom-up unexpectedness.
- **Alternatives:** Fixed beta_td=0.4 constant; separate attention module per drive
- **Rationale:** The original attention module had a fixed beta_td=0.4 that provided weak, static top-down biasing regardless of goal type or priority. This meant all drives influenced attention identically — a D1 (error-minimization) goal had the same top-down influence as a D5 (energy-optimization) goal. The drive-dependent blend makes attention responsive to the current motivational state. Precision-weighted similarity ensures that goal dimensions with specific targets (high precision) dominate the similarity computation over dimensions where the goal is agnostic (low precision). Passing prediction enables proper bottom-up salience (unexpectedness = |chunk - prediction|) instead of falling back to stale chunk salience.
- **v3.0 trace:** §3.2 Def 3.4 (Attention), §3.3 Def 3.5 (MDIM drives), Phase 4 gap report finding G-005

---

*End of Decision Log (as of Phase 4 gap audit, Week 1-2 critical + G-005/G-006/G-012 fixes).*
