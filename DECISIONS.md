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
