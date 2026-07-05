# PHCA v3.0 — Engineer's Playbook: Day-Zero Execution Plan

**Document Type:** Execution Plan / Engineer Onboarding  
**Status:** FINAL — READY FOR SPRINT 0  
**Parent Document:** Implementation Blueprint (`10-implementation-blueprint.md`)  
**Specification:** PHCA v3.0 (`09-phca-v3-patch.md`)  
**Audience:** Development Team (3–4 engineers — no cognitive science background assumed)  
**Date:** June 29, 2026

---

## TABLE OF CONTENTS

1. [Daily Workflow & Team Norms](#1-daily-workflow--team-norms)
2. [Repository Structure & Onboarding](#2-repository-structure--onboarding)
3. [Ticket Board: Phase 3.1 — Core Engine](#3-ticket-board-phase-31--core-engine)
4. [Ticket Board: Phase 3.2 — Full Architecture](#4-ticket-board-phase-32--full-architecture)
5. [Ticket Board: Phase 3.3 — Benchmarking & Validation](#5-ticket-board-phase-33--benchmarking--validation)
6. [Ticket Board: Phase 3.4 — Iteration & Refinement](#6-ticket-board-phase-34--iteration--refinement)
7. [Component Implementation Checklists (21 Components)](#7-component-implementation-checklists)
8. [Integration Test Plan](#8-integration-test-plan)
9. [Acceptance Test Plan (v3.0 §1.3)](#9-acceptance-test-plan)
10. [Benchmark Suite Implementation Plan](#10-benchmark-suite-implementation-plan)
11. [Scheduling & Dependency Grid (52 Weeks)](#11-scheduling--dependency-grid)
12. [Risk Monitoring & Contingency Triggers](#12-risk-monitoring--contingency-triggers)
13. [Phase Transition Gates](#13-phase-transition-gates)
14. [Glossary of Key Terms](#14-glossary-of-key-terms)
15. [Emergency Contacts & Escalation](#15-emergency-contacts--escalation)

---

## 1. DAILY WORKFLOW & TEAM NORMS

### 1.1 A Typical Day

| Time | Activity | Tool |
| :--- | :--- | :--- |
| 09:00 | Standup (15 min) — what I did yesterday, what I'll do today, blockers | Slack / Discord |
| 09:15 | Code: implement ticket from the board | VS Code |
| 12:00 | Lunch | — |
| 13:00 | Code / review (alternate days: write code Mon/Wed/Fri, review Tue/Thu) | GitHub PRs |
| 16:00 | Run tests + benchmark (minimum: the test for today's ticket) | `pytest`, `cargo test` |
| 16:30 | Commit + push. If PR is ready, request review from team lead | Git + GitHub |
| 17:00 | Log decision in the decision log if any design choice was made | `DECISIONS.md` |

### 1.2 Team Composition

| Role | Person | Responsibility |
| :--- | :--- | :--- |
| **ML/Systems Lead** | TBD | Architecture decisions, code review, Phase 3.1–3.4 oversight |
| **Backend/Performance Engineer** | TBD | Rust components (RBTA, HPM), performance profiling, data persistence |
| **ML Engineer** | TBD | Python components (G', TSPL, MDIM, Attention), PyTorch integration |
| **Researcher/Evaluator** | TBD | Benchmark design, metric analysis, v3.0 spec compliance verification |

### 1.3 Communication

- **Blockers:** Tag `@lead` in Slack immediately — never wait for standup
- **v3.0 questions:** Tag `@researcher` — they own spec interpretation
- **PR reviews:** Respond within 24 hours. Merge after 1 approval (Phase 3.1) or 2 approvals (Phase 3.2+)
- **Decision log:** Every implementation decision gets logged in `DECISIONS.md` with the date, option chosen, alternatives, and rationale

### 1.4 Coding Standards

- **Python:** PEP8 (line length 100). Type hints required on all public functions. `ruff` for linting, `black` for formatting
- **Rust:** `rustfmt` defaults. Clippy must pass with no warnings. No `unsafe` blocks without explicit lead approval
- **Naming:** Classes: `PascalCase`. Functions: `snake_case`. Constants: `UPPER_SNAKE_CASE`. Modules: `snake_case`
- **Tests:** Every ticket must have at least one test. Tests must be deterministic (fixed random seed). No flaky tests allowed
- **Performance:** Every PR must include a benchmark measurement (in the PR description) showing no regression on the ticket's component

---

## 2. REPOSITORY STRUCTURE & ONBOARDING

### 2.1 Prerequisites

| Tool | Version | Why |
| :--- | :--- | :--- |
| **OS** | Linux (Ubuntu 22.04+) or macOS 13+ | Rust build targets, perf profiling |
| **Python** | 3.11+ | PyTorch 2.x, Pyro, pgmpy |
| **Rust** | 1.75+ (nightly for some profiling) | RBTA, HPM runtime components |
| **Node.js** | 20+ (optional) | For Mermaid.js rendering / docs preview |
| **Docker** | 24+ | Reproducible benchmark environments |
| **Git** | 2.40+ | Version control |
| **Make** | — | Build automation |

### 2.2 Quick Setup (5 minutes)

```bash
# 1. Clone the repository
git clone https://github.com/ali-baneshi/phca-v3.git
cd phca-v3

# 2. Install Python dependencies
pip install uv  # or use pip if uv not available
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt

# 3. Install Rust toolchain + build
rustup default stable
cargo build --release

# 4. Verify setup
make test-all        # runs all unit tests (Python + Rust)
make bench-level-0   # runs Level 0 benchmark
make lint            # runs ruff + clippy

# 5. Open the project
code .
```

### 2.3 Repository Directory Layout

```
phca-v3/
├── README.md                    # Project overview + links
├── SETUP.md                     # Step-by-step setup (this section)
├── CONTRIBUTING.md              # PR workflow, coding standards
├── DECISIONS.md                 # Decision log (append only)
├── Makefile                     # Build + test targets
├── requirements.txt             # Python dependencies (pinned)
├── Cargo.toml                   # Rust workspace
├── Cargo.lock                   # Rust dependency lock
│
├── docs/
│   ├── architecture.md          # High-level architecture diagram
│   ├── api-reference.md         # Full public API docs
│   └── v3.0-spec/               # Local copies of v3.0 specs
│       ├── 07-rigorous-whitepaper.md
│       ├── 08-external-audit-report.md
│       ├── 09-phca-v3-patch.md
│       ├── 10-implementation-blueprint.md
│       └── 11-engineers-playbook.md          ← YOU ARE HERE
│
├── python/
│   ├── phca/                    # Main Python package
│   │   ├── __init__.py
│   │   ├── asi/                 # ASI + Step 0 sanitizer
│   │   │   ├── sanitizer.py
│   │   │   ├── grounding_adapter.py
│   │   │   └── tests/
│   │   ├── world_model/         # G' (Probabilistic Graph)
│   │   │   ├── graph.py
│   │   │   ├── inference.py
│   │   │   ├── similarity.py    # k-NN replacement for VSA
│   │   │   └── tests/
│   │   ├── prediction/          # Prediction Engine + PEU
│   │   │   ├── engine.py
│   │   │   ├── error_unit.py
│   │   │   ├── ensemble.py
│   │   │   └── tests/
│   │   ├── learning/            # TSPL (P, E, S streams)
│   │   │   ├── tspl.py
│   │   │   ├── ewc.py
│   │   │   ├── gem.py
│   │   │   ├── skill_compilation.py
│   │   │   └── tests/
│   │   ├── motivation/          # MDIM (6 drives)
│   │   │   ├── mdim.py
│   │   │   ├── drives.py        # D1–D6 definitions + setpoints
│   │   │   ├── pareto_front.py
│   │   │   └── tests/
│   │   ├── regulation/          # Criticality Regulator
│   │   │   ├── pid_controller.py
│   │   │   ├── observables.py   # Φ, H, λ_max, C estimation
│   │   │   ├── orthogonality.py
│   │   │   └── tests/
│   │   ├── attention/           # Precision-Weighted Attention
│   │   │   ├── attention.py
│   │   │   └── tests/
│   │   ├── memory/              # M1–M5 hierarchy
│   │   │   ├── m1_sensory.py
│   │   │   ├── m2_working.py
│   │   │   ├── m3_episodic.py
│   │   │   ├── m4_semantic.py
│   │   │   ├── m5_procedural.py
│   │   │   ├── consolidation.py
│   │   │   └── tests/
│   │   ├── hpm/                 # HPM Grammar Runtime
│   │   │   ├── parser.py        # Python reference impl
│   │   │   ├── type_checker.py
│   │   │   ├── resource_verifier.py
│   │   │   └── tests/
│   │   ├── resilience/          # Failure Detection & Recovery
│   │   │   ├── detector.py
│   │   │   ├── recovery.py
│   │   │   └── tests/
│   │   ├── core/                # Cognitive cycle orchestrator
│   │   │   ├── cycle.py         # Main 21-step cycle
│   │   │   └── tests/
│   │   └── config.py            # Global configuration + defaults
│   │
│   ├── benchmarks/              # Φ-IQ Evaluation Suite
│   │   ├── runner.py            # Main benchmark runner CLI
│   │   ├── level_0.py
│   │   ├── level_1.py
│   │   ├── level_2.py
│   │   ├── level_3.py
│   │   ├── level_4.py
│   │   ├── level_5.py
│   │   └── metrics.py           # Φ-IQ composite computation
│   │
│   ├── environments/            # Simulated environments
│   │   ├── grid_world.py        # Phase 3.1: 2D navigation
│   │   ├── pendulum.py          # Level 0 stationary prediction
│   │   ├── inverted_pendulum.py # Level 1 reactive control
│   │   ├── mujoco_bridge.py     # MuJoCo wrapper (Phase 3.2+)
│   │   └── multi_agent.py       # Multi-agent env (Phase 3.3)
│   │
│   └── tests/                   # Integration + system tests
│       ├── test_phase_3_1.py
│       ├── test_phase_3_2.py
│       ├── test_acceptance.py   # v3.0 §1.3 success criteria
│       └── conftest.py          # Shared fixtures
│
├── rust/
│   ├── rpta/                    # RBTA Constraint Enforcer
│   │   ├── Cargo.toml
│   │   ├── src/
│   │   │   ├── lib.rs
│   │   │   ├── enforcer.rs
│   │   │   ├── composition.rs   # Theorem 2.1/3.1 composition rules
│   │   │   └── bounds.rs        # ResourceBounds type
│   │   └── tests/
│   │
│   ├── hpm-runtime/             # HPM Grammar Runtime (Rust impl)
│   │   ├── Cargo.toml
│   │   ├── src/
│   │   │   ├── lib.rs
│   │   │   ├── parser.rs
│   │   │   ├── type_checker.rs
│   │   │   ├── resource_verifier.rs
│   │   │   └── operators.rs
│   │   └── tests/
│   │
│   └── common/                  # Shared Rust types
│       ├── Cargo.toml
│       └── src/
│           ├── lib.rs
│           ├── state_vector.rs
│           └── module_types.rs
│
└── scripts/
    ├── setup.sh                 # Full environment setup
    ├── run_benchmarks.sh        # Run all benchmarks
    ├── generate_report.py       # Generate benchmark report
    └── profile_cycle.py         # Profile single cognitive cycle
```

### 2.4 Running Tests

```bash
# Run ALL tests (Python + Rust)
make test-all

# Run Python tests only
make test-python
# Or: pytest python/tests/ -v

# Run Rust tests only
make test-rust
# Or: cd rust && cargo test

# Run a single test file
pytest python/phca/asi/tests/test_sanitizer.py -v --benchmark-skip

# Run a specific test
pytest python/tests/test_acceptance.py::test_cycle_latency -v

# Run benchmarks (not tests — they take longer)
make bench-level-0
make bench-level-1
make bench-all  # Levels 0–5 (this takes days!)
```

### 2.5 PR Workflow

1. **Create a branch:** `git checkout -b <ticket-id>-short-description` (e.g., `phca-3.1-001-asi-sanitizer`)
2. **Implement the ticket.** Commit often. Use meaningful commit messages.
3. **Run tests before pushing:** `make test-all && make lint`
4. **Push:** `git push -u origin <branch>`
5. **Open a PR on GitHub.** Fill out the PR template (what was done, what was tested, benchmark results)
6. **Request review** from the ML/Systems Lead
7. **Address review feedback.** Re-run tests.
8. **Merge** after approval. Use "Squash and merge" to keep history clean.
9. **Delete the branch.**

### 2.6 Debugging Guide

| Symptom | Likely Cause | How to Debug |
| :--- | :--- | :--- |
| NaN in state vector | ASI sanitizer not catching edge case | Add `pdb.set_trace()` in `sanitizer.py`; check IEEE 754 handling |
| Cycle latency > 500ms | G' inference too slow | Run `python scripts/profile_cycle.py` to identify bottleneck |
| RBTA violation unexpectedly | Wrong `B_time` bound | Check `ResourceBounds` in config for that module |
| TSPL not learning | Learning rate α too low or gradient zero | Log gradient norms; check Fisher diagonal magnitude |
| MDIM generates same goal repeatedly | Meta-stable state not triggering | Check Pareto front computation; log deficits daily |
| Float non-determinism between runs | GPU non-determinism | Set `torch.use_deterministic_algorithms(True)` |

### 2.7 Profiling

```bash
# Python: profile the full cognitive cycle
python scripts/profile_cycle.py --cycles 100 --output profile.json

# Rust: benchmark the RBTA enforcer
cd rust/rpta && cargo bench

# Linux: CPU profiling
perf record python scripts/profile_cycle.py --cycles 1000
perf report
```

---

## 3. TICKET BOARD: PHASE 3.1 — CORE ENGINE (Weeks 1–12)

**Goal:** Running predictive core on grid-world with RBTA enforcement, G' world model, P-Stream learning.  
**Components:** ASI v0, M1/M2, G', Prediction Engine, PEU, P-Stream, RBTA Basic, Grid-World Env.  
**VSA:** Excluded. E-Stream/S-Stream/MDIM/CR/Attention/HPM: Not implemented (use fixed defaults).

### 3.1 Ticket List

#### Week 1–2: Foundation

```
PHCA-3.1-001 — Set up repository scaffolding + CI pipeline
- Component: Infrastructure
- v3.0 Ref: None (tooling)
- Acceptance: `make test-all` passes on clean checkout; CI runs on PR
- Deps: None (start ticket)
- Effort: 2 days | Priority: P0 | Risk: Low
- Validation: `test_scaffolding` — repo structure matches layout

PHCA-3.1-002 — Implement grid-world environment (5×5, 10×10, 20×20)
- Component: Environment (grid_world.py)
- v3.0 Ref: Blueprint §D.1
- Acceptance: Agent can move N/S/E/W/STAY; env returns state vector + terminal flag
- Deps: None (start ticket)
- Effort: 2 days | Priority: P0 | Risk: Low
- Validation: `test_grid_world_basic` — verify all 5 actions, 3 grid sizes
```

#### Week 2–4: ASI + Memory

```
PHCA-3.1-003 — Implement shared data types (StateVector, ResourceBounds, GoalVector)
- Component: Common / python/phca/config.py
- v3.0 Ref: Blueprint Appendix D.1
- Acceptance: All Python @dataclass types defined with type annotations; serializable via pickle or numpy
- Deps: PHCA-3.1-001 (scaffolding)
- Effort: 1 day | Priority: P0 | Risk: Low
- Validation: `test_datatypes_serialization` — round-trip serialization + equality

PHCA-3.1-004 — Implement ASI Step 0 Sanitizer
- Component: ASI (sanitizer.py)
- v3.0 Ref: §2.2 Patch B (NaN/Inf/|v|>V_max → hold last valid, halve precision, raise after 7 failures)
- Acceptance: NaN → replaced; precision halves; ASI_SENSOR_FAILURE raised after 7 consecutive failures
- Deps: PHCA-3.1-003 (StateVector type)
- Effort: 3 days | Priority: P0 | Risk: Low
- Validation: `test_asi_sanitizer.py` (5 test cases: NaN, Inf, overflow, valid pass-through, recovery)

PHCA-3.1-005 — Implement M1 Sensory Buffer + M2 Working Memory
- Component: Memory (m1_sensory.py, m2_working.py)
- v3.0 Ref: §3.1 Memory Hierarchy Table
- Acceptance: M1 holds 10×d_sensor samples; M2 holds 7±2 chunks; oldest evicted on overflow
- Deps: PHCA-3.1-003 (StateVector)
- Effort: 2 days | Priority: P0 | Risk: Low
- Validation: `test_m1_capacity`, `test_m2_capacity`, `test_m2_eviction`
```

#### Week 3–6: RBTA + G'

```
*PHCA-3.1-006 — Implement RBTA Constraint Enforcer (Rust, basic: runtime + memory only)
- Component: RBTA (rust/rpta/)
- v3.0 Ref: §2.1 Def 2.2, Def 2.3 (basic bounds only; no composition tree yet)
- Acceptance: Detects runtime > B_time; detects memory > B_mem; returns OK/WARNING/VIOLATION
- Deps: PHCA-3.1-003 (ResourceBounds type — FFI bridge to Rust)
- Effort: 5 days | Priority: P0 | Risk: Medium (Rust FFI)
- Validation: `test_rbta_single_module`, `test_rbta_time_violation`, `test_rbta_mem_violation`
- NOTE: This is on the critical path but has 45d slack — start early, don't block other tickets

PHCA-3.1-007 — Implement G' Probabilistic Graph (nodes, edges, CPDs)
- Component: World Model (graph.py)
- v3.0 Ref: §2.2 Def 2.4b (simplified, no VSA)
- Acceptance: Can create graph with N nodes + E edges; supports discrete CPDs; supports temporal edges
- Deps: PHCA-3.1-003 (StateVector)
- Effort: 5 days | Priority: P0 | Risk: Medium (first ML component)
- Validation: `test_g_creation`, `test_g_temporal_edge`, `test_g_cpd_update`
```

#### Week 5–8: Prediction + Learning

```
PHCA-3.1-008 — Implement G' forward inference (junction tree for small graphs)
- Component: World Model (inference.py)
- v3.0 Ref: §2.2 Def 2.5 (prediction via G' only)
- Acceptance: Given evidence (current state + action), returns P(next_state) for |V| ≤ 100 graphs
- Deps: PHCA-3.1-007 (G' graph structure)
- Effort: 5 days | Priority: P0 | Risk: High — "If G' inference > 50ms, switch to pgmpy exact" (Risk R7)
- Validation: `test_g_forward_deterministic` (confidence = 1.0), `test_g_forward_random` (confidence ≈ 0.0)

*PHCA-3.1-009 — Implement Prediction Engine + Prediction Error Unit (single model)
- Component: Prediction (engine.py, error_unit.py)
- v3.0 Ref: §2.2 Def 2.5 (single-model ensemble, Phase 3.1)
- Acceptance: Engine calls G'.predict(); PEU computes δ = observed - predicted; horizon 1–10 supported
- Deps: PHCA-3.1-008 (G' forward inference)
- Effort: 3 days | Priority: P0 | Risk: Low
- Validation: `test_prediction_engine_basic`, `test_peu_error_computation`

*PHCA-3.1-010 — Implement P-Stream (Procedural TSPL) with skill compilation
- Component: Learning (tspl.py, skill_compilation.py)
- v3.0 Ref: §3.1 Def 3.2 (P-Stream only: highest α, lowest λ, highest η), Def 3.3.3 (skill compilation)
- Acceptance: P-Stream updates G' parameters via δ_t; skill compiles (freezes) at ≥95% accuracy
- Deps: PHCA-3.1-009 (PEU provides δ_t)
- Effort: 5 days | Priority: P0 | Risk: Medium (stochastic — need fixed seed for determinism)
- Validation: `test_p_stream_learning`, `test_skill_compilation_freeze`
```

#### Week 8–11: Integration

```
PHCA-3.1-011 — Implement cognitive cycle orchestrator (Steps 0–15, Phase 3.1 subset)
- Component: Core (cycle.py)
- v3.0 Ref: Blueprint §B Cognitive Cycle (Phase 3.1: Steps 0–7, 9, 14–15, 19)
- Acceptance: Full 500ms cycle; ASI→M2→G'→Prediction→PEU→TSPL→action; RBTA enforces
- Deps: PHCA-3.1-004 through PHCA-3.1-010
- Effort: 5 days | Priority: P0 | Risk: High (first integration point)
- Validation: `test_cognitive_cycle_basic`, `test_cycle_latency_under_500ms`

PHCA-3.1-012 — Phase 3.1 integration + acceptance tests
- Component: Tests / Infrastructure
- v3.0 Ref: §1.3 criteria 1 (cycle latency) + Phase 3.1 acceptance criteria
- Acceptance: Grid-world navigation learned; RBTA violation detected on slow module; cycle < 500ms
- Deps: PHCA-3.1-011
- Effort: 5 days | Priority: P0 | Risk: Medium
- Validation: All Phase 3.1 gate conditions pass (see §13.1)
```

**Total Phase 3.1 tickets:** 12 tickets / ~43 engineering-days (8 person-weeks with 2 engineers)

**Path markings:** `*` = critical path ticket. These cannot slip without delaying Phase 3.1 delivery.

---

## 4. TICKET BOARD: PHASE 3.2 — FULL ARCHITECTURE (Weeks 13–28)

**Goal:** Complete 21-component architecture with anti-forgetting, MDIM, Criticality Regulator, HPM, VSA (conditional).  
**Components added:** ASI v1, E-Stream, S-Stream, EWC, GEM, M3/M4/M5, Consolidation, MDIM, CR, Attention, HPM, V (conditional).

### 4.1 Ticket List (Abbreviated)

```
PHCA-3.2-001 — Upgrade ASI to v1: Grounding Level Adapter (Levels 0, 1, 2)
- v3.0 Ref: §2.5.1 Def 2.5a; Effort: 3d; Deps: PHCA-3.1-011; Risk: Low
- Validation: test_asi_level_0, test_asi_level_1, test_asi_level_2

PHCA-3.2-002 — Implement M3 Episodic Memory (SQLite MVCC)
- v3.0 Ref: §2.3.2 Patch C (MVCC semantics); Effort: 5d; Deps: PHCA-3.1-005; Risk: Medium
- Validation: test_m3_mvcc_snapshot, test_m3_write_after_snapshot_invisible

PHCA-3.2-003 — Implement E-Stream (Episodic TSPL with GEM)
- v3.0 Ref: §3.1 Def 3.2 (α_E, η_E), Def 3.3.2 (GEM); Effort: 5d; Deps: PHCA-3.2-002, PHCA-3.1-010; Risk: Medium
- Validation: test_e_stream_learning, test_gem_projection

PHCA-3.2-004 — Implement S-Stream (Semantic TSPL with EWC)
- v3.0 Ref: §3.1 Def 3.2 (α_S, λ_S), Def 3.3.1 (EWC); Effort: 5d; Deps: PHCA-3.1-010; Risk: Medium
- Validation: test_s_stream_learning, test_ewc_penalty, test_fisher_diagonal

PHCA-3.2-005 — Implement Consolidation Scheduler (E→S transfer with MVCC)
- v3.0 Ref: §2.3.2 Patch C, Theorem 3.3; Effort: 5d; Deps: PHCA-3.2-002, PHCA-3.2-003, PHCA-3.2-004; Risk: High
- Validation: test_consolidation_e_to_s, test_mvcc_atomicity

PHCA-3.2-006 — Implement M4 Semantic Memory (RocksDB / SQLite + write-lock)
- v3.0 Ref: §2.3.2 Patch C (write-lock semantics); Effort: 3d; Deps: PHCA-3.2-005; Risk: Medium
- Validation: test_m4_write_lock, test_m4_read_during_write

PHCA-3.2-007 — Implement M5 Procedural Memory (frozen skills)
- v3.0 Ref: §3.1 Table (no-lock, read-only after compilation); Effort: 2d; Deps: PHCA-3.1-010; Risk: Low
- Validation: test_m5_read_after_compilation

*PHCA-3.2-008 — Implement MDIM: 6 Drives (D1–D6) with homeostasis
- v3.0 Ref: §3.3 Def 3.8 (drives), Def 3.9 (goal generation); Effort: 5d; Deps: PHCA-3.2-003, PHCA-3.2-004; Risk: Medium
- Validation: test_mdim_drive_setpoints, test_mdim_softmax_goal_selection

*PHCA-3.2-009 — Implement MDIM Pareto Front + Meta-Stable State
- v3.0 Ref: §2.4 Def 3.10 (Pareto front), Def 3.11 (meta-stable state), Theorem 3.2; Effort: 5d; Deps: PHCA-3.2-008; Risk: High
- Validation: test_pareto_front_detection, test_meta_stable_suppression, test_oscillation_prevention

*PHCA-3.2-010 — Implement Criticality Regulator (PID + integral windup protection)
- v3.0 Ref: §2.3 Def 2.8; Effort: 5d; Deps: PHCA-3.1-011 (needs all modules for Φ estimation); Risk: Medium
- Validation: test_pid_basic, test_integral_windup_protection, test_criticality_convergence

PHCA-3.2-011 — Implement Criticality Regulator Orthogonality Constraint
- v3.0 Ref: §2.6 Def 2.8a (covariance monitoring); Effort: 3d; Deps: PHCA-3.2-010; Risk: Medium
- Validation: test_covariance_monitoring, test_parameter_freeze

PHCA-3.2-012 — Implement Precision-Weighted Attention (k-WTA + Gumbel noise)
- v3.0 Ref: §D.3 Def 5.1; Effort: 5d; Deps: PHCA-3.1-005 (needs M2), PHCA-3.2-008 (needs goal); Risk: Medium
- Validation: test_attention_basic, test_gumbel_noise, test_precision_weighting

*PHCA-3.2-013 — Implement HPM Grammar Runtime (Python reference, then Rust)
- v3.0 Ref: §3.2 Def 3.4–3.7; Effort: 10d; Deps: PHCA-3.1-006 (needs ResourceBounds compositing); Risk: High
- Validation: test_hpm_valid_composition, test_hpm_type_error, test_hpm_resource_verification

PHCA-3.2-014 — Implement World Model V (VSA) — CONDITIONAL
- v3.0 Ref: §2.2 Def 2.4 (V component); Effort: 10d; Deps: PHCA-3.1-007 (G' needed for priors); Risk: High
- Validation: test_vsa_binding, test_vsa_retrieval, test_vsa_vs_g_similarity
- NOTE: Only implement if G' similarity search is insufficient at Level 3 benchmarks (decision point)

PHCA-3.2-015 — Upgrade Prediction Engine to dual-model ensemble (G' + V)
- v3.0 Ref: §2.2 Def 2.5 (conditional on grounding level); Effort: 3d; Deps: PHCA-3.2-014 (if VSA included); Risk: Low

PHCA-3.2-016 — Phase 3.2 integration + full cycle tests (all 21 steps)
- Effort: 10d; Deps: All Phase 3.2 tickets except VSA (optional)
- Validation: All Phase 3.2 gate conditions pass (see §13.2)
```

**Total Phase 3.2 tickets:** 16 tickets / ~76 engineering-days (15 person-weeks with 2 engineers)

---

## 5. TICKET BOARD: PHASE 3.3 — BENCHMARKING & VALIDATION (Weeks 29–40)

**Goal:** Run all 6 benchmark levels; validate all 5 success criteria from v3.0 §1.3.

### 5.1 Ticket List (Abbreviated)

```
*PHCA-3.3-001 — Implement Failure Detection & Recovery Matrix (all 6 categories A–F)
- v3.0 Ref: §4.1 (30+ failure modes); Effort: 10d; Deps: PHCA-3.2-016; Risk: High
- Validation: test_failure_detection_all_modes, test_recovery_within_10_cycles

*PHCA-3.3-002 — Implement Φ-IQ Evaluation Suite Runner + CLI
- v3.0 Ref: §1.3 success criteria, §5 Φ-IQ composite; Effort: 5d; Deps: PHCA-3.3-001; Risk: Low
- Validation: test_phi_iq_computation, test_cli_help

PHCA-3.3-003 — Level 0 benchmark (stationary prediction)
- v3.0 Ref: Blueprint §D.2 Level 0; Effort: 3d; Deps: PHCA-3.3-002; Risk: Low
- Acceptance: MSE < 0.1 after 1000 steps on pendulum

PHCA-3.3-004 — Level 1 benchmark (reactive control)
- v3.0 Ref: Blueprint §D.2 Level 1; Effort: 5d; Deps: PHCA-3.3-002; Risk: Medium
- Acceptance: Stability within 100 steps on inverted pendulum

PHCA-3.3-005 — Level 2 benchmark (goal pursuit)
- v3.0 Ref: Blueprint §D.2 Level 2; Effort: 5d; Deps: PHCA-3.3-002; Risk: Medium
- Acceptance: Goal completion > 80% over 100 trials

PHCA-3.3-006 — Level 3 benchmark (self-motivated exploration)
- v3.0 Ref: Blueprint §D.2 Level 3; Effort: 5d; Deps: PHCA-3.3-002; Risk: Medium
- Acceptance: ≥ 1 novel goal per 100 cycles

PHCA-3.3-007 — Level 4 benchmark (continual learning, 100 tasks)
- v3.0 Ref: Blueprint §D.2 Level 4; Effort: 10d; Deps: PHCA-3.3-002; Risk: High
- Acceptance: Forgetting ≤ 5% after 100 sequential tasks

PHCA-3.3-008 — Level 5 benchmark (multi-agent coordination)
- v3.0 Ref: Blueprint §D.2 Level 5; Effort: 10d; Deps: PHCA-3.3-002; Risk: High
- Acceptance: Joint success > 60%, synergy ratio > 1.0

PHCA-3.3-009 — Comprehensive benchmark report generation
- v3.0 Ref: §1.3 criteria 1–5; Effort: 5d; Deps: PHCA-3.3-003 through 008; Risk: Low
- Deliverable: benchmarks/results/final_report_{date}.json + .pdf
```

**Total Phase 3.3 tickets:** 9 tickets / ~58 engineering-days (12 person-weeks with 2 engineers)

---

## 6. TICKET BOARD: PHASE 3.4 — ITERATION & REFINEMENT (Weeks 41–52)

**Goal:** Parameter optimization, bottleneck fixes, VSA decision, final documentation.

### 6.1 Ticket List

```
PHCA-3.4-001 — Parameter sweep infrastructure (grid search over 16 hyperparameters)
- Effort: 5d; Deps: PHCA-3.3-009; Risk: Low

PHCA-3.4-002 — Parameter sweep execution (Phase 3.4 hyperparameter table)
- Effort: 10d; Deps: PHCA-3.4-001; Risk: Medium (compute cost)

PHCA-3.4-003 — Bottleneck identification + optimization (target: < 500ms cycle)
- Effort: 10d; Deps: PHCA-3.4-002; Risk: Medium

PHCA-3.4-004 — VSA permanence decision gate (keep or cut permanently)
- Effort: 3d (decision); Deps: PHCA-3.3-006 (Level 3 results); Risk: Low
- Decision criteria: If G' similarity achieves < 20% degradation vs. V on analogical reasoning → CUT VSA

PHCA-3.4-005 — Component pruning analysis (identify components not contributing to Φ-IQ)
- Effort: 5d; Deps: PHCA-3.4-002; Risk: Medium

PHCA-3.4-006 — Re-run full benchmark suite (post-optimization)
- Effort: 10d (compute time); Deps: PHCA-3.4-003, PHCA-3.4-004, PHCA-3.4-005; Risk: Low

PHCA-3.4-007 — Final documentation: API reference, architecture diagram, deployment guide
- Effort: 10d; Deps: PHCA-3.4-006; Risk: Low

PHCA-3.4-008 — Final report: "PHCA v3.0 — Validated Architecture Specification"
- Effort: 5d; Deps: PHCA-3.4-004, PHCA-3.4-006, PHCA-3.4-007; Risk: Low
```

**Total Phase 3.4 tickets:** 8 tickets / ~58 engineering-days (12 person-weeks with 2 engineers)

---

## 7. COMPONENT IMPLEMENTATION CHECKLISTS

### 7.1 ASI (Abstract Sensorimotor Interface)

**File:** `python/phca/asi/sanitizer.py`, `python/phca/asi/grounding_adapter.py`

**Data Structures:**
```python
@dataclass
class ASIConfig:
    sensor_dim: int
    v_max: float                        # V_max: maximum plausible sensor value
    epsilon_confidence: float = 0.01    # ε_confidence: minimum precision threshold
    asi_failure_limit: int = Field(init=False)  # floor(d / 3)

    def __post_init__(self):
        self.asi_failure_limit = self.sensor_dim // 3

class ASISanitizer:
    last_valid: np.ndarray              # v_j^{(t-1)}
    precision: np.ndarray               # p_j per sensor
    failure_count: np.ndarray           # consecutive failures per sensor
```

**Public API:**
```python
class ASISanitizer:
    def sanitize(self, raw: np.ndarray) -> tuple[np.ndarray, np.ndarray, ASIStatus]: ...
    def reset(self) -> None: ...

class ASIGroundingAdapter:
    def adapt(self, state: StateVector, level: int) -> tuple[StateVector | None, StateVector | None]: ...
```

**Internal Algorithm (Sanitizer):**
```python
def sanitize(self, raw: np.ndarray) -> tuple[np.ndarray, np.ndarray, ASIStatus]:
    clean = np.copy(raw)
    for j in range(raw.shape[0]):
        if np.isnan(raw[j]) or np.isinf(raw[j]) or abs(raw[j]) > self.config.v_max:
            clean[j] = self.last_valid[j]
            self.precision[j] *= 0.5
            self.failure_count[j] += 1
            if self.precision[j] < self.config.epsilon_confidence:
                return clean, self.precision, ASIStatus.SENSOR_FAILURE(j)
        else:
            self.last_valid[j] = raw[j]
            self.failure_count[j] = 0
    return clean, self.precision, ASIStatus.OK
```

**Error Handling:**
- NaN/Inf → hold last valid value; halve precision
- precision < ε_confidence → raise `ASI_SENSOR_FAILURE(j)` → trigger B1 recovery
- > d/3 sensors failed simultaneously → global sensor failure recovery mode

**Logging:**
```python
# Required log lines (use structlog):
log.info("asi.sanitize.failure", sensor=j, value=raw[j], precision=precision[j], cycle=current_cycle)
log.warning("asi.sanitize.threshold_exceeded", sensor=j, precision=precision[j])
log.critical("asi.sanitize.global_failure", failed_sensors=failure_count, limit=ASI_FAILURE_LIMIT)
```

**Unit Tests (≥5):**
1. `test_nan_replaced_by_last_valid` — NaN in → last valid value out
2. `test_precision_halves_on_failure` — precision 1.0 → 0.5 after single NaN
3. `test_sensor_failure_after_7_consecutive` — 7th consecutive failure raises status
4. `test_valid_value_passes_through` — no modification to valid inputs
5. `test_recovery_after_failure` — after valid value, precision resets to 1.0
6. `test_asi_failure_limit` — > d/3 sensors failed triggers global failure
7. `test_grounding_level_0_feature_encoder` — level 0 returns encoded features

**Performance Targets:**
- Sanitization: < 10μs for d ≤ 1024
- Grounding adapter: < 1ms (level 0), < 10μs (level 1), < 100μs (level 2 with V)
- Memory: ASISanitizer state = 3 × d × 4 bytes ≈ 12KB for d=1024

---

### 7.2 RBTA Constraint Enforcer

**File:** `rust/rpta/src/enforcer.rs`

**Data Structures:**
```rust
#[derive(Debug, Clone)]
pub struct ResourceBounds {
    pub b_time: f64,        // seconds
    pub b_mem: f64,         // bytes
    pub b_energy: f64,      // Joules (estimated from FLOPs)
    pub entropy_floor: f64, // minimum acceptable H(beliefs)
}

#[derive(Debug)]
pub struct ConstraintViolation {
    pub module_id: String,
    pub bound_type: BoundType,  // TIME | MEM | ENERGY | ENTROPY | SENSOR
    pub measured: f64,
    pub allowed: f64,
}

#[derive(Debug, PartialEq)]
pub enum EnforcerAction { Continue, Interrupt, Terminate }

pub struct RBTAEnforcer {
    bounds: HashMap<String, ResourceBounds>,
    composition_tree: Option<HPMNode>,
}
```

**Public API:**
```rust
impl RBTAEnforcer {
    pub fn new(config: EnforcerConfig) -> Self;
    pub fn check_cycle(
        &self,
        runtime_log: &HashMap<String, f64>,
        memory_log: &HashMap<String, f64>,
        energy_log: &HashMap<String, f64>,
        belief_entropies: &HashMap<String, f64>,
        sensor_validity: &BitVec,
    ) -> (Vec<ConstraintViolation>, EnforcerStatus, EnforcerAction);
    pub fn update_composition_tree(&mut self, tree: HPMNode);
}
```

**Internal Algorithm:**
```rust
pub fn check_cycle(&self, ...) -> (...) {
    let mut violations = Vec::new();
    for (module_id, bounds) in &self.bounds {
        if runtime_log[module_id] > bounds.b_time { violations.push(...); }
        if memory_log[module_id] > bounds.b_mem { violations.push(...); }
        if energy_log[module_id] > bounds.b_energy { violations.push(...); }
        if belief_entropies[module_id] < bounds.entropy_floor { violations.push(...); }
    }
    
    // Composite bound check (Theorem 3.1 monotonicity)
    if let Some(ref tree) = self.composition_tree {
        self.check_composite_bounds(tree, &runtime_log, &mut violations);
    }
    
    let status = match violations.len() { 0 => OK, 1..=2 => WARNING, _ => VIOLATION };
    let action = match status { OK => Continue, WARNING => Interrupt, VIOLATION => Terminate };
    (violations, status, action)
}
```

**Error Handling:**
- If `module_id` missing from a log, treat as 0.0 (module did not run)
- If `module_id` not in `self.bounds`, skip it (unregistered module)
- Rust panic → caught at FFI boundary; returned as `TERMINATE` action

**Logging:**
```rust
info!("rpta.violation", module = %module_id, type = ?bound_type, measured, allowed);
warn!("rpta.warning", count = violations.len());
error!("rpta.terminate", count = violations.len());
```

**Unit Tests (≥5):**
1. `test_single_module_time_violation` — runtime > B_time → violation
2. `test_single_module_all_bounds_ok` — no violations → OK status
3. `test_sequence_composition_time` — composite time = sum + τ_comp
4. `test_parallel_composition_time` — composite time = max + τ_sync
5. `test_entropy_floor_violation` — H(beliefs) < entropy_floor → violation
6. `test_nested_composition_monotonicity` — M = ((A∘B)∥C) — all bounds dominated

**Performance Targets:**
- Single module check: < 1μs
- Full 20-module check: < 100μs
- Composite tree (depth 5): < 50μs
- Total: < 200μs per cycle
- Memory: < 1MB for bounds storage

---

### 7.3 World Model G' (Probabilistic Graph)

**File:** `python/phca/world_model/graph.py`, `inference.py`, `similarity.py`

**Data Structures:**
```python
@dataclass
class StateNode:
    name: str
    cpd_type: Literal["discrete", "gaussian", "conditional_gaussian"]
    parents: list[str]
    params: np.ndarray | dict  # CPD parameters (table or mean/std)

@dataclass
class TemporalEdge:
    source: str
    target: str
    lag: int  # e.g., 1 for X_i^{(t)} → X_j^{(t+1)}
    params: np.ndarray

class WorldModelGPrime:
    nodes: dict[str, StateNode]
    temporal_edges: list[TemporalEdge]
    causal_edges: list[tuple[str, str]]
    similarity_index: KDTree  # for analogical retrieval (replaces V)
    state_history: list[StateVector]  # for k-NN queries
```

**Public API:**
```python
class WorldModelGPrime:
    def predict(self, state: StateVector, action: np.ndarray) -> tuple[StateVector, float]: ...
    def learn(self, state_t: StateVector, action: np.ndarray, state_t1: StateVector, error: float) -> None: ...
    def add_node(self, node: StateNode) -> None: ...
    def add_temporal_edge(self, edge: TemporalEdge) -> None: ...
    def add_causal_edge(self, source: str, target: str) -> None: ...
    def similarity_search(self, query: StateVector, k: int = 5) -> list[tuple[StateVector, float]]: ...
```

**Internal Algorithm (Predict):**
```python
def predict(self, state: StateVector, action: np.ndarray) -> tuple[StateVector, float]:
    evidence = {f"X_{i}_t": state.values[i] for i in range(len(state.values))}
    evidence["action_t"] = action
    # Build query network with temporal edges
    network = self._build_forward_network(evidence)
    if len(self.nodes) <= 50:
        posterior = pgmpy.inference.VariableElimination(network).query(variables=["X_t1_0"], evidence=evidence)
    else:
        posterior = pyro.infer.Importance(network, num_samples=500).run(evidence)
    predicted_values = posterior.mean if hasattr(posterior, 'mean') else posterior
    confidence = 1.0 - entropy(predicted_values) / max_entropy(self._state_dim)
    return StateVector(values=predicted_values, ...), confidence
```

**Error Handling:**
- Missing evidence variable → use prior marginal (no crash)
- Inference failure (graph disconnected) → log warning, return zero vector with confidence = 0.0
- KDTree empty → return empty list from similarity_search

**Logging:**
```python
log.info("gprime.predict", state=state.values[:5], horizon=1, confidence=confidence)
log.warning("gprime.inference_fallback", method="prior", reason="disconnected_graph")
log.info("gprime.similarity_search", query=query.values[:3], matches=len(results))
```

**Unit Tests (≥5):**
1. `test_deterministic_chain` — A→B→C chain: confidence = 1.0 for B given A
2. `test_random_variable` — independent random variable: confidence ≈ 0.0
3. `test_temporal_edge_learning` — X_i^{(t)}→X_j^{(t+1)} learns lag-1 dependency after 100 steps
4. `test_similarity_search` — k-NN returns nearest states from history
5. `test_graph_creation_validation` — invalid edge (source not in nodes) → ValueError

**Performance Targets:**
- Forward inference (|V|≤50, |E|≤200): < 10ms (exact junction tree)
- Forward inference (|V|≤200, |E|≤1000): < 50ms (sampling)
- Similarity search (k=5, history=10K): < 1ms
- Learning update: < 5ms
- Total: < 20ms per cycle

---

### 7.4 Prediction Engine

**File:** `python/phca/prediction/engine.py`, `error_unit.py`, `ensemble.py`

**Data Structures:**
```python
class PredictionEngine:
    ensemble_weights: list[float]  # [w_G] for Phase 3.1; [w_G, w_V] for Phase 3.2+
    meta_lr: float = 0.01
    prediction_history: deque[tuple[StateVector, StateVector, float]]  # (state, pred, error), maxlen=1000
```

**Public API:**
```python
class PredictionEngine:
    def predict(self, state: StateVector, goal: GoalVector | None, horizon: int,
                grounding_level: int) -> tuple[StateVector, float]: ...
    def update_ensemble(self, observation: StateVector, prediction: StateVector) -> None: ...
```

**Internal Algorithm:**
```python
def predict(self, state, goal, horizon, grounding_level):
    if grounding_level in (0, 1):
        prediction, confidence = self.gprime.predict(state, self.last_action)
    elif grounding_level == 2:
        prediction, confidence = self.vsa.retrieve(self.grounding_adapter.adapt(state, 2))
    return prediction, confidence

def update_ensemble(self, observation, prediction):
    error = np.linalg.norm(observation.values - prediction.values)
    # Meta-gradient update (Phase 3.1: single weight = 1.0, no-op)
    for k in range(len(self.ensemble_weights)):
        grad = 2 * (observation.values - prediction.values) @ prediction.values
        self.ensemble_weights[k] -= self.meta_lr * grad
```

**Error Handling:**
- horizon > known_horizon → confidence decays logarithmically (Parrondo's bound)
- No G' available (level 2, no VSA) → raise PredictionUnavailable → fallback to random action

**Unit Tests (≥5):**
1. `test_horizon_1_returns_value` — horizon 1 returns a finite prediction
2. `test_confidence_deterministic` — deterministic environment → confidence = 1.0
3. `test_prediction_diverges` — horizon > train_horizon → confidence → 0.0
4. `test_meta_gradient_update` — ensemble weight shifts toward better model
5. `test_grounding_level_2_no_vsa` — level 2 without VSA raises PredictionUnavailable

**Performance Targets:**
- Single-model ensemble: < 1ms overhead
- Meta-gradient update: < 2ms
- Total (including G' inference): < 25ms for horizon ≤ 5

---

### 7.5 TSPL (Three-Stream Predictive Learning)

**File:** `python/phca/learning/tspl.py`, `ewc.py`, `gem.py`, `skill_compilation.py`

**Data Structures:**
```python
@dataclass
class StreamConfig:
    alpha: float      # learning rate
    lambda_: float    # elastic consolidation
    eta: float        # exploration noise
    ew_lambda: float  # EWC penalty (S-Stream only)
    gem_buffer_size: int  # GEM buffer (E-Stream only)

class TSPL:
    configs: dict[StreamID, StreamConfig]
    theta: dict[str, np.ndarray]  # current parameters
    theta_protected: dict[str, np.ndarray]  # EWC-protected snapshot
    fisher_diagonal: dict[str, np.ndarray]  # Fisher information (diagonal)
    episodic_buffer: list[Episode]  # for GEM (E-Stream only)
```

**Public API:**
```python
class TSPL:
    def update(self, stream: StreamID, prediction_error: float, 
               state: StateVector, prediction: StateVector) -> tuple[dict, bool]: ...
    def freeze_skill(self, skill_id: str) -> None: ...
    def compute_fisher(self, dataset: Iterable[tuple[StateVector, StateVector]]) -> None: ...
```

**Internal Algorithm (P-Stream example):**
```python
def update_p_stream(self, prediction_error, state, prediction):
    config = self.configs[P_STREAM]
    gradient = self._compute_gradient(prediction_error, state, prediction)
    
    # Unified TSPL update (v3.0 Def 3.2):
    theta_new = {}
    for key in self.theta:
        theta_new[key] = (self.theta[key] 
            - config.alpha * gradient[key]
            - config.lambda_ * (self.theta[key] - self.theta_protected[P_STREAM][key])
            + config.eta * np.random.randn(*self.theta[key].shape))
    
    # Skill compilation check (v3.0 Def 3.3.3):
    accuracy = self._estimate_skill_accuracy()
    skill_compiled = accuracy >= 0.95
    
    return theta_new, skill_compiled
```

**Error Handling:**
- Fisher diagonal has zero entries → add ε=1e-8 to prevent division by zero
- GEM projection fails (norm=0) → skip projection for this task

**Logging:**
```python
log.info("tspl.update", stream=stream.name, error=prediction_error, 
         alpha=config.alpha, lambda_=config.lambda_)
log.info("tspl.skill_compiled", skill_id=skill_id, accuracy=accuracy)
log.warning("tspl.fisher_zero_entry", parameter=key, epsilon_added=True)
```

**Unit Tests (≥5):**
1. `test_p_stream_highest_alpha` — P-Stream α > E-Stream α > S-Stream α
2. `test_s_stream_ewc_preserves_old_task` — EWC penalty reduces forgetting on Task A after Task B
3. `test_e_stream_gem_projection` — GEM gradient projection prevents loss increase
4. `test_skill_compilation_freeze` — params frozen after accuracy ≥ 0.95
5. `test_fisher_diagonal_computation` — Fisher diagonal has correct shape + non-negative values

**Performance Targets:**
- Gradient computation (< 10K params): < 5ms
- EWC penalty: < 2ms
- GEM projection (buffer=5): < 10ms
- Total per stream per cycle: < 20ms

---

### 7.6 MDIM (Multi-Drive Intrinsic Motivation)

**File:** `python/phca/motivation/mdim.py`, `drives.py`, `pareto_front.py`

**Data Structures:**
```python
@dataclass
class HomeostaticDrive:
    name: str              # "D1" .. "D6"
    current_value: float
    set_point: float
    threshold: float       # θ_i: deviation threshold for homeostasis
    corrective_actions: list[str]

class MDIM:
    drives: dict[str, HomeostaticDrive]  # D1–D6
    temperature: float = 1.0             # softmax temperature (T_MDIM)
    meta_stable: bool = False
    meta_stable_config: MetaStableConfig
    goal_stack: list[GoalVector]          # bounded by D_max
```

**Public API:**
```python
class MDIM:
    def update_drives(self, inputs: DriveInputs) -> None: ...
    def generate_goal(self, context: GoalContext) -> GoalVector: ...
    def is_on_pareto_front(self) -> bool: ...
    def meta_stable_check(self) -> bool: ...
```

**Internal Algorithm:**
```python
def generate_goal(self, context):
    # Compute deficits
    deficits = {d.name: abs(d.current_value - d.set_point) for d in self.drives.values()}
    
    # Pareto front check (v3.0 §2.4 Def 3.10)
    on_front = self.is_on_pareto_front(deficits["D1"], deficits["D3"], deficits["D5"])
    
    if self.meta_stable or (on_front and all(
        deficits[d] < self.drives[d].threshold for d in ["D1", "D3", "D5"])):
        # Meta-stable state: only non-conflicting drives D2, D4, D6
        self.meta_stable = True
        weights = softmax([deficits[d] for d in ["D2", "D4", "D6"]], self.temperature)
        goal_type = categorical_sample(weights, ["D2", "D4", "D6"])
    else:
        self.meta_stable = False
        weights = softmax(list(deficits.values()), self.temperature)
        goal_type = categorical_sample(weights, list(self.drives.keys()))
    
    # Contextualize (bounded to ≤ 200ms)
    goal = self._contextualize(goal_type, context.state, context.m3, context.m4)
    self.goal_stack.append(goal)
    return goal
```

**Error Handling:**
- Goal stack exceeds D_max → pop oldest before pushing new (v3.0 audit fix)
- Contextualization exceeds 200ms → RBTA enforcer will catch it → interrupt → retry with simpler context

**Logging:**
```python
log.info("mdim.goal_generated", drive=goal_type, deficits=deficits, meta_stable=self.meta_stable)
log.warning("mdim.goal_stack_pruned", old_size=len(stack), new_size=D_max)
log.info("mdim.pareto_front", on_front=on_front, d1=d1, d3=d3, d5=d5)
```

**Unit Tests (≥5):**
1. `test_drive_homeostasis` — each drive returns to setpoint after perturbation
2. `test_softmax_goal_selection` — distribution over drives is categorical
3. `test_pareto_front_detection` — (low, low, low) is Pareto-optimal
4. `test_meta_stable_suppression` — D1/D3/D5 suppressed in meta-stable state
5. `test_goal_stack_bounded` — stack never exceeds D_max
6. `test_empowerment_computation` — D6 = I(s_{t+1}; a_t | s_t) for 2-state MDP

**Performance Targets:**
- Drive deficit computation: < 100μs
- Pareto front check: < 500μs
- Goal contextualization: < 200ms (v3.0 hard bound)
- Total: < 200ms per cycle

---

### 7.7 Remaining Components (Abbreviated Checklists)

Each remaining component follows the same pattern. Key specifications:

| Component | File | Key Method | Perf Budget | Critical Tests |
| :--- | :--- | :--- | :--- | :--- |
| **Attention** | `attention/attention.py` | `select(WM, goal, precision) -> List[ChunkID]` | < 2ms | k-WTA selection; Gumbel noise; precision weighting |
| **Criticality Regulator** | `regulation/pid_controller.py` | `regulate(Φ_current, T, η, α) -> (ΔT, Δη, Δα)` | < 5ms | PID convergence; integral windup; orthogonality covariance |
| **Consolidation Scheduler** | `memory/consolidation.py` | `scheduled_consolidation() -> bool` | < 1ms | MVCC snapshot isolation; write atomicity; lost-update prevention |
| **HPM Grammar** | `hpm/parser.py`, `type_checker.py` | `validate(spec: str) -> HPMNode` | < 2ms | Type safety; resource additivity (SEQ/PAR); nested composition |
| **Failure Detection** | `resilience/detector.py` | `detect(all_states) -> List[FailureEvent]` | < 1ms | All 30+ modes detectable; cascade detection |
| **Φ-IQ Evaluator** | `benchmarks/metrics.py` | `compute(results: dict) -> float` | N/A (offline) | Monotonic across levels; dimension computation |

Full specifications for each are in Blueprint Section B and will be expanded in the `docs/api-reference.md` file during Phase 3.1 implementation.

---

## 8. INTEGRATION TEST PLAN

### 8.1 Phase 3.1 Integration Tests

| Test ID | Name | What It Tests | Stubs/Mocks | Expected Outcome |
| :--- | :--- | :--- | :--- | :--- |
| IT-3.1-1 | ASI→M2 Pipeline | Sanitizer output flows to WM correctly | Grid-world env stub | After 1 cycle, M2 has 1 chunk |
| IT-3.1-2 | G'→PE→PEU→TSPL Loop | Prediction error flows through PEU to TSPL update | Deterministic env (no randomness) | Parameters change in response to δ_t |
| IT-3.1-3 | Full Cognitive Cycle | All 15 Phase 3.1 cycle steps execute | Grid-world env | Cycle completes in < 500ms |
| IT-3.1-4 | RBTA Enforcement | Artificially slow module triggers violation | Slow module stub (sleep 100ms) | RBTA returns VIOLATION |
| IT-3.1-5 | P-Stream Skill Compilation | Agent learns grid-world navigation | Fixed 5×5 grid, single goal | After 500 episodes, accuracy ≥ 95% |

### 8.2 Phase 3.2 Integration Tests

| Test ID | Name | What It Tests | Stubs/Mocks | Expected Outcome |
| :--- | :--- | :--- | :--- | :--- |
| IT-3.2-1 | E-Stream + GEM Continual Learning | Agent trained on Task A, then Task B | 2-task sequence | Task A accuracy drops < 5% after Task B |
| IT-3.2-2 | S-Stream + EWC Continual Learning | Same as IT-3.2-1 but semantic protection | 2-task sequence | Task A accuracy drops < 2% (stronger protection) |
| IT-3.2-3 | Consolidation E→S Transfer | Episodes consolidated into semantic memory | M3 with 100 episodes | After sleep cycle, M4 has ≥50 new facts |
| IT-3.2-4 | MDIM Goal Generation | Agent in empty environment generates goals | Empty grid-world, no external task | ≥1 goal generated per 100 cycles |
| IT-3.2-5 | Criticality Regulation | PID maintains Φ near Φ_critical | Full system, 1000 cycles | Φ within bounds ≥ 90% of cycles |
| IT-3.2-6 | HPM Composition | Valid vs. invalid compositions | Pre-built module specs | Valid passes, invalid fails |
| IT-3.2-7 | Full Phase 3.2 Cycle | All 21 steps execute | MuJoCo reacher task | Cycle < 500ms with attention active |

### 8.3 Phase 3.3 Integration Tests

| Test ID | Name | What It Tests | Stubs/Mocks | Expected Outcome |
| :--- | :--- | :--- | :--- | :--- |
| IT-3.3-1 | Failure Detection Matrix | All 30+ failure modes detected | Injected failure per mode | Each mode detected |
| IT-3.3-2 | Failure Recovery Bounded | Recovery within 10 cycles | Injected B1 (distribution shift) | Recovery complete by cycle 10 |
| IT-3.3-3 | Φ-IQ Suite Runs End-to-End | Runner executes all 6 levels | Full system | Results JSON produced |
| IT-3.3-4 | Multi-Agent Coordination | 2 agents coordinate | Shared grid-world | Joint success > 60% |

### 8.4 Regression Prevention (CI)

The following tests MUST pass on every PR commit (automated in CI):

```yaml
# .github/workflows/ci.yml — on: [pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Install Python deps
        run: pip install -r requirements.txt
      - name: Build Rust
        run: cd rust && cargo build --release
      - name: Lint
        run: make lint  # ruff (Python) + clippy (Rust)
      - name: Unit tests (Python)
        run: pytest python/phca/ -v --timeout=30 -x
      - name: Unit tests (Rust)
        run: cd rust && cargo test
      - name: Integration tests (Phase 3.1)
        run: pytest python/tests/test_phase_3_1.py -v --timeout=120 -x
      - name: Cycle latency check
        run: python scripts/profile_cycle.py --cycles=10 --max-ms=600
```

### 8.5 Stub/Mock Strategy

| Component | Phase 3.1 Stub | Phase 3.2+ Stub |
| :--- | :--- | :--- |
| **ASI** | Real (trivial) | Real |
| **RBTA** | Real | Real |
| **G'** | Real | Real |
| **V (VSA)** | N/A (excluded) | Stub: returns zero vector + confidence=0.5 |
| **Prediction Engine** | Real | Real |
| **TSPL (P-Stream)** | Real | Real |
| **TSPL (E-Stream)** | Stub: no-op | Real |
| **TSPL (S-Stream)** | Stub: no-op | Real |
| **MDIM** | Stub: returns fixed goal | Real |
| **Criticality Regulator** | Stub: returns fixed params | Real |
| **Attention** | Stub: returns all M2 contents | Real |
| **HPM Grammar** | Stub: passes all | Real |
| **Consolidation** | N/A (M3 not exists) | Real |
| **Failure Detection** | Stub: returns empty | Real (Phase 3.3) |

---

## 9. ACCEPTANCE TEST PLAN

### 9.1 Criterion 1: Cycle Latency (< 500ms with |WM| = 7)

| Field | Value |
| :--- | :--- |
| **Test ID** | AT-1 |
| **v3.0 Ref** | §1.3 criterion 1 |
| **Test command** | `python scripts/profile_cycle.py --cycles=1000 --max-ms=500 --wm-size=7` |
| **Environment** | Target hardware: Intel i7-12700H or equivalent, 16GB RAM, no GPU |
| **Data collection** | `time.perf_counter_ns()` before/after each cycle. Recorded to `logs/cycle_latency_{timestamp}.json` |
| **Pass threshold** | Median cycle time < 500ms over 1000 consecutive cycles |
| **Fail threshold** | Median > 500ms or any single cycle > 2000ms |
| **Automation** | ✅ Automated in CI (nightly benchmark) |
| **Reporting** | `results/acceptance/latency_{date}.json` with histogram plot |

### 9.2 Criterion 2: Forgetting Rate (< 5% after 100 sequential tasks)

| Field | Value |
| :--- | :--- |
| **Test ID** | AT-2 |
| **v3.0 Ref** | §1.3 criterion 2 |
| **Test command** | `python -m phca.benchmarks.runner --level=4 --output=results/acceptance/forgetting.json` |
| **Environment** | Same as AT-1 (CPU only) |
| **Data collection** | Per-task accuracy tracked in SQLite (`logs/benchmark_level_4.db`). Δ_perf computed for each task pair. |
| **Pass threshold** | For every task k, accuracy after training tasks k..100 ≥ accuracy_at_completion(k) × 0.95 |
| **Fail threshold** | Any task has accuracy drop > 5% |
| **Automation** | 🔄 Semi-automated (requires ~7 days of compute; triggered manually) |
| **Reporting** | `results/acceptance/forgetting_{date}.json` with per-task accuracy matrix |

### 9.3 Criterion 3: Goal Autonomy (≥ 1 novel goal per 100 cycles)

| Field | Value |
| :--- | :--- |
| **Test ID** | AT-3 |
| **v3.0 Ref** | §1.3 criterion 3 |
| **Test command** | `python -m phca.benchmarks.runner --level=3 --output=results/acceptance/goals.json` |
| **Environment** | Empty grid-world (10×10, no obstacles, no external goals) |
| **Data collection** | MDIM goal log (every goal generated: drive type, target state, cycle number). Novelty: state not visited in previous 500 cycles |
| **Pass threshold** | ≥ 1 novel goal per 100 cycles averaged over 5000 cycles |
| **Fail threshold** | < 1 novel goal per 200 cycles |
| **Automation** | ✅ Automated (runs in ~5 hours) |
| **Reporting** | `results/acceptance/goals_{date}.json` with goal novelty timeline |

### 9.4 Criterion 4: Criticality Maintenance (Φ ∈ [0.9Φc, 1.1Φc] for ≥ 90% cycles)

| Field | Value |
| :--- | :--- |
| **Test ID** | AT-4 |
| **v3.0 Ref** | §1.3 criterion 4 |
| **Test command** | `python -m phca.benchmarks.runner --level=3 --output=results/acceptance/criticality.json` (Level 3 provides diverse enough dynamics for Φ estimation) |
| **Environment** | Full system on grid-world with MDIM active |
| **Data collection** | Φ logged every cycle via `regulation/observables.py`. Sliding window Φ_critical computed every 100 cycles |
| **Pass threshold** | ≥ 90% of cycles have Φ ∈ [0.9Φ_critical, 1.1Φ_critical] over 5000 cycles |
| **Fail threshold** | < 70% |
| **Automation** | ✅ Automated |
| **Reporting** | `results/acceptance/criticality_{date}.json` with Φ timeline + histogram |

### 9.5 Criterion 5: Failure Recovery (≥ 80% within 10 cycles)

| Field | Value |
| :--- | :--- |
| **Test ID** | AT-5 |
| **v3.0 Ref** | §1.3 criterion 5 |
| **Test command** | `python tests/test_acceptance.py::test_failure_recovery -v` |
| **Environment** | Full system with Failure Detection & Recovery Matrix active |
| **Data collection** | For each of the 30+ failure modes: inject failure at cycle t, observe recovery cycle t+r. Failed if r > 10 or recovery never achieved |
| **Pass threshold** | ≥ 80% of failure modes recover within 10 cycles |
| **Fail threshold** | < 60% |
| **Automation** | ✅ Automated (runs in ~2 hours for all modes) |
| **Reporting** | `results/acceptance/recovery_{date}.json` with per-mode recovery time matrix |

---

## 10. BENCHMARK SUITE IMPLEMENTATION PLAN

### 10.1 Architecture

```
python/benchmarks/
├── __init__.py
├── runner.py              # Main CLI: python -m phca.benchmarks.runner --level=0..5 --output=results.json
├── level_0.py             # Stationary prediction (pendulum)
├── level_1.py             # Reactive control (inverted pendulum)
├── level_2.py             # Goal pursuit (grid-world navigation)
├── level_3.py             # Self-motivated exploration (empty grid-world)
├── level_4.py             # Continual learning (100 task sequence)
├── level_5.py             # Multi-agent coordination (2 agents)
├── metrics.py             # Φ-IQ computation, per-dimension metrics
└── report.py              # JSON schema + visualization generation
```

### 10.2 Runner CLI

```python
# python -m phca.benchmarks.runner --help
Usage: python -m phca.benchmarks.runner [OPTIONS]

Options:
  --level INTEGER     Benchmark level(s) 0-5 (default: all). Repeatable.
  --output TEXT       Output file path (default: results/benchmark_{level}_{timestamp}.json)
  --seeds TEXT        Comma-separated list of random seeds (default: 42,43,44,45,46)
  --device TEXT       Device to run on (cpu, cuda) (default: cpu)
  --profile           Enable profiling (default: off)
  --max-cycles INT    Maximum cycles per benchmark (default: level-specific)
  --help              Show this message and exit.

Examples:
  python -m phca.benchmarks.runner --level=0 --output=results/level_0.json
  python -m phca.benchmarks.runner --level=0 --level=1 --level=2 --seeds=42,43
  python -m phca.benchmarks.runner --level=4  # Takes ~7 days
```

### 10.3 Metric Computation (metrics.py)

```python
@dataclass
class BenchmarkResult:
    level: int
    seed: int
    metrics: dict[str, float]          # per-dimension metrics
    phi_iq: float                      # composite Φ-IQ
    raw_data: dict                      # raw timestream data (optional, large)
    config: dict                       # agent configuration snapshot

def compute_phi_iq(results: list[BenchmarkResult]) -> dict:
    """Compute Φ-IQ composite from per-level results."""
    dimensions = {
        "prediction_accuracy":  results[0].metrics["mse"],
        "adaptation_speed":     results[1].metrics["settling_time"],
        "goal_complexity":      results[3].metrics["novel_goals_per_100"],
        "transfer_efficiency":  results[4].metrics["forward_transfer"],
        "resource_efficiency":  results[2].metrics["path_efficiency"],
        "failure_rate":         results[5].metrics["failure_rate"],
    }
    weights = {"w1": 1/6, "w2": 1/6, "w3": 1/6, "w4": 1/6, "w5": 1/6, "w6": 1/6}
    phi_iq = sum(dimensions[k] * weights[f"w{i+1}"] for i, k in enumerate(dimensions) if k != "failure_rate")
    phi_iq -= dimensions["failure_rate"] * weights["w6"]
    return {"dimensions": dimensions, "phi_iq": phi_iq, "weights": weights}
```

### 10.4 Results JSON Schema

```json
{
  "benchmark": {
    "level": 4,
    "seed": 42,
    "started_at": "2026-10-15T09:00:00Z",
    "completed_at": "2026-10-22T16:30:00Z"
  },
  "agent_config": {
    "active_components": ["ASI", "G'", "PE", "TSPL-P", "TSPL-E", "TSPL-S", "MDIM", "CR", "ATTN"],
    "hyperparameters": {
      "alpha_P": 0.05,
      "alpha_E": 0.005,
      "lambda_S": 1.0,
      "K_p": 1.0
    }
  },
  "metrics": {
    "per_task_accuracy": [0.95, 0.93, ..., 0.94],
    "forgetting_rate": 0.023,
    "forward_transfer": 0.12,
    "backward_transfer": -0.01
  },
  "phi_iq": {
    "composite": 0.74,
    "dimensions": {
      "prediction_accuracy": 0.82,
      "adaptation_speed": 0.65,
      "goal_complexity": 0.71,
      "transfer_efficiency": 0.78,
      "resource_efficiency": 0.69,
      "failure_rate": 0.08
    },
    "weights": {
      "w1": 0.167,
      "w2": 0.167,
      "w3": 0.167,
      "w4": 0.167,
      "w5": 0.167,
      "w6": 0.167
    }
  },
  "system_info": {
    "platform": "Linux-6.8-x86_64",
    "cpu": "Intel i7-12700H",
    "ram_gb": 31.2,
    "python_version": "3.11.5",
    "pytorch_version": "2.2.0"
  }
}
```

### 10.5 Visualization

Each benchmark run automatically generates:

1. `{level}_metrics_{date}.png` — Per-dimension metric bar chart
2. `{level}_timeline_{date}.png` — Metric over time (if raw data captured)
3. `report_{date}.png` — Φ-IQ spider/radar chart (all dimensions)
4. `report_{date}.json` — Full results in JSON format

These are generated by `report.py` using `matplotlib`:

```python
def generate_report(results: list[BenchmarkResult], output_dir: str) -> str:
    """Generate all visualizations and return path to report directory."""
    # Spider chart of Φ-IQ dimensions
    # Bar chart of per-level metrics
    # Timeline plots for metrics over time
```

---

## 11. SCHEDULING & DEPENDENCY GRID

### 11.1 Week-by-Week Schedule

| Week | Phase | Tickets | Focus Area | Milestone | Risk Check |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1** | 3.1 | 001, 002 | Scaffolding + grid-world | Repo setup done | — |
| **2** | 3.1 | 003, 004 | Data types + ASI sanitizer | Sensors flowing | — |
| **3** | 3.1 | 005, 006 | M1/M2 + RBTA (start) | WM working | — |
| **4** | 3.1 | 006, 007 | RBTA (finish) + G' nodes | **M1.1: ASI→WM pipeline** | R7 (G' perf) |
| **5** | 3.1 | 007, 008 | G' nodes + G' inference | Graph creation | R7 (inference) |
| **6** | 3.1 | 008 | G' inference (finish) | G' forward prediction | R7 (profile) |
| **7** | 3.1 | 009 | Prediction Engine + PEU | **M1.2: G' prediction** | — |
| **8** | 3.1 | 010 | P-Stream TSPL | Learning active | R2 (P-Stream) |
| **9** | 3.1 | 010, 011 | P-Stream + cycle orchestrator | Cycle integration | — |
| **10** | 3.1 | 011 | Cycle orchestrator | **M1.3: P-Stream navigates** | R1 (latency) |
| **11** | 3.1 | 011, 012 | Cycle + acceptance tests | Integration tests | R1 (latency) |
| **12** | 3.1 | 012 | Acceptance tests + gate | **GATE 1: Phase 3.1 done** | — |
| **13** | 3.2 | 001, 002 | ASI v1 + M3 MVCC | — | — |
| **14** | 3.2 | 002, 003 | M3 (finish) + E-Stream start | M3 operational | — |
| **15** | 3.2 | 003, 004 | E-Stream (GEM) + S-Stream start | — | R3 (Fisher) |
| **16** | 3.2 | 004, 005 | S-Stream (EWC) + Consolidation start | **M2.1: Anti-forgetting** | R4 (GEM stall) |
| **17** | 3.2 | 005, 006 | Consolidation + M4 | E→S transfer | R8 (MVCC) |
| **18** | 3.2 | 006, 007 | M4 + M5 | Full memory hierarchy | — |
| **19** | 3.2 | 008 | MDIM D1–D6 | Drives implemented | — |
| **20** | 3.2 | 009 | MDIM Pareto front | **M2.2: Goal generation** | R5 (Pareto) |
| **21** | 3.2 | 010 | Criticality Regulator PID | PID working | R6 (windup) |
| **22** | 3.2 | 011 | Orthogonality constraint | CR complete | R6 (oscillation) |
| **23** | 3.2 | 012 | Precision-Weighted Attention | Attention active | — |
| **24** | 3.2 | 013 | HPM Grammar Runtime (start) | **M2.3: HPM parsing** | — |
| **25** | 3.2 | 013 | HPM Grammar (finish) | HPM verification | — |
| **26** | 3.2 | 014 | VSA (conditional) | VSA evaluation | R10 (VSA) |
| **27** | 3.2 | 015, 016 | Dual ensemble + integration | Full cycle tests | — |
| **28** | 3.2 | 016 | Integration + acceptance | **GATE 2: Phase 3.2 done** | R5, R6, R8 |
| **29** | 3.3 | 001 | Failure Detection Matrix | **M3.1: Detection active** | — |
| **30** | 3.3 | 001, 002 | Recovery + Φ-IQ Suite | Suite runner | — |
| **31** | 3.3 | 003 | Level 0 benchmark | MSE < 0.1 | — |
| **32** | 3.3 | 004 | Level 1 benchmark | Stability < 100 steps | — |
| **33** | 3.3 | 005 | Level 2 benchmark | **M3.2: Goal pursuit** | — |
| **34** | 3.3 | 006 | Level 3 benchmark | Novel goals | — |
| **35** | 3.3 | 007 | Level 4 benchmark (starts) | Continual learning | R2 (forgetting) |
| **36** | 3.3 | 007 | Level 4 (continues, ~7 days) | — | — |
| **37** | 3.3 | 007, 008 | Level 4 done + Level 5 start | **M3.3: Continual done** | R7 (multi-agent) |
| **38** | 3.3 | 008 | Level 5 (continues, ~10 days) | — | R7 |
| **39** | 3.3 | 008, 009 | Level 5 done + report | **M3.4: Benchmarks done** | — |
| **40** | 3.3 | 009 | Report generation | **GATE 3: Phase 3.3 done** | R8 (Φ-IQ) |
| **41** | 3.4 | 001 | Parameter sweep infrastructure | Sweep harness | — |
| **42** | 3.4 | 002 | Parameter sweeps (execution) | — | — |
| **43** | 3.4 | 002, 003 | Sweeps + bottleneck identification | **M4.1: Best config** | — |
| **44** | 3.4 | 003 | Bottleneck optimization | — | — |
| **45** | 3.4 | 004 | VSA permanence decision | **M4.2: VSA decision** | — |
| **46** | 3.4 | 005 | Component pruning analysis | — | — |
| **47** | 3.4 | 006 | Re-run full benchmark suite | Final numbers | — |
| **48** | 3.4 | 006 | Re-run (continues, ~10 days) | — | — |
| **49** | 3.4 | 007 | Documentation | API reference | — |
| **50** | 3.4 | 007, 008 | Documentation + final report | **M4.3: Final report** | — |
| **51** | 3.4 | 008 | Final report completion | **M4.4: Docs complete** | — |
| **52** | — | — | Buffer / contingency / handoff | **GATE 4: Project complete** | — |

### 11.2 Critical Path Visualization

```
W1    W4    W8    W12   W16   W20   W24   W28   W32   W36   W40   W44   W48   W52
[001]─[004]─[008]─[011]─[012]
  │     │     │     │     └── GATE 1 (Phase 3.1 complete)
  │     │     │     └── Cognitive cycle working
  │     │     └── G' inference (GATE 1.2)
  │     └── ASI+M1/M2 pipeline (GATE 1.1)
  └── Scaffolding

                    [013]─[016]
                      │     └── GATE 2 (Phase 3.2 complete)
                      └── HPM Grammar critical

                              [001]─[002]─[003]─[004]─[005]─[006]─[007]─[008]─[009]
                                │     │     │     │     │     │     │     │     └── GATE 3
                                │     │     │     │     │     │     │     └── L5 done
                                │     │     │     │     │     │     └── L4 done
                                │     │     │     │     │     └── L3 done
                                │     │     │     │     └── L2 done (GATE 3.2)
                                │     │     │     └── L1 done
                                │     │     └── L0 done (GATE 3.1)
                                │     └── Suite runner
                                └── Failure detection

                                              [002]─[003]─[004]─[006]─[007]─[008]
                                                │     │     │     │     │     └── GATE 4
                                                │     │     │     │     └── Final docs
                                                │     │     │     └── Final bench
                                                │     │     └── VSA decision (GATE 4.2)
                                                │     └── Optimization
                                                └── Parameter sweep
```

**Slack analysis:**
- RBTA (PHCA-3.1-006): has 45 days slack (week 3 → week 10 deadline). Can slip 6 weeks.
- HPM Grammar (PHCA-3.2-013): has 30 days slack (week 24 → week 28 deadline). Can slip 4 weeks.
- VSA (PHCA-3.2-014): has 50 days slack (week 26 → week 40 VSA decision point). Can be cut entirely.

**Parallelization opportunities:**
- Phase 3.1 weeks 3–6: RBTA (Rust) and G' (Python) can be built in parallel by 2 engineers.
- Phase 3.2 weeks 13–16: M3/M4/M5 (SQLite/RocksDB) and TSPL (Python) can be built in parallel.
- Phase 3.2 weeks 19–23: MDIM (Python) and Criticality Regulator (Python) can be built in parallel.

---

## 12. RISK MONITORING & CONTINGENCY TRIGGERS

### 12.1 Operational Monitoring Plan

| Risk ID | Risk | Monitoring Metric | Alert Threshold | Escalation Path | Contingency Action | Expected Recovery |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **T1** | RBTA Constraint Enforcer bottleneck | Per-cycle enforcement time (μs) | > 200μs avg over 10 cycles | Engineer → Lead | Disable entropy floor checking; batch constraint checks | 2 hours |
| **T2** | TSPL catastrophic forgetting | Per-task accuracy, Δ_perf (logged daily) | Δ_perf > 5% decline on any task | Researcher → Lead | Activate replay buffer; decrease α_E by 50% | 1 day |
| **T3** | MDIM drive thrashing | Drive deficit variance across D1/D3/D5 | Variance > 2σ over 100 cycles | Engineer → Researcher | Widen Pareto front thresholds; disable meta-stable | 4 hours |
| **T4** | CR PID oscillation | Covariance(T, η, α) over W=100 window | max(Σ) > Σ_max for 50+ consecutive cycles | Engineer → Lead | Freeze slowest parameter (T); clamp integral | 1 hour |
| **T5** | VSA performance degrades | Analogical task accuracy | VSA accuracy < G' similarity accuracy | Researcher → Lead | Cut VSA permanently (Phase 3.2 → 3.3 transition) | 1 day (decision) |
| **T6** | Multi-agent FLP failure | Consensus timeout count | > 10% cycles with consensus failure | Engineer → Researcher | Switch agents to autonomy fallback | 30 min |
| **T7** | G' inference too slow | Per-cycle G' inference time (ms) | > 50ms avg over 10 cycles | Engineer → Lead | Switch inference from sampling to exact junction tree | 4 hours |
| **T8** | Consolidation race condition | M3 version counter integrity | Version counter mismatch detected | Engineer → Lead | Rollback to last consistent snapshot; skip consolidation cycle | 2 hours |
| **T9** | ASI sanitization misses edge case | Undetected NaN count | > 0 NaN values in clean_vector | Engineer → Lead | Log → add case to sanitizer → patch → re-run tests | 1 hour |
| **X1** | Edge-of-chaos not optimal | Phase 3.4 comparison (CR enabled vs. disabled) | CR-disabled achieves higher Φ-IQ (p < 0.05) | Researcher | Document CR as unnecessary; disable by default | — |
| **X2** | Multi-memory not transferable | Single-memory baseline vs. full TSPL comparison | Single-memory beats TSPL on Φ-IQ | Researcher | Simplify memory hierarchy; merge E+S streams | 2 weeks |
| **X3** | Φ-IQ not valid metric | Correlation with held-out human evaluation | r < 0.3 correlation | Researcher | Report per-dimension scores; de-emphasize composite | — |
| **X4** | D6 Empowerment useless | D6 disabled vs. enabled comparison | No significant Φ-IQ change | Researcher | Remove D6; keep D1–D5 | 1 day |

### 12.2 Alert Severity Colors

| Severity | Color | Response Time | Notification |
| :--- | :--- | :--- | :--- |
| **Critical** | 🔴 | < 1 hour | Slack @channel + phone call to Lead |
| **High** | 🟠 | < 4 hours | Slack @lead + GitHub issue |
| **Medium** | 🟡 | < 24 hours | GitHub issue |
| **Low** | 🟢 | < 1 week | Decision log entry |

### 12.3 Dashboard (Suggested)

The following metrics should be visible on a team dashboard (Grafana or similar):

1. **Cycle latency** — P50, P95, P99 over last 100 cycles (🔴 if P95 > 500ms)
2. **RBTA violation count** — per module, per bound type (🔴 if any violation)
3. **MDIM goal rate** — goals per 100 cycles (🟡 if < 0.5 per 100)
4. **Φ criticality** — % of cycles within bounds (🔴 if < 80%)
5. **Prediction error** — moving average MSE (🟢 if stable, 🟡 if trending up)
6. **TSPL forgetting** — Δ_perf per active task (🔴 if > 5%)
7. **CI test pass rate** — % of passing tests on main (🔴 if < 95%)

---

## 13. PHASE TRANSITION GATES

### 13.1 Gate 1: Phase 3.1 → 3.2 (Week 12)

**Conditions to proceed:**
- [ ] All Phase 3.1 tickets are closed (PHCA-3.1-001 through 012)
- [ ] `test_phase_3_1.py` passes (all integration tests IT-3.1-1 through 5)
- [ ] Cycle latency: median < 500ms, P95 < 1000ms (AT-1)
- [ ] P-Stream learns navigation: grid-world success rate > 80% after 1000 episodes
- [ ] RBTA enforcer: detects artificial timeout violation (IT-3.1-4)
- [ ] Code coverage: ≥ 70% on Phase 3.1 components
- [ ] All Rust code passes `cargo clippy` with no warnings
- [ ] All Python code passes `ruff` with no errors
- [ ] Decision log is up to date

**Evidence required:**
- PR merge history (all Phase 3.1 PRs merged)
- `make test-all` output log from CI
- Latency profile (`python scripts/profile_cycle.py --cycles=1000`) results JSON
- Grid-world learning curve (generated by P-Stream training script)
- Decision log (`DECISIONS.md`)

**Reviewers:** ML/Systems Lead + Researcher

**Approval:** Documented as a GitHub issue comment on `PHCA-3.1-012`. If not approved:
- **Minor issues (≤ 3 bugs):** File bugs, approve conditionally
- **Major issues (> 3 bugs or latency > 750ms):** Hold gate; allocate 2 additional weeks

**Failure action:** Phase 3.1 extension by 2 weeks (weeks 13–14). Phase 3.2 shifted to weeks 15–30. Contingency: cut RBTA entropy floor checks to save 50μs.

### 13.2 Gate 2: Phase 3.2 → 3.3 (Week 28)

**Conditions to proceed:**
- [ ] All Phase 3.2 tickets closed (PHCA-3.2-001 through 016)
- [ ] TSPL forgetting rate < 5% on 2-task sequence (IT-3.2-1)
- [ ] MDIM generates ≥ 1 goal per 100 cycles in empty environment
- [ ] CR maintains Φ within [0.9Φc, 1.1Φc] for ≥ 85% of cycles (approach target of 90%)
- [ ] HPM grammar rejects type-invalid composition (IT-3.2-6)
- [ ] Full 21-step cognitive cycle completes in < 500ms (Phase 3.2 configuration)
- [ ] All 5 acceptance tests (AT-1 through AT-5) have baseline measurements (may not pass yet)
- [ ] VSA decision recorded (included or rationale for exclusion)

**Reviewers:** ML/Systems Lead + Researcher + External Auditor (for spec compliance)

### 13.3 Gate 3: Phase 3.3 → 3.4 (Week 40)

**Conditions to proceed:**
- [ ] All Phase 3.3 tickets closed (PHCA-3.3-001 through 009)
- [ ] AT-1: Cycle latency < 500ms ✅
- [ ] AT-2: Forgetting rate < 5% ✅
- [ ] AT-3: Goal autonomy ≥ 1 per 100 ✅
- [ ] AT-4: Criticality ≥ 90% ✅
- [ ] AT-5: Failure recovery ≥ 80% ✅
- [ ] Φ-IQ increases monotonically across Levels 0–5
- [ ] 33 v3.0 verification checks passed (Appendix A of Blueprint)
- [ ] Comprehensive benchmark report generated

**Reviewers:** ML/Systems Lead + Researcher + External Auditor (mandatory)

### 13.4 Gate 4: Phase 3.4 Complete (Week 52)

**Conditions to close:**
- [ ] All Phase 3.4 tickets closed (PHCA-3.4-001 through 008)
- [ ] Parameter sweep complete: best configuration documented
- [ ] Bottleneck optimization: cycle latency < 500ms on target hardware
- [ ] VSA permanence decision: final (keep or cut)
- [ ] Component pruning: decision log entry for each pruned component
- [ ] Final benchmark report: all 5 AT criteria pass
- [ ] Full documentation: API reference, architecture diagram, deployment guide
- [ ] Decision log: complete record of all design decisions

**Deliverable:** `PHCA v3.0 — Validated Architecture Specification` final report

---

## 14. GLOSSARY OF KEY TERMS

| Term | Definition (Engineer-Friendly) |
| :--- | :--- |
| **ASI** | Abstract Sensorimotor Interface. The thin wrapper between the cognitive core and the environment. Receives sensor vectors, sanitizes them, and passes them to working memory. |
| **RBTA** | Resource-Bounded Temporal Automata. A formal way of saying "every module has a max time, memory, and energy budget per cycle, and we check those budgets at runtime." |
| **G'** | The probabilistic world model. A Bayesian network (graph of random variables with causal edges) that can predict what will happen next given the current state and action. |
| **VSA (V)** | Vector Symbolic Architecture. A way of representing concepts as high-dimensional vectors (1000+ dimensions) that can be combined (bound), added (bundled), and permuted. Like word embeddings on steroids. **Excluded from Phase 3.1.** |
| **TSPL** | Three-Stream Predictive Learning. The learning system with 3 parallel "streams" — procedural (skills, learns fast), episodic (events, medium), semantic (facts, slow). All learn from prediction error. |
| **EWC** | Elastic Weight Consolidation. A trick to prevent forgetting: identify important parameters (via Fisher Information Matrix), penalize changing them. Used by DeepMind in 2017. |
| **GEM** | Gradient Episodic Memory. Another anti-forgetting trick: constrain gradient updates so they don't increase the loss on past tasks. Used by Facebook AI in 2017. |
| **MDIM** | Multi-Drive Intrinsic Motivation. 6 "drives" (like hunger/thirst in animals) that generate goals when they deviate from their set point. D1=prediction error, D2=criticality, D3=learning progress, D4=curiosity, D5=energy efficiency, D6=empowerment. |
| **Pareto Front** | A set of tradeoff configurations where you can't improve one thing without making another worse. Used to prevent MDIM from oscillating between conflicting drives. |
| **Criticality Regulator** | A PID controller (like a thermostat) that keeps the system at the "edge of chaos" — the sweet spot between too ordered (boring) and too chaotic (random). |
| **HPM** | Hierarchical Predictive Module. A typed grammar for composing cognitive modules. Like a programming language for cognition — you write `SEQUENCE(sensor, predictor)` and the system verifies it makes sense. |
| **Φ-IQ** | Phi-Intelligence Quotient. A single number claiming to measure how "intelligent" the system is across 6 dimensions (prediction, adaptation, goals, transfer, efficiency, failure rate). |
| **MVCC** | Multi-Version Concurrency Control. A database technique (used by PostgreSQL) that lets readers see a consistent "snapshot" while writers create a new version. Used in the episodic memory. |
| **RBTA Constraint Enforcer** | A Rust module that checks, every cycle, whether each module stayed within its time/memory/energy budget. If not, it raises a violation. |
| **Cognitive Cycle** | The main loop. 21 steps (Phase 3.2+) from sensor input to action output. Target: < 500ms for the whole loop. |
| **WM | M2** | Working Memory. Holds 7±2 "chunks" (like the human brain's working memory limit). The current focus of attention. |
| **Set Point** | The "ideal" value for a homeostatic drive. Like a thermostat set to 72°F. MDIM drives try to return to their set point (not maximize/minimize indefinitely). |

---

## 15. EMERGENCY CONTACTS & ESCALATION

### 15.1 Escalation Paths

| Issue | First Contact | Second Contact | Third Contact |
| :--- | :--- | :--- | :--- |
| **Technical implementation question** | Pair programming / team Slack | ML/Systems Lead | Architect (email) |
| **v3.0 spec interpretation conflict** | Researcher | External Auditor | — |
| **Performance regression (latency > 750ms)** | Backend/Performance Engineer | ML/Systems Lead | Architect |
| **Test failure in CI (blocking merge)** | Ticket author | Engineer on rotation | — |
| **Security vulnerability** | Lead → all team | Security contact | — |
| **Hardware failure** | DevOps/Infra Engineer | ML/Systems Lead | — |
| **Team conflict / blocker** | ML/Systems Lead | Project Manager | — |
| **External audit question** | Researcher | Lead | External Auditor |

### 15.2 Communication Channels

| Channel | Purpose | Who Has Access |
| :--- | :--- | :--- |
| **Slack #phca-team** | Daily chat, standup updates, quick questions | All team |
| **Slack #phca-alerts** | Automated CI failures, benchmark alerts, threshold violations | All team (read-only for non-engineers) |
| **GitHub Issues** | Bug tracking, feature requests, decision tracking | All team + stakeholders |
| **GitHub PRs** | Code review | All team |
| **Shared calendar** | Standups (daily 09:00), spec reviews (weekly Wed 14:00), integration tests (biweekly Fri 14:00) | All team |

### 15.3 On-Call Rotation

| Week | Primary (Incidents) | Secondary (PR Review) |
| :--- | :--- | :--- |
| 1–4 | ML/Systems Lead | Backend Engineer |
| 5–8 | Backend Engineer | ML Engineer |
| 9–12 | ML Engineer | Researcher |
| ... | Rotate every 4 weeks | |

During on-call, the primary responds to ALL alerts within the response time specified in §12.2. The secondary handles PR review so the primary can focus on incidents.

---

*End of Document — PHCA v3.0 Engineer's Playbook: Day-Zero Execution Plan*

**Status:** FINAL — Ready for Sprint 0  
**Next action:** Complete tickets PHCA-3.1-001 (scaffolding) and PHCA-3.1-002 (grid-world) in Week 1  
**Document maintainer:** ML/Systems Lead  
**Last updated:** June 29, 2026
