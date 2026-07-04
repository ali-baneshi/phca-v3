# PHCA v3.0 — Implementation Status Matrix

Maps **whitepaper success criteria** and **blueprint components** to current
code, gate scripts, and measured outcomes. Last aligned with STATUS.md:
2026-07-04.

Legend: **Implemented** | **Partial** | **Measured** | **Not implemented** | **Stub**

---

## Whitepaper §1.3 Success Criteria

| Criterion | Target | Status | Gate / evidence |
|---|---|---|---|
| Cycle latency | < 500 ms | **Measured PASS** | Φ-IQ pass criteria; mean ~10–17 ms MLP |
| Forgetting rate | < 5% after 100 sequential tasks | **Not implemented** | No 100-task benchmark exists |
| Goal autonomy | ≥ 1 novel goal / 100 cycles | **Partial** | L3 gates drive diversity > 0.1, not novel-goal rate |
| Criticality maintenance | Φ ∈ [0.9Φc, 1.1Φc] for ≥ 90% cycles | **Not implemented** | APC regulates error volatility, not integrated Φ |
| Failure recovery | ≥ 80% mitigated within 10 cycles | **Not implemented** | `phca/resilience/` is stub only |

---

## Verified Invariants (A1–A5)

| Invariant | Code enforcement | Falsification | Notes |
|---|---|---|---|
| **A1** Resource boundedness | `rbta_enforcer.py` | `assumption_validation.py --ci` | **Measured PASS** |
| **A2** Temporal causality | Pipeline order in `cycle.py` | `assumption_validation.py --ci` (A2 monitor) | **Measured PASS** |
| **A3** Incomplete knowledge | Entropy floor in regulation | `assumption_validation.py --ci` | **Measured PASS** |
| **A4** Prediction as primary | G′ predict every cycle | A4 test on continuous MPC | **Partial** — discrete GridWorld uses hybrid geometry+confidence selector |
| **A5** Feedback-driven adaptation | PEU → TSPL → G′.learn | Weight freeze/active test | **Measured PASS** |

See [docs/phca_causal_evidence.md](docs/phca_causal_evidence.md) for behavioral
evidence separate from invariant tests.

---

## Blueprint Components

| Component | Spec source | Code location | Status |
|---|---|---|---|
| ASI sanitizer | v3 patch | `phca/asi/sanitizer.py` | **Implemented** |
| Grounding adapter (L0/L2) | v3 patch | — | **Not implemented** (ASI always level 1) |
| RBTA enforcer | Whitepaper §2.1 | `phca/regulation/rbta_enforcer.py` | **Implemented** (Python; Rust removed D-084) |
| G′ world model | Whitepaper §2.2 | `phca/world_model/` | **Implemented** (Gaussian / graph / MLP) |
| V (VSA ensemble) | Whitepaper §2.2 | — | **Not implemented** |
| S (script library) | Whitepaper §2.2 | — | **Not implemented** |
| Prediction engine | Blueprint | `phca/prediction/engine.py` | **Implemented** |
| PEU | Blueprint | `phca/prediction/error_unit.py` | **Implemented** |
| TSPL P-Stream | Blueprint | `phca/learning/tspl.py` | **Implemented** |
| TSPL E/S streams + EWC/GEM | Blueprint | — | **Removed** Phase 3.3 (D-020) |
| MDIM (6 drives) | Whitepaper §3 | `phca/motivation/mdim.py` | **Implemented** |
| APC (adaptive PID) | Blueprint | `phca/regulation/pid_controller.py` | **Implemented** (error volatility, not SOC Φ) |
| Attention | Blueprint | `phca/attention/attention.py` | **Implemented** |
| HPM runtime | Blueprint | `phca/hpm/parser.py` | **Partial** — `compute_bounds()` only |
| M1 sensory | Blueprint | `phca/memory/m1_sensory.py` | **Implemented** |
| M2 working | Blueprint | `phca/memory/m2_working.py` | **Implemented** |
| M3 episodic | Blueprint | `phca/memory/m3_episodic.py` | **Implemented** (SQLite, VACUUM D-108) |
| M4 semantic | Blueprint | `phca/consolidation/scheduler.py` | **Partial** — statistical pattern facts |
| M5 procedural | Blueprint | — | **Not implemented** |
| M6 meta-memory | Whitepaper | — | **Not implemented** |
| Failure matrix A–F | Blueprint | `phca/resilience/` | **Stub** |
| GridWorld | Phase 3.1 | `phca/environments/grid_world.py` | **Implemented** |
| MuJoCo (3 envs) | Phase 4–7 | `phca/environments/mujoco_env.py` | **Implemented** |
| Cognitive Observatory | Phase 7–12 | `phca/monitoring/` | **Complete** (schema, replay, report parity, scrub perf, multi-session compare) |
| Φ-IQ L0–L3 | Blueprint | `scripts/benchmark.py` | **Implemented + measured** |
| Φ-IQ L4–L5 | Blueprint | — | **Not implemented** |

---

## Evaluation Gates

| Gate | Script | CI? | Current status (2026-07-04) |
|---|---|---|---|
| Φ-IQ regression (L0 quick) | `check_benchmark_gate.py` | **Yes** | PASS |
| Φ-IQ full (MLP L0–L3) | `scripts/benchmark.py --use-mlp` | No (nightly) | PASS (0.7403) |
| MuJoCo smoke | `check_benchmark_gate.py --mujoco` | No (nightly) | PASS |
| Causal behavior L1–L3 | `phca_causal_eval.py --gate` | Smoke only | PASS (Gaussian default; use `--use-mlp` for deployment mode) |
| Assumption validation | `assumption_validation.py --ci` | No (nightly) | **5/5 PASS** (A1–A5 incl. A2) |
| OOD calibration | `ood_calibration.py` | No (nightly) | Monotonic PASS |
| Nightly stress | `nightly_stress.py` | Scheduled workflow | Fill-phase PASS @ 1k; post-cap @ 10k |
| Observatory integrity | `phca_replay.py --check` | Unit tests | PASS |

---

## Action Selection Modes

| Environment | Mode | A4 "prediction-primary"? |
|---|---|---|
| GridWorld discrete | Manhattan + confidence + MDIM blend | **Partial** |
| Cartpole | 3-bin discrete | **Partial** |
| Pendulum continuous | MPC: sample K actions, predict, pick best ŝ′ | **Yes** (A4 measured) |
| Reacher continuous | Same MPC path, dim 2 | **Yes** |

Details: [docs/action_selection.md](docs/action_selection.md)

---

## Open Backlog (from STATUS.md)

| ID | Item | Phase |
|---|---|---|
| P10-1 | Full offline report ↔ dashboard parity | **Done** (Phase 10) |
| P11-1 | 3000+ cycle scrub without severe lag | **Done** (Phase 11) |
| AD-3 | Unify benchmark entry points | 4.1 (deprecated runner) |
| AD-4 | M5 procedural memory | 4.3 |
| SC-1 | Grounding adapter | Deferred |
