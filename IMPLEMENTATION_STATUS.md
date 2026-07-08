# PHCA v3.0 — Implementation Status Matrix

Maps **whitepaper success criteria** and **blueprint components** to current
code, gate scripts, and measured outcomes. Last aligned with STATUS.md:
2026-07-06.

**Maturation audits (2026-07-07):** [docs/maturity_audit_2026-07-07.md](docs/maturity_audit_2026-07-07.md),
[docs/static_audit_2026-07-07.md](docs/static_audit_2026-07-07.md).

Legend: **Implemented** | **Partial** | **Measured** | **Not implemented** | **Stub** | ✅ **Done**

---

## Whitepaper §1.3 Success Criteria

| Criterion | Target | Status | Gate / evidence |
|---|---|---|---|
| Cycle latency | < 500 ms | **Measured PASS** | Φ-IQ pass criteria; mean ~10–17 ms MLP |
| Forgetting rate | < 5% after 100 sequential tasks | **Partial** | Level-4-lite: `scripts/benchmark_level4.py` (10–20 GridWorld tasks, P-Stream + replay); full 100-task AT-2 not CI-gated |
| Goal autonomy | ≥ 1 novel goal / 100 cycles | **Measured PASS** | MDIM tracks `_seen_goal_signatures` → `novel_goal_rate` in snapshot; L3 gates `novel_goal_rate > 0.01` (✅ Done 2026-07-08) |
| Criticality maintenance | Φ ∈ [0.9Φc, 1.1Φc] for ≥ 90% cycles | **Not implemented** | APC regulates error volatility, not integrated Φ |
| Failure recovery | ≥ 80% mitigated within 10 cycles | **Partial (MVP)** | `phca/resilience/` B1/B4/B5/C1/F5 + `scripts/benchmark_recovery.py` |

---

## Verified Invariants (A1–A5)

| Invariant | Code enforcement | Falsification | Notes |
|---|---|---|---|
| **A1** Resource boundedness | `rbta_enforcer.py` | `assumption_validation.py --ci` | **Measured PASS** |
| **A2** Temporal causality | Pipeline order in `cycle.py` | `assumption_validation.py --ci` (A2 monitor) | **Measured PASS** |
| **A3** Incomplete knowledge | `_epistemic_entropy()` → RBTA entropy_floor (MC-dropout mutual info) | `assumption_validation.py --ci` | **Measured PASS** |
| **A4** Prediction + Spatial Heuristics as Hybrid Cognitive Map | G′ predict every cycle + spatial heuristics for discrete navigation | A4 test on continuous MPC + hybrid selector metrics | **Hybrid Cognitive Map** (D-136) — GridWorld: confidence-gated task_lock (τ=0.6) with Manhattan/BFS geometry + blended G′ fallback; continuous MPC remains prediction-primary (measured) |
| **A5** Feedback-driven adaptation | PEU → TSPL → G′.learn | Weight freeze/active test | **Measured PASS** |

See [docs/phca_causal_evidence.md](docs/phca_causal_evidence.md) for behavioral
evidence separate from invariant tests.

---

## Blueprint Components

| Component | Spec source | Code location | Status |
|---|---|---|---|
| ASI sanitizer | v3 patch | `phca/asi/sanitizer.py` | **Implemented** |
| Grounding adapter (L0/L2) | v3 patch | — | **Not implemented** (ASI always level 1); see NoiseInjector in `python/phca/asi/noise_injector.py` for noise-stress proxy |
| **NoiseInjector** | Resilience | `python/phca/asi/noise_injector.py` | **Implemented** — configurable Gaussian noise, decay, warmup; `scripts/benchmark_noise_closedloop.py` + `scripts/benchmark_noise_robustness.py` |
| **Hybrid ablation** | A4 analysis | `scripts/benchmark_hybrid_ablation.py` | **Implemented** — 30-seed Mann-Whitney ablation: Manhattan vs Prediction vs Hybrid; `experiments/hybrid_map_ablation.yaml` |
| RBTA enforcer | Whitepaper §2.1 | `phca/regulation/rbta_enforcer.py` | **Implemented** (Python; Rust removed D-084) |
| G′ world model | Whitepaper §2.2 | `phca/world_model/` | **Implemented** (Gaussian / graph / MLP) |
| V (VSA ensemble) | Whitepaper §2.2 | — | **Not implemented** |
| S (script library) | Whitepaper §2.2 | — | **Not implemented** |
| Prediction engine | Blueprint | `phca/prediction/engine.py` | **Implemented** |
| PEU | Blueprint | `phca/prediction/error_unit.py` | **Implemented** |
| TSPL P-Stream | Blueprint | `phca/learning/tspl.py` | **Implemented** |
| TSPL E/S streams + EWC/GEM | Blueprint | — | **Removed** Phase 3.3 (D-020) |
| MDIM (6 drives) | Whitepaper §3 | `phca/motivation/mdim.py` | **Implemented** — deficit/target normalization before softmax (✅ Done 2026-07-06) |
| APC (adaptive PID) | Blueprint | `phca/regulation/pid_controller.py` | **Implemented** (error volatility, not SOC Φ) |
| Attention | Blueprint | `phca/attention/attention.py` | **Implemented** |
| HPM runtime | Blueprint | `phca/hpm/parser.py` | **Partial** — `compute_bounds()` only |
| M1 sensory | Blueprint | `phca/memory/m1_sensory.py` | **Implemented** |
| M2 working | Blueprint | `phca/memory/m2_working.py` | **Implemented** |
| M3 episodic | Blueprint | `phca/memory/m3_episodic.py` | **Implemented** (SQLite, VACUUM D-108) |
| M4 semantic | Blueprint | `phca/consolidation/scheduler.py` | **Partial** — statistical pattern facts |
| M5 procedural | Blueprint | — | **Not implemented** |
| M6 meta-memory | Whitepaper | — | **Not implemented** |
| Failure matrix A–F | Blueprint | `phca/resilience/` | **Enhanced** — B1, B4, B5, C1, F5 detect + recover; E1 FallbackController with dual-signal (entropy + cascade) |
| GridWorld | Phase 3.1 | `phca/environments/grid_world.py` | **Implemented** |
| MuJoCo (3 envs) | Phase 4–7 | `phca/environments/mujoco_env.py` | **Implemented** |
| Cognitive Observatory | Phase 7–20 | `phca/monitoring/` | **Complete** (schema, replay, report, scrub, compare, anomalies, explain, API, supervisor, multi-agent, query, reproduce) |
| Φ-IQ L0–L3 | Blueprint | `scripts/benchmark.py` | **Implemented + measured** |
| Φ-IQ L4-lite (forgetting) | Blueprint | `scripts/benchmark_level4.py` | **Partial** — GridWorld continual, AT-2-lite gate |
| Φ-IQ L5 | Blueprint | — | **Not implemented** |

---

## Evaluation Gates

| Gate | Script | CI? | Current status (2026-07-06) |
|---|---|---|---|
| Φ-IQ regression (L0 quick) | `check_benchmark_gate.py` | **Yes** | PASS |
| Φ-IQ full (MLP L0–L3) | `scripts/benchmark.py --use-mlp` | No (nightly) | PASS (0.7317) |
| MuJoCo smoke | `check_benchmark_gate.py --mujoco` | No (nightly) | PASS |
| Causal behavior L1–L3 | `phca_causal_eval.py --gate` | Smoke only | PASS (Gaussian default; use `--use-mlp` for deployment mode) |
| Assumption validation | `assumption_validation.py --ci` | No (nightly) | **5/5 PASS** (A1–A5 incl. A2) |
| OOD calibration | `ood_calibration.py` | No (nightly) | Monotonic PASS |
| Nightly stress | `nightly_stress.py` | Scheduled workflow | Fill-phase PASS @ 1k; post-cap @ 10k |
| Observatory integrity | `phca_replay.py --check` on `fixtures/multi_agent_short/` and `fixtures/reacher_short/` | **Yes** | PASS |
| Scientific reproduction | `make reproduce` / `make reproduce-quick` | Local manifest | Run locally; `logs/reproduce_report.json` may be dry-run |
| Forgetting rate (AT-2-lite) | `scripts/benchmark_level4.py` | No | Local gate; L4b **PASS** (D-137: eval start-position confound fixed) — see `docs/l4_root_cause_verdict.md` |
| L4 ablation matrix | `scripts/run_l4_ablation.py` | No | T3 local; `make bench-level4-ablation` |
| Maturation T1 gates | `make maturation-test` | No | 45 tests static+forgetting+resilience+maturation |
| Cognitive recovery | `scripts/benchmark_recovery.py` | No | Local gate; B1/C1/F5 injectable scenarios |
| **Scientific validation suite** | `make validate-science` / `run_validation_suite.py` | Local | **Complete** (2026-07-05); 24 experiments × 30 seeds → `results/validation/` |

### Scientific metrics (2026-07-05 audit)

| Metric / hypothesis | Implementation | Notes |
|---|---|---|
| `cross_context_reuse` | `emergence.py` | Uses `env_goal_relocated` + `goal_switched` trace flags (D-128) |
| H004 interaction | `aggregate_validation.py` | Reads `interaction_test.json`; multiseed traces in `run_experiment.py` |
| `transfer_efficiency` | `phi_iq.py` | Proxy only — not cross-task transfer; see `forgetting.py` + Level-4-lite |
| H003 synergy | `synergy.py` | Noisy; smoke (3 seeds) may flip verdict vs full (30) |
| `goal_thrash_events` | `runner._collect_failures` | MDIM drive changes — not env goal relocation |

See [docs/doc_drift_audit_2026-07-05.md](docs/doc_drift_audit_2026-07-05.md).

**Scope note:** Observatory Phases 7–20 are complete. Whole PHCA blueprint items
(M5, L4–L5, grounding adapter, etc.) remain in backlog below.

---

## Action Selection Modes

| Environment | Mode | A4 "prediction-primary"? |
|---|---|---|
| GridWorld discrete | Confidence-gated task_lock (τ=0.6) + Manhattan/BFS geometry + blended G′ fallback | **Hybrid Cognitive Map** (D-136) |
| Cartpole | 3-bin discrete | **Partial** |
| Pendulum continuous | MPC: sample K actions, predict, pick best ŝ′ | **Yes** (A4 measured) |
| Reacher continuous | Same MPC path, dim 2 | **Yes** |

Details: [docs/action_selection.md](docs/action_selection.md)

---

## Fixes (2026-07-08)

| Feature | Code | Status | Evidence |
|---|---|---|---|
| Goal novelty metric (replace drive-diversity proxy) | `python/phca/motivation/mdim.py` — `_seen_goal_signatures`, `_novel_goal_count`, `snapshot()` | ✅ **Done** | `novel_goal_rate` in MDIM snapshot; L3 pass criteria uses `novel_goal_rate > 0.01` instead of `active_drives > 0.1` |
| B1 eta boost timing (applied before TSPL, not after) | `python/phca/core/cycle.py` — APC regulation skips eta overwrite when B1 recovery active | ✅ **Done** | Recovery B1 eta boost persists through APC regulation phase |
| Eval contamination (learn disabled during benchmark eval phase) | `scripts/benchmark_level4.py` — `_run_eval_on_task` sets `interventions.enable_gprime_learn=False` | ✅ **Done** | gprime_learn + M3 replay frozen during eval; weights uncontaminated |
| M3 replay total visible in default benchmark output | `scripts/benchmark_level4.py` — `m3_replay_total` printed always, not only under `--diagnostic` | ✅ **Done** | Print unconditional; stored in seed_entry |

## Core Infrastructure Fixes (2026-07-06)

| Feature | Code | Status | Evidence |
|---|---|---|---|
| Confidence-gated discrete action selection (autonomous goal-setting / conditional action branch) | `python/phca/core/cycle.py` — `TASK_LOCK_CONFIDENCE_THRESHOLD=0.6` | ✅ **Done** | Low confidence → blended G′ per-candidate scorer; high confidence → Manhattan greedy (D-112 preserved) |
| MDIM deficit/target drive normalization | `python/phca/motivation/mdim.py` — `generate_goal()` | ✅ **Done** | `deficits_norm = deficits / targets` before softmax; `goal_switch_boost` capped at 2.0 |
| Unified epistemic entropy for A3 + D4 (MC-Dropout) | `python/phca/core/cycle.py` — `_epistemic_entropy()` | ✅ **Done** | `0.01 + gprime._last_mutual_info` (MC-dropout); wired to D4 `model_entropy` and `belief_entropies["G'"]`; A3 CI PASS (min_entropy=0.0408) |

---

## Completed Observatory items (from STATUS.md)

| ID | Item | Phase |
|---|---|---|
| P7-4 | JSONL/replay/report parity | **Done** (Phase 7) |
| P8-1 | Seek/scrub without panel desync | **Done** (Phase 8) |
| P10-1 | Full offline report ↔ dashboard parity | **Done** (Phase 10) |
| P11-1 | 3000+ cycle scrub without severe lag | **Done** (Phase 11) |
| P12-1 | Multi-session `--compare` | **Done** (Phase 12) |
| P13-1 | Session anomaly detection | **Done** (Phase 13) |
| P14-1 | Action explainability | **Done** (Phase 14) |
| P15-1 | Stable `phca.monitoring` public API | **Done** (Phase 15) |
| P16-1 | Supervisor + crash recovery | **Done** (Phase 16) |
| P17-1 | Multi-agent Observatory | **Done** (Phase 17) |
| P18-1 | Cognitive-moment query | **Done** (Phase 18) |
| P19-1 | Scientific reproduction manifest | **Done** (Phase 19) |
| P20-1 | Research maturity sign-off | **Done** (Phase 20) |

## Open Backlog (blueprint / cognition)

| ID | Item | Phase |
|---|---|---|
| AD-3 | Unify benchmark entry points | 4.1 | **Done** — `runner.py` deprecated; use `scripts/benchmark.py` |
| AD-4 | M5 procedural memory | 4.3 |
| SC-1 | Grounding adapter | Deferred |
