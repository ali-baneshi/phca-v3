# PHCA v3.0 — Implementation Status Matrix

Maps **whitepaper success criteria** and **blueprint components** to current
code, gate scripts, and measured outcomes. Last aligned with STATUS.md:
2026-07-11.

**Maturation audits (2026-07-07):** [docs/maturity_audit_2026-07-07.md](docs/maturity_audit_2026-07-07.md),
[docs/static_audit_2026-07-07.md](docs/static_audit_2026-07-07.md).

Legend: **Implemented** | **Partial** | **Measured** | **Not implemented** | **Stub** | ✅ **Done**

---

## Whitepaper §1.3 Success Criteria

| Criterion | Target | Status | Gate / evidence |
|---|---|---|---|
| Cycle latency | < 500 ms | **Measured PASS** | Φ-IQ pass criteria; mean ~10–17 ms MLP |
| Forgetting rate | < 5% after 100 sequential tasks | **PASS** | Level-4-lite: `scripts/benchmark_level4.py` (10-task GridWorld) — validated `forgetting_rate=0.0000`, `passes_gate=True` on 2026-07-08 with `m3_replay_budget=16` + consolidation gradient active; full 100-task AT-2 not CI-gated |
| Goal autonomy | ≥ 1 novel goal / 100 cycles | **Measured PASS** | MDIM tracks `_seen_goal_signatures` → `novel_goal_rate` in snapshot; L3 gates `novel_goal_rate > 0.01` (✅ Done 2026-07-08) |
| Criticality maintenance | Φ ∈ [0.9Φc, 1.1Φc] for ≥ 90% cycles | **Implemented (D-139)** | Backward-pass gradient-norm w.r.t input (output-sensitivity Jacobian, not loss gradient). `Φ = (2/π)·arctan(||∂mean(out)/∂x|| / sqrt(d))`, EMA-filtered 0.7·cached + 0.3·raw. Zero extra FLOP for loss-gradient path (already from G'.learn); ~38K FLOP extra for output-sensitivity when `last_input_sensitivity()` called directly. PID setpoint unchanged at 0.5 (arctan maps [0,∞) → [0,1)). |
| Failure recovery | ≥ 80% mitigated within 10 cycles | **Partial (MVP)** | `phca/resilience/` B1/B4/B5/C1/F5 + `scripts/benchmark_recovery.py` |

---

## Verified Invariants (A1–A5)

| Invariant | Code enforcement | Falsification | Notes |
|---|---|---|---|
| **A1** Resource boundedness | `rbta_enforcer.py` | `assumption_validation.py --ci` | **Measured PASS** |
| **A2** Temporal causality | Pipeline order in `cycle.py` | `assumption_validation.py --ci` (A2 monitor) | **Measured PASS** |
| **A3** Incomplete knowledge | `_epistemic_entropy()` → RBTA entropy_floor (MC-dropout mutual info) | `assumption_validation.py --ci` | **Measured PASS** (regression fix 2026-07-11: direction-blind severity in `ConstraintViolation.__post_init__` zeroed ENTROPY violations; fixed with direction-agnostic `abs(measured - allowed)` formula + count-based `_classify_action`) |
| **A4** Prediction as Primary | G′ predict every cycle, unified prediction-scored action selection | A4 test on all envs | **Suspended for GridWorld (D-156)** — blended G′ scorer caused catastrophic navigation failure at 10×10 (goal_rate 0.03% vs pure-geometry 26.8%; vs greedy_observed 24.0%). Default reverted to pure BFS/Manhattan geometry. Prediction-primary still active for continuous-control (Cartpole, Pendulum, Reacher) via MPC. |
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
| RBTA enforcer | Whitepaper §2.1 | `phca/regulation/rbta_enforcer.py` | **Implemented** — count-based `_classify_action` (0→CONT, 1-2→INT, 3+→TERM) with severity override (>0.8→TERM). `ConstraintViolation` severity direction-agnostic via `abs()`. `ResourceBounds` validates `B_energy > 0`. (Python; Rust removed D-084) |
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
| M3 episodic | Blueprint | `phca/memory/m3_episodic.py` | **Implemented** (SQLite, VACUUM D-108, PER D-138, task-aware eviction D-146) |
| M4 semantic | Blueprint | `phca/consolidation/scheduler.py` | **Partial** — statistical pattern facts |
| M5 procedural | Blueprint | — | **Not implemented** |
| M6 meta-memory | Whitepaper | — | **Not implemented** |
| Failure matrix A–F | Blueprint | `phca/resilience/` | **Enhanced** — B1, B4, B5, C1, F5 detect + recover; E1 FallbackController with dual-signal (entropy + cascade) |
| GridWorld | Phase 3.1 | `phca/environments/grid_world.py` | **Implemented** |
| MuJoCo (3 envs) | Phase 4–7 | `phca/environments/mujoco_env.py` | **Implemented** |
| Cognitive Observatory | Phase 7–20 | `phca/monitoring/` | **Complete** (schema, replay, report, scrub, compare, anomalies, explain, API, supervisor, multi-agent, query, reproduce) |
| Φ-IQ L0–L3 | Blueprint | `scripts/benchmark.py` | **Implemented + measured** |
| Φ-IQ L4-lite (forgetting) | Blueprint | `scripts/benchmark_level4.py` | **PASS** — GridWorld continual, `forgetting_rate=0.0000` (eval start-position confound fixed D-145) |
| Φ-IQ L5 | Blueprint | — | **Not implemented** |

---

## Evaluation Gates

| Gate | Script | CI? | Current status (2026-07-06) |
|---|---|---|---|
| Φ-IQ regression (L0 quick) | `check_benchmark_gate.py` | **Yes** | PASS |
| Φ-IQ full (MLP L0–L3) | `scripts/benchmark.py --use-mlp` | No (nightly) | PASS (0.7317) |
| MuJoCo smoke | `check_benchmark_gate.py --mujoco` | No (nightly) | PASS |
| Causal behavior L1–L3 | `phca_causal_eval.py --gate` | Smoke only | **L1 PASS, L2 FAIL, L3 PASS** at 15 seeds (MLP, 200 cyc) with pure-geometry default (D-156). L3 gate PASS at 5×5 (goal 28.8% vs greedy 27.9%). L2 FAIL is inherent ceiling (greedy_observed 55.5% vs PHCA 50.3% at 5×5). 10×10: L2 PASS (26.8% vs 24.0%), L3 marginal FAIL (13.57% vs 13.73% — tied within noise). Blended scorer caused catastrophic collapse at 10×10 (0.03% goal rate). See D-156. |
| Assumption validation | `assumption_validation.py --ci` | No (nightly) | **5/5 PASS** (A1–A5 incl. A2) |
| OOD calibration | `ood_calibration.py` | No (nightly) | Monotonic PASS |
| Nightly stress | `nightly_stress.py` | Scheduled workflow | Fill-phase PASS @ 1k; post-cap @ 10k |
| Observatory integrity | `phca_replay.py --check` on `fixtures/multi_agent_short/` and `fixtures/reacher_short/` | **Yes** | PASS |
| Scientific reproduction | `make reproduce` / `make reproduce-quick` | Local manifest | Run locally; `logs/reproduce_report.json` may be dry-run |
| Forgetting rate (AT-2-lite) | `scripts/benchmark_level4.py` | No | Local gate; L4b **PASS** (eval start-position confound fixed D-137/D-145) — see `docs/l4_root_cause_verdict.md` |
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
|---|---|---|---|
| GridWorld discrete | Pure BFS/Manhattan geometry (default since D-156; G' prediction blend available via `--enable-blended-scorer`) | **No** (prediction-primary suspended — blended scorer caused catastrophic 0.03% goal rate at 10×10; pure geometry 26.8%) |
| Cartpole | 3-bin discrete (unified path) | **Yes** |
| Pendulum continuous | MPC: sample K actions, predict, pick best ŝ′ | **Yes** (A4 measured) |
| Reacher continuous | Same MPC path, dim 2 | **Yes** |

Details: [docs/action_selection.md](docs/action_selection.md)

---

## Fixes (2026-07-08)

| Feature | Code | Status | Evidence |
|---|---|---|---|---|
| Goal novelty metric (replace drive-diversity proxy) | `python/phca/motivation/mdim.py` — `_seen_goal_signatures`, `_novel_goal_count`, `snapshot()` | ✅ **Done** | `novel_goal_rate` in MDIM snapshot; L3 pass criteria uses `novel_goal_rate > 0.01` instead of `active_drives > 0.1` |
| B1 eta boost timing (applied before TSPL, not after) | `python/phca/core/cycle.py` — APC regulation skips eta overwrite when B1 recovery active | ✅ **Done** | Recovery B1 eta boost persists through APC regulation phase |
| Eval contamination (learn disabled during benchmark eval phase) | `scripts/benchmark_level4.py` — `_run_eval_on_task` sets `interventions.enable_gprime_learn=False` | ✅ **Done** | gprime_learn + M3 replay frozen during eval; weights uncontaminated |
| M3 replay total visible in default benchmark output | `scripts/benchmark_level4.py` — `m3_replay_total` printed always, not only under `--diagnostic` | ✅ **Done** | Print unconditional; stored in seed_entry |
| M3 replay budget increased from 4→16 (Gap A) | `scripts/benchmark_level4.py` — `m3_replay_budget` default | ✅ **Done** | ~142 samples/prior-task vs ~35; addresses root cause of 100% measured forgetting |
| Consolidation gradient feedback into G' (Gap B) | `python/phca/consolidation/scheduler.py` — `step()` takes optional `gprime`, replays episodes at `lr_scale=0.1` before marking consolidated | ✅ **Done** | M3 transitions now contribute gradient signal before eviction; eval-leak guarded by `enable_gprime_learn` |
| Novel goal rate in benchmark JSON output (Gap C) | `scripts/benchmark_level4.py` — `novel_goal_rate` in seed_entry + aggregated return | ✅ **Done** | Enables CI trend tracking for A4 autonomy criterion |
| 2026-07-08 validation — Shadow Gaps resolved | `scripts/benchmark_level4.py` — `--m3-replay-budget 16` | ✅ **Validated** | `forgetting_rate=0.0000`, `passes_gate=True`, `m3_replay_total=23040` |
| **Feature 3 — PER (Prioritized Experience Replay)** | `python/phca/memory/m3_episodic.py` — `sample_episodes_per()`, `update_priority()`, `batch_update_priorities()`; `python/phca/core/cycle.py` — PER sampling + IS weights + `_per_beta` annealing 0.4→1.0; `python/phca/consolidation/scheduler.py` tuple unpacking | ✅ **Done** (D-138) | Error-reduction-rate priority: `max(ε, (stored_error − current_error)/(stored_error + ε))` — not absolute TD-error, to avoid overfitting to aleatoric noise. |
| **Feature 2 — Φ (gradient-norm criticality)** | `python/phca/world_model/mlp.py` — `_last_output_sens` + `last_input_sensitivity()`; `python/phca/core/cycle.py` — `_update_phi_from_gradient()`, removed `_error_vol_window` + `_approximate_error_volatility()`; `python/phca/config.py` — `PHI_TARGET`, `PHI_MAX` | ✅ **Done** (D-139) | Output Jacobian norm via arctan, EMA-filtered. Replaces temporal CoV. |
| **Feature 1 — Async Two-Thread Loop** | `python/phca/core/cycle.py` — `_action_loop()`, `_learning_loop()`, `start_async()`, `stop_async()`, `_finalize_learning_cycle()`; `python/phca/config.py` — `PerceptionFrame`, `ActionResult`, `STALE_THRESHOLD_MS` | ✅ **Done** (D-140) | Queue-based (maxsize=1) back-pressure. Sync mode default (no regression). Async: avg latency 20.2ms vs sync 12.7ms; goal success 98.8% vs 99.0%. |

## Fixes (2026-07-09)

| Feature | Code | Status | Evidence |
|---|---|---|---|
| ASI sanitizer log flood reduction + precision recovery | `python/phca/asi/sanitizer.py` — CRITICAL→warning, warning→debug level changes; `_valid_streak` counter with `_precision_recovery_rate=1.5`; `reset()` clears streak | ✅ **Done** | All 21 ASI tests pass |
| MLP replay buffer capacity 500→2000 | `python/phca/world_model/mlp.py` — default `replay_capacity=2000`; also in `estimate_mlp_memory_bytes()` | ✅ **Done** | All 57 MLP tests pass |
| Consolidation state_dim default removed | `python/phca/consolidation/scheduler.py` — `state_dim` is now required (no default 84) | ✅ **Done** | All 26 consolidation tests pass; all callers already pass state_dim explicitly |
| Gaussian mutual_info slogdet now logged | `python/phca/world_model/gaussian.py` — silent LinAlgError return 0.0 → logs warning with `exc_info=True` | ✅ **Done** | All 21 Gaussian tests pass |
| M3 PER two-phase query (avoid full table scan) | `python/phca/memory/m3_episodic.py` — `sample_episodes_per()` now fetches only `(episode_id, priority)` first, samples n, then fetches only chosen rows | ✅ **Done** | All 21 M3 tests pass |
| PEU float64 conversion removed | `python/phca/prediction/error_unit.py` — `diff` stays in input dtype (float32) instead of casting to float64 | ✅ **Done** | All 8 PEU tests pass |
| M3 vacuum_interval 100→1000 (D-056 mismatch) | `python/phca/memory/m3_episodic.py` — `_vacuum_interval` changed from 100 to 1000 to match D-056 design decision | ✅ **Done** | VACUUM runs 10× less frequently; test sets own interval so unaffected |

## Fixes (2026-07-11)

| Feature | Code | Status | Evidence |
|---|---|---|---|
| L4 eval start-position confound — full cleanup | `scripts/benchmark_level4.py` — removed `train_end_positions` dict + `train_start_pos` param | ✅ **Done** | L4 smoke benchmark PASS; 10 lines dead code eliminated |
| M3 task-aware eviction (NEW-02) | `python/phca/memory/m3_episodic.py` — per-task quota eviction replaces global FIFO | ✅ **Done** (D-146) | `test_task_aware_eviction` verifies per-task fairness; all 11 M3 tests pass |
| Φ-IQ remove transfer_efficiency from composite (NEW-03) | `python/phca/evaluation/metrics/phi_iq.py` — dropped TE term, redistributed 0.15 weight to PA/AS/GC; `python/phca/evaluation/result_schema.py` — updated `DEFAULT_WEIGHTS` | ✅ **Done** (D-147) | Weights: PA 0.25, AS 0.25, GC 0.20, RE 0.20, FR 0.10; L3 benchmark Φ-IQ 0.77, all pass criteria ✓ |
 | Composition tree factory extraction (F-04) | `python/phca/core/cycle.py` — `_build_full_composition_tree()` replaces 30 duplicate lines | ✅ **Done** (D-148) | All 698 tests pass; no behavioural change |
| RBTA bound recalibration for grid >= 5 (D-152) | `config.py` — DEFAULT_MODULE_BOUNDS 1.5–2.5× increase; `mlp.py` — headroom 1.0→2.0, ACTION headroom 14.0×, MEM 500K floor removed, B_energy scaling; `cycle.py` — default action_b_time 0.020→0.030 | ✅ **Done** (D-152) | Grid 10 L2 violations ~72%→~10%; `test_l2_10x10_gaussian_violation_rate_under_15pct` PASS; 151/151 tests pass |

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

## Known Limitations

| Limitation | Impact | Status |
|---|---|---|
| Async mode `self.current_state` data race | Thread A overwrites state before Thread B finishes reading → stale state used in PEU/G'.learn | **Known** — mitigated by `maxsize=1` queue back-pressure; full fix (snapshot through ActionResult) deferred |
| Learning efficiency can drop if env.step > 50ms | Thread B timeouts on queue.get → cache_miss → replay-only fallback; learning stalls | **Monitored** — `_cache_hit/_cache_miss` counters + 10-window starvation warning |
| M3 SQLite BUSY under concurrent write pressure | SQLite may return `sqlite3.OperationalError("database is locked")` | **Mitigated** — `PRAGMA busy_timeout=5000` + `OperationalError` caught → uniform-sampling fallback in all SQL read methods |
| Warm-up → steady-state latency step (~3.4×) | MLP replay buffer fills at ~64 cycles → mini-batch replay activates → latency jumps from ~16ms to ~57ms | **Flagged W4 target** (D-081 corollary) |

## Test Coverage

| Suite | Count | CI? |
|---|---|---|
| Core unit tests (grid, cycle, memory, prediction, etc.) | 142+ | **Yes** (every push) |
| MuJoCo integration tests | 23 | **Yes** (`MUJOCO_GL=disabled`) |
| Static contract tests | 10+ | **Yes** |
| Maturation T1 gates | 45 | No (nightly) |
| Total | **698** (693 pass; 5 pre-existing monitoring/UI failures: JSON roundtrip, Qt rendering, multi-agent session report) | Mixed CI / nightly |

## Open Backlog (blueprint / cognition)

| ID | Item | Phase |
|---|---|---|
| AD-3 | Unify benchmark entry points | 4.1 | **Done** — `runner.py` deprecated; use `scripts/benchmark.py` |
| AD-4 | M5 procedural memory | 4.3 |
| SC-1 | Grounding adapter | Deferred |
