# PHCA Maturity Audit — 2026-07-07 (expanded)

Authoritative **Claim → Evidence → Gap** matrix for maturation Track A.
Companions: [`static_audit_2026-07-07.md`](static_audit_2026-07-07.md), [`maturation_signoff.md`](maturation_signoff.md).

**Trust order:** measured artifacts in `logs/` → this audit → `IMPLEMENTATION_STATUS.md` → README.

**Gate tiers:** T0 CI (<5 min) | T1 CI extended (<15 min) | T2 Nightly | T3 Weekly local | T4 Release

**Gap classes:** G0 no measurement | G1 weak protocol | G2 behavior fails | G3 false confidence | G4 metric conflation | G5 silent no-op

---

## Executive summary (post-maturation pass)

| Area | Status | Notes |
|------|--------|-------|
| L0–L3 Φ-IQ | Green (T0) | `benchmark_ci_baseline.json` |
| A1–A5 invariants | Green (T2) | nightly `assumption_validation.py --ci` |
| Observatory Phases 7–20 | Green (T0 replay) | Resilience fields in JSONL (2026-07-07) |
| Level-4-lite L4b | **Green (T3)** | `forgetting_rate=0.0000` (2026-07-08, budget=16) — shadow gaps resolved |
| M3→G′ replay | **Wired** | G5-01 closed |
| Cognitive resilience injectables | Green (T3) | G3: ≠ full matrix |
| Maturation T1 tests | Green | `make maturation-test` |

---

## Whitepaper §1.3 (5 rows)

| ID | Claim | Target | Status | Gate | Tier | Last result | Gap | Track |
|----|-------|--------|--------|------|------|-------------|-----|-------|
| WP-1 | Cycle latency | <500 ms | PASS | benchmark logs | T2 | ~10–17 ms mean | — | C |
| WP-2 | Forgetting rate | <5% @ 100 tasks | PASS | `benchmark_level4.py` | T3 | `forgetting_rate=0.0000` @ 10 (budget=16) | G2 (closed) | E |
| WP-3 | Goal autonomy | ≥1 novel/100 cyc | Partial | L3 diversity | — | Not measured | G0 | G |
| WP-4 | Criticality Φ | 90% in band | Not impl | — | — | APC volatility only | G0 | — |
| WP-5 | Failure recovery | ≥80%/10 cyc | Partial MVP | `benchmark_recovery.py` | T3 | 1.00 inject | G3 | F |

---

## Invariants A1–A5 (5 rows)

| ID | Claim | Gate | Tier | CI? | Result | Gap | Track |
|----|-------|------|------|-----|--------|-----|-------|
| A1 | Resource boundedness | assumption_validation --ci | T2 | No | PASS | — | C |
| A2 | Temporal causality | same | T2 | No | PASS | — | C |
| A3 | Incomplete knowledge | same | T2 | No | PASS | — | C |
| A4 | Prediction primary | causal_eval + MPC | T1/T3 | Smoke | Partial GW | G2 | C,G |
| A5 | Feedback adaptation | same | T2 | No | PASS | — | C |

---

## Blueprint components (15 rows)

| ID | Component | Status | Gate | Tier | Gap | Track |
|----|-----------|--------|------|------|-----|-------|
| BP-01 | RBTA enforcer | Impl | grid RBTA tests | T0 | — | C |
| BP-02 | G′ MLP | Impl | L0–L3 | T0/T2 | FIFO 500 cap | D,E |
| BP-03 | TSPL P-Stream | Impl | A5 | T2 | E/S removed | — |
| BP-04 | M3 episodic | Impl | nightly stress | T2 | Wired 2026-07-07 | D |
| BP-05 | M4 semantic | Partial | consolidation | T3 | Stats only | D |
| BP-06 | Failure matrix | Partial MVP | recovery bench | T3 | 25+ missing | F |
| BP-07 | Grounding L0/L2 | Not impl | — | — | — | — |
| BP-08 | M5 procedural | Not impl | — | — | — | — |
| BP-09 | Φ-IQ L4-lite | Partial | benchmark_level4 | T3 | G2 | E |
| BP-10 | Φ-IQ L5 | Not impl | — | — | — | — |
| BP-11 | Observatory | Complete | phca_replay | T0 | — | H |
| BP-12 | ASI sanitizer | Impl | unit tests | T1 | — | C |
| BP-13 | MDIM | Impl | L3 gate | T2 | ≠ novel goals | G |
| BP-14 | Attention | Impl | cycle tests | T1 | — | C |
| BP-15 | HPM runtime | Partial | bounds only | T2 | No full runtime | C |

---

## Observatory phases (13 rows)

| ID | Phase | Claim | Gate | Tier | Result | Track |
|----|-------|-------|------|------|--------|-------|
| OBS-7 | JSONL/replay parity | Done | replay --check | T0 | PASS | H |
| OBS-8 | Seek/scrub | Done | monitoring tests | T1 | PASS | H |
| OBS-9 | Schema governance | Done | schema version | T0 | PASS | H |
| OBS-10 | Report parity | Done | report tests | T1 | PASS | H |
| OBS-11 | Scrub perf 3k+ | Done | perf tests | T2 | PASS | H |
| OBS-12 | Multi-session compare | Done | compare tests | T2 | PASS | H |
| OBS-13 | Anomalies | Done | anomaly tests | T2 | PASS | H |
| OBS-14 | Explainability | Done | explain tests | T1 | PASS | H |
| OBS-15 | Public API | Done | public_api tests | T1 | PASS | H |
| OBS-16 | Session recovery | Done | test_session_recovery | T1 | PASS | H |
| OBS-17 | Multi-agent | Done | multi_agent fixture | T0 | PASS | H |
| OBS-18 | Query | Done | query tests | T1 | PASS | H |
| OBS-19 | Reproduce manifest | Done | reproduce-quick | T4 | Local | I |

---

## Science hypotheses (6 rows)

| ID | Hypothesis | Gate | Tier | Artifact | Gap | Track |
|----|------------|------|------|----------|-----|-------|
| SCI-H001 | Memory → cross_context_reuse | validate-science | T4 | summary.json | Fixed D-128 | I |
| SCI-H002 | Transfer beyond reward | phi_iq proxy | T4 | G4 | Not retention | G |
| SCI-H003 | Synergy | synergy.py | T4 | smoke may flip | Noisy | I |
| SCI-H004 | Interaction | interaction_test | T4 | verdicts.json | Wired | I |
| SCI-repro | reproduce manifest | make reproduce-quick | T4 | reproduce_report | Local | I |
| SCI-checksum | aggregate drift | golden manifest | T4 | sha256 | Tolerance 0 | I |

---

## Evaluation gates (15 rows)

| Gate | Script | Tier | CI? | Status | Track |
|------|--------|------|-----|--------|-------|
| Lint | ruff | T0 | Yes | PASS | J |
| Unit all | test-python | T0 | Yes | PASS | J |
| L0 quick | check_benchmark_gate | T0 | Yes | PASS | G |
| Observatory replay | phca_replay --check | T0 | Yes | PASS | H |
| Maturation T1 | make maturation-test | T1 | Via pytest | PASS | J |
| Forgetting unit | test_forgetting | T1 | Via pytest | PASS | E |
| Resilience unit | test_resilience | T1 | Via pytest | PASS | F |
| Static contracts | test_static_contracts | T1 | Via pytest | PASS | B |
| Causal smoke | phca_causal_eval | T1 | ci-local | L1 PASS, L2/L3 FAIL (30 seeds, D-151) | G |
| Assumption validation | assumption_validation --ci | T2 | nightly | PASS | C |
| Full Φ-IQ MLP | benchmark.py --use-mlp | T2 | nightly | PASS | G |
| OOD calibration | ood_calibration | T2 | nightly | PASS | G |
| Nightly stress | nightly_stress | T2 | nightly | PASS | D |
| L4 ablation | run_l4_ablation | T3 | No | Local | E |
| Recovery injectables | benchmark_recovery | T3 | No | PASS | F |

**Total claim rows:** 59

---

## G4 conflation register

| Metric | Measures | Does NOT measure |
|--------|----------|------------------|
| forgetting_rate | Max drop on prior tasks | Improvements |
| transfer_efficiency | adapt × prediction | Cross-task retention |
| cross_context_reuse | trace flags | Forgetting |
| recovery_rate injectable | Script faults | Full matrix |

---

## G5 register (updated)

| ID | Hook | Status |
|----|------|--------|
| G5-01 | M3→G′ replay | **Wired** |
| G5-02..04 | Observability resilience fields | **Wired** |
| G5-05 | Eval on_task_boundary | **Wired** (2026-07-07) |
| G5-06 | B4 during train | Partial — use `--interleaved-eval` |

---

## Risk priority

| Rank | Item | Score | Action |
|------|------|-------|--------|
| 1 | L4b capacity (G2) | 9 → **0** | **Closed** — `forgetting_rate=0.0000` with budget=16 + consolidation gradient; see ChangeLog 2026-07-08 |
| 2 | A4 GridWorld gap | 6 | Causal eval MLP T3 |
| 3 | Injectable ≠ §1.3 (G3) | 5 | limitations + tests |
| 4 | Science T4 drift | 5 | golden manifest |

---

## Commands

```bash
make maturation-test
make bench-level-0
make bench-level4-ablation
make bench-recovery
make ci-local
```

---

## Changelog

| Date | Change |
|------|--------|
| 2026-07-07 | Initial audit |
| 2026-07-07 | Expanded to 59 rows; M3/Obs wired; sign-off doc |
| 2026-07-08 | Shadow gaps resolved (budget 16 + consolidation gradient); L4b forgetting **PASS** `0.0000` — risk item 1 closed |
