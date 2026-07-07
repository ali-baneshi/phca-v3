# PHCA Maturity Audit — 2026-07-07

Authoritative **Claim → Evidence → Gap** matrix for maturation Track A.
Companion: [`static_audit_2026-07-07.md`](static_audit_2026-07-07.md).

**Trust order:** measured artifacts in `logs/` → this audit → `IMPLEMENTATION_STATUS.md` → README.

**Gate tiers:** T0 CI (<5 min) | T1 CI extended (<15 min) | T2 Nightly | T3 Weekly local | T4 Release

**Gap classes:** G0 no measurement | G1 weak protocol | G2 behavior fails | G3 false confidence | G4 metric conflation | G5 silent no-op

---

## Executive summary

| Area | Status | Blocking issue |
|------|--------|----------------|
| L0–L3 Φ-IQ | Green (T0) | None |
| A1–A5 invariants | Green (T2) | Not in T0 CI |
| Observatory Phases 7–20 | Green (T0 replay) | Cognitive resilience fields not in JSONL |
| Level-4-lite forgetting | **Red (T3)** | `forgetting_rate=1.0` @ 10 tasks; tasks 0,7 collapse |
| Cognitive resilience injectables | Green (T3) | G3: ≠ full failure matrix |
| Scientific validation | Complete artifact | T4 regression not automated |
| M3→G′ replay path | **G5 unwired** | `sample_episodes` has zero callers |

**Root-cause hypothesis (Track E):** Verdict **B + partial C** — M3 task-tagged episodes never feed G′ learning; G′ uses 500-slot FIFO internal buffer overwritten across 800 train cycles/task sequence. Mitigation hooks (`protect_parameters`, `replay_boost`) run but cannot retain early-task transitions.

---

## Whitepaper §1.3 success criteria

| ID | Claim | Target | Status | Gate | Tier | Last result | Gap | Track |
|----|-------|--------|--------|------|------|-------------|-----|-------|
| WP-1 | Cycle latency | <500 ms | Measured PASS | Φ-IQ benchmark logs | T2 | ~10–17 ms mean MLP | — | C |
| WP-2 | Forgetting rate | <5% @ 100 tasks | Partial | `scripts/benchmark_level4.py` | T3 | **FAIL** 1.0 @ 10 tasks | G2 | E |
| WP-3 | Goal autonomy | ≥1 novel goal/100 cycles | Partial | L3 diversity >0.1 only | — | Not measured | G0 | G |
| WP-4 | Criticality Φ | ∈[0.9Φc,1.1Φc] 90% cycles | Not implemented | — | — | APC = error volatility | G0 | — |
| WP-5 | Failure recovery | ≥80% mitigated /10 cycles | Partial MVP | `benchmark_recovery.py` | T3 | 1.00 injectables | G3 | F |

**Artifact:** `logs/benchmark_level4.json` (2026-07-07 config: 10 tasks, 80 train, 20 eval, 3 seeds, MLP 5×5).

---

## Verified invariants A1–A5

| ID | Claim | Gate | Tier | CI? | Last result | Gap | Track |
|----|-------|------|------|-----|-------------|-----|-------|
| A1 | Resource boundedness | `assumption_validation.py --ci` | T2 | No | PASS | — | C |
| A2 | Temporal causality | same | T2 | No | PASS | — | C |
| A3 | Incomplete knowledge | same | T2 | No | PASS | — | C |
| A4 | Prediction primary | MPC path + causal eval | T1/T3 | Smoke | **Partial** GridWorld | G2 | C,G |
| A5 | Feedback adaptation | same + weight freeze test | T2 | No | PASS | — | C |

---

## Blueprint components (selected)

| ID | Component | Status | Gate / evidence | Tier | Gap | Track |
|----|-----------|--------|-----------------|------|-----|-------|
| BP-01 | RBTA enforcer | Implemented | A1 + grid RBTA tests | T0/T1 | — | C |
| BP-02 | G′ MLP | Implemented | L0–L3, causal smoke | T0/T1 | Internal replay 500 cap | D,E |
| BP-03 | TSPL P-Stream | Implemented | A5 | T2 | E/S removed D-020 | — |
| BP-04 | M3 episodic | Implemented | nightly stress | T2 | `sample_episodes` unwired G5 | D |
| BP-05 | M4 semantic | Partial | consolidation stats | T3 | Statistical facts only | D |
| BP-06 | Failure matrix | Partial MVP | B1,B4,B5,C1,F5 | T3 | 25+ modes missing | F |
| BP-07 | Grounding L0/L2 | Not implemented | — | — | — | — |
| BP-08 | M5 procedural | Not implemented | — | — | — | — |
| BP-09 | Φ-IQ L4-lite | Partial | benchmark_level4 | T3 | G2 | E |
| BP-10 | Φ-IQ L5 | Not implemented | — | — | — | — |
| BP-11 | Observatory | Complete | phca_replay --check | T0 | Resilience fields missing JSONL | H |

---

## Evaluation gates inventory

| Gate | Script | Tier | CI? | Status (2026-07-07) | Owner track |
|------|--------|------|-----|---------------------|-------------|
| Φ-IQ L0 quick | `check_benchmark_gate.py` | T0 | Yes | PASS | G |
| Φ-IQ full MLP | `benchmark.py --use-mlp` | T2 | No | PASS (0.7317) | G |
| MuJoCo | `check_benchmark_gate.py --mujoco` | T2 | Partial job | PASS | C |
| Causal L1–L3 | `phca_causal_eval.py --gate` | T1 smoke | No | PASS Gaussian | G |
| Assumption validation | `assumption_validation.py --ci` | T2 | No | 5/5 PASS | C |
| OOD calibration | `ood_calibration.py` | T2 | No | Monotonic PASS | G |
| Nightly stress | `nightly_stress.py` | T2 | Scheduled | PASS @1k/10k | D |
| Observatory replay | `phca_replay.py --check` | T0 | Yes | PASS | H |
| Forgetting AT-2-lite | `benchmark_level4.py` | T3 | No | **FAIL** | E |
| Recovery injectables | `benchmark_recovery.py` | T3 | No | PASS 1.00 | F |
| Science suite | `make validate-science` | T4 | No | Artifact 2026-07-05 | I |
| Reproduce | `make reproduce-quick` | T4 | No | Local | I |
| Unit: forgetting | `test_forgetting.py` | T1 | Via test-python | 11 pass | E |
| Unit: resilience | `test_resilience.py` | T1 | Via test-python | 11 pass | F |
| Static contracts | `test_static_contracts.py` | T1 | Via test-python | Documents G5 | B |

**Test count:** 130 collected (`pytest python/tests/` fast path).

---

## G4 metric conflation register (must not merge in reports)

| Metric | Location | Actually measures | Does NOT measure |
|--------|----------|-------------------|------------------|
| `forgetting_rate` | `forgetting.py` | Max relative drop on prior tasks | Improvement on later tasks |
| `transfer_efficiency` | `phi_iq.py` | adaptation × prediction | Cross-task retention |
| `cross_context_reuse` | `emergence.py` | env_goal_relocated + goal_switched | Forgetting rate |
| `goal_thrash_events` | validation runner | MDIM drive changes | Env goal relocation |
| RSS nightly soak | `nightly_stress.py` | M3/M4 growth | Cognitive forgetting |
| Session recovery | `session_recovery.py` | JSONL crash finalize | In-cycle B1–F5 |
| Recovery rate injectable | `benchmark_recovery.py` | Scripted fault injection | Full failure matrix |

---

## Risk priority (Impact × Likelihood)

| Rank | Item | Score | Next action |
|------|------|-------|-------------|
| 1 | M3 not wired to G′ learn (G5) | 9 | Track B confirm → Track E wire |
| 2 | L4 protocol: 0 baselines on many tasks (G1) | 8 | Track E diagnostic + baseline rule |
| 3 | Observability missing resilience fields (G5) | 7 | Track H export |
| 4 | B4 uses eval history only during eval pass | 7 | Track F verify detector timing |
| 5 | A4 partial GridWorld | 6 | Track C causal eval MLP |
| 6 | Injectable recovery ≠ §1.3 (G3) | 5 | Track F document + false-positive test |
| 7 | Science suite T4 drift | 5 | Track I golden checksums |

---

## Baseline commands (reproducibility)

```bash
# T1 fast
ruff check python/ --no-cache
PYTHONPATH=python python -m pytest python/tests/test_forgetting.py python/tests/test_resilience.py python/tests/test_static_contracts.py -q

# T0 CI equivalent
make ci-local

# T3 continual
PYTHONPATH=python python scripts/benchmark_level4.py --tasks 10 --task-cycles 80 --eval-cycles 20 --seeds 3 --use-mlp --output logs/benchmark_level4.json

# T3 recovery
PYTHONPATH=python python scripts/benchmark_recovery.py --scenarios b1,c1,f5 --output logs/benchmark_recovery.json
```

---

## Changelog

| Date | Change |
|------|--------|
| 2026-07-07 | Initial maturation audit (Track A); L4 FAIL from `logs/benchmark_level4.json`; G5 M3 replay identified |
