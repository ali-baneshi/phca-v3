# Documentation Drift Audit — 2026-07-05

Systematic comparison of **living documentation claims** vs **runtime implementation**
and **committed validation artifacts**. Use this file when updating README, STATUS,
or hypothesis manifests.

**Trust order:** `STATUS.md` → named artifacts in `docs/benchmark_artifacts.md` →
`IMPLEMENTATION_STATUS.md` → this audit → archive/research docs.

---

## 1. Drift categories

| Tier | Meaning | Action |
|------|---------|--------|
| **Critical** | Scientific claim ≠ code behavior | Fix code and/or docs |
| **Living-stale** | Living doc numbers/formulas outdated | Sync living docs |
| **Historical** | Phase snapshot; metrics may be stale | Banner only; do not cite numbers |
| **Aspirational** | Whitepaper/blueprint targets not built | Keep banners; no live claims |

---

## 2. Claim / implementation / evidence matrix

### Critical (Category A)

| Claim (doc) | Implementation | Evidence | Status |
|-------------|----------------|----------|--------|
| H001: memory → cross-context reuse under **goal relocation** (`hypotheses.yaml`, `grid_size=10`) | `cross_context_reuse()` used `goal_switched` = MDIM `drive_id` change (`trace.py`); `relocate_goal()` not traced | `results/validation`: reuse 0 full & ablated | **Fixed:** `env_goal_relocated` in trace |
| H001 manifest `grid_size=10` vs `no_memory.yaml` `grid_size=5` | Mismatch | `experiments/ablations/no_memory.yaml` | **Fixed:** manifest → 10 |
| H004 interaction / novel behaviour | `aggregate_validation.py` stub; `interaction_test.json` not read | `hypothesis_verdicts.json` always Partially_supported | **Fixed:** wire interaction_test |
| `interaction_test` 30 seeds | `run_experiment.py` ran trace compare on base seed only | `interaction_test.json` single-trace | **Fixed:** multiseed aggregate |
| H002 transfer beyond local reward | `transfer_efficiency = adaptation × prediction` (`phi_iq.py`) | `phi_iq_metric.md` disclaims; hypothesis text misleading | Docs updated |
| H003 desync reduces synergy | Noisy metric; smoke (3 seeds) flips verdict vs full (30) | smoke H003 Validated, full Refuted | Document smoke caveat |

### Living-stale (Category B)

| Claim | Doc value | Actual (2026-07-05) | File to update |
|-------|-----------|----------------------|----------------|
| Test count | 743 | **768** (`pytest python/`) | README, STATUS, DOCUMENTATION_MAP, reproducibility |
| L1/L3 action diversity | `/5.0` fixed | `unique_actions / env.action_space_size` | `phi_iq_metric.md` |
| L3 goal_complexity fallback | MDIM only | action diversity if no MDIM | `phi_iq_metric.md` |
| `behavioral_compression` | not documented | seeded RNG (`emergence.py`) | `phi_iq_metric.md` / limitations |
| APC orthogonality | not documented | zero-variance guard before `corrcoef` | `architecture.md` note |
| Canonical Φ-IQ | 0.7323 | unchanged (`logs/benchmark_report.json`) | OK |
| Suite aggregate Φ-IQ | 0.526±0.189 | `results/validation/summary.json` | OK |

### Historical / aspirational (Category C)

| Document | Issue |
|----------|-------|
| `docs/PHCA_Cognitive_Observatory_Architecture_FA.md` | Phases 13–20 marked backlog; English/STATUS: complete |
| `docs/monitoring_plan.md` | References removed `phca-monitor.py` |
| `docs/archive/phase20_completion_report.md` | 699 tests vs 768 |
| `research/outputs/10-implementation-blueprint.md` | Rust, 21 components — unbuilt |
| `gate_phase3.3_final.md` | Wrong path to release notes |
| Whitepaper §1.3 | Forgetting, Φ criticality, failure recovery — not implemented |

### Undocumented architecture (Category D)

| Path | Scripts | Emergence/synergy |
|------|---------|-------------------|
| Scaling benchmarks | `scripts/benchmark.py` | **No** — Φ-IQ only |
| Ablations / OOD | `run_experiment.py` → `evaluation/runner.py` | **Yes** — trace-based |

Documented in `reproducibility.md` after this audit.

---

## 3. Canonical numbers reference

| Number | Command / source | Notes |
|--------|------------------|-------|
| Overall Φ-IQ **0.7323** | `benchmark.py --use-mlp --cycles=200` seed 42 | 5×5 grid; canonical |
| L0–L3 breakdown | `logs/benchmark_report.json` | Per-level table in `phi_iq_metric.md` |
| Suite aggregate **0.526±0.189** | `make validate-science` → `summary.json` | Mean of 20 experiment file means |
| full_system **0.631±0.029** | `results/validation/ablations/full_system.json` | 30 seeds |
| Scaling slip=0 (30 seeds) | 5×5 **0.700**, 10×10 **0.325**, 20×20 **0.147** | `results/validation/scaling/` |
| Predictive validity **r≈0.992** | `phi_iq_validation.py` 100 seeds | H006; tautology caveat in docs |
| **768** tests | `PYTHONPATH=python pytest python/` | Core + evaluation alignment tests |
| Causal gate L1–L3 | `phca_causal_eval.py` | **FAIL** (L2/L3 at 30 seeds, D-151) |

**Do not cite** benchmark numbers from `docs/archive/*` without cross-checking named `logs/` and DECISIONS (STATUS.md is frozen historical).

---

## 4. Metric semantics (post-fix glossary)

| Field | Meaning |
|-------|---------|
| `goal_switched` (trace) | MDIM intrinsic drive id changed between cycles |
| `env_goal_relocated` (trace) | `GridWorld.relocate_goal()` called before this cycle |
| `cross_context_reuse` | Action-profile similarity before/after **context switches** (`env_goal_relocated` or `goal_switched`) |
| `goal_thrash_events` (failure log) | `drive_id` changes in cycle history — **not** env goal relocation |
| `transfer_efficiency` | `adaptation_speed × prediction_accuracy` — not cross-task transfer |
| `synergy` | MI(confidence, next action) minus marginal floor |

---

## 5. Files that must not be updated in isolation

When changing any of these, update the linked set:

1. **Φ-IQ formula/levels:** `phi_iq.py`, `phi_iq_metric.md`, `benchmark.py`, `check_benchmark_gate.py`
2. **Hypothesis verdicts:** `hypotheses.yaml` (claims only), `aggregate_validation.py`, `results/validation/README.md`
3. **Test counts:** `STATUS.md`, `README.md`, `DOCUMENTATION_MAP.md`, `reproducibility.md`
4. **Causal edges:** `causal_graph.yaml`, `causal_graph.md`, ablation manifests
5. **Validation suite:** `run_validation_suite.py`, `reproducibility.md`, `IMPLEMENTATION_STATUS.md`

---

## 6. Remediation log (this pass)

| Item | Action |
|------|--------|
| `env_goal_relocated` trace field | Implemented |
| `no_memory.yaml` grid_size 10 | Aligned |
| H004 evaluator | Wired to `interaction_test.json` |
| Living docs sync | README, STATUS, phi_iq_*, reproducibility, IMPLEMENTATION_STATUS |
| Stale doc banners | FA observatory, monitoring_plan, archive README |
| Smoke re-run | `results/validation_smoke/` + README update |

**Out of scope (future):** full `make validate-science` (~3 h), decomposition regression study, whitepaper §1.3 criteria implementation.
