# Claim Correction Sheet — 2026-08-02

Honesty patches tracked during the investigation. Status synced at **D-198** docs honesty sweep.

## Demote / banner

| Target | Correction | Status |
|--------|------------|--------|
| `STATUS.md` | Banner + L4b “fixed” clarified as D-137 confound / superseded 37.83% FAIL | **Applied** D-194 / D-198 |
| `docs/architecture.md` | Banner + Step-9 geometry default; A4 env-scoped; TE weight / Φ-IQ historical | **Applied** D-194 / D-198 |
| `docs/l4_root_cause_verdict.md` | Banner + Result marked historical/vacated | **Applied** D-194 / D-198 |
| `docs/action_selection.md` | Default-first banner; A4 env-scoped; D-197 blended-at-power | **Applied** D-198 |
| `docs/limitations.md` | Geometry default; STATUS→investigations; RBTA 30-seed rates; Φ-IQ historical labels | **Applied** D-194 / D-198 |
| `docs/phca_causal_evidence.md` | Overnight SoT honesty; remove “run 10×10”; 15-seed viewport means labeled | **Applied** D-198 |
| `DOCUMENTATION_MAP.md` | STATUS → Historical; cite logs/DECISIONS; trust order D-198 | **Applied** D-198 |
| `docs/maturity_audit_2026-07-07.md` | Dated demotion banner | **Applied** D-198 |
| `docs/phi_iq_metric.md` | Planner-contamination banner; L4 SoT; drop STATUS cite | **Applied** D-198 |

## Live wording tightenings

| Target | Correction | Status |
|--------|------------|--------|
| `README.md` | L3 FAIL overnight SoT; trust order logs→DECISIONS; Cartpole fallthrough; D-197/D-198 | **Applied** D-198 |
| `IMPLEMENTATION_STATUS.md` | Geometry default row; causal D-197; Φ-IQ historical; Cartpole A4 No | **Applied** D-198 |
| `CONTRIBUTING.md` / `SETUP.md` / `reproducibility.md` | Drop STATUS triage / hardcoded counts; overnight harness | **Applied** D-198 |
| `docs/benchmark_artifacts.md` | Overnight / diagnosis / L4-30s index | **Applied** D-198 |
| Observatory `session_report.py` | Geometry dominance flags | **Applied** (prior) |

## Metric labeling (eval docs / README)

| Metric | Must be labeled as | Status |
|--------|-------------------|--------|
| GridWorld goal_rate (default) | Planner competence under geometry | **LIVE** README / causal honesty |
| Φ-IQ L2 under default | Contaminated by planner GC | **Applied** phi_iq banner + reports |
| L4 forgetting_rate | Goal-rate retention under geometry; prefer eval PE | **Mitigated** |
| Causal gate PASS | Not prediction-primary; see `secondary_prediction` | **Applied** |
| CI L0 Φ-IQ PASS | Smoke / regression floor only | **Mitigated** |

## Residual wording debt (non-blocking)

- Deeper body polish inside demoted `docs/architecture.md` OOD/assumption sections (banner + key rows fixed).
- Second overnight dir (`logs/overnight_20260803_*`) may be cited when finished — not required for D-198.

## Do not correct into false negatives

- Continuous MPC **is** prediction-primary.
- G′ still learns on MLP path when learn is on.
- Viewport causal PASS vs greedy_observed is real — still planner/frontier under default geometry (H2).
