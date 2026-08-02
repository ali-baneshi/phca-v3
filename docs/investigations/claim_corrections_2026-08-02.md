# Claim Correction Sheet — 2026-08-02

Honesty patches tracked during the investigation. Status synced at closeout (D-196).

## Demote / banner

| Target | Correction | Status |
|--------|------------|--------|
| `STATUS.md` | Banner: “Frozen 2026-07-08; not authoritative after D-137. Use README + DECISIONS + IMPLEMENTATION_STATUS.” | **Applied** D-194 |
| `docs/architecture.md` | Banner: pre-D-156 action/A4; see D-158 and README A4 row. | **Applied** D-194 |
| `docs/l4_root_cause_verdict.md` | Banner: superseded by D-145/D-168 and `logs/benchmark_level4_30s.json` (37.83% FAIL). | **Applied** D-194 |
| `docs/action_selection.md` | Header warning + geometry-first default / blended opt-in body. | **Applied** D-194 (body restructured; keep header) |
| `docs/limitations.md` | Replace confidence≥0.6 hybrid with unconditional geometry default + opt-in. | **Applied** D-194 |
| `docs/phca_causal_evidence.md` | Align viewport tables to 30-seed; remove stale “run 10×10”; honesty pointer for secondary PE. | **Partial** — 30-seed tables already present; secondary PE honesty pointer added D-196 |
| `DOCUMENTATION_MAP.md` | Update DECISIONS range; stop calling STATUS source-of-truth for gates. | **Partial** — STATUS marked frozen/historical; some “cite STATUS” lines remain as residual wording debt |

## Live wording tightenings

| Target | Correction | Status |
|--------|------------|--------|
| `IMPLEMENTATION_STATUS.md` | Where D-156 0.03% collapse is cited, always add D-159 RBTA-artifact caveat inline. | **Applied** D-194 |
| Observatory `session_report.py` | Treat `pure_geometry_ablation` ≥50% like `task_lock_planner` for limitations flags. | **Applied** (`pure_geometry_ablation_dominant`, `geometry_dominated_action_selection`) |
| Cycle docstring | Align “15-step” class doc with module header “12-step / labels 0–19.” | **Applied** D-194 |
| `_select_action` comment ~1714 | Remove “prediction-scored path always runs” — false under default. | **Applied** D-194 |

## Metric labeling (eval docs / README)

| Metric | Must be labeled as | Status |
|--------|-------------------|--------|
| GridWorld goal_rate (default) | Planner competence under geometry (+ seed variance), not G′-control | LIVE guidance |
| Φ-IQ L2 under default | Contaminated by planner GC (`action_selection_interpretation`) | **Mitigated** D-194 addendum |
| L4 forgetting_rate | Goal-rate retention under geometry; prefer `eval_prediction_error` / `mean_eval_prediction_error` | **Mitigated** D-194 |
| Causal gate PASS | Beats random + greedy_observed on ≥75% metrics — not proof of prediction-primary; see gate `secondary_prediction` (PE dual-report, not gated) | **Mitigated** D-195 addendum + D-196 pointer |
| CI L0 Φ-IQ PASS | Smoke / regression floor only | **Mitigated** D-194 |

## Residual wording debt (not blocking)

- Deeper polish of `DOCUMENTATION_MAP.md` citation paths that still mention STATUS for gates.
- Optional fuller rewrite of demoted subsidiary bodies beyond banners (out of closeout scope).

## Do not correct into false negatives

- Continuous MPC **is** prediction-primary (S4: ~7 predicts/select) — keep that claim.
- G′ still learns on MLP path (PE early→late improves when learn on) — do not claim “nothing learns.”
- Viewport causal PASS is real vs greedy_observed — but still UNKNOWN how much is frontier/geometry vs G′ (D-195 H2: under default geometry, G′ is off the action path).
