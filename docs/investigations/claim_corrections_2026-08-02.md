# Claim Correction Sheet — 2026-08-02

Suggested honesty patches. **Not applied in this investigation pass** — list only.

## Demote / banner

| Target | Correction |
|--------|------------|
| `STATUS.md` | Add banner: “Frozen 2026-07-08; not authoritative after D-137. Use README + DECISIONS + IMPLEMENTATION_STATUS.” |
| `docs/architecture.md` | Banner: pre-D-156 action/A4; see D-158 and README A4 row. Fix Step-9 language away from “prediction-primary default.” |
| `docs/l4_root_cause_verdict.md` | Banner: superseded by D-145/D-168 and `logs/benchmark_level4_30s.json` (37.83% FAIL). |
| `docs/action_selection.md` | Keep header warning; rewrite body so default path is geometry-first; move blended to “opt-in path.” |
| `docs/limitations.md` | Replace confidence≥0.6 hybrid narrative with unconditional geometry default + agreement-gated opt-in. |
| `docs/phca_causal_evidence.md` | Align viewport tables to 30-seed artifacts; remove stale “run 10×10” recommendation. |
| `DOCUMENTATION_MAP.md` | Update DECISIONS range to current; stop calling STATUS source-of-truth for gates. |

## Live wording tightenings

| Target | Correction |
|--------|------------|
| `IMPLEMENTATION_STATUS.md` | Where D-156 0.03% collapse is cited, always add D-159 RBTA-artifact caveat inline. |
| Observatory `session_report.py` | Treat `pure_geometry_ablation` ≥50% like `task_lock_planner` for `expected_limitations` / `goal_success_can_mask_prediction_path_quality`. **Applied 2026-08-02** (`pure_geometry_ablation_dominant`, `geometry_dominated_action_selection`). |
| Cycle docstring | Align “15-step” class doc with module header “12-step / labels 0–19.” |
| `_select_action` comment ~1714 | Remove “prediction-scored path always runs” — false under default. |

## Metric labeling (eval docs / README)

| Metric | Must be labeled as |
|--------|-------------------|
| GridWorld goal_rate (default) | Planner competence under geometry (+ seed variance), not G′-control |
| Φ-IQ L2 under default | Contaminated by planner GC |
| L4 forgetting_rate | Goal-rate retention under geometry; prefer `eval_prediction_error` for model retention |
| Causal gate PASS | Beats random + greedy_observed on ≥75% metrics — not proof of prediction-primary |
| CI L0 Φ-IQ PASS | Smoke / regression floor only |

## Do not correct into false negatives

- Continuous MPC **is** prediction-primary (S4: ~7 predicts/select) — keep that claim.
- G′ still learns on MLP path (PE early→late improves when learn on) — do not claim “nothing learns.”
- Viewport causal PASS is real vs greedy_observed — but still UNKNOWN how much is frontier/geometry vs G′.
