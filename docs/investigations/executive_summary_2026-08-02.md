# Investigation Executive Summary — 2026-08-02

## Verdict

PHCA v3 under **default discrete GridWorld config** is a **geometry-primary controller** with a learning/regulation stack that still runs. Continuous MuJoCo remains **prediction-primary (MPC)**. Several headline gates (L4 forgetting, Φ-IQ L2 narrative, causal “PASS” at some scales) are easy to over-read as evidence of prediction-driven cognition.

## What actually decides / learns / binds

| Concern | Default GridWorld | Continuous MuJoCo |
|---------|-------------------|-------------------|
| Action | BFS/Manhattan (`pure_geometry_ablation`) | G′-scored MPC |
| G′ role | Predict + learn (side channel); **0** select-time predicts | ~7 predicts/select |
| TSPL / M3 replay | No-op on discrete graph (honesty skip) | Wired on MLP |
| RBTA | Can STAY / truncate / skip learn | Same |
| Trustworthy metric | PEU / eval PE / causal `secondary_prediction` (model fit) | Error improvement + violations |
| Contaminated metric | goal_rate, Φ-IQ L2 GC, L4 FR, causal scenario PASS under geometry | Less so |

## Mitigated / closed this arc (D-194+)

1. **G1-INV-04** — L4 vacuous PASS blocked (coverage gate).
2. **G5-INV-13** — Async RBTA carry applied; dead PerceptionFrame queue removed.
3. **G3-INV-07 / H-04** — Observatory flags geometry-dominated sessions.
4. **G5-INV-08/09** — Discrete G′ coverage warn; TSPL/M3 theater skipped on discrete graph.
5. **G3-INV-02 / G3-INV-06 / G3-INV-20** — Φ-IQ / CI L0 / A4 scope labeling.
6. **G4-INV-12** — Mean-per-seed FR reported alongside aggregate.
7. **G5-INV-01** — Default geometry locked by tests (documented, not “fixed” into blended).

See [hardening_backlog_2026-08-02.md](hardening_backlog_2026-08-02.md).

## Residual open

1. **G2-INV-05** — Causal mixed L2/L3 **behavior** still open. Honesty + diagnosis done (D-195); secondary PE dual-report on gate (not gated). No default flip.
2. **G4-INV-03** — L4 FR still confounded by geometry despite dual PE labeling.
3. **G0-INV-18** — Blueprint backlog (M5/M6/VSA/L5) catalog only.
4. **G2-INV-19** — Historical Φ-IQ not re-run at full power (documented).

## Causal diagnosis (D-195) + overnight confirm (D-197)

Short diagnostic (5×100) then **30×200 overnight** (`logs/overnight_20260802_103502/`):

- **H2 hard:** learn-off ≡ geometry on L2 scenario metrics; PE diverges sharply (~3.4 → ~23).
- **H3 at power:** blended **hurts** L2 on 5×5 and 10×10; no default flip.
- **H1:** rejected at power on 5×5 (loses goal_rate); partial only on 10×10 geometry.
- Pattern: 5×5 L2 FAIL / 10×10 L2 PASS / L3 FAIL under geometry.
- Gate exposes `secondary_prediction` / `prediction_error_mean` (report-only).

Notes: [causal_diagnosis_g2inv05_2026-08-02.md](causal_diagnosis_g2inv05_2026-08-02.md),
[overnight_analysis_2026-08-02.md](overnight_analysis_2026-08-02.md). Docs honesty sync: **D-198**.

## Deliverables

See [README.md](README.md). Machine probes in `logs/investigation_*.json` and diagnosis JSON above. Tests: `python/phca/core/tests/test_action_selection_defaults.py`, `python/tests/test_causal_eval.py`.

## Still out of scope

- New 30-seed causal/L4/Φ-IQ re-gates as primary deliverable.
- Forcing blended scorer as default / expanding discrete graph beyond 10 dims.
- “Fixing” PHCA to beat greedy_observed without new evidence.
- Blueprint features (M5/M6/VSA/L5).
