# Investigation Executive Summary — 2026-08-02

## Verdict

PHCA v3 under **default discrete GridWorld config** is a **geometry-primary controller** with a learning/regulation stack that still runs. Continuous MuJoCo remains **prediction-primary (MPC)**. Several headline gates (L4 forgetting, Φ-IQ L2 narrative, causal “PASS” at some scales) are easy to over-read as evidence of prediction-driven cognition.

## What actually decides / learns / binds

| Concern | Default GridWorld | Continuous MuJoCo |
|---------|-------------------|-------------------|
| Action | BFS/Manhattan (`pure_geometry_ablation`) | G′-scored MPC |
| G′ role | Predict + learn (side channel); **0** select-time predicts | ~7 predicts/select |
| TSPL / M3 replay | No-op on discrete graph | Wired on MLP |
| RBTA | Can STAY / truncate / skip learn | Same |
| Trustworthy metric | PEU / eval PE (model fit) | Error improvement + violations |
| Contaminated metric | goal_rate, Φ-IQ L2 GC, L4 FR | Less so |

## Hottest confirmed gaps

1. **G5-INV-01** — Default action = geometry (locked by test).
2. **G1-INV-04** — L4 vacuous PASS (seed 1542, 8/10 tasks excluded).
3. **G4-INV-03 / G4-INV-12** — L4 FR confounded; aggregate ≠ mean per-seed.
4. **G5-INV-08/09** — Discrete G′ 10-dim cap; TSPL/M3 replay absent.
5. **G3-INV-07** — Observatory now flags geometry dominance (**H-04 done**).
6. **G5-INV-13** — Async carry/staleness incomplete.

## Deliverables

See [README.md](README.md). Machine probes in `logs/investigation_*.json`. Tests: `python/phca/core/tests/test_action_selection_defaults.py`.

## Remediation follow-up (D-194, Waves 1–3)

Implemented: L4 vacuous-PASS guard + dual PE; doc demotions; discrete G′ coverage/TSPL honesty;
async carry + entropy_na; ASI stale safe-mode; M3 fallback flag; recovery MVP label.
See [hardening_backlog_2026-08-02.md](hardening_backlog_2026-08-02.md).

## Still out of scope

- New 30-seed causal/L4 reruns.
- Forcing blended scorer as default / expanding discrete graph beyond 10 dims.
- Blueprint features (M5/M6/VSA/L5).
