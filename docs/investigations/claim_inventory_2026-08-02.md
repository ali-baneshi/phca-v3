# Claim Inventory Freeze — 2026-08-02

Trust order used: measured `logs/` / `results/` → `DECISIONS.md` (D-145+) → README / IMPLEMENTATION_STATUS / maturity_audit → subsidiary docs.

## Live claims (still authoritative)

| ID | Claim | Owner | Status | Evidence |
|----|-------|-------|--------|----------|
| C-A4 | A4 is environment-scoped: GridWorld discrete = geometry inductive bias; continuous MPC = prediction-primary | D-158, README A4 row | LIVE | `InterventionConfig.disable_blended_scorer=True`; probe `S1_default_*` / `S4_pendulum_mpc` |
| C-GEO | Default discrete action = pure BFS/Manhattan | D-156, D-161, interventions.py docstring | LIVE | `selector_mode=pure_geometry_ablation` 100% in default probes |
| C-BLEND | Blended scorer is opt-in; does not universally beat geometry post-RBTA fix | D-159, D-161 | LIVE | D-161 30-seed re-eval; README mixed table |
| C-RBTA-ART | D-156 0.03% “collapse” was RBTA bound-scaling artifact | D-159 addendum, README footnote | LIVE | Ungated blended ~77.8% after fix |
| C-L4 | L4-lite forgetting FAIL 37.83% @ 30 seeds; metric confounded | D-168, round-14, `logs/benchmark_level4_30s.json` | LIVE | `forgetting_rate=0.3783`, `passes_gate=false` |
| C-CAUSAL | Causal gate mixed; small-seed PASS overstated | D-151, D-161 | LIVE | 5×5 L2 FAIL; 10×10 L2 PASS; L3 FAIL; viewports PASS |
| C-A1A3A5 | A1–A3/A5 measured PASS with A3 scope limit (~3/12 modules real entropy) | assumption_validation, F-03, D-159 | LIVE | Probe T4: 9 modules hardcoded entropy 0.1 |
| C-OBS | Observatory Phases 7–20 complete ≠ cognitive maturity | IMPLEMENTATION_STATUS, maturation_signoff | LIVE | CI replay check is integrity-only |

## Demoted / historical (do not cite as current truth)

| Document | Why demoted | Superseded by |
|----------|-------------|---------------|
| `STATUS.md` | Frozen 2026-07-08 / D-137; misses D-145–D-168 | README, IMPLEMENTATION_STATUS, DECISIONS |
| `docs/architecture.md` | Pre-D-156 A4 / action-selection language; old Φ-IQ weights | README, D-147/D-158 |
| `docs/l4_root_cause_verdict.md` | Verdict 0% PASS via D-137 eval-position fix | D-145, D-168, L4 30s artifact |
| `docs/action_selection.md` body | Describes blended as primary path (header warns) | interventions.py + D-161 |
| `docs/limitations.md` hybrid≥0.6 section | Pre-D-156 confidence hybrid | D-156 default geometry |
| `docs/phca_causal_evidence.md` mixed 15/30 tables | Partial drift; “run 10×10” already done | D-161 + causal logs |
| `DOCUMENTATION_MAP.md` DECISIONS “through D-127” | Stale map | DECISIONS through D-190+ |

## Decision anchors (D-145 onward, investigation-relevant)

| Decision | One-line effect |
|----------|-----------------|
| D-145 | Removed train-end eval-position confound for L4 (restored random starts) |
| D-151 | 30-seed standard; causal L2/L3 FAIL at power |
| D-152 | RBTA bound recalibration / ACTION headroom |
| D-153–D-155 | Blended helps L1, hurts L3; pure geometry L3 ablation |
| D-156 | Default `disable_blended_scorer=True` (later causal story revised) |
| D-157 | Adaptive confidence gating (opt-in path) |
| D-158 | A4 environment-scoped reframing |
| D-159 | Agreement gating + RBTA artifact correction |
| D-161 | Post-fix 30-seed: geometry remains default; neither mode universal |
| D-168 | PER sampled current task only — anti-forgetting wire broken until fixed |
| D-190 | Cross-link D-168 ↔ F-06/D-145 |

## Probe artifact index

- `logs/investigation_probes_2026-08-02.json` — T1/T3/T5/T6 structural + runtime
- `logs/investigation_t2_analysis.json` — L4/causal artifact parse
- `logs/investigation_t3_t7_probes.json` — learn-off, entropy, Observatory flags
