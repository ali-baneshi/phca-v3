# Gap Register — PHCA v3 Investigation 2026-08-02

Gap classes (maturity audit): **G0** no measurement | **G1** weak protocol | **G2** behavior fail | **G3** false confidence | **G4** metric conflation | **G5** silent no-op.

Updated **2026-08-02** after remediation Waves 1–3 (D-194) and causal diagnosis (D-195).

| ID | Class | Type | Sev | Finding | Status |
|----|-------|------|-----|---------|--------|
| G5-INV-01 | G5 | BUG | P0 | Default discrete = geometry | **Mitigated** (locked by tests; documented) |
| G3-INV-02 | G3 | EVAL | P0 | Φ-IQ L2 planner-dominated under default | **Mitigated** (`action_selection_interpretation` on Φ-IQ reports; metric formula unchanged) |
| G4-INV-03 | G4 | EVAL | P0 | L4 FR confounded by geometry | **Mitigated** (dual PE + labeling; confound remains) |
| G1-INV-04 | G1 | EVAL | P0 | L4 vacuous PASS (seed 1542) | **Closed** (coverage gate) |
| G2-INV-05 | G2 | EVAL | P0 | Causal mixed L2/L3 | **Open** (behavioral). Honesty mitigated. **Diagnosed (D-195):** H2 accepted (learn-off ≡ geometry); H3 accepted (blended moves metrics; hurts 5×5, mixed 10×10); H1 partial/budget-dependent. No default flip. See `causal_diagnosis_g2inv05_2026-08-02.md` |
| G3-INV-06 | G3 | EVAL | P0 | CI L0-only Φ-IQ | **Mitigated** (labeled smoke) |
| G3-INV-07 | G3 | DOC | P0 | Observatory vs cognition / geometry flags | **Closed** (session_report + docs) |
| G5-INV-08 | G5 | BUG | P1 | Discrete G′ 10-dim cap silent | **Mitigated** (metrics + warn; cap unchanged) |
| G5-INV-09 | G5 | BUG | P1 | TSPL/M3 no-op on discrete G′ | **Closed** (skip + flags) |
| G5-INV-10 | G5 | BUG | P1 | `effective_stage_order` unused | **Closed** (metadata docstring) |
| G5-INV-11 | G5 | BUG | P1 | M1 write-only | **Closed** (documented) |
| G4-INV-12 | G4 | EVAL | P1 | Aggregate vs mean per-seed FR | **Closed** (`mean_per_seed_forgetting_rate` + aggregation note) |
| G5-INV-13 | G5 | BUG | P1 | Async carry / dead queue / age | **Closed** |
| G3-INV-14 | G3 | BUG | P1 | Synthetic RBTA entropy 0.1 | **Closed** |
| G5-INV-15 | G5 | BUG | P2 | ASI stale-state silent continue | **Closed** (safe mode @ 3) |
| G3-INV-16 | G3 | EVAL | P2 | Resilience MVP ≠ full matrix | **Mitigated** (labeled) |
| G5-INV-17 | G5 | BUG | P2 | M3 silent :memory: fallback | **Closed** (flag + event) |
| G0-INV-18 | G0 | MISSING | P3 | Blueprint backlog | **Catalog** |
| G2-INV-19 | G2 | EVAL | P1 | Historical Φ-IQ not re-run | **Documented** |
| G3-INV-20 | G3 | EVAL | P1 | A4 assumption = MPC structural | **Mitigated** (`scope=continuous_mpc_pendulum_only` in assumption_validation) |

## Closed / not reproduced this pass

| Prior claim | Result |
|-------------|--------|
| Blended always collapses to ~0% at 10×10 | Superseded by D-159 (RBTA artifact) |
| L4 0% forgetting PASS | Superseded; 37.83% FAIL at 30 seeds |
| M3→G′ replay unwired | Wired for MLP; discrete guarded as ineffective (G5-INV-09) |
