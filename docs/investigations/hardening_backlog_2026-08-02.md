# Hardening Backlog — 2026-08-02

Prioritized surgical fixes. Separate **cognitive-cycle** from **Observatory** and **blueprint-missing**.

## P0 — False confidence / claim integrity

| Item | Area | Action | Gap IDs | Status |
|------|------|--------|---------|--------|
| H-01 | Eval | L4 vacuous-PASS guard when excluded >50% or <2 valid | G1-INV-04 | **DONE** D-194 |
| H-02 | Eval | Dual-report L4 goal FR + mean eval PE + coverage flags | G4-INV-03 | **DONE** D-194 |
| H-03 | Docs | Demote banners on STATUS / architecture / l4 / action_selection / … | claim_corrections | **DONE** D-194 |
| H-04 | Observatory | Flag `pure_geometry_ablation` dominant sessions | G3-INV-07 | **DONE** 2026-08-02 |
| H-05 | CI | Document L0 smoke / regression floor only | G3-INV-06 | **DONE** D-194 |

## P1 — Implemented-path integrity

| Item | Area | Action | Gap IDs | Status |
|------|------|--------|---------|--------|
| H-06 | World model | Surface discrete G′ dim coverage + warn <0.25 | G5-INV-08 | **DONE** D-194 |
| H-07 | Learning | Skip TSPL/M3 theater on discrete G′; effectiveness flags | G5-INV-09 | **DONE** D-194 |
| H-08 | Cycle | `effective_stage_order` = metadata only | G5-INV-10 | **DONE** D-194 |
| H-09 | Async | Apply RBTA carry in action loop; remove dead queue/age | G5-INV-13 | **DONE** D-194 |
| H-10 | RBTA | Entropy only for G′/MDIM/ATTN; others entropy_na | G3-INV-14 | **DONE** D-194 |
| H-11 | Tests | Lock default selector_mode + learn-off invariance | G5-INV-01 | **DONE** |
| H-16 | M1 | Document write-only trace buffer | G5-INV-11 | **DONE** D-194 |

## P2 — Runtime robustness

| Item | Area | Action | Gap IDs | Status |
|------|------|--------|---------|--------|
| H-12 | ASI | Stale-state anomaly + safe STAY after 3 failures | G5-INV-15 | **DONE** D-194 |
| H-13 | M3 | `fell_back_to_memory` + `m3_fallback_memory` event | G5-INV-17 | **DONE** D-194 |
| H-14 | Async | Remove dead PerceptionFrame scaffolding | G5-INV-13 | **DONE** D-194 |
| H-15 | Resilience | Recovery benchmark labeled injectable MVP | G3-INV-16 | **DONE** D-194 |

## P3 — Blueprint backlog (do not file as bugs)

| Item | Notes |
|------|-------|
| M5 procedural memory | TSPL compiles skills; no persistent library |
| M6 meta-memory | Absent |
| VSA dual ensemble | Absent |
| Grounding adapter L0/L2 | NoiseInjector is stress proxy only |
| Full HPM runtime | `compute_bounds()` only |
| Φ-IQ L5 | Not implemented |
| Full failure matrix A–F | Partial B1/B4/B5/C1/F5/E1 |

## Explicit non-goals this backlog

- Broad rewrite of action selection to force blended default (needs new evidence, D-161).
- Treating Observatory UX polish as cognitive competence work.
- Re-litigating whitepaper Φ-IQ L5 without L4 metric fix.
