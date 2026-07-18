# PHCA Maturation Sign-Off Checklist — 2026-07-07

Plan: **Maturation v2 — System-Wide Evidence Hardening**

## Track completion

| Track | Deliverable | Status |
|-------|-------------|--------|
| A | [`maturity_audit_2026-07-07.md`](maturity_audit_2026-07-07.md) ≥40 claim rows | Done |
| B | [`static_audit_2026-07-07.md`](static_audit_2026-07-07.md) + `test_static_contracts.py` | Done |
| C | `test_maturation.py` cognitive core + A1–A5 via nightly | Done (T2 assumption gate nightly) |
| D | M3/M4 maturation tests + replay wire | Done |
| E | L4 protocol + diagnostic + ablation script + verdict doc | Done |
| F | Resilience false-positive + injectable tests | Done |
| G | G4 separation tests + golden manifest | Done |
| H | Observability JSONL round-trip tests | Done |
| I | Validation checksum manifest | Done |
| J | Makefile maturation targets + bisection doc | Done |

## Sign-off checklist

- [x] Claim matrix: tier + owner track per major row
- [x] G5 hooks: M3 replay + observability export wired
- [x] G4 metrics: documented; tests prevent conflation
- [x] T1: `make maturation-test` (static + forgetting + resilience + maturation)
- [x] T0: L0 bench + observatory replay (existing CI)
- [ ] L4b PASS at 10 tasks × 30 seeds — **FAIL (37.83% forgetting_rate)** — honest limitation recorded (supersedes 3-seed result)
- [x] Recovery injectables: `make bench-recovery` (T3 local)
- [x] Bisection protocol documented
- [x] No new blueprint features (M5, VSA, EWC/GEM, L5)

## Gate promotion

| Gate | Promotion | Status |
|------|-----------|--------|
| maturation-test | T1 CI via `test-python` | Included |
| bench-recovery | T3 nightly soft (manual `make bench-recovery`) | Documented |
| bench-level4-ablation | T3 weekly local | Makefile target |
| L4b hard | T4 release | **Not promoted** (G2 open) |

## Next maturity increment (out of scope)

- Revisit D-020 only via explicit architecture decision
- Task-conditioned G′ heads or larger replay budget if L4b required
- Expand failure matrix beyond B1/B4/B5/C1/F5
