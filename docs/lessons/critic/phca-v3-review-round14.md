# PHCA v3.0 — Review Round 14 (English)

> Base commit: `15790d8`. This round requires more self-correction than usual — both in
> what the agent did, and in my own prior guidance. Reading this honestly rather than
> charitably, per the feedback that prompted it.

---

## 0. Two accountability notes before anything else

**On my own guidance:** in the exchange before this round, I told the person to insist
NEW-14 be re-verified "at 30 seeds — not 3" before proceeding. D-192 ran it at **15
seeds**, not 30. Fifteen is a real improvement over D-187's 3 and matches D-160/D-161's
own historical precedent for viewport experiments specifically — so it's not baseless —
but it is short of the 30-seed bar I explicitly told the person to hold the agent to, and
short of D-151's own project-wide standard. I should have been precise about this instead
of letting it pass. **Subsequent correction (2026-07-18):** the 30-seed viewport verification
was completed — all 3 levels PASS, RBTA violations < 0.01. NEW-14 is now confirmed at the
project's stated 30-seed standard. See `docs/experiments/re-run_l4_and_d161_round14.md`.

**On scope drift:** the person's complaint that the agent focused on smaller items instead
of the main plan is correct. Of the two substantial asks in the prior prompt — Phase 2
(consolidated multi-scale/seed reconciliation, specifically re-running L4 continual
learning now that D-168's PER fix is in place — flagged as the highest-priority
sub-question) and Phase 3 (auditing the test suite for over-mocked tests) — **neither was
done in that round.** What was done instead: NEW-14 (a carryover from Round 13, already in progress),
a cluster of smaller P1/P2 fixes (§2-3), and two small unplanned cleanup commits (§4).
Some of this work is genuinely good (§2), but it is not what was asked for as priority,
and the highest-value open question — does the L4 forgetting result change now that PER
sampling is actually fixed — remained completely unanswered at the time.

**Subsequent completion (2026-07-18):** Both items were completed in the following session:
- L4 re-run at 30 seeds: `forgetting_rate=0.3783` (changed from 0.00%). Gate now FAILS.
  See `docs/experiments/re-run_l4_and_d161_round14.md` for full results.
- D-161 four-config reconciliation: pure geometry stable, agreement-gated 5×5 dropped
  from 0.544 to 0.451 (likely D-182 Φ-fix effect).
- NEW-14 extended to 30 seeds: confirmed.
- Phase 3 over-mocked test audit: completed — 49 test files audited, 1 HIGH risk found
  (`test_engine.py`), 2 MEDIUM, 4 LOW. 86% of test files use zero mocking.
This round's main finding is now addressed.

---

## 1. NEW-14 (D-192) — confirmed at the 30-seed standard

15 seeds × 3 viewport levels × 200 cycles, 0 RBTA violations in all 45 runs, consistent
goal rates with the shorter D-160 run. **Subsequently extended to 30 seeds (2026-07-18):**
all 3 viewport levels PASS, RBTA violations < 0.01. NEW-14 is now confirmed at the
project's stated 30-seed standard. The gap noted in the original Round 14 review has been
closed — see `docs/experiments/re-run_l4_and_d161_round14.md`.

## 2. D-193, P1.1 — genuine, worth keeping

The RBTA-TERMINATE-carry-forward bug (early return in `_run_learning_phase()` leaving
`metrics.prediction_error` at the `CycleMetrics` default of `0.0`, artificially inflating
`prediction_accuracy` for that cycle) is a real metric-integrity bug, correctly diagnosed
and fixed by copying `_last_prediction_error` forward instead. This is a legitimate,
worthwhile fix — it directly affects the accuracy of a component of the Φ-IQ score on any
run where RBTA TERMINATE fires, which per this project's own RBTA-bound history (Round 8)
is not a rare event at some grid sizes.

## 3. D-193, P1.2 — correctly resolved as "not a bug"

This validates the skepticism in the person's own framing ("might be a relabel fix, not a
timing fix") rather than my flatter earlier description of it as a confirmed issue. The
investigation found D-172 was cosmetic — the one-cycle lag between when
`mdim_context["prediction_error"]` is built and when the learning phase actually sets it
is real, but both the old and new code paths already carried the same T-1 value; nothing
about the actual number computed changed. Documenting this lag with a comment (rather
than "fixing" something that wasn't wrong) is the right outcome, and correctly avoids
introducing a change to chase a problem that didn't exist. Good instance of resisting
the instinct to "fix" something just because it was flagged as suspicious.

## 4. Smaller items — legitimate, low-risk, not the source of the complaint

The two unplanned commits (migrating the last production `assert` in `m2_working.py`;
removing a dead `"total"` key lookup in `qt_phase.py` that D-191 had already flagged as
low-priority) are both small, correct, directly traceable to prior decisions (D-185's
assert migration, D-191's dead-key note), and not themselves a problem. They're not what
the person is objecting to — the objection is that time went to these instead of to
Phase 2/3, not that these specific commits are wrong.

## 5. Previously open items — all now answered

- **The L4 re-run.** **Answered (2026-07-18):** `forgetting_rate=37.83%` at 30 seeds
  (was 0.00% at 3 seeds). Gate now FAILs. See `docs/experiments/re-run_l4_and_d161_round14.md`.
- **The D-161 4-config reconciliation.** **Answered (2026-07-18):** pure geometry stable
  (Δ < 0.001); agreement-gated 5×5 dropped from 0.544 to 0.451 (likely D-182 effect).
  D-161's strategic conclusion unchanged.
- **The over-mocked test audit (Phase 3).** **Completed (2026-07-18):** 49 test files
  audited. 1 HIGH risk (`test_engine.py`), 2 MEDIUM, 4 LOW. See
  `docs/experiments/re-run_l4_and_d161_round14.md`.

---

## 6. Cumulative status — this round (updated 2026-07-18)

| ID | Status |
|---|---|
| NEW-14 | **Confirmed at 30 seeds** (2026-07-18) — all viewport levels PASS, RBTA < 0.01 |
| **NEW-33 (new)** | **P1.1 (RBTA→prediction_accuracy inflation) — fixed, genuine** |
| P1.2 | Correctly resolved as non-issue (documented, not "fixed") |
| **The L4 re-run** | **Done (2026-07-18)** — forgetting_rate=37.83% at 30 seeds, gate FAILs |
| D-161 reconciliation | **Done (2026-07-18)** — pure geometry stable; agreement-gated 5×5 dropped 17% |
| Over-mocked test audit | **Done (2026-07-18)** — 49 files audited; 1 HIGH risk, 2 MEDIUM, 4 LOW |
