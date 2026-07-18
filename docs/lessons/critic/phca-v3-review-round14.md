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
of letting it pass. See §1 for whether it matters here.

**On scope drift:** the person's complaint that the agent focused on smaller items instead
of the main plan is correct. Of the two substantial asks in the prior prompt — Phase 2
(consolidated multi-scale/seed reconciliation, specifically re-running L4 continual
learning now that D-168's PER fix is in place — flagged as the highest-priority
sub-question) and Phase 3 (auditing the test suite for over-mocked tests) — **neither was
done.** What was done instead: NEW-14 (a carryover from Round 13, already in progress),
a cluster of smaller P1/P2 fixes (§2-3), and two small unplanned cleanup commits (§4).
Some of this work is genuinely good (§2), but it is not what was asked for as priority,
and the highest-value open question — does the L4 forgetting result change now that PER
sampling is actually fixed — remains completely unanswered. That is the main finding of
this round, more than any individual code issue.

---

## 1. NEW-14 (D-192) — real progress, but not yet at the stated standard

15 seeds × 3 viewport levels × 200 cycles, 0 RBTA violations in all 45 runs, consistent
goal rates with the shorter D-160 run. This is a believable, well-instrumented result and
a genuine improvement over D-187's 3-seed number. **It should not yet be described as
fully closing NEW-14 per this project's own D-151 standard** — the honest status is
"strong evidence at 15 seeds, not yet verified at the project's own stated 30-seed bar."
This is a small gap, cheap to close (rerun at 30 seeds, most of the work is already
built), and worth closing precisely *because* this project has specifically burned a
review cycle before (D-151 itself) on a result that looked solid at low seed count and
flipped at 30.

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

## 5. What is still completely open — and is the real priority

- **The L4 re-run.** D-168 fixed PER to sample prior tasks instead of the current one.
  Nobody has yet re-run the Level-4-lite continual-learning benchmark with this fix
  active to see whether the forgetting_rate number changes, or whether it's still
  dominated by the geometric fallback per the original F-06 finding. This is, by a wide
  margin, the single most scientifically important open question in the entire project
  right now, and it has been open since Round 12.
- **The D-161 4-config reconciliation.** Roughly 25 decision entries have landed since
  D-161 (including D-182's Φ fix, which changes whether the PID controller and MDIM's D2
  drive receive a live vs. frozen signal — plausibly relevant to D-161's own numbers).
  Nobody has confirmed D-161's comparison table still holds.
- **The over-mocked test audit (Phase 3).** Not started.

---

## 6. Cumulative status — this round

| ID | Status |
|---|---|
| NEW-14 | Improved (15 seeds), **not yet at the 30-seed standard** — cheap to close |
| **NEW-33 (new)** | **P1.1 (RBTA→prediction_accuracy inflation) — fixed, genuine** |
| P1.2 | Correctly resolved as non-issue (documented, not "fixed") |
| **The L4 re-run** | **Still not done — highest priority, unchanged since Round 12's ask** |
| D-161 reconciliation | Still not done |
| Over-mocked test audit | Still not done |
