# PHCA v3.0 — Review Round 8 (English)

> Base commit: `39021de` ("round-39"). Covers D-159 and D-160, spanning what the commit
> history shows as roughly 8 rounds of work (32-39) condensed into 2 decision-log entries.
> This round contains a finding that requires me to partially revise my own Round 6/7
> conclusions — see §1, which I'm leading with rather than burying, in the same spirit
> this project applies to its own corrections.

---

## 1. The headline finding: the D-156 "catastrophic collapse" was misdiagnosed

This is the most important thing in this round, and it affects how several of my own
prior findings should be read, so I want to be precise about it rather than gloss over it.

**What D-156 (Round 6) reported:** at 10×10, the blended (learned-model-scored) action
selector collapsed to 0.03% goal-reaching — worse than random — while pure geometry
achieved 26.8%. This became the basis for defaulting to pure geometry, for the D-157/158
confidence-gating effort, for my own Round 7 deep-dive into *why* the confidence signal
was saturated, and for D-158's broader philosophical reframing of A4 as
environment-scoped.

**What D-159's addendum found, several weeks of work later:** a completely unrelated bug.
`gprime_stress_bounds()` returned a fixed `B_time=0.080s` regardless of grid size,
which did not scale the way the build-time bounds did. At 10×10, this mismatch caused
**99% of cycles to register an RBTA time violation**, which (per the TERMINATE path
described all the way back in Round 1 of this review) forces the agent into a safe
STAY-equivalent action almost every cycle — **regardless of which action-selection
strategy is in use.** When this bound bug was fixed and the exact D-156 configuration
(ungated blended scorer, 10×10, MLP) was re-run, goal rate was **77.8%**, not 0.03%.

**Let me state plainly what this means:** the original 0.03% number was not measuring
"the learned model makes catastrophically bad navigation decisions." It was measuring
"an agent that gets force-stopped by a resource-bound bug on 99% of cycles cannot reach
a goal, no matter what action-selection logic is choosing among the 1% of cycles it's
allowed to act freely." The blended scorer was, in a real sense, innocent of the crime
it was convicted of in D-156.

### What still holds, and what needs revision

To be fair to the intervening work, not everything built on top of D-156 turns out to be
wrong:
- **Pure geometry is still measurably better than ungated blended** even after the fix
  (97.1% vs 77.8%, per D-159's own follow-up ablation, NEW-09 table). So the *qualitative*
  conclusion — geometry is currently the more reliable default — still holds. It's the
  *magnitude* and the *causal story* that were wrong, not the direction.
- **Agreement-based gating is a genuine improvement** or the underlying signal problem I
  found in Round 7 (one-step confidence saturation) — that finding is independently
  verified and unaffected by the RBTA-bound-bug revelation. Agreement gating recovers
  goal rate to 94.1%, closing most (not all) of the gap to pure geometry.
- **What needs to be walked back:** the philosophical weight placed on "why does
  prediction-driven navigation fail so badly in GridWorld" (my Round 6/7 analysis, and
  D-158's reframing) was partly answering a question built on a confound. The honest
  version of the question is now smaller: prediction-driven navigation is *somewhat*
  worse than geometry (77.8% vs 97.1%), not catastrophically broken — and D-158's
  "inductive bias" language, while I'd already flagged it as doing more rhetorical work
  than the evidence supported (Round 7, §3), turns out to have been even further ahead of
  the evidence than I'd realized at the time, since some of that evidence was itself
  wrong.

**A note on my own accountability here, since this review has held the project to this
standard throughout:** I did not catch the RBTA-bound-bug possibility in Rounds 6 or 7,
despite spending real effort reproducing the confidence-saturation finding by running the
code myself. In hindsight, a 99% RBTA violation rate would have been visible if I'd
logged `rbta_violation_rate` alongside confidence in my Round 7 reproduction — I logged
confidence and threshold, not violations. That's a gap in my own diagnostic thoroughness,
not just the project's. I'm noting it explicitly because the alternative — quietly
updating my framework without acknowledging the miss — would be exactly the kind of
undisclosed revision this review exists to catch when the project does it.

---

## 2. Verification of the fixes in this round

### D-159, item 1 — Agreement-based gating

```python
self._agreement_buffer: list = []  # 1.0 if blended-scorer agreed with geometry, else 0.0
...
if len(self._agreement_buffer) >= window and agreement_rate < self.interventions.agreement_threshold:
    ...  # trigger fallback to pure geometry
```
Confirmed present and wired as described: default window 30 cycles, threshold 0.3. This
is a materially better signal than one-step confidence (Round 7's NEW-10) because it's
measuring the thing that actually matters for the failure mode — sustained disagreement
between the two policies — rather than a proxy (one-step accuracy) that turned out to be
uninformative. Good fix, independent of §1's revelation.

### D-159, item 4 — F-03 (entropy floor for 9 modules) resolved as an honest no-op

```python
"ASI": ResourceBounds(..., entropy_floor=0.0),
"WM": ResourceBounds(..., entropy_floor=0.0),
... (7 more modules)
```
This is worth being precise about what it does and doesn't fix. It does **not** add real
entropy measurement for these 9 modules — that capability gap (flagged since Round 1)
remains. What it does do: previously, these modules had a nonzero floor (0.01) being
compared against a hardcoded placeholder value (0.1) that could never trigger a
violation — an *appearance* of enforcement that was actually vacuous. Setting the floor
to 0.0 makes the non-enforcement explicit and mathematically unambiguous (entropy ≥ 0
always, so a 0.0 floor can never be violated, by construction, not by placeholder luck).
**This is a legitimate resolution in the same spirit as D-144's RBTA simplification** —
replacing a misleading appearance of rigor with an honest, visible no-op — but it should
be described that way (A3 enforcement narrowed to 3 modules, by decision, not fixed
system-wide) rather than as "F-03 fixed" without qualification. The README's A3
description doesn't yet reflect this narrowing explicitly; worth a one-line addition.

### D-159 addendum — RBTA bound scaling fix, independently verified

```python
def estimate_mlp_gprime_time_bound(state_dim, base_time=0.080, ref_dim=REF_STATE_DIM):
    return base_time * max(1.0, state_dim / ref_dim)

def grid_floor(state_dim, ref_dim=REF_STATE_DIM, base_floor=0.01):
    return base_floor / max(1.0, state_dim / ref_dim)
```
Both confirmed present and match the decision log's description (time bound scales up
with grid size, entropy floor scales down since larger-grid MLPs show lower measured
MC-dropout mutual information for legitimate reasons, not degraded knowledge). This is a
sound fix to a real, previously-undetected miscalibration.

---

## 3. D-160 — Partial observability (viewport) is a substantial, well-documented new feature

This is new scope, not a fix to a prior finding, so I evaluated it fresh rather than
against an existing critique.

**What it does:** adds a genuine limited-viewport mode to the core `GridWorld` (previously
partial observability only existed in a benchmark-script-level wrapper, `ScenarioGridWorld`).
The agent now only "knows" about walls and the goal within a configurable Manhattan-distance
radius, tracked cumulatively as it explores (`_known_walls`, `_observed_cells`). When the
goal hasn't been seen, `get_goal_position()` returns `None`, and a new frontier-exploration
heuristic (`_find_frontier_cell`) gives the agent a proxy objective (nearest unexplored
cell) instead of leaving it with no directional signal at all.

**What I'd flag as genuinely good practice here, worth naming explicitly:** the decision
log's own "Open issues" section is a model of the kind of honesty this review has been
asking for throughout — it lists, unprompted, that (a) there are now two separate partial-
observability implementations that should eventually be unified, (b) the standalone BFS
planner bypasses the new `observed_grid` abstraction in at least one code path, (c) the
frontier heuristic is a reasonable-but-not-optimal choice, and (d) the validating
experiment used a small sample (15 seeds, one grid size). None of this was something I
had to dig for — it was disclosed in the same entry that reports the positive result.
That is exactly the reporting standard the rest of this review has been checking for, and
it's worth reinforcing when it happens well, not just when it doesn't.

**One thing worth independently verifying in a future round, since it wasn't stated
explicitly:** the RBTA violation rates reported for the viewport experiments (9-12%,
"dropped from ~60%") are still nonzero and notably higher than the 2.6-2.7% seen in the
non-viewport NEW-09 ablation table in §1. Given this round's central lesson — that RBTA
violation rate can silently dominate a goal-rate outcome — it would be worth confirming
these remaining 9-12% violations are benign (e.g., occasional INTERRUPT, not masking a
second confound) before trusting the viewport goal-rate numbers as cleanly as the
now-corrected 10×10 L2 numbers.

---

## 4. Minor items

- **Committed `__pycache__` directories.** This pull included compiled `.pyc` files under
  `python/environments/__pycache__/` and several `python/phca/*/__pycache__/` paths,
  later partially cleaned up in a merge-conflict resolution commit. This is ordinary repo
  hygiene debris (a missing or incomplete `.gitignore` entry), not an architectural
  concern, but worth a one-line fix (`__pycache__/` and `*.pyc` in `.gitignore`) so it
  doesn't recur and bloat the repository over time.
- **README currency, checked directly this round:** the A4 table row *was* updated to
  include the D-159 agreement-gating fix — good, this closes the lag pattern flagged in
  Rounds 6 and 7 for this specific row. It does not yet mention the §1 finding (the RBTA
  bound bug as the real root cause of D-156's number), which is reasonable given how
  recently that was found, but should be added in the next documentation pass so a
  reader doesn't come away thinking "blended scorer collapses catastrophically" is still
  the operative finding.

---

## 5. Cumulative status table — all findings, eight rounds

| ID | Description | Found | Status |
|---|---|---|---|
| F-01, F-02/NEW-01, F-04, F-05, F-06, F-08, NEW-02, NEW-03 | (see prior rounds) | R1-R4 | Fixed |
| F-03 | Entropy floor placeholder for 9 modules | R1 | Resolved as honest no-op (§2) — real measurement still absent, now clearly labeled rather than implied |
| NEW-04 | Causal gate fails at 30 seeds | R4 | Substantially reframed by §1 — much of the "fail" was RBTA-bound-driven, not a genuine navigation-quality gap |
| NEW-05, NEW-06 | Blended-scorer trade-off; RBTA bound justification | R5 | Still directionally valid; magnitudes need re-reading in light of §1 |
| NEW-07 | 10×10 blended-scorer collapse (0.03%) | R6 | **Retracted as stated** — was an RBTA-bound artifact, not a scorer failure (§1) |
| NEW-08 | README lag | R6 | Recurred and was fixed again this round for the A4 row specifically |
| NEW-09 | Negative-feedback-loop hypothesis under-isolated | R6 | **Now tested directly** (D-159 addendum): the loop exists but is a second-order effect, not the primary cause of D-156's number |
| NEW-10 | Confidence-gating signal saturated/uninformative | R7 | Fixed — replaced with agreement-based gating (§2) |
| NEW-11 | "Inductive bias" framing risk | R7 | Partially materialized and partially resolved — README now carries the NEW-10 caveat, but not yet the §1 caveat |
| NEW-12 | Threshold doc/code mismatch | R7 | Fixed — code now matches D-158's stated 0.9 |
| **NEW-13 (new)** | **README A4 row does not yet reflect that D-156's headline number was an RBTA-bound artifact, not a genuine scorer failure** | R8 | **Open — the natural next documentation fix** |
| **NEW-14 (new)** | **Viewport (D-160) RBTA violation rate (9-12%) not yet independently confirmed benign, given this round's lesson about violation-driven confounds** | R8 | **Open — cheap to check, high value given §1** |
| **NEW-15 (new, minor)** | **Committed `__pycache__` artifacts** | R8 | **Open — trivial `.gitignore` fix** |

---

## 6. What this round changes about how to read the whole review

If there's one thing worth carrying forward from this round specifically, it's a
methodological one, not a finding about PHCA: **when a headline number changes
dramatically between two configurations, checking the resource-enforcement layer
(RBTA violation rate, in this project) should be one of the first things verified,
before building a causal story about the *decision-making* layer.** A safety/enforcement
mechanism that can silently override behavior on 99% of cycles is, almost by definition,
capable of producing results that look like they're about the mechanism being overridden
rather than about the enforcement layer itself. This project's own architecture (RBTA as
a hard override, not a log-only warning — a design choice praised as a strength as early
as Round 1 of this review) is exactly what made this confound possible in the first
place. That's not a contradiction — a safety mechanism that can actually change behavior
is more valuable than one that can't, precisely because it's real enough to also be a
source of surprising results when miscalibrated. It just means every future benchmark
number in this project (and arguably in any project with a similarly "live" enforcement
layer) should be reported alongside its violation rate as a matter of course, not as an
afterthought.

## 7. Recommendations

1. Add the §1 caveat to the README's A4 row and to `docs/phca_causal_evidence.md`
   (NEW-13) — cheap, and the most important documentation gap left standing.
2. Independently confirm the viewport experiments' 9-12% RBTA violation rate is benign
   (NEW-14) before those numbers are cited elsewhere as cleanly as the corrected 10×10
   L2 numbers.
3. Add `.gitignore` entries for `__pycache__/` and `*.pyc` (NEW-15).
4. Going forward, adopt "report violation rate alongside every headline goal-rate number"
   as a standing practice (§6) — this would have caught D-156's confound at the source
   rather than after several rounds of downstream analysis.
5. Unify the two partial-observability implementations (`ScenarioGridWorld`'s
   `partial_map` vs. the new core `partial_obs_radius`), as D-160 already flags in its
   own open-issues list.
