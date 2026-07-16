# PHCA v3.0 — Review Round 11 (English)

> Base commit: `bf6954a` ("round-43"). This is a genuinely strong round — the two-phase
> prompt (independent architect pass, then compare against Round 10) appears to have
> worked as intended: two new, real findings (D-165, D-167) were caught independently,
> not just prompted by the review file. Verified all six new decision entries (D-162
> through D-167) directly against code.

---

## 1. NEW-19 (nightly always-PASS) — verified genuinely fixed

Read the new `nightly` Makefile target in full. It now uses a `FAILED=""` shell
accumulator, appends a step name on each failure (`FAILED="${FAILED} static-phi-iq"`,
etc.), and the final block is:
```makefile
if [ -n "$$FAILED" ]; then echo "  Nightly hardening: FAILED —$$FAILED"; exit 1; \
else echo "  Nightly hardening: ALL PASS"; fi
```
I sanity-checked this exact pattern in isolation (three dummy steps, one failing) and
confirmed it correctly reports the failing step names and exits 1. This is a real fix,
not a re-statement of the old one — D-162's rationale correctly diagnoses why the
previous attempt failed (`|| echo` making every line exit 0) and the new pattern doesn't
have that flaw. **Also worth noting as a quality improvement, not just a bug fix:** the
`nightly-mujoco` prerequisite was folded inline as step 2/7 with its own accumulator
entry, rather than being a separate `make` prerequisite target — this removes a second,
different way the recipe could previously have aborted early (a failing prerequisite
target aborts the whole `make` invocation before the main recipe's body even starts,
regardless of any `||` handling inside the body). That's a real, related class of bug
that wasn't explicitly called out as fixed but appears to have been fixed as a side
effect of the restructuring. Good.

## 2. D-167 — a genuine, independently-found bug, verified fixed

This is the most valuable finding of the round, and it was not something Round 10 asked
for — it came from the Phase 1-2 architect pass finding pattern C/D territory on its own
(a signal that gates a decision was being sampled in a way that biases it). Confirmed in
`cycle.py`: the probe buffer append (used to calibrate the confidence-gating threshold
from Round 7's NEW-10 fix) was previously filtered to only collect data when
`selector_mode` was `"pure_geometry_ablation"` or `"adaptive_geometry_fallback"` — i.e.,
only during fallback cycles. The fix removes that filter; probe data is now collected
every cycle regardless of which selector chose the action, which is the correct
population to calibrate a prediction-accuracy signal from (the signal is "how accurate
was G' about the transition that actually happened," which has nothing to do with which
policy picked the action).

**Correctly scoped by the entry itself:** D-167 honestly notes this bug was "latent" —
inert under the current default (`disable_blended_scorer=True`, so those two selector
modes are the only ones that ever run, meaning the old filter was accidentally
equivalent to "collect everything" under current settings). This is good, precise
severity-scoping, consistent with this project's better decision-log entries (D-161 is
still the reference standard, and this one matches it). It correctly flags that the bug
*would* have mattered the moment someone flipped the default to test the blended
scorer — exactly the kind of forward-looking caveat that prevents a future regression
from being a surprise.

## 3. D-165 — verified implemented, just under a different name than I initially checked for

My first pass grepped literally for `warn_once` and found nothing, which would have been
a real gap (a decision entry describing something that wasn't actually built). On a
closer look, the behavior is implemented via a `self._warned_estimated_bounds` boolean
guard inside `_collect_runtime_log()`, logging `rbta.estimated_bounds` once with the
detail `energy=runtime×50 memory=formulaic entropy=hardcoded_0.1_for_9_of_12_modules`.
This is the correct behavior (log once, not every cycle) even though the identifier
doesn't literally match the entry's shorthand description — worth a one-line correction
in the entry text (`warn_once` → the actual guard-flag name) purely for future
grep-ability, but not a functional gap. I'm noting my own false-negative here for the
same reason Round 8 noted missing the RBTA-bound confound: better to show the check and
the correction than silently fix my own read after the fact.

This also substantively extends F-03 (Round 1): the original finding was specifically
about the entropy placeholder; this docstring is the first place the codebase also
states plainly, in one spot, that energy and memory bounds are formulaic estimates too,
not measured. That's a more complete and honest picture of A1's current enforcement
scope than existed before this round.

## 4. D-166 — verified, good durability improvement

`InterventionConfig.disable_blended_scorer`'s docstring now states the default's
rationale in-code. This is a better fix than only updating README, precisely because
this review has repeatedly found README lagging behind code/decisions by a round or
more — a docstring next to the field itself can't drift the same way, since anyone
reading the field necessarily reads the explanation next to it.

## 5. What's still open (unchanged this round)

- **NEW-14** (viewport RBTA violation rate, 9-12%, not yet independently confirmed
  benign): still just restated from the existing D-161 numbers, not freshly
  investigated. Genuinely still open.
- **NEW-20** (decision-log discipline): resumed as of this round — D-162 through D-167
  are all present, including honest "(retroactive)" authorship notes for the three that
  covered rounds 40-42's undocumented work. Good recovery; worth watching that it holds
  going forward rather than lapsing again.

## 6. Cumulative status — this round

| ID | Status |
|---|---|
| NEW-19 | **Fixed, verified correct** (§1) |
| NEW-20 | **Resumed** (§5) |
| D-165 (new, minor) | Fixed; entry's `warn_once` shorthand doesn't match the actual guard-flag name — cosmetic, one-line correction |
| **NEW-21 (new)** | **D-167 — confirmed fixed; genuinely good independent catch this round** |
| NEW-14 | Still open |

---

## 7. Assessment of the process itself

Worth stating plainly since it's useful signal for how to prompt future rounds: asking
for an independent architect-level pass *before* reading the prior review file produced
real value here (D-165, D-167) beyond what the prior file asked for. That's the
intended effect of decoupling "understand independently" from "react to a specific
list" — worth keeping as the standard shape for future prompts, not just a one-off.
