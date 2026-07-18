# PHCA v3.0 — Review Round 12 (English)

> Base commit: `290c081` ("round-45"). This is the largest and most substantive single
> round of fixes in the whole review history — 18 new decision entries (D-168 through
> D-185, plus D-173/D-174 filling in gaps), spanning M3 replay, RBTA enforcement scope,
> the async action loop, entropy computation, and a genuinely significant dead-code find
> in the Φ/error-volatility feedback loop. Two fixes below are, in my assessment, more
> consequential than anything found since Round 8's RBTA-bound bombshell — I verified
> both directly in source before writing this up.

---

## 1. D-168 — the PER replay mechanism was reinforcing the wrong task (verified, high severity)

This connects directly back to Round 1's original F-06 finding and deserves to be read
alongside it. `_replay_m3_prior_tasks()` calls `m3.sample_episodes_per(...)` when M3
supports prioritized replay — the "headline" anti-forgetting path this project has
touted since D-138. Confirmed in source: this call passed
`task_id=self._current_task_id`. Since the method's entire purpose is to replay *prior*
tasks so the model doesn't forget them, sampling the *current* task instead meant PER-based
replay was, for its entire existence, reinforcing exactly the task the model was already
training on — providing **zero** forgetting-mitigation benefit. Confirmed fixed:
`task_id=None` now samples across all tasks.

**Why this matters beyond "a bug got fixed":** Round 1 flagged that the L4 "0% forgetting"
headline result likely reflected a memoryless geometric planner's robustness to goal
relocation rather than genuine retention by the learned components. This bug shows that
even setting that confound aside, **the mechanism that was supposed to provide genuine
retention (PER-prioritized cross-task replay) could not have been contributing to any
forgetting-resistant behavior that was observed**, because it was never sampling the
right data. This doesn't necessarily change the L4 headline number (which per Round 1's
analysis was likely dominated by the geometric fallback anyway), but it closes a gap that
would have mattered a great deal the moment the project tries to demonstrate genuine
learned-model retention independent of the geometric fallback — which D-161 already
flags as a needed future test. Good, and worth a one-line cross-reference from this entry
back to F-06/D-145 in a future documentation pass, since the two findings are about the
same claim from different angles.

## 2. D-182 — Φ (error_volatility) has been dead code since D-139 (verified, high severity)

Also independently verified. `_update_phi_from_gradient()`'s actual computation
(`normalized = ...`, `self._cached_phi = ...`) was unreachable — sitting after a `return`
inside a code path that could never execute post-guard-restructure. This means **Φ (the
gradient-based criticality signal introduced in D-139, roughly a week+ of project history
ago) never varied** — `_cached_phi` sat frozen at its init value the entire time, and
every downstream consumer (the PID controller in APC, and MDIM's D2 drive, per the
entry's own trace) was receiving a constant instead of a live signal. Confirmed fixed:
the guard now returns early only for the non-MLP case, and the computation runs
unconditionally otherwise.

**This is the same "dead-wired output" pattern this review has now found four separate
times** (Attention→learning in Round 1, MDIM's task_lock override also Round 1, and now
this) — a value is computed, assigned somewhere with every appearance of being consumed,
and simply isn't, with zero errors anywhere in the pipeline to surface it. It's worth
naming why this keeps happening in this specific codebase rather than treating each
instance as isolated: this architecture has many places where a "regulate before act"
signal is threaded through several dataclasses and dict contexts before reaching its
consumer (see docs/architecture.md's own cycle diagram — REG happens before ACT, feeding
several downstream modules from one computed value). That's a lot of hops for a value to
survive intact, and this project's control-flow style (early-return guards added
incrementally over many rounds, as D-182's own history shows) creates exactly the
conditions where a later refactor can silently strand a computation after a return
without any test catching it (unit tests apparently checked that `_update_phi_from_gradient`
didn't crash, not that `_cached_phi` actually changed value over a run — worth adding
that specific regression test if one doesn't already exist: assert `_cached_phi` takes at
least two distinct values over a 50-cycle MLP run).

## 3. Other verified fixes, grouped by theme

**Replay/M3 correctness (D-169, D-175, D-176, D-178):** the PER "death spiral"
(`priority > PER_EPSILON` excluding floor-priority regressed episodes permanently, fixed
to `>=`) is a real, subtle bug with a genuinely correct fix — a two-character change with
a well-reasoned justification for why it's sufficient (floor priority still sorts last,
so regressed episodes are eligible without dominating). The logging/documentation
additions (D-175, D-178) and the PER-beta-annealing extraction (D-176, following the same
pattern as D-148/D-149) are solid hygiene, low-risk.

**RBTA enforcement gaps (D-177):** confirms `runtime_log["ENV"]` was being collected but
never checked against any bound because `DEFAULT_MODULE_BOUNDS` had no `"ENV"` entry
outside the MuJoCo builder. This is the same category of gap as the original F-03 finding
(a monitored value with no real enforcement behind it) — worth noting these keep
surfacing in different corners of RBTA's bound table, which suggests a systematic audit
of `DEFAULT_MODULE_BOUNDS` against everything `_collect_runtime_log` actually populates
would be higher-leverage than fixing them one at a time as each is independently
discovered. Recommend this as a specific Phase-1-style task for the next round: enumerate
every key `_collect_runtime_log` can write, and confirm each has a corresponding bound.

**Async path (D-179):** the sticky-flag bug (RBTA flags never reset at the top of
`_action_loop`, so a single TERMINATE could permanently lock the async agent into neutral
actions) is a genuine, serious bug in a code path that's easy to under-test precisely
because it's async — confirmed the reset now exists at the same three call sites as the
sync path. Good catch, and a plausible explanation for any past "async mode acts strange
after an RBTA event" reports if they existed.

**Entropy computation (D-180):** the ATTN entropy clip-before-vs-after-normalize ordering
bug is a real numerical correctness issue directly inside the belief-entropy machinery
this review has tracked since Round 1 (F-03). Worth flagging precisely because it means
even the *3* modules previously verified as having "real" entropy measurement (G′, MDIM,
ATTN) had at least one of the three producing a mathematically invalid distribution
(sum ≠ 1) before this fix — a good reminder that "this one is real, not a placeholder"
and "this one is numerically correct" are different claims, and the first was verified in
earlier rounds without the second being checked as carefully.

**Assert→ValueError migration (D-183, D-185) and gaussian fixes (D-183's overflow check,
D-184's empty-evidence edge case):** straightforward, correct, low-risk robustness work.
The motivation (asserts silently vanish under `python -O`) is a real and often-overlooked
production risk; good that it was caught and addressed as a sweep rather than one at a
time.

**CI baseline staleness (D-173):** a genuinely important find, structurally similar to
NEW-16/NEW-19 from Rounds 9-10 but in a different part of the CI surface — the regression
baseline JSON was computed under pre-D-147 Φ-IQ weights, making the PR-level
`benchmark-level-0` gate compare incommensurate numbers (0.5775 old-weight vs. 0.8185
current-weight) since D-147 shipped. This means the PR-level Φ-IQ regression gate may
have been silently non-functional (either always trivially passing or comparing noise)
for however long D-147 has been merged. Fixed by regenerating the baseline. **Worth a
specific check in the next round:** confirm there's now a process or test that prevents
this specific staleness from recurring the next time the Φ-IQ formula changes — e.g., a
CI step that fails loudly if the formula's weight-hash doesn't match a hash stored
alongside the baseline file, rather than relying on someone remembering to regenerate it.

## 4. NEW-14 — instrumented, not yet resolved (honest partial progress)

D-174 adds a `violations_by_type` breakdown (TIME/MEM/ENERGY/ENTROPY counts) to the
causal-eval output — real, useful infrastructure for answering NEW-14, but the entry is
explicit that this enables the investigation rather than completing it. The 9-12%
viewport RBTA violation rate's root cause is still not determined as of this round. This
is honestly scoped in the decision entry (it's filed as "instrumentation," not as a
fix) — good practice, flagging only because NEW-14 should stay open, not be marked closed
by this entry.

## 5. Minor hygiene note

`DECISIONS.md` now contains **two entries both numbered D-167** (one from the prior
round's confidence-gating-probe fix, one appearing again mid-file in this round's diff,
verbatim identical text). This is almost certainly a merge artifact rather than a new
issue, but worth a quick cleanup pass — a duplicated decision number undermines the
document's own utility as a citable reference (a future entry or external note citing
"D-167" would be ambiguous, or in this case not ambiguous since they're identical, but
it's the kind of small inconsistency worth catching before it compounds).

---

## 6. Cumulative status — this round

| ID | Status |
|---|---|
| **NEW-24 (new)** | **D-168 — PER replay sampled current task, not prior tasks; zero forgetting-mitigation benefit until fixed.** ✅ Fixed, verified |
| **NEW-25 (new)** | **D-182 — Φ/error_volatility dead code since D-139; PID + MDIM D2 received a frozen signal.** ✅ Fixed, verified |
| **NEW-26 (new)** | **D-169 — PER death spiral permanently excluding regressed episodes.** ✅ Fixed |
| **NEW-27 (new)** | **D-179 — async action loop RBTA sticky-flag bug, permanent neutral-action lock after one TERMINATE.** ✅ Fixed |
| **NEW-28 (new)** | **D-180 — ATTN entropy normalization order bug (invalid probability distribution).** ✅ Fixed |
| **NEW-29 (new)** | **D-173 — CI regression baseline stale relative to D-147 weights, likely non-functional gate.** ✅ Fixed — recommend a staleness-prevention check (§3) |
| **NEW-30 (new, minor)** | **D-177 — RBTA collected ENV timing but had no bound to check it against.** ✅ Fixed; recommend a full bounds-vs-collected-keys audit (§3) |
| NEW-14 | Instrumented (D-174), root cause still open |
| Duplicate D-167 entry | Minor — cleanup only |

This round is a strong example of the independent-architect-pass approach paying off at
scale rather than just finding one or two items — worth continuing the same prompt
structure, possibly with the specific addition of "enumerate every value a
dataclass/context dict carries and confirm each is read somewhere" as a standing check,
since that's the exact shape of both D-182 (this round) and the original Attention
finding (Round 1).
