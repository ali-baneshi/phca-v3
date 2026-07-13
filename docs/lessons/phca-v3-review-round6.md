# PHCA v3.0 — Review Round 6 (English)

> Base commit: `7656681` ("round-31"). This round contains the most consequential
> finding of the entire review process. Read §1 first even if you skip everything else.

---

## 1. The headline finding: the system has come full circle, and that is a *good* thing, done honestly

Across Rounds 1–5, a major thread of this review was: "PHCA's action selection silently
falls back to a memoryless BFS/Manhattan planner whenever a goal exists, undermining the
architecture's central claim that prediction drives action (A4)." Round 1's fix removed
that fallback and made the G′-scored blend run unconditionally. Round 4 and 5 confirmed
this fix was real and that it measurably improved simple scenarios (L1).

**This round, the project ran the harder tests that fix deserved — a 10×10 grid and a
targeted L3 ablation — and found that the blended scorer doesn't just underperform at
scale, it catastrophically breaks the agent:**

> At 10×10, PHCA with the blended scorer reaches the goal in **0.03%** of episodes
> (1 success in 3,000 seed×cycles) and wanders *farther* from the goal than a random
> agent (mean distance 7.10 vs. random's presumably lower baseline). Pure geometry, on
> the same grid, reaches **26.8%** — beating `greedy_observed`'s 24.0% — and passes the
> gate convincingly. (D-156)

The response to this was, again, the right one: rather than keep the "architecturally
pure" blended scorer as the default because it's philosophically more defensible, the
team **flipped the default back to pure geometry** (`InterventionConfig.disable_blended_scorer
= True`) and documented exactly why, including the negative result, in `DECISIONS.md` and
`IMPLEMENTATION_STATUS.md`.

This is worth stating plainly: **the system's actual behavior today is, once again, "use
geometry when a goal exists, mostly ignore the learned model for action selection" — the
exact behavior that Round 1 of this review flagged as undermining A4.** The difference
between now and Round 1 is entirely about *why* and *how it's communicated*: it was
previously an undocumented, silent bypass discovered by reading code; it is now a
documented, deliberate, evidence-backed default with a clear escape hatch
(`--enable-blended-scorer`) and a clear near-term plan (adaptive confidence-gating, D-156
item 2) to eventually make the blended scorer viable. That is a completely different, and
much better, situation to be in — but the practical, benchmarked behavior of the shipped
system is closer to where it started than the intervening four rounds might suggest, and
that needs to be stated clearly rather than buried under "all previous findings: fixed."

---

## 2. What this means for previously "resolved" findings — necessary corrections

This finding requires walking back the certainty of some earlier verdicts. Not because
those verdicts were wrong at the time — they accurately described the code as it existed
in Rounds 1–5 — but because the *default configuration* has now changed underneath them.

| Finding | Round 5 verdict | Corrected verdict (Round 6) |
|---|---|---|
| F-05 (MDIM 6-drive bypass) | ✅ Fixed | ✅ **Still fixed** — this is independent of `disable_blended_scorer`; MDIM's `winner=1` hard override is gone regardless of which action-selection path runs. Verified: the flag lives in action selection (`_select_action`), not in `MDIM.generate_goal()`. |
| F-06 (L4 forgetting confound) | ✅ Fixed | ✅ **Still fixed at the eval-protocol level** (random start position genuinely restored) — ⚠️ **but weakened in spirit**: since `benchmark_level4.py` never sets `disable_blended_scorer=False`, the L4 continual-learning benchmark's action selection is, by default, back to pure geometry. The eval protocol confound is gone, but the underlying mechanism producing `goal_reached` is — once again — a memoryless planner, not the learned/predictive system. The "0% forgetting" claim (wherever it currently stands — worth re-running given all the changes) still says more about a classical planner's robustness than about neural retention. |
| A4 invariant ("prediction as the axis of action") | Treated as resolved in the discrete case | 🔴 **Not currently true of the default configuration.** `IMPLEMENTATION_STATUS.md` says this plainly (see §3). |

This is not a criticism of the engineering — confidence-gating toward a reliable
fallback when a learned model isn't trustworthy is a completely reasonable, arguably
*correct* systems-safety decision (it's philosophically consistent with A1, actually: a
resource-bounded agent should not act on unreliable predictions just to look more
"cognitive"). It is, however, a reason the review needs to track "what does the *shipped
default* actually do" as a separate axis from "is the code capable of doing the
principled thing," because those two have now diverged twice in six rounds.

---

## 3. Documentation consistency check: README vs. IMPLEMENTATION_STATUS.md

This is a new, concrete, easily-fixed finding.

**`IMPLEMENTATION_STATUS.md`, line 117** (updated this round, honest and current):
> "GridWorld discrete | Pure BFS/Manhattan geometry (default since D-156; G' prediction
> blend available via `--enable-blended-scorer`) | **No** (prediction-primary suspended —
> blended scorer caused catastrophic 0.03% goal rate at 10×10; pure geometry 26.8%)"

**`README.md`, line 231** (the A1–A5 invariants table, *not* updated this round):
> "**A4** ... **Discrete GridWorld: hybrid cognitive map** — confidence-gated
> Manhattan/BFS geometry + G′ blended scorer (D-136)."

These two files now describe the same subsystem inconsistently. `IMPLEMENTATION_STATUS.md`
correctly says prediction-primary behavior is *suspended by default* and explains why in
one sentence, including the specific catastrophic number. `README.md`'s invariants table —
which is far more likely to be the first (and possibly only) thing an external reader of
this repository sees — still describes a "hybrid" system without any indication that the
predictive half is currently switched off by default pending further work. A reader who
only reads the README would come away believing the blended scorer is live.

**Recommended fix (small, mechanical):** update the README A4 row to mirror
`IMPLEMENTATION_STATUS.md`'s honesty, e.g.: *"Discrete GridWorld: pure geometric
navigation is the current default (D-156) after the blended G′ scorer was found to
collapse goal-reaching at 10×10 (0.03% vs. 26.8%); the blended scorer remains available
via `--enable-blended-scorer` and is the target architecture pending G′ reliability
improvements."* This is a two-minute edit that closes a real gap between the project's
internal honesty (which is consistently excellent, as this whole review has noted) and
its external-facing summary (which currently lags behind it by one round).

---

## 4. The negative feedback loop finding (D-155) — a genuinely interesting result worth double-checking

D-155 reports something conceptually important and slightly counter-intuitive:

> "An unexpected secondary finding: pure geometry also improves G′ prediction error
> (6.54 vs 6.92). This suggests a negative feedback loop: blended scorer → erratic
> exploration → confusing G′ training data → worse predictions → even worse blended
> actions."

This is a plausible and well-articulated hypothesis, and if true it's a genuinely useful
systems-level insight (a classic exploration/exploitation data-quality spiral). But it is
currently a hypothesis inferred from a single before/after comparison (two `pred_error`
numbers, 6.92 vs 6.54, roughly a 5.5% difference), not something isolated by a controlled
experiment. Before this becomes load-bearing for future design decisions (e.g., deciding
how to implement adaptive confidence-gating), it's worth being a little careful about
alternative explanations for the same two numbers:

- **Reversion to the mean / noise:** with 15 seeds, a ~5.5% difference in mean prediction
  error is not obviously outside plausible seed-to-seed variance. Was this checked against
  a confidence interval or just a point estimate?
- **Direction-of-causality ambiguity:** it's equally consistent with the data that G′'s
  predictions were already going to be mediocre in this environment (partial
  observability limits how good *any* short-horizon predictor can be), and that mediocre
  predictions *cause* both the erratic exploration *and* the higher measured error,
  rather than exploration causing the error. The "loop" framing assumes a specific
  causal direction that the single comparison doesn't establish.

**Suggested follow-up:** a cleaner test of the feedback-loop hypothesis would hold
exploration behavior fixed (e.g., force both conditions to follow the *same* pure-geometry
trajectory during a data-collection phase, only varying whether G′ is trained on that
data vs. trained on blended-scorer-collected data) and compare resulting prediction error.
That isolates "does the *data distribution* from erratic exploration hurt training" from
"is erratic exploration itself just a symptom of already-bad predictions." This is a
Tier-3 nice-to-have, not urgent — but worth flagging before this hypothesis gets cited as
established fact in later documentation, the way some earlier five-seed results were
until D-151 caught it.

---

## 5. Verification of D-153/D-154 (the items directly requested in Round 5)

### ✅ D-153 — NEW-05 trade-off, confirmed and well-documented

The table in D-153 matches the numbers I derived independently in Round 5 exactly
(L1 +7.8%, L3 −15.4%), and the additional observation that prediction error moved in
*opposite* directions for the two levels (L1 got worse, L3 got better) while behavior
also diverged in opposite directions is a genuinely sharp piece of analysis — it correctly
rules out the simplest possible explanation ("better predictions → better behavior,
worse predictions → worse behavior") and forces the more nuanced "reliability-dependent
usefulness" framing that later shows up validated in D-155/D-156.

### ✅ D-154 — Variance regression test, confirmed present and reasonable

```
$ grep -n "test_action_latency_spread_within_20x" python/tests/test_grid_rbta_bounds.py
92:def test_action_latency_spread_within_20x():
```
Confirmed to exist, and the measured p99/p50 ≈ 5.3× (well inside the 20× gate) matches
what D-152's addendum reports. This closes NEW-06 as requested — the variance guard is
now independent of the absolute RBTA bound, exactly as recommended.

---

## 6. Cumulative status table — all findings, six rounds

| ID | Description | Found | Status |
|---|---|---|---|
| F-01 | Attention weights never consumed by G′ learning | R1 | ✅ Fixed |
| F-02/NEW-01 | RBTA severity classification bugs | R1/R2 | ✅ Fixed |
| F-03 | Belief-entropy floor only real for 1 of 12 modules | R1 | ⚠️ Partial (3/12 real, 9 placeholder) |
| F-04 | Duplicated composition-tree/hpm_spec (×3) | R1 | ✅ Fixed |
| F-05 | MDIM 6-drive bypass in all benchmark levels | R1 | ✅ Fixed, independent of §2's caveat |
| F-06 | L4 "forgetting=0%" driven by eval-position confound | R1 | ✅ Protocol fixed — ⚠️ mechanism caveat, see §2 |
| F-08 | `goal_autonomy_achieved` threshold too lenient | R1 | ✅ Fixed |
| NEW-02 | M3 FIFO eviction starves early tasks | R3 | ✅ Fixed |
| NEW-03 | Φ-IQ `transfer_efficiency` double-counts | R3 | ✅ Fixed |
| NEW-04 | Causal gate L2/L3 fail at 30 seeds | R4 | 🟡 Reframed by D-153/155/156 — root cause now understood (unreliable G′ under partial observability), not just "documented as failing" |
| NEW-05 | Blended-scorer fix helped L1, hurt L3 | R5 | ✅ Confirmed by targeted ablation (D-155) |
| NEW-06 | RBTA bound loosening needs independent justification | R5 | ✅ Addressed (D-152 addendum + D-154 variance test) |
| **NEW-07 (new)** | **Blended scorer catastrophically fails at 10×10 (0.03% vs 26.8%); now disabled by default system-wide** | R6 | 🟡 **Understood and mitigated via default flip; adaptive confidence-gating is the real fix and remains unimplemented** |
| **NEW-08 (new)** | **README A4 table not updated to reflect D-156's default change; contradicts `IMPLEMENTATION_STATUS.md`** | R6 | 🔴 **Open — trivial fix** |
| **NEW-09 (new)** | **Negative-feedback-loop hypothesis (D-155) is plausible but not yet isolated from confounds** | R6 | 🟡 **Open — needs a controlled follow-up before being treated as established** |

---

## 7. Prioritized recommendations

1. **(Two minutes) Fix NEW-08** — update the README A4 row to match
   `IMPLEMENTATION_STATUS.md`. This is the cheapest possible fix in this entire review and
   closes a real external-facing inconsistency.
2. **(Medium) Prototype adaptive confidence-gating** — this is the one piece of unfinished
   work that every one of D-151, D-153, D-155, and D-156 converges on as "the actual fix."
   A minimal version: use pure geometry whenever G′'s epistemic confidence (already
   computed via MC-dropout, per Round 1's findings) is below some threshold, and let the
   blended scorer engage only above it. This is very close to the *old* task-lock
   confidence gate that Round 1 removed — the difference is that gate should now be tied
   to measured G′ reliability rather than a fixed cycle-count/task-lock heuristic, and
   should be validated at both 5×5 and 10×10 before being called done, given how badly the
   ungated version failed to generalize across scales.
3. **(Cheap, protects NEW-09) Add a controlled ablation** isolating data-distribution
   effects from exploration-quality effects before the "negative feedback loop" framing
   is used to justify design decisions elsewhere.
4. **Re-run the L4 continual-learning benchmark** now that action selection defaults to
   pure geometry *and* the eval-position confound is fixed, and report the resulting
   `forgetting_rate` with an explicit note (per §2) that it currently reflects a
   deterministic planner's robustness to goal relocation, not neural memory retention —
   consistent with how D-156 now describes the discrete action-selection subsystem
   elsewhere in the docs.
5. F-03 remains the only fully-untouched item from Round 1; still low priority (fail-safe,
   not fail-open).
