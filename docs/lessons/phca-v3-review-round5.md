# PHCA v3.0 — Review Round 5 (English)

> Base commit: `1ae0ad3` ("round-30"). This round covers Decisions D-144 through D-152
> (the work done since the last review), verified directly against code and — where
> possible — against actual benchmark output data, not just source reading.
> Written in English per request; more sections than prior rounds, as requested.

---

## 0. Executive Summary

This was the most important round so far, for a specific reason: **the project ran the
exact experiment the previous review recommended (NEW-04), got an uncomfortable result,
and reported it honestly in both `DECISIONS.md` and `README.md` instead of hiding or
reframing it.** That is a genuine, verifiable act of scientific integrity, and it
deserves to be stated plainly before anything else in this document.

At the same time, the underlying result is serious: **even after three full rounds of
architectural fixes (removing the task-lock bypass, fixing MDIM, fixing Attention→learning,
etc.), PHCA still cannot beat a simple greedy `if-else` heuristic at adequate statistical
power (30 seeds) on Levels 2 and 3 of the causal evaluation gate.** This is not a wiring
bug anymore — the wiring is now correct. This is a question about whether the learned
world model and the action-selection architecture, as currently built, produce a
measurable behavioral advantage over classical baselines in the environments tested.

This document is organized as:
1. Verification of D-144–D-152 (what was fixed, and how well)
2. A new, previously-invisible finding: the Round-1 fix has a **double-edged effect** —
   it helped Level 1 and appears to have made Level 3 *worse*
3. Deep analysis of the causal-gate failure (D-151) — what it does and doesn't mean
4. A critical look at the RBTA bound recalibration (D-152) — is it calibration or
   goalpost-moving?
5. A full cumulative status table across all five review rounds
6. Prioritized recommendations

---

## 1. Verification of D-144 through D-152

| Decision | Claim | Verified status |
|---|---|---|
| D-144 | Revert `_classify_action` to count-based + severity override | ✅ Confirmed in round 3 |
| D-145 | L4 eval start-position confound — full cleanup | ✅ Confirmed in round 4 |
| D-146 | M3 task-aware eviction | ✅ Confirmed in round 4 |
| D-147 | Remove `transfer_efficiency` from Φ-IQ | ✅ Confirmed in round 4 |
| D-148 | Composition tree factory extraction | ⚠️ Partial (round 4) |
| D-149 | Complete F-04 — extract `_build_hpm_spec()` | ✅ New — closes the gap left by D-148 |
| D-150 | Fix `goal_autonomy` threshold (0.004 → 0.01) | ✅ New — verified below |
| D-151 | **Causal gate L2/L3 FAIL at 30 seeds** | ✅ New — the central finding of this round |
| D-152 | RBTA bound recalibration for grid ≥ 5 | ⚠️ New — needs scrutiny, see §4 |

### 1.1 D-149 — F-04 is now genuinely closed

```
$ grep -n "_build_hpm_spec\|hpm_spec = {" python/phca/core/cycle.py
```
confirms there is now exactly one place that builds the tree structure and it is reused
by both the minimal and normal branches, plus the final RBTA check. The duplication
identified in round 1 and partially fixed in round 4 is now fully consolidated.

### 1.2 D-150 — Threshold correction verified

`goal_autonomy_achieved` moved from `novel_rate > 0.004` to `novel_rate > 0.01`, matching
the whitepaper's stated target of "≥1 novel goal / 100 cycles." This is a genuine
tightening (2.5× stricter), not a loosening, and the L3 benchmark still clears it with
real margin (`novel_goal_rate = 0.0333` vs. threshold `0.01`). This closes F-08 cleanly —
better than I expected, since a natural failure mode for this kind of fix is to nudge the
number just enough to keep passing; instead they went to the value the whitepaper
actually specifies.

---

## 2. New finding: the Round-1 action-selection fix is a genuine trade-off, not a pure win

I compared the **old** 30-seed causal-eval baseline (`causal_eval.json`, pre-Round-1) with
the **new** 30-seed rerun (`causal_eval_round4.json`, post all three rounds of fixes).
This is a clean before/after comparison of the exact same experiment.

| Level | PHCA goal_rate (old) | PHCA goal_rate (new) | greedy_observed (new) | Gate (old) | Gate (new) |
|---|---|---|---|---|---|
| L1 | 0.79 (tied with baselines) | **0.85** | 0.79 | PASS | **PASS, and now a real margin** |
| L2 | 0.55 | 0.56 | 0.60 | FAIL | FAIL (unchanged) |
| L3 | 0.34 | **0.29** | 0.29 | **PASS** | **FAIL** |

Two things stand out:

**Level 1 genuinely improved and is no longer just imitating the baseline.** In the old
run, PHCA's numbers were *identical* to `greedy_observed`/`bfs_search` down to two decimal
places — the smoking-gun evidence from round 4 that the old geometry bypass made PHCA
behave as a disguised BFS planner. In the new run, PHCA (0.85) now clearly exceeds
`greedy_observed` (0.79). This is real, positive, measurable evidence that the Round-1
fix (removing the hard bypass, always running the G′-scored blend) did what it was
supposed to do in a simple, low-noise environment.

**Level 3 got worse, and a previously-passing gate now fails.** This is not something
the changelog claims or discusses, and it's worth being precise about the likely cause:
Level 3 in `phca_causal_eval.py` involves noisier/partial observability and dynamic
obstacles (harder conditions than L1). The old geometric bypass — a deterministic,
memoryless BFS/Manhattan planner — was *robust* to those conditions almost by
construction, since it doesn't depend on a noisy learned signal at all. The new blended
scorer weights G′'s predictions more heavily, and under partial observability, G′'s
predictions are presumably less reliable, so the blend performs worse than the
old "cheat" did. In other words: **the fix that resolved the architectural/philosophical
objection (bypass = fake cognition) came at a real, measurable behavioral cost in the
harder scenario, and nothing in the repo currently flags or explains this specific
regression.**

This deserves attention independent of D-151, because it's a distinct causal claim: it's
not just "PHCA doesn't beat baselines," it's "the fix made one specific scenario
measurably worse than it was before the fix." I'd treat this as a new finding:

> **NEW-05 — Removing the geometric bypass improved L1 (simple/low-noise) but degraded
> L3 (partial observability/dynamic obstacles) enough to flip a previously-passing gate
> to failing.** This should be verified with a targeted ablation (run L3 with and without
> the blended scorer, holding everything else fixed) before concluding it's the sole
> cause, but the directional evidence is strong and the timing lines up exactly with the
> Round-1 change.

---

## 3. Deep dive on D-151 — what the causal-gate failure does and doesn't mean

D-151 is refreshingly precise about what happened, and it's worth restating carefully
because it would be easy to over- or under-interpret:

- **What it does mean:** at 30 seeds × 200 cycles, PHCA's *behavioral output* (where the
  agent ends up, how fast, how much reward) is statistically indistinguishable from — or
  slightly worse than — a simple greedy heuristic in two of three tested scenarios. The
  5-seed nightly config that had been reporting "PASS" was underpowered; this is now
  corrected in the README.
- **What it does *not* necessarily mean:** it does not, by itself, mean the world model
  G′ has learned nothing, or that the architecture is worthless. A resource-bounded,
  partial-knowledge agent (A1/A3) is not expected to match a full-information planner
  (`greedy_full_info` is explicitly *not* gated, for exactly this reason). What's
  concerning is that it also doesn't beat `greedy_observed`, which has access to the
  *same* information PHCA does. That is the fair comparison, and that's the one that fails.

D-151's own hypothesized root causes are reasonable, and I'd rank them by how easy they
are to falsify cheaply:

1. **(Cheapest to test) Grid too small.** A 5×5 grid has at most ~25 states; a greedy
   heuristic with full local observability can be near-optimal almost by construction,
   leaving very little room for a learned predictive model to add value even if it's
   working correctly. D-151 already recommends this as the next step (10×10 grid). I'd
   go further and suggest this should have been the *first* environment tested for the
   causal gate from the beginning — a 5×5 grid was likely never going to discriminate
   between "genuinely predictive cognition" and "a slightly-elaborate greedy rule,"
   independent of any implementation bug.
2. **G′ convergence within 200 cycles.** Worth checking `prediction_error_mean` from the
   `causal_eval_round4.json` data directly (it's logged per-run) — if prediction error is
   still high/non-converged at cycle 200, that's a training-budget problem, not an
   architecture problem.
3. **Action-selection still not using prediction scores effectively.** Given the L1
   improvement documented in §2, I'd consider this the least likely of the three at this
   point — the mechanism clearly does *something* differently and better in the easy
   case. This makes hypothesis (1) more likely than (3), not less.

**Recommendation:** before doing (1) the 10×10 grid experiment, do the five-minute check
of hypothesis (2) — pull `prediction_error_mean` across cycles from existing logs and
look at whether it's still decreasing at cycle 200. If it's flat or noisy, that changes
where engineering effort should go (learning-rate schedule, replay buffer tuning) versus
a bigger grid.

---

## 4. Critical look at D-152 — RBTA bound recalibration

This decision deserves more scrutiny than the changelog gives it, because "loosen the
resource bounds until the enforcer stops firing" is a pattern that — done carelessly —
would directly undermine A1 (the whole *point* of RBTA is that resource bounds are real
constraints, not decorative numbers).

Reading D-152 closely, there's a legitimate argument buried in it: **PHCA's cognitive
cycle does structurally more work per step than a greedy baseline** (MDIM goal
generation, correlation tracking, attention modulation, TSPL planning all run every
cycle), and the original bounds were "calibrated to idealized nanosecond-level estimates,
not real measured timings." That's a real category of bug — bounds set from a
back-of-envelope estimate rather than profiling — and correcting them against measured
reality is legitimate engineering, not cheating.

That said, two specific numbers in D-152 are worth flagging as needing independent
justification, not just acceptance on the changelog's word:

- **ACTION time bound headroom raised to 14×.** A 14× headroom is large. D-152 frames
  this as "covers stochastic 2.0s spikes from goal-conditioned action selection" — but a
  100× spread between typical (0.02s) and worst-case (2.0s) action-selection latency is
  itself a symptom worth investigating on its own terms, not just accommodating. What
  causes a single action-selection call to occasionally take 100× longer than typical?
  If it's occasional cold-start recomputation (e.g., BFS re-planning after obstacle
  changes) that's understandable; if it's unbounded/pathological, loosening the bound
  just hides it from RBTA rather than fixing it.
- **The gate itself was loosened** (`< 0.10` → `< 0.15` for the grid-10 violation-rate
  test) to accommodate the new bounds. This is the part that most resembles
  "moving the goalpost until the test passes," even if each individual bound change has
  a defensible rationale. It would strengthen this decision considerably to include a
  short empirical note on *why* 15% (rather than, say, 12% or 10%) is the right cutoff —
  otherwise it reads as "whatever number the current implementation happens to hit."

**Suggested follow-up (cheap):** add a profiling-based unit test that asserts the
*measured* p50/p95/p99 action-selection latency stays within some multiple of the p50 —
e.g., `p99 < 20 * p50` — so that if the 100× spread grows further in a future change, CI
catches the regression in variance even if the absolute bound is loosened. Right now
nothing would flag it if that spread got worse from here.

---

## 5. Cumulative status table — all findings, all rounds

| ID | Description | Round found | Status |
|---|---|---|---|
| F-01 | Attention weights never consumed by G′ learning | 1 | ✅ Fixed (R1) |
| F-02 / NEW-01 | RBTA severity classification bugs | 1 / 2 | ✅ Fixed (R2, R3) |
| F-03 | Belief-entropy floor only meaningfully checked for one module | 1 | ⚠️ Partially fixed — G′/MDIM/ATTN real, 9 modules still placeholder `0.1` |
| F-04 | Duplicated composition-tree/hpm_spec structures (×3) | 1 | ✅ Fixed (R4 partial, D-149 complete) |
| F-05 | MDIM 6-drive system bypassed by task_lock in all benchmark levels | 1 | ✅ Fixed (R1) |
| F-06 | L4 "forgetting=0%" driven by eval-position confound, not memory | 1 | ✅ Fixed (D-145, R4) |
| F-08 | `goal_autonomy_achieved` threshold too lenient (0.004) | 1 | ✅ Fixed (D-150) |
| NEW-02 | M3 FIFO eviction starves early-task episodes | 3 | ✅ Fixed (D-146) |
| NEW-03 | Φ-IQ `transfer_efficiency` double-counts correlated signal | 3 | ✅ Fixed (D-147) |
| NEW-04 | Causal gate L2/L3 fail at adequate statistical power | 4 | 🔴 **Open, honestly documented (D-151)** — deepest unresolved issue |
| NEW-05 | Round-1 action-selection fix improved L1 but regressed L3 | 5 (this round) | 🟡 **New — needs targeted ablation to confirm causal link** |
| NEW-06 | RBTA bound loosening (D-152) partly justified, partly needs independent evidence | 5 (this round) | 🟡 **New — recommend adding a variance-based regression test** |

Only one item from the original round-1 list (F-03) remains genuinely unaddressed at
this point, and it's a minor one (most of the system's entropy monitoring is real; nine
peripheral modules use a placeholder). Everything else that was fixed appears to be
fixed for real reasons, not cosmetic ones — which is a meaningfully different situation
than where this review started three rounds ago.

---

## 6. Prioritized recommendations for the next round

1. **Cheap, high-value: pull `prediction_error_mean` over cycles from the existing
   `causal_eval_round4.json` data** (no new experiment needed) to check whether G′ has
   converged by cycle 200 in L2/L3. This directly informs which of D-151's three
   hypotheses to chase first.
2. **Run the 10×10 grid experiment D-151 already proposed.** If PHCA beats
   `greedy_observed` at 10×10 but not 5×5, that's a genuinely interesting and defensible
   result (the architecture needs "room" to show an advantage) and should be reported as
   such, honestly, alongside the 5×5 result rather than replacing it.
3. **Run a targeted ablation for NEW-05**: L3 causal-eval with the blended G′-scorer vs.
   a temporarily-restored pure-geometry path, everything else held fixed, to confirm
   whether the Round-1 fix is really the cause of the L3 regression before it's
   attributed there permanently in documentation.
4. **Add a variance-based latency regression test** for ACTION time (NEW-06), independent
   of the absolute bound, so future changes to the 100× p50/p99 spread get caught even as
   the raw bound continues to be tuned.
5. **F-03** (9 modules with placeholder entropy) — lowest priority; cosmetic completeness
   rather than a live risk, since it's fail-safe (can never falsely trigger) rather than
   fail-open (can never trigger, but this was already known).
