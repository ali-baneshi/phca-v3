# PHCA v3.0 — Review Round 7 (English)

> Base commit: `7f4f9be`. This round covers D-157 (adaptive confidence-gating) and D-158
> (the A4 reframing), the new 1000-cycle MuJoCo stability logs, and the README rewrite.
> It also contains a finding I verified independently by running the code myself, not
> just by reading it — see §2, which I consider the most important result of this round.

---

## 1. Overview of what changed

Four things happened since Round 6:

1. **D-157**: implemented adaptive confidence-gating for the blended scorer — when G′'s
   mean confidence across candidate actions falls below a threshold, fall back to pure
   geometry instead of trusting the prediction.
2. **D-158**: a deliberate reframing of the A4 invariant as environment-dependent (pure
   geometry in GridWorld treated as an "inductive bias" rather than a violation), plus an
   upgrade of D-157's fixed threshold to a self-calibrating one, plus a proposed new
   experimental design (G′-gated vs. pure geometry, rather than PHCA vs. `greedy_observed`).
   Notably, this decision entry is attributed to "Systems Reviewer (Round 6)" — i.e., it
   is presented as adopting this review's own recommendation. Worth naming plainly since
   it affects how I should read this decision: I am, in effect, being asked to evaluate
   a proposal that credits itself to my own prior suggestion, so I have tried to be
   especially careful in §4 not to grade my own homework charitably.
3. **New 1000-cycle MuJoCo stability logs** (`pendulum_1000_mlp.json`,
   `reacher_1000_mlp.json`, `pendulum_310_probe_test.json`) — directly answering Round 6's
   request to empirically verify the "continuous path is unaffected" claim rather than
   asserting it structurally.
4. **A large README rewrite** (853 lines) that adopted the honesty-focused restructuring
   from the previous round's edits nearly wholesale, including the "Current default
   performance" table with the real 5×5/10×10 pure-geometry numbers.

---

## 2. The central finding: adaptive confidence-gating does not work, and I can show exactly why

D-157/D-158 present confidence-gating as the safety net that makes the blended scorer
viable again — the mechanism that should prevent the 10×10 catastrophe from recurring
while still letting G′ contribute when it's actually trustworthy. I checked this by
finding an existing log (`logs/causal_eval_10x10_l2_adaptive_scorer.json`) that runs
exactly this configuration, and then reproduced it directly by importing the code myself
and instrumenting a live run, so this isn't a single-file coincidence.

**The log:** at 10×10, Level 2, with the blended scorer enabled and adaptive gating on
by default, PHCA's goal rate is **0.03%** — numerically identical to the ungated collapse
found in D-156. The safety net did not fire.

**Why, verified directly:** I ran a 200-cycle instance of the cycle at grid size 10 with
`disable_blended_scorer=False` and logged `_cached_confidences` and `_calibration_threshold`
every cycle. The result:

```
mean confidence, cycles 1-30:    0.998
mean confidence, cycles 150-200: 0.994
calibration probe at cycle 100:  mean_accuracy=0.9934, mean_confidence=0.9958 -> threshold raised 0.65 -> 0.8
calibration probe at cycle 200:  mean_accuracy=0.9930, mean_confidence=0.9947 -> threshold raised 0.8 -> 0.875
```

**G′'s confidence is pinned at approximately 0.99 for the entire run, and the
self-calibration probe reports approximately 99% "accuracy" throughout — while the agent
is, per the causal-eval log for the identical configuration, reaching the goal 0.03% of
the time.** The gate (`mean_conf < threshold`) can only fire when confidence drops below
threshold; since confidence never meaningfully drops below roughly 0.99, and the
self-calibrating mechanism responds to sustained high (mis-)confidence by *raising* the
threshold further (0.65 to 0.875 in just 200 cycles), the gate essentially never engages.
The blended scorer runs almost unchecked, and the result is the same catastrophic
failure as the ungated version, because the safety net that was supposed to catch it
never trips.

**Root cause, and why it's not a simple bug.** The calibration probe measures one-step
prediction accuracy on a buffer of recent `(state, action, next_state)` triples
(`_run_calibration_probe`, confirmed by reading it). In a GridWorld, one-step transitions
are a genuinely easy regression problem — the agent moves by at most one cell in a known
direction, and a modestly-trained MLP will get this right (or close to right) the large
majority of the time. That is a completely different question from "if I chain 20-40 of
these one-step predictions together to navigate to a distant goal, does the accumulated
error stay small enough to reach it." Small, apparently-negligible one-step errors compound
over a long horizon in exactly the way that produces the observed outcome: a model that
looks 99% accurate on the metric being measured, while being nearly useless for the task
the metric was meant to be a proxy for.

This is a more specific and more actionable diagnosis than "the blended scorer is
unreliable" (D-156's framing) or "gate on confidence" (D-157/158's fix). **The problem is
not that G′ is sometimes wrong and needs a gate — it's that the confidence signal being
gated on on measures the wrong thing.** A gate built on a metric that's structurally
decoupled from the failure mode it's meant to catch will not catch that failure mode, no
matter how well-tuned the threshold or how sophisticated the self-calibration around it.
Raising the threshold further (which is what the self-calibrating mechanism does when it
sees sustained high "accuracy") makes this strictly worse, not better, since it can never
correct for a confidence signal that was never measuring the relevant thing to begin with.

**What would actually test the right thing:** a gating signal derived from *rollout*
consistency — e.g., compare G′'s N-step-ahead compounded prediction against the actual
trajectory, or use the discrepancy between the blended scorer's chosen action and the
pure-geometry suggestion as an input to the gate, rather than a property of the one-step
prediction in isolation. This is a concrete, scoped follow-up, not a restart of the whole
effort.

---

## 3. D-158's reframing — a fair-minded but critical read

Separating this from §2's empirical finding is important, because D-158 is a
communication and scoping decision, and deserves to be judged on those terms rather than
folded into the confidence-gating bug above.

**What's genuinely good about it:**
- Narrowing a falsified universal claim ("prediction drives action everywhere") to a
  scoped one ("prediction drives action in continuous control, where it has held up; in
  GridWorld, geometry is used as a prior") is legitimate, ordinary scientific practice,
  not evasion — provided the narrower claim is actually true, which as of Round 6's
  MuJoCo data (§5 below) it now has real support for, not just a structural argument.
- Proposing "G′-gated vs. pure geometry" as the comparison of interest, instead of "PHCA
  vs. `greedy_observed`," is a better-designed experiment. The old framing conflates two
  questions (does PHCA beat a simple heuristic; does G′ add value over that heuristic)
  into one number. The new framing isolates the second question, which is the one that
  actually matters for A4.

**What deserves a more skeptical read:**
- The phrase "treated as an inductive bias, not a violation of A4" is doing more rhetorical
  work than the evidence currently supports. Calling something an inductive bias is
  usually a claim that the prior knowledge is *known to help* (the way, say, convolutional
  weight-sharing is a defensible inductive bias because it's proven to help image tasks).
  Here, geometry isn't being used as a *bias on top of* a working predictive model — as
  of §2, it is functionally the *entire* mechanism, because the thing meant to let G′
  contribute (the gate) doesn't work. Describing the current state as "geometry as
  inductive bias" reads more accurately as "geometry as the whole mechanism, with an
  aspiration that G′ will eventually contribute" — which is a real and reasonable
  position, but a different, more modest one than the phrase implies.
- The practical effect of relabeling is that A4's status in the README's invariant table
  can now be described in a way that sounds more resolved than it is, without any code
  changing. That's exactly the pattern this review has flagged before (the gap between
  DECISIONS.md's technical precision and how findings get summarized upward). D-158
  itself is precise and honest in its own text — the risk is entirely in how a shorter,
  future summary of it gets written, and I'd flag this pre-emptively rather than after
  it happens: a future README or whitepaper line that says "A4: satisfied (GridWorld
  geometry is an intentional inductive bias, D-158)" without also saying "and the
  confidence-gating mechanism intended to let G′ contribute doesn't currently work" would
  cross from scoping into obscuring.

**My overall verdict:** the reframing itself is legitimate and I would not ask for it to
be reverted. What I'd ask for is that it not be treated as closing the issue. The
underlying technical gap (G′ doesn't yet measurably help GridWorld navigation, and the
mechanism meant to let it help safely doesn't work) is exactly as open after D-158 as it
was after D-156 — D-158 changed how the gap is described and how future experiments
should be designed, which has real value, but it did not close the gap.

---

## 4. Documentation-vs-code discrepancy (small, but part of a pattern)

D-158's text states the self-calibrating threshold "starts at 0.9." The actual code
(`cycle.py` line 286) initializes `self._calibration_threshold = 0.65` — matching D-157's
original fixed value, not D-158's stated starting point. This is minor on its own, but it
is the same category of small drift between the decision log's prose and the shipped code
that this review has now flagged multiple times (Round 4's `hpm_spec` duplication, Round
6's README lag). None of these individually matter much; as a repeated pattern across
seven rounds, it suggests decision-log entries are sometimes written slightly ahead of
(or reflecting an intended-but-not-yet-final version of) the code they describe. A cheap
process fix: land the code change first, then write the entry from the actual diff,
rather than the reverse.

---

## 5. The MuJoCo 1000-cycle data — a genuine, well-earned positive result

This directly answers a gap flagged in the previous round of this review (the "continuous
path unaffected" claim was previously a structural argument, not a measurement). It now is:

| Env | Cycles | Early error | Late error | Improves | RBTA violations | All finite |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Pendulum | 1000 | 1.845 | 0.719 | Yes (61% reduction) | 0 | Yes |
| Reacher | 1000 | 18.51 | 2.09 | Yes (89% reduction) | 0 | Yes |
| Pendulum (probe config) | 310 | 4.29 | 0.66 | Yes (85% reduction) | 0 | Yes |

This is a clean, legitimate result: the world model's predictions genuinely improve over
an extended run, with zero resource violations and no numerical instability, in the one
domain where G′ actually drives action selection unconditionally. It is good evidence
that the architecture's core prediction-and-learning loop works as intended when the
action-selection mechanism around it doesn't have the specific problem found in §2. This
is worth stating plainly and positively — it's real support for the narrower version of
A4 that D-158 wants to stand behind.

**One loose thread worth a one-line follow-up, not a concern:** Reacher reports
`terminal_episodes: 1000` (every single cycle recorded as episode-terminal) while
Pendulum reports `0`. This is very likely just Reacher-v5's normal fixed-length episode
resets rather than a problem — the error curve still improves cleanly across the full run
either way — but it's different enough from Pendulum's number that it's worth a one-line
confirmation in whatever documentation cites this table, so a future reader doesn't have
to re-derive that it's benign the way I just did.

---

## 6. README verification

The rewrite (853 lines) substantially adopted the structure and content from the previous
round's edits: the "Current status of the core hypothesis" section, the hedged
continuous-path language, and the "Current default performance" table (with the real
5×5/10×10 pure-geometry numbers, including the 5×5 L2 failure) are all present and intact
in the current file. This is good — that content didn't get lost or diluted in the
larger rewrite.

**What's not yet reflected:** D-157 gets one clause in the A4 table row ("adaptive
confidence-gating (D-157) as a safety net"), stated as if it is a working mitigation. Per
§2, it currently is not. D-158's reframing language ("inductive bias," environment-scoped
A4) does not appear in the README at all yet. This is the same lag pattern noted in Round
6 (NEW-08) recurring one round later with new content — not a new category of issue, but
worth tracking as a standing item rather than a one-time fix, since it seems to
recur naturally with the pace of decisions outrunning the README's update cycle.

---

## 7. Cumulative status table — all findings, seven rounds

| ID | Description | Found | Status |
|---|---|---|---|
| F-01 through F-08, NEW-02/03 | (see Round 6 table) | R1-R3 | ✅ Fixed (unchanged this round) |
| NEW-04 | Causal gate fails at adequate power | R4 | 🟡 Root cause now well understood (§2); not yet closed |
| NEW-05 | Blended-scorer fix traded L1 for L3 | R5 | ✅ Confirmed |
| NEW-06 | RBTA bound loosening needed justification | R5 | ✅ Addressed |
| NEW-07 | Blended scorer collapses at 10x10; defaulted off | R6 | 🟡 Understood; D-157/158 is the attempted fix |
| NEW-08 | README lagged D-156 | R6 | ✅ Fixed this round — but see the recurrence noted in §6 |
| NEW-09 | Negative-feedback-loop hypothesis under-isolated | R6 | Still open, not addressed this round |
| **NEW-10 (new)** | **Adaptive confidence-gating (D-157/158) does not work: G' confidence is saturated near 0.99 regardless of actual navigation reliability, so the gate never fires** | R7 | 🔴 **Open — root cause identified (§2), fix scoped (rollout-consistency-based gating signal) but not implemented** |
| **NEW-11 (new)** | **D-158's "inductive bias" framing risks reading as resolved when the underlying mechanism (NEW-10) is not** | R7 | 🟡 **Open — a communications risk to pre-empt, not yet realized in current README text** |
| **NEW-12 (new)** | **D-158 states calibration threshold starts at 0.9; code initializes at 0.65** | R7 | 🟢 **Minor — cosmetic, part of a recurring small doc/code lag pattern** |
| **NEW-13 (new, positive)** | **MuJoCo continuous-control path empirically confirmed stable and improving over 1000 cycles** | R7 | ✅ **Closed — real data now supports what was previously a structural argument** |

---

## 8. Prioritized recommendations

1. **Redesign the confidence-gating signal (NEW-10).** This is now the single most
   consequential open item — more so than it appeared last round, because it explains
   why the "safety net" everyone is currently relying on doesn't actually work. The
   concrete next step: replace or supplement one-step prediction accuracy with a
   rollout-consistency signal (N-step compounded prediction error, or agreement between
   the blended and pure-geometry choices) as the gating criterion, and re-run the
   10×10 L2 adaptive-scorer experiment to see whether goal rate recovers toward the
   26.8% pure-geometry baseline (success) or stays near 0.03% (signal that the problem is
   deeper than the gating criterion).
2. **Add one caveat sentence to the README's A4 row** noting that adaptive
   confidence-gating (D-157) is implemented but has not yet been shown to work
   (§2) — cheap, and pre-empts NEW-11 before it becomes a real overclaim rather than a risk.
3. **Sync D-158's stated threshold value with the code**, or update the code to match —
   whichever is actually intended (NEW-12, trivial).
4. Everything else from Round 6's priority list (NEW-09's controlled ablation, F-03's
   placeholder entropy) remains open and unchanged this round.
