# PHCA v3.0 — Review Round 13 (English)

> Base commit: `54c1e51`. This round closed out every item from the Round 12 prompt
> (D-186 through D-191) and, notably, includes an honest "clean" result (D-191: a full
> failure-pattern sweep that found nothing new) — which is itself useful signal, not a
> non-event. Verified the two most checkable claims (D-187's viewport fix, D-189's
> staleness guard) directly in source.

---

## 1. D-186 — the field-audit technique, applied as instructed, found more dead wiring

Confirms the Phase-1 field-audit approach from the Round 12 prompt generalizes: four dead
`mdim_context` keys were found and removed (`cycle`, `prediction_confidence`,
`consolidation_facts`, `task_lock` — each computed and inserted every cycle, never read
by `compute_drives()`), following the exact same shape as D-182 and the original
Attention finding. Also caught and fixed a real async/sync skew (`violations_by_type`
populated in `step()` but not in `_finalize_learning_cycle()`) and a rationale
non-update bug (the no-state discrete path left a stale `last_action_rationale` in
place). The decision to leave four dead `CycleMetrics` fields in place rather than
removing them (to preserve the serialization protocol, deferred as a separate
deprecation pass) is a reasonable, appropriately conservative call — removing a
serialized field is a different risk class than removing an internal dict key, and
treating them differently is the right level of caution.

## 2. D-187 — NEW-14 resolved, verified directly in source

Checked `scripts/phca_causal_eval.py:472-485` myself: the viewport bound-widening
(`_scale = 1.0 + (10.0 - po_radius) * 0.1`, applied multiplicatively to every RBTA
bound's time/memory/energy and divisively to entropy floors) is real and matches the
decision entry's description. The finding that this already-existing scaling (from
D-160) produces exactly 0 violations across all 9 runs (3 viewport levels × 3 seeds) is a
clean, believable result — this isn't a new fix, it's confirmation that an existing
mechanism was already sufficient, which is a perfectly good outcome for an
investigation to land on.

**One scope observation worth logging, not a bug:** this bound-widening lives in the
`phca_causal_eval.py` script itself, applied ad hoc to a `cycle` object after
construction — not inside `build_for_env`/`build_for_mujoco` or any shared builder path.
This means any *other* caller that constructs a viewport-enabled cycle (the main
`benchmark.py` suite, the Observatory, a future script) would not automatically get this
widening unless it duplicates the same seven lines. This is the same category of
observation D-160 itself already flagged honestly (two separate partial-observability
implementations that should eventually be unified) — worth folding this bound-widening
into that same future unification pass rather than treating it as separately resolved,
since right now it's correct-but-only-in-one-place.

## 3. D-188, D-189 — regression tests for exactly the right property

D-188's test now asserts `_cached_phi` takes more than one distinct value over 50 MLP
cycles, which is precisely the property D-182's bug violated and a "doesn't crash" test
would have missed — good, targeted test design. D-189's `weight_hash` mechanism (MD5 of
sorted `DEFAULT_WEIGHTS`, stored in the baseline, checked at gate time) directly closes
the D-173 recurrence risk; I verified `_compute_weight_hash`, `_check_weight_hash`, and
the `--update-baseline-hash` mode all exist as described. Both of these are good examples
of a pattern worth continuing generally: when a bug is found, ask not just "how do I fix
this instance" but "what test would have caught this, and does a version of that
vulnerability exist elsewhere" — D-188 and D-189 both did this well.

## 4. D-191 — a clean audit result, and why it's worth taking seriously as a data point

Four specific failure patterns were checked systematically (dead keys beyond
mdim_context; async/sync field skew beyond what D-186 found; rationale non-update beyond
what D-186 fixed; tests structurally vulnerable to a frozen-signal bug like D-182) and
none produced a new fix. This matters for calibrating how much more there likely is to
find via this specific technique: two full rounds of systematic field-tracing (Round 12
prompt + this round's continuation) surfaced a bounded, now-apparently-exhausted set of
issues (D-168, D-182, D-186's four items), not an open-ended stream. That's a reasonable
point to conclude this particular technique has done most of its work for this codebase,
and future rounds should diversify method rather than re-running the same field audit
expecting more yield from it. The one loose end noted (`qt_phase.py` reading a
`belief_entropies["total"]` key that's never written, silently falling back to "first
available value") is correctly triaged as harmless/low-priority rather than inflated into
a new finding — appropriate proportionality.

## 5. D-190 — good documentation practice, low-risk

The cross-reference from D-168 back to F-06/D-145 is exactly the kind of connective
documentation this review has asked for before (Round 12's prompt, item 8) — a future
reader tracing the L4 forgetting claim now sees both angles (eval-protocol confound +
broken replay mechanism) from either entry point.

---

## 6. Cumulative status — this round

| ID | Status |
|---|---|
| **NEW-31 (new)** | **D-186 — 4 more dead mdim_context keys + async violations_by_type gap + rationale non-update.** ✅ Fixed |
| NEW-14 | **✅ Resolved (D-187), verified** |
| **NEW-25 regression test** | **✅ Added (D-188)** |
| **NEW-29 recurrence guard** | **✅ Added (D-189)** |
| Duplicate D-167 entry | Not explicitly addressed this round — recommend confirming in next pass |
| — | D-191's clean sweep: no new open items from this technique |
| **NEW-32 (new, low priority)** | Viewport bound-widening lives only in `phca_causal_eval.py`, not in a shared builder path — fold into D-160's already-planned unification pass |

## 7. Suggested shift in method for the next round

Given §4's observation, the next round's Phase 1 should diversify rather than repeat the
field-audit technique verbatim. Reasonable next angles, in rough priority order:
1. **Confirm the duplicate D-167 cleanup** (small, quick).
2. **Multi-scale/multi-seed re-verification**: several headline numbers in this project
   (Φ-IQ post-D-147, the D-161 4-config comparison, viewport results) have each been
   individually re-verified at some point, but not all together, on the current
   HEAD, in one pass — worth a single consolidated re-run to confirm nothing regressed
   as a side effect of the last ~25 decision entries' worth of changes.
3. **Test-suite audit for the D-188 pattern elsewhere**: D-188 fixed one aggregate-only
   test; D-191's Pattern 4 check was a good start but was scoped to searching for the
   *same* kind of assertion shape — worth also checking for tests that mock or stub a
   component so thoroughly that they'd pass even if the real component were fully
   disconnected (a different, complementary risk to the frozen-signal pattern).
