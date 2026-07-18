# PHCA v3.0 — Review Round 10 (English)

> Base commit: `5caa2f8` ("round-42", following a somewhat tangled history — see §4).
> This round is intentionally broader than Round 9, per feedback that the previous round
> narrowed too quickly onto CI without a full system pass. §1 is the centerpiece finding:
> the CI fix from Round 9 introduced a new, more serious problem than the one it fixed.

---

## 1. The central finding: the nightly workflow now always reports "ALL PASS", regardless of what actually happened

This is worse than the problem it replaced, and it's worth being direct about that.

**What Round 9 (NEW-16) asked for:** restructure the nightly Makefile recipe so that one
known, already-diagnosed failure (the L2 causal gate at 5×5, per D-161) doesn't abort and
hide the other six checks.

**What was actually implemented:** every one of the 7 steps in the `nightly` target was
wrapped as `<command> || echo "⚠️ [N/7] ... FAILED (continuing)"`. This does stop one
step's failure from aborting the rest — that part is correct and matches the ask. But it
has a side effect that appears not to have been checked: **wrapping a command in
`|| echo ...` makes that shell line's exit status always 0**, because the `echo` fallback
itself succeeds. Since nothing collects or checks each step's actual pass/fail status, and
the recipe ends with an unconditional:
```makefile
@echo "============================================"
@echo "  Nightly hardening: ALL PASS"
@echo "============================================"
```
**the `make nightly` target — and therefore the `nightly.yml` GitHub Actions job that
calls it — will now exit 0 and print "ALL PASS" even if all seven checks fail.** The
original problem (one known failure silently hid six unrelated checks) has been replaced
by a strictly worse one (any number of failures, known or unknown, are now silently
invisible, forever, since there's no longer even a red badge to prompt someone to look).

**What makes this particularly worth flagging carefully:** a genuinely good, precise fix
*was* made one layer down, in `scripts/phca_causal_eval.py` itself — the script was
changed to distinguish an expected L2 failure (per D-161, does not raise `sys.exit(1)`)
from a critical L3/L4 failure (still raises `sys.exit(1)`). That is exactly the right
fix, at exactly the right layer — it preserves the ability to detect a *new* problem
(an L3 regression) while not crying wolf about the *known* one (L2). **But the Makefile's
blanket `|| echo ... continuing` on top of it throws this distinction away**: whether
`phca_causal_eval.py --gate` exits 0 (L2-only, expected) or 1 (L3 regression, critical),
the Makefile step now prints the exact same generic warning and the recipe still ends
with "ALL PASS" either way. Two correct-looking fixes, made at two different layers,
combined to cancel out the one that actually mattered (surfacing a real L3 regression).

**Recommended fix, concretely:**
```makefile
nightly: nightly-mujoco
	@FAILED=""; \
	echo "[1/7] Static Φ-IQ benchmark..."; \
	{ ... } || FAILED="$$FAILED static-phi-iq"; \
	... (repeat per step, appending to $$FAILED on nonzero exit) ...; \
	echo "[6/7] Causal behavior gate..."; \
	python scripts/phca_causal_eval.py --levels level2,level3 --cycles 200 --seeds 30 \
	    --use-mlp --gate --output logs/phca_causal_eval_nightly.json \
	    || FAILED="$$FAILED causal-gate(check-if-L3-only-see-D161)"; \
	...; \
	if [ -n "$$FAILED" ]; then \
	    echo "Nightly hardening: FAILED —$$FAILED"; exit 1; \
	else \
	    echo "Nightly hardening: ALL PASS"; \
	fi
```
The key property any fix needs: **the final message and exit code must be computed from
actual step outcomes, not printed unconditionally.** If the intent is specifically that a
step-6 exit of 1 should sometimes be tolerated (L2-only) and sometimes not (L3
regression) — that distinction already exists correctly inside the python script's own
exit code (0 vs 1) as of the fix described above; the Makefile doesn't need to
re-implement that logic, it just needs to stop discarding the exit code it's already
being given.

---

## 2. Process finding: six rounds of real changes, zero new decision-log entries

`DECISIONS.md` has no entry after D-161, despite the repository showing six further
commits (`round-40` through `round-42`, plus what the git log shows as a second,
overlapping `round-40`-`round-42` sequence — see §4) that include the CI restructuring in
§1, the `gaussian.py` numerical-stability fix (§3), and the `phca_causal_eval.py` gate
logic split. This breaks the project's own stated norm — the README's Contributing
section says explicitly: *"every change is logged as a `D-XXX` entry in DECISIONS.md."*

This is worth flagging with some urgency, not just as a paperwork gap: **§1's regression
is exactly the kind of thing a decision-log entry would have caught before merge**, since
writing "here's the fix, here's why it's correct, here's what I verified" for the nightly
Makefile change would have prompted a check of what `make nightly`'s exit code actually
does under a failure — the same self-checking discipline that caught the RBTA bound bug
(Round 8) and the confidence-gating failure (Round 7) elsewhere in this project's history.
The decision log isn't just documentation here; it appears to function as this project's
primary quality gate, and skipping it for exactly the six commits that touched CI
correctness is a bad place for it to lapse.

---

## 3. Other changes this round, verified

### `python/phca/world_model/gaussian.py` — legitimate, well-reasoned numerical fix

Both `posterior()` and `conditional_covariance()` now check `np.linalg.cond(Σ_EE)` and
use the faster `np.linalg.solve()` for well-conditioned matrices (falling back to
`np.linalg.pinv()` only when the condition number exceeds 1e12 or `solve()` raises). This
correctly balances the two failure modes previously seen in this codebase's history:
`inv()` alone can raise on singular matrices, while unconditional `pinv()` (a prior fix)
is robust but slow enough to have reportedly caused SVD timeouts. This looks sound.

**One minor efficiency note, not a correctness issue:** `np.linalg.cond()` itself is
computed via SVD internally, meaning the "fast path" still pays an SVD-adjacent cost on
every call before deciding whether to take the fast or slow branch. This likely still
represents a net win (cond-number computation is typically cheaper than a full
solve-versus-pinv comparison), but if this function is on a hot path, it would be worth
confirming with a quick profile that the `cond()` check itself isn't eating a meaningful
fraction of the savings — a cheap thing to check, not a red flag on its own.

### `scripts/phca_causal_eval.py` — see §1; correct in isolation

Already covered above: this is a good, precise fix that correctly distinguishes expected
from critical gate failures. Flagging again here only to be clear that the criticism in
§1 is about the Makefile layer on top of it, not this script.

---

## 4. Repo hygiene: a tangled commit history worth a quick look

`git log --oneline` shows `round-40` through `round-42` appearing **twice**, in two
different positions in history (once immediately before `round-43`-`round-45`, and again
—identically named— after them, ending at the current tip `round-42` /  `5caa2f8`). This
is consistent with a branch that was rebased, force-pushed, or merged in a way that
duplicated round numbers rather than a real problem with any specific commit's content,
but it's worth a `git log --all --graph` check to confirm there isn't an orphaned branch
with the "other" round-43-45 sitting unmerged somewhere, since round numbering suggests
those commits (previously reviewed in Round 9, e.g. D-161) might not be reachable from
the current tip in the same order they were reviewed. Given D-161 *is* present in the
current file, the content made it through either way — this is a housekeeping note, not
a data-loss concern, but tidying the branch history (or at minimum understanding how it
happened) would reduce confusion for the next person reading `git log`.

---

## 5. Cumulative status table — new items this round

| ID | Description | Found | Status |
|---|---|---|---|
| **NEW-19 (new, high severity)** | **`make nightly` now unconditionally reports "ALL PASS" and exits 0 regardless of actual step outcomes — a regression introduced while fixing NEW-16, and more severe than the original issue** | R10 | **Open — concrete fix provided in §1; this should be the top priority for the next round** |
| **NEW-20 (new, moderate)** | **Six consecutive rounds of real changes (including the NEW-19 regression itself) were not logged in DECISIONS.md, breaking the project's own stated contribution norm and removing the self-check that has historically caught issues like NEW-19** | R10 | **Open — process fix: resume decision-log entries before the next merge, retroactively document rounds 40-42** |
| NEW-16 | Original nightly gate silently short-circuits on known L2 failure | R9 | Superseded by NEW-19 — the underlying intent was addressed but the implementation regressed further |
| NEW-17 | Phantom `noise-injector` CI job in README | R9 | **Fixed** — verified as a real, separate job in `ci.yml` with its own steps |
| NEW-18 | `-x` fail-fast, no caching, no `workflow_dispatch` | R9 | **Fixed** — all three confirmed: `--maxfail=5` everywhere, `cache: 'pip'` on every job, `workflow_dispatch` added |
| NEW-14 | Viewport RBTA violation rate (9-12%) not yet confirmed benign | R8 | Still open, not addressed in rounds 40-42 |

All findings from Rounds 1-8 remain as last recorded.

---

## 6. Why this round matters beyond the specific bug

The pattern across §1 and §2 is worth naming as its own lesson, since it's a new variant
of something this review has now seen several times in different forms: **a fix made with
good intentions, verified narrowly (does step 6 now avoid aborting the recipe? yes), but
not verified against the actual property that mattered (does the final reported status
still reflect reality?).** The same discipline that this project has applied well
elsewhere — state what you tested, what you didn't, and what would change the answer
(D-161 is still the gold standard for this) — needed to be applied to the CI fix itself,
and wasn't, likely because it landed without a decision-log entry to force that
self-check. Fixing NEW-19 and resuming NEW-20 together should close this loop.
