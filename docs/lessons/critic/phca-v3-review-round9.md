# PHCA v3.0 — Review Round 9 (English)

> Base commit: `4e852aa` ("round-45"). D-161 verified against the decision log and
> matches the summary given. The main new content this round is a direct audit of
> `.github/workflows/` and the `Makefile` CI targets, per the request to look specifically
> at CI problems — and there is at least one significant, concrete, high-confidence one.

---

## 1. D-161 verification — genuinely rigorous, no issues found

Read in full against `DECISIONS.md`. It does exactly what was asked: tests both configs
at both scales with 30 seeds (matching D-151's own standard), reports RBTA violation rate
alongside goal rate at every data point (the practice recommended at the end of Round 8),
and reaches a specific, falsifiable, non-final conclusion — including an important
independent confirmation that **pure geometry itself fails the L2 gate at 5×5** (0.552 vs
0.598 against `greedy_observed`), which is the same number I found and flagged in the
prompt for this round. This is now a load-bearing, well-documented, honestly-reported
architectural gap: not a bug, and not something either action-selection mode currently
solves. Good work, no correction needed here.

---

## 2. The main finding: the nightly GitHub Actions workflow is very likely failing right now, on every scheduled run

This follows directly from §1 and is worth stating with the same directness the project
itself uses for its own findings.

**The chain of facts, each independently verified:**

1. `.github/workflows/nightly.yml` runs `make nightly NIGHTLY_CYCLES=11000` on a daily
   cron schedule.
2. The `nightly` Makefile target, step 6 of 7, runs:
   ```
   python scripts/phca_causal_eval.py --levels level2,level3 --cycles 200 --seeds 30 \
       --use-mlp --gate --output logs/phca_causal_eval_nightly.json
   ```
3. `scripts/phca_causal_eval.py`'s `--grid-size` argument defaults to `5` (confirmed:
   `parser.add_argument("--grid-size", type=int, default=5, ...)`), and this command line
   does not override it.
4. There is no `--enable-blended-scorer` flag here, so this run uses the current default,
   `disable_blended_scorer=True` (pure geometry).
5. **D-161, produced in this same repository, documents that pure geometry fails the L2
   causal gate at 5×5** (0.552 vs 0.598, 0 of 3 required metrics against
   `greedy_observed`).
6. `--gate` causes the script to call `sys.exit(1)` when the overall gate fails
   (confirmed at the `if args.gate and not passed: sys.exit(1)` line).
7. A nonzero exit from a `make` recipe step aborts the target and propagates a failing
   exit code to the calling GitHub Actions step, which marks the job (and since this is a
   single-job workflow, the whole `nightly.yml` run) as failed.

**Putting this together: every scheduled nightly run should currently be failing at step
6/7, specifically on the Level 2 causal gate, for a reason the project has already
diagnosed and written up in its own decision log — but that diagnosis was never
propagated into the CI configuration that triggers on it.** This is not a hypothetical;
it follows deterministically from code that's already in the repo, and I'd recommend
checking the Actions tab directly to confirm the badge state, since I can't view that
from here.

**Why this matters beyond "a CI badge is red":** a red nightly badge that everyone already
expects to be red, for a reason already written down, is a much lower-severity problem
than it looks. But if it's been failing silently for a while, it likely means **nothing
else in the nightly suite has been effectively checked either** — steps 1-5 and 7 only
run if step 6 doesn't abort them first (this is a single sequential `make` recipe, not
independent parallel jobs), so if step 6 has been failing since D-161 was merged, the
static Φ-IQ regression gate, assumption validation, OOD calibration, stress soak, and
session anomaly gate (steps 1-5 and 7) have effectively **not been running at all** in CI
since then, even though they're all still "in the workflow" on paper.

**Recommended fix, in order of preference:**
- Cheapest: change the nightly gate call to `--levels level3` only (which passes at 5×5
  per D-161), or add `--grid-size 10` (where geometry passes L2), until D-161's open item
  (closing the 5×5 L2 gap) is resolved — with a comment linking to D-161 explaining why.
- More correct: split the single sequential `make nightly` recipe into independent steps
  in the workflow YAML (using `continue-on-error` or separate steps) so that one known,
  already-documented failure doesn't silently suppress the other six checks. This is a
  structural CI improvement independent of the specific gate issue.
- Least preferred but simplest: temporarily mark the level2 gate as expected-to-fail
  (e.g., `--gate` without `sys.exit(1)` for level2 specifically, with a loud log message)
  until the underlying navigation-quality gap is closed, so at minimum the other six
  checks keep running and reporting.

---

## 3. Secondary finding: the README's CI table describes a job that doesn't exist

The CI table (README, "Continuous integration" section) lists six jobs, including:

| Job | Tier | What it checks |
| :--- | :--- | :--- |
| **noise-injector** | T0 | ASI NoiseInjector unit tests in `python/phca/asi/tests/` |

`.github/workflows/ci.yml` was read in full: it defines exactly five jobs — `lint`,
`test-python`, `observatory-check`, `benchmark-level-0`, `mujoco-gate`. There is no job
named or functioning as `noise-injector`, and no step within any of the five that runs
`python/phca/asi/tests/` as an isolated, separately-reported check — those tests exist
and do run, but only as part of the broad `python -m pytest python/tests/ python/phca/`
call in `test-python`, indistinguishable in the Actions UI from any other test in that
sweep. The same phantom job is referenced two more times in the README (the CI job table
and the `make ci-local` command-line comment), plus once in the Makefile's own `ci-local`
recipe list — none of which materialize a distinct job or step. This looks like a
description that was accurate at some earlier point (there may once have been a separate
job) and was not updated when the workflow was consolidated. Low severity on its own, but
worth fixing alongside §2 since both stem from the same underlying pattern: the CI
configuration and its description have drifted apart, and nothing currently catches that
drift automatically.

---

## 4. Minor CI hygiene notes

- **`test-python` and `ci-local` both use `-x`** (pytest stop-on-first-failure). This
  means a single early failing test hides the pass/fail status of every test after it in
  the same run, which slows down debugging (each CI round-trip can surface at most one
  new failure) and could mask a second, unrelated regression landing in the same PR as a
  first one. Removing `-x` (or using `--maxfail=5` as a middle ground) would give a
  fuller picture per run at a modest cost in total CI time.
- **No dependency caching** (`actions/cache` is not used anywhere in either workflow
  file). Every job reinstalls `requirements.txt` (and often `requirements-dev.txt` /
  `requirements-mujoco.txt`) from scratch. Not a correctness issue, but a straightforward
  CI-minutes optimization available via `actions/setup-python`'s built-in `cache: pip`
  option.
- **`ci.yml` has no `workflow_dispatch` trigger** (only `push`/`pull_request`), so it
  cannot be manually re-run from the Actions UI without an empty commit or a PR
  push — `nightly.yml` already has this and it would be a one-line addition to `ci.yml`.

---

## 5. Cumulative status table — new items this round

| ID | Description | Found | Status |
|---|---|---|---|
| **NEW-16 (new, high severity)** | **Nightly workflow's Level 2 causal gate call almost certainly fails on every scheduled run, per D-161's own documented 5×5 result, silently short-circuiting the rest of the nightly suite** | R9 | **Open — recommend checking the Actions tab to confirm, then applying one of the three fixes in §2** |
| **NEW-17 (new, low severity)** | **README's CI table describes a `noise-injector` job that does not exist as a distinct job or step anywhere in the actual workflow files** | R9 | **Open — trivial fix: either add the job or correct the README/Makefile references** |
| **NEW-18 (new, minor)** | **CI hygiene: `-x` fail-fast hides multi-failure visibility; no dependency caching; `ci.yml` missing `workflow_dispatch`** | R9 | **Open — all three are cheap, standard fixes** |

All findings from Rounds 1-8 remain as last recorded; D-161 (§1) closes no open items
but substantially strengthens the evidentiary basis for NEW-04/NEW-07's current status.
