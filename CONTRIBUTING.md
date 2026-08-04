# PHCA v3.0 — Contributing Guide

## PR Workflow

1. **Branch:** `git checkout -b <ticket-id>-description` (e.g., `phca-3.1-001-scaffold`)
2. **Implement:** Write code. Commit often with meaningful messages.
3. **Test:** `make test-all && MUJOCO_GL=disabled make test-mujoco && make lint` before pushing
4. **Push:** `git push -u origin <branch>`
5. **Open PR:** Fill in the PR template (what was done, what was tested, benchmark results)
6. **Review:** Request review from the ML/Systems Lead
7. **Merge:** Squash and merge after approval. Delete the branch.

## Coding Standards

### Python
- **Style:** PEP8 (line length 100)
- **Types:** Type hints required on all public functions
- **Format:** `black` is optional for touched files; repository-wide formatting is not yet normalized
- **Lint:** `ruff` is the enforced gate — zero errors before merge

### All Languages
- **Seeds:** Use fixed random seeds (`seed=42`) in all tests
- **Determinism:** Tests must be deterministic — no flaky tests
- **PR Description:** Every PR must include benchmark measurements showing no regression

## Decision Log

Every significant design decision must be logged in `DECISIONS.md`:
- Date
- Author
- Decision (what was chosen)
- Alternatives considered
- Rationale
- v3.0 specification reference

## PR Review Checklist

Reviewers must check:
- [ ] Does the code match the v3.0 specification?
- [ ] Are all public functions type-annotated?
- [ ] Are there unit tests covering the change?
- [ ] Do all tests pass (`make test-python` + `make test-mujoco`)?
- [ ] Is there a performance measurement (no regression)?
- [ ] Is the decision logged in `DECISIONS.md` if applicable?
- [ ] Is there any dead code or commented-out code?
- [ ] For monitoring changes: do Observatory tests pass with `QT_QPA_PLATFORM=offscreen`?
- [ ] For dashboard/replay changes: do scrub/rebuild tests pass (`test_playback_store.py`, per-panel `test_*_dashboard.py`)?

## Monitoring / Observatory changes

Dashboard, replay, and session-report PRs must pass the full monitoring suite (**386 tests**):

```bash
mkdir -p .tmp
TMPDIR=.tmp QT_QPA_PLATFORM=offscreen PYTHONPATH=python \
  python -m pytest python/phca/monitoring/tests/ -q
```

Test counts: run `make test-all` / `pytest` for current totals (do not hardcode from frozen STATUS.md).

Scrub/rebuild behavior is guarded by `test_playback_store.py` (including 500-frame JSON
immutability). Replay banner changes need coverage in the relevant `test_*_dashboard.py` module.

## Hardening Changes

For changes touching cognition, benchmarks, or long-run stability:

```bash
MUJOCO_GL=disabled PYTHONPATH=python python scripts/benchmark.py --use-mlp --cycles=200
MUJOCO_GL=disabled PYTHONPATH=python python scripts/assumption_validation.py --ci
make nightly NIGHTLY_CYCLES=1000   # full hardening suite (causal L3 FAIL expected — see DECISIONS D-161/D-197)
```

## Getting Help

- Tag `@lead` in Slack for blockers
- Tag `@researcher` for v3.0 spec interpretation
- Check [research/outputs/11-engineers-playbook.md](research/outputs/11-engineers-playbook.md) for ticket details
