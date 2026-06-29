# PHCA v3.0 — Contributing Guide

## PR Workflow

1. **Branch:** `git checkout -b <ticket-id>-description` (e.g., `phca-3.1-001-scaffold`)
2. **Implement:** Write code. Commit often with meaningful messages.
3. **Test:** `make test-all && make lint` before pushing
4. **Push:** `git push -u origin <branch>`
5. **Open PR:** Fill in the PR template (what was done, what was tested, benchmark results)
6. **Review:** Request review from the ML/Systems Lead
7. **Merge:** Squash and merge after approval. Delete the branch.

## Coding Standards

### Python
- **Style:** PEP8 (line length 100)
- **Types:** Type hints required on all public functions
- **Format:** `black` (default settings)
- **Lint:** `ruff` — zero errors before merge

### Rust
- **Format:** `rustfmt` defaults
- **Lint:** `cargo clippy` — zero warnings
- **Unsafe:** No `unsafe` blocks without explicit lead approval

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
- [ ] Do all tests pass (`make test-all`)?
- [ ] Is there a performance measurement (no regression)?
- [ ] Is the decision logged in `DECISIONS.md` if applicable?
- [ ] Is there any dead code or commented-out code?

## Getting Help

- Tag `@lead` in Slack for blockers
- Tag `@researcher` for v3.0 spec interpretation
- Check `docs/11-engineers-playbook.md` for ticket details
