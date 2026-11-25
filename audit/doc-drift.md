# Documentation drift

## Tier violations

`docs/DOCUMENTATION_MAP.md` declares `logs/` artifacts and `DECISIONS.md` as the
top trust tier. In practice:

- Most numbers under `logs/` are ignored by `.gitignore` and were never committed,
  so the top tier is mostly absent from a clone.
- `DECISIONS.md` (307,922 bytes, 2,780 lines) is append-only prose. It contains
  at least one decision (D-159) that retracts the headline finding of D-153/156,
  but the README's opening still leads with the retracted number.
- `STATUS.md` is frozen at D-137 yet still sits at the repository root.

## Contradictions found

| Document A | Document B | Conflict |
|---|---|---|
| `README.md` top section | `README.md` D-159 addendum | 0.03% collapse vs 77.8% after bound fix |
| `docs/limitations.md` ("single-agent only") | `monitoring/multi_agent.py` | multi-agent exists |
| `docs/limitations.md` (M5 not implemented) | `core/cycle.py` header ("M5 procedural memory") | one of them is false |
| `README.md` ("12-step cycle") | `cycle.py` docstring (21-step blueprint) | numbering scheme, explained only in a README aside |
| `docs/limitations.md` (10–17 ms) | measurement | 4.8 ms steady state |
| `docs/ACTION_SELECTION` docs | `README` status | experimental selector described as available without its caveat |

## Documents that are historical but not marked

- `STATUS.md`, `IMPLEMENTATION_STATUS.md`, `gate_phase3.3_final.md`,
  `STRESS_TEST_SCENARIOS.md`, `AUDIT_ACTION_SELECTION_2026-09-16.md`.
- `docs/archive/*` is correctly archived; `docs/lessons/*` (32 files: course material,
  critic rounds, en/fa/zh lessons) is course material unrelated to the runtime.
- `research/` (724 KB) is theory input with no executable code.

## Documentation volume

131 markdown files, 1.6 MB under `docs/` plus 724 KB under `research/`. Only eight
are needed by a reader who wants to run the code: README, SETUP, ARCHITECTURE,
LIMITATIONS, DEMO, DECISIONS (short), CONTRIBUTING, CHANGELOG.

## Drift mechanism

The repository has no doc test. Nothing checks that a documented flag exists, that
a table's number matches an artifact, or that a limitation is still true. Adopting
"every number names its command" and deleting unregenerable tables is the only
durable fix; editing stale tables just resets the clock.
