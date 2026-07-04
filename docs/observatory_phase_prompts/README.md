# Observatory Phase Prompts (Phases 13–20)

Copy-paste **one English prompt** into a **new Cursor chat** to plan and implement each remaining Observatory phase. Phases 7–12 are complete; see [STATUS.md](../../STATUS.md).

## How to use

1. Open the `.txt` file for the phase you want.
2. Copy the entire prompt block.
3. Paste into a new chat (Plan mode first is recommended).
4. Confirm the agent's plan, then let it implement.
5. Run tests and update living docs before moving to the next phase.

## Recommended order

```
Phase 13 → 14 → 15 → 16 → 17 → 18 → 19 → 20
```

Phases 13–16 deepen single-agent observability; 17–18 add advanced analysis; 19–20 are scientific validation and sign-off. Run Phase 20 only after 13–19 are complete (or consciously deferred).

## Prompt files

| Phase | Focus | File |
|-------|-------|------|
| 13 | Anomaly detection | [phase13_anomaly_detection.txt](phase13_anomaly_detection.txt) |
| 14 | Action explainability | [phase14_action_explainability.txt](phase14_action_explainability.txt) |
| 15 | Observability API | [phase15_observability_api.txt](phase15_observability_api.txt) |
| 16 | Production hardening | [phase16_production_hardening.txt](phase16_production_hardening.txt) |
| 17 | Multi-agent Observatory | [phase17_multi_agent.txt](phase17_multi_agent.txt) |
| 18 | Interactive analysis | [phase18_interactive_analysis.txt](phase18_interactive_analysis.txt) |
| 19 | Scientific validation | [phase19_scientific_validation.txt](phase19_scientific_validation.txt) |
| 20 | Research maturity sign-off | [phase20_sign_off.txt](phase20_sign_off.txt) |

## Existing code to extend (all phases)

| Capability | Location |
|------------|----------|
| Spike / moment flags | `python/phca/monitoring/cognitive_panels.py` |
| Session report metrics | `python/phca/monitoring/session_report.py` |
| `action_rationale` in JSONL | `python/phca/core/cycle.py`, Action tab in `qt_dashboard.py` |
| Integrity gate | `scripts/phca_replay.py --check` |
| Multi-session compare | `compare_session_reports()` (Phase 12) |
| Nightly hardening | `make nightly` in `Makefile` |

## Principles (every phase)

1. Correctness and replay honesty before visual polish.
2. Shared helpers for dashboard, report, and `--check`.
3. Behavioral tests + sync `STATUS.md`, `DECISIONS.md`, `docs/observability.md`.
4. Surgical diffs; no schema break without migration.

See also [PHCA_Cognitive_Observatory_Architecture.md](../PHCA_Cognitive_Observatory_Architecture.md).
