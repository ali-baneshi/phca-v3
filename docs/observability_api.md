# PHCA Observability API (Phase 15)

Stable programmatic access to Observatory session data for external tools, notebooks, and CI pipelines.

**Import surface:**

```python
from phca.monitoring import (
    load_session_frames,
    build_session_report,
    compare_all_agents,
    export_session_csv,
    frame_from_json,
    frames_for_agent,
    is_multi_agent_session,
    MomentQuery,
    query_frames,
    query_session_dir,
    navigate_match,
    normalize_observability_json,
    session_agent_ids,
)
```

Submodule imports (`from phca.monitoring.observability import …`, `from phca.monitoring.render import frame_from_json`, etc.) remain supported for existing scripts.

## Stable vs internal

| Stable (`from phca.monitoring import …`) | Internal (no stability guarantee) |
|---|---|
| Symbols in `phca.monitoring.__all__` | `qt_dashboard.py` — PyQt5 Cognitive Observatory UI |
| | `render.py` — matplotlib live dashboard |
| | `camera_render.py` — RGB frame helpers |
| | `action_explain.py` — explain-chain formatting |
| | `session_anomalies.py` — anomaly detection |
| | `retention_slope.py` — RSS slope helpers |

Do not import `qt_dashboard` or `render` from library code that must run without PyQt5/matplotlib.

## Quick start

```python
from pathlib import Path
from phca.monitoring import (
    build_session_report,
    export_session_csv,
    load_session_frames,
    write_session_report,
)

session = Path("logs/sessions/20260704_120000")
frames = load_session_frames(session)
report = build_session_report(
    {"env": "Reacher-v5", "cycles": len(frames)},
    [f'{{"cycle_id": {f.cycle_id}}}' for f in frames],  # or read JSONL lines directly
)
csv_path = export_session_csv(session)  # writes session/scalars.csv
```

### Moment query (Phase 18)

```python
from phca.monitoring import MomentQuery, query_session_dir

matches = query_session_dir(
    "logs/sessions/20260704_120000",
    MomentQuery(spike=True, violation=True),
)
for m in matches:
    print(m.cycle_id, m.flags)
```

For a session directory with `meta.json` + `timeseries.jsonl`:

```python
from phca.monitoring import write_session_report, export_session_csv

report = write_session_report("logs/sessions/20260704_120000")
export_session_csv("logs/sessions/20260704_120000")
```

## Schema contract

- **Current version:** `schema_version = 1` (`OBSERVABILITY_SCHEMA_VERSION`).
- **Legacy v0:** JSONL records without `schema_version` are treated as v0 and normalized on load.
- **Normalization:** Always pass JSON objects through `normalize_observability_json()` before `frame_from_json()`.
- **Fail-closed:** Unknown future schema versions raise `ValueError`; mixed-version JSONL fails `phca_replay.py --check` unless `--allow-incomplete`.

Full field reference (recorded vs live-only fields, `action_rationale` extensions): see [observability.md](observability.md) § JSONL schema.

### `meta.json`

New sessions include `observability_schema_version: 1`. Session reports and CSV export do not require `meta.json` schema fields beyond what `build_session_report` already reads.

## CSV export (`export_session_csv`)

Writes one row per cycle to `<session_dir>/scalars.csv` (or a custom `out_path`).

| Column | Type | Description |
|--------|------|-------------|
| `cycle_id` | int | Cognitive cycle index |
| `agent_id` | int | Agent index (0 default for legacy sessions) |
| `agent_label` | str | Optional agent label |
| `timeline_step` | int | Aligned step index (`-1` legacy) |
| `schema_version` | int | Frame schema version (0 legacy, 1 current) |
| `env_kind` | str | `grid`, `mujoco_rgb`, `continuous`, … |
| `prediction_error` | float | G′ prediction error scalar |
| `prediction_confidence` | float | Model confidence |
| `latency_ms` | float | Cycle wall time |
| `active_drive_id` | int | Active MDIM drive |
| `violations_count` | int | RBTA violation count |
| `goal_reached` | bool | Goal reached this cycle |
| `rss_bytes` | int | Process RSS |
| `rbta_action` | str | RBTA decision (`CONTINUE`, …) |
| `explored` | bool | Explore vs exploit |
| `best_score` | float | Best candidate score (empty if null) |
| `mechanism` | str | `explore`, `prediction`, `greedy_fallback`, … |
| `spike` | bool | Prediction-error spike moment |
| `learn_burst` | bool | G′ learn burst (≥ 5 ms) |
| `decision_shift` | bool | Best-score jump moment |
| `violation` | bool | RBTA violation moment |
| `learn_ms` | float | `module_timings.gprime_learn` |
| `module_ms_total` | float | Sum of all module timings |
| `gprime_learn_ms` | float | G′ learn timing |
| `bottleneck_module` | str | Slowest pipeline module key |

Column order is fixed in `export.SESSION_SCALAR_COLUMNS`.

## Versioning policy

### JSONL schema

- **Additive changes** (new optional fields) do **not** bump `schema_version`.
- **Breaking changes** (rename/remove required fields) require `schema_version = 2` plus an updated `normalize_observability_json()` (see D-110, D-118).
- Phase 14 `action_rationale` extensions are additive under schema v1.

### Python API

- `phca.monitoring.__all__` is the stability boundary.
- New symbols may be added to `__all__` in minor releases.
- Removing or renaming `__all__` symbols requires a deprecation note in `DECISIONS.md` and `STATUS.md`.

## Module map

| Module | Role |
|--------|------|
| `observability.py` | `ObservabilityFrame`, `ObservabilityStore`, schema constants |
| `session_io.py` | Frame load/save, JSONL/session loaders |
| `multi_agent.py` | Multi-agent session helpers (`frames_for_agent`, `--check` validators) |
| `session_query.py` | Cognitive-moment query (`MomentQuery`, `query_frames`, `navigate_match`) |
| `export.py` | `export_session_csv` |
| `session_report.py` | Offline session aggregates and compare |
| `cognitive_panels.py` | Cognitive moment helpers |
| `playback.py` | `PlaybackClock`, `CyclePacer` |
| `belief_projection.py` | PCA projection (Qt-free) |
| `overview_narrative.py` | Overview narrative strings (Qt-free) |
