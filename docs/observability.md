# PHCA Cognitive Observatory — Observability Architecture

The Cognitive Observatory is a **PyQt5 live dashboard** plus **JSONL session recording**, **offline replay**, and **session reports**. One `ObservabilityFrame` is produced per cognitive cycle and is the ground truth for dashboard, JSONL, and reports.

**Phase 8 (largely complete):** PyQt `--qt` replay with seek/scrub, rolling-window history rebuild across all panels, transport controls, and honest replay banners for live-only fields.

## Thread model

```
CognitiveCycle.step()          [cycle thread]
    └─ ObservabilityFrame.from_cycle()
           └─ ObservabilityStore.push()     [ring buffer]

phca_observatory._tick()       [Qt main thread]
    ├─ SessionRecorder.record() → timeseries.jsonl
    └─ PlaybackClock.push() → DashboardController.update() → 7 tabs
```

Dashboard and report code **must not mutate** `ObservabilityFrame` instances or input JSON
after capture (`build_session_report()` deep-reads lines without mutating parsed dicts).
Replay reconstructs frames via `frame_from_json()` with deep-copied dict/list fields.

## Session directory layout

| File | Purpose |
|------|---------|
| `meta.json` | Run metadata; `cycles` = **requested** run length; `recorded_cycles` written on recorder close |
| `timeseries.jsonl` | One compact JSON object per cycle (~2 KB) |
| `session_report.json` | Offline aggregate (rebuild with `write_session_report()` — do not trust stale files) |
| `session.mp4` / `.gif` | Optional dashboard video (not in JSONL) |

## JSONL schema

### Schema governance (Phase 9)

Every new JSONL frame includes `schema_version: 1`. Legacy sessions without
`schema_version` are treated as schema v0 and remain replay/report compatible.
`meta.json` includes `observability_schema_version` for newly recorded sessions.

Replay and report loading pass JSON objects through
`normalize_observability_json()` before constructing an `ObservabilityFrame`.
Unknown future schema versions fail closed in `--check` and replay/report
loading; mixed schema versions in a single JSONL fail `--check` unless
`--allow-incomplete` is explicitly used for forensic inspection.

### Recorded fields (replay/report)

`schema_version`, scalars, `grid`, `obs_vector`, `predicted_state`, `goal_ref`, `gprime_uncertainty`, `attention_indices`, `attention_saliences`, `candidate_scores`, `action_rationale`, `module_timings` (**ms**), `rbta_bounds` (**time in seconds**), `rbta_violations`, `runtime_log`, `memory_log`, `energy_log`, `drive_*`, `goal_stack`, `goal_history`, `pareto_front`, `meta_stable`, `m3_top_error`, `dim_names`, `action_names`, retention caps, etc.

### Live-only (not in JSONL)

| Field | Replay behavior |
|-------|-----------------|
| `env_frame` | Camera RGB; schematic or STALE in replay |
| `candidate_rollouts` | Empty; Action tab shows banner |
| `m3_recent`, `m4_relevant`, `m4_top` | Partial via `m3_top_error`; Memory tab banner |
| `sanitized_state`, `goal_target`, `prediction_precision`, `per_dim_peu`, `last_action_vector`, `attention_weights/precisions` | Synthesize from `obs_vector`/`goal_ref` where safe (`belief_reference()`); otherwise banner |
| `drive_goals` | Goals tab radial inset live-only |

### Unit rules (RBTA)

- `module_timings[*]` → **milliseconds**
- `rbta_bounds[*].time` → **seconds** (converted to ms by `rbta_time_bound_ms()`)
- Near-bound ratio: `flow_timing_ratio(mod, timings, bounds)` = measured_ms / bound_ms
- Near-bound modules: `flow_near_bound_modules(f, top_k=3)` returns modules with ratio **> 0.5**, sorted descending
- Pipeline time budget: `pipeline_time_budget_ms()` sums RBTA time bounds for pipeline modules

## Playback and scrub (Phase 8)

[`PlaybackClock`](python/phca/monitoring/playback.py) decouples display from cycle production.

| Mode | Buffer | Cursor behavior |
|------|--------|-----------------|
| **live** | `push()` appends frames | Cursor catches tail at `speed`; `follow_live()` re-attaches to newest |
| **replay** | `set_frames()` fixed list | Linear advance at `speed` frames per heartbeat |

**Seek / jump:** `seek(i)` sets `scrubbing=True` and emits with `force_rebuild=True`. The controller passes a rolling prefix `frames[max(0, i-200) : i+1]` (up to 201 frames) to every panel's `rebuild_histories()`.

**Sequential step:** when the cursor advances by exactly one frame, `rebuild=False` — panels append to local history only.

**Pause / scrub freeze:** while `paused` or `scrubbing`, the heartbeat does not advance the cursor. `set_autoscale_frozen(True)` holds axis bounds to prevent jitter during scrub.

## Transport controls

[`_TransportBar`](python/phca/monitoring/qt_dashboard.py) provides:

- Scrub slider (seek to any cycle)
- Play / pause
- Speed slider (continuous slow-mo via `throttle_period()`)
- Step forward (one frame while paused)

**Keyboard** ([`ObservatoryWindow.keyPressEvent`](python/phca/monitoring/qt_dashboard.py)):

| Key | Action |
|-----|--------|
| Space | Toggle play / pause |
| Right | Step forward one frame |
| Left | Seek back one frame |
| Home | Seek to cycle 0 |
| End / Esc | Follow live tail (exit scrub) |

## Panel rebuild contract

All tab views implement `rebuild_histories(frames: List[ObservabilityFrame])`. On seek or jump, `DashboardController` invokes rebuild on all 11 views: Overview, Cognitive Flow, Action Selection, Phase Space (trajectory, radar, per-dim), Retention, RBTA bounds, Goals & Motivation, Memory & Belief.

Live sequential ticks call `update(frame)` only; history grows by append unless a jump is detected.

## Replay modes

```bash
# PyQt dashboard from JSONL (canonical Phase 8 path — full fidelity + scrub)
PYTHONPATH=python python scripts/phca_replay.py logs/sessions/<ts>/ --qt

# Offline session report
PYTHONPATH=python python scripts/phca_replay.py logs/sessions/<ts>/ --report

# Matplotlib legacy reconstruct (deprecated for full review)
PYTHONPATH=python python scripts/phca_replay.py logs/sessions/<ts>/ --from-jsonl

# Play recorded video only
PYTHONPATH=python python scripts/phca_replay.py logs/sessions/<ts>/
```

### Replay banners

Banners appear when `PlaybackClock.mode == "replay"` and mark genuinely unavailable live-only data. They must **not** appear during live observatory scrubbing (the ring buffer retains live-only fields).

Panels that draw replay banners include Action Selection, Cognitive Flow, Phase Space, and Memory & Belief. Recorded fields (`module_timings`, `rbta_bounds`, `candidate_scores`, `gprime_uncertainty`, `goal_ref`, counts/caps, `m3_top_error`) remain authoritative in replay without banners.

## Session integrity (`--check`)

```bash
PYTHONPATH=python python scripts/phca_replay.py --check logs/sessions/<ts>/
```

Fails unless:

1. JSONL is non-empty
2. Line count equals `meta.cycles` (unless `--allow-incomplete`)
3. `cycle_id` is contiguous `0..N-1`
4. All lines parse; `frame_from_json` smoke on first/mid/last
5. Video (if present) is non-zero; `ffprobe` validates stream when available
6. `schema_version` is supported; mixed versions fail unless `--allow-incomplete`

When `recorded_cycles` is present, `--check` also reports whether JSONL line count matches it. A mismatch fails unless `--allow-incomplete` is explicitly used; empty JSONL always fails.

Use `--allow-incomplete` to skip count/contiguity checks (still fails on empty JSONL).

## Dashboard / report parity

Shared helpers in [`python/phca/monitoring/cognitive_panels.py`](python/phca/monitoring/cognitive_panels.py):

- `rbta_time_bound_ms`, `flow_timing_ratio`, `flow_near_bound_modules`
- `build_moment_series`, `cognitive_moment`, `append_cognitive_moment`

[`session_report.py`](python/phca/monitoring/session_report.py) and Flow/Retention/RBTA panels use the same normalization. Session reports count near-bound cycles using the same `flow_near_bound_modules()` threshold as the live Flow panel. Reports are built from JSONL only and must not mutate input dicts.

## Manual smoke checklist

1. **Live 50-cycle run**
   ```bash
   QT_QPA_PLATFORM=offscreen PYTHONPATH=python python scripts/phca_observatory.py --cycles=50 --mlp
   ```
2. **Replay scrub all 7 tabs** (seek to 0, mid, end; verify histories rebuild)
   ```bash
   PYTHONPATH=python python scripts/phca_replay.py logs/sessions/<ts>/ --qt
   ```
3. **Write session report**
   ```bash
   PYTHONPATH=python python scripts/phca_replay.py logs/sessions/<ts>/ --report
   ```
4. **Large replay** (500+ cycles) — scrub seek; `test_dashboard_controller_scrub_500_jsonl_frames_no_mutation` guards JSON immutability
5. **Integrity check** — complete session PASS; interrupted session FAIL
6. **Video export** — `--record-video` with tab cycling if enabled

## Automated tests

```bash
mkdir -p .tmp
TMPDIR=.tmp QT_QPA_PLATFORM=offscreen PYTHONPATH=python \
  python -m pytest python/phca/monitoring/tests/ -q
```

**232 tests** (2026-07-03). Key modules:

| Module | Coverage |
|--------|----------|
| `test_playback_store.py` | Seek/rebuild semantics, 500-frame scrub immutability |
| `test_observability_integrity.py` | RBTA units, `--check`, schema governance, report parity, JSON immutability |
| `test_*_dashboard.py` | Per-panel smoke, replay banners, scrub rebuild |
| `test_cognitive_panels.py` | Shared helper contracts |
| `test_session_report.py` | Offline report from JSONL |
| `test_r2_extensions.py` | Moment parity, retention/memory/goals anchors |

## Known gaps (Phase 9+)

| Gap | Target phase |
|-----|--------------|
| Offline report does not cover every dashboard subview | Phase 10 |
| Large sessions (3000+ cycles) may lag on scrub rebuild | Phase 11 |
| Legacy matplotlib replay (`render.py`, `phca_visualise.py`) not full-fidelity | Deprecated |
| `meta.cycles` is requested count; compare with `recorded_cycles` via `--check` | Ongoing |
| GridWorld overview hides camera QLabel; grid body is the camera substitute | By design |

## Troubleshooting (live Observatory)

### Wayland / `libdecor-gtk.so`

On Wayland (`echo $XDG_SESSION_TYPE`), PyQt5 may print:

```text
Failed to load plugin 'libdecor-gtk.so': failed to init
```

This is **not** a crash. The cognitive cycle and JSONL recording continue normally.
Install the system decoration plugin (`libdecor-gtk` on Arch/Manjaro) or run with
`QT_QPA_PLATFORM=xcb` if you prefer XWayland. See [SETUP.md](../SETUP.md).

### Startup warnings

- **`pgmpy` FutureWarning with `--mlp`:** should not appear after lazy-import
  hardening; if it does, use a venv from `requirements.txt`.
- **No video file on `--check`:** expected unless you passed `--record-video`.

## Related documents

- [PHCA_Cognitive_Observatory_Architecture.md](PHCA_Cognitive_Observatory_Architecture.md) — full Observatory architecture and 20-phase roadmap
- [architecture.md](architecture.md) — core PHCA 12-step cycle
