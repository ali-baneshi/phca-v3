# PHCA Cognitive Observatory — Observability Architecture

The Cognitive Observatory is a **PyQt5 live dashboard** plus **JSONL session recording**, **offline replay**, and **session reports**. One `ObservabilityFrame` is produced per cognitive cycle and is the ground truth for dashboard, JSONL, and reports.

**Phases 8–16 complete:** PyQt replay/scrub (Phases 8–12), anomaly detection (13), action explainability (14), stable `phca.monitoring` API (15 — [observability_api.md](observability_api.md)), subprocess supervisor + crash recovery (16 — § Production operations). **Phase 17 complete:** multi-agent `agent_id` on frames, aligned local runner, per-agent dashboard timeline, per-agent reports (§ Multi-agent sessions). **Phase 18 complete:** cognitive-moment query CLI + dashboard filters (§ Interactive query). Phases 19–20 are backlog — see [STATUS.md](../STATUS.md).

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
| `meta.json` | Run metadata; `cycles` = **requested** run length; `status` lifecycle; `recorded_cycles` on close |
| `timeseries.jsonl` | One compact JSON object per cycle (~2 KB) |
| `session_report.json` | Offline aggregate; written after live runs or recovery |
| `session.mp4` / `.gif` | Optional dashboard video (not in JSONL) |
| `<record-root>/.latest` | Pointer to active session dir while `status=running` |

#### `meta.json` lifecycle fields (Phase 16)

| Field | Values | When set |
|-------|--------|----------|
| `status` | `running`, `complete`, `incomplete`, `empty`, `corrupt` | `start()` → `running`; normal `close()` → `complete`; crash/early exit → `incomplete`/`empty` |
| `started_at` | ISO-8601 UTC | `start()` |
| `closed_at` | ISO-8601 UTC | `close()`, `abort()`, or `finalize_session()` |
| `recorded_cycles` | int | Actual JSONL lines written |
| `recovery_reason` | string | `abort()`, supervisor recover, or manual `--recover` |

### Post-run trust (live launcher)

After a recorded session, `phca_observatory.py` prints a structured summary:

```text
=== Session summary ===
  dir:      logs/sessions/<ts>
  cycles:   N / N
  verify:   PASS (contiguous cycle_id, schema v1)
  report:   logs/sessions/<ts>/session_report.json
=== Exit 0 ===
```

Verify runs `phca_replay.py --check` by default; use `--no-verify` to skip. Non-zero
exit if verify or report write fails.

**Review mode (default):** the window stays open after completion — transport scrub
works, Overview shows a session-results panel (early/late metrics, mechanism mix,
optional cross-session + Φ-IQ context). Close the window to exit; use `--close-at-end`
for CI auto-close.

## Dashboard coordination (Phase 10 — complete)

- **Session status strip** — env, `env_kind`, camera mode, cycle/lag, schema, JSONL count.
- **LIVE / REPLAY / REVIEW data-contract banners** — shared strings in `cognitive_panels.py`.
- **Cross-tab moment badges** — tab titles suffix `• SPIKE` / `VIOL` / `DECISION` when active.
- **Mechanism rollup** — Retention tab rolling 256-cycle histogram (parity with `session_report`).
- **Overview session results** — post-run panel from `session_report.json` with optional `--compare-report` and `--benchmark-report`.
- **Review mode** — window stays open after run (default); `--close-at-end` for CI.
- **MuJoCo JSONL honesty** — non-grid sessions omit misleading `agent_pos`/`goal_pos`.
- **Scrub perf (Phase 11 — complete)** — decimated rolling rebuild + lazy per-tab rebuild on seek; budget tests ≤2s/≤4s at 3000+ cycles.

## Dashboard tab index

| # | Tab | Primary focus |
|---|-----|---------------|
| 0 | Overview | Grid/MuJoCo world, drives, error/confidence, session-results (review) |
| 1 | Cognitive Flow | Pipeline timings, near-bound, violations |
| 2 | Action Selection | Scores, explore/exploit, rollouts (live-only) |
| 3 | Phase Space & Trajectory | Trajectory, radar, per-dim traces |
| 4 | Retention & Resources | RSS/M3/M4, mechanism rollup, RBTA bounds table |
| 5 | Memory & Belief | M1/M2/M3 snapshots, `m3_top_error` |
| 6 | Goals & Motivation | Drives D1–D6, goal stack, pareto |

Tab labels match `OBSERVATORY_TAB_LABELS` in [`cognitive_panels.py`](../python/phca/monitoring/cognitive_panels.py).

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

`schema_version`, `agent_id`, `agent_label`, `timeline_step`, scalars, `grid`, `obs_vector`, `predicted_state`, `goal_ref`, `gprime_uncertainty`, `attention_indices`, `attention_saliences`, `candidate_scores`, `action_rationale`, `module_timings` (**ms**), `rbta_bounds` (**time in seconds**), `rbta_violations`, `runtime_log`, `memory_log`, `energy_log`, `drive_*`, `goal_stack`, `goal_history`, `pareto_front`, `meta_stable`, `m3_top_error`, `dim_names`, `action_names`, retention caps, etc.

| Field | Default | Description |
|-------|---------|-------------|
| `agent_id` | `0` | Agent index within a session (legacy single-agent = 0) |
| `agent_label` | `""` | Optional human label (`grid_a`, `agent_1`, …) |
| `timeline_step` | `-1` | Shared step index when agents run aligned; `-1` = legacy/single-agent |

#### Multi-agent sessions (Phase 17)

- **Recording layout:** one `timeseries.jsonl` with interleaved lines (`agent_id` per line). Not per-agent subdirectories.
- **Aligned runner:** `phca_observatory.py --agents N` steps N independent cycles in lockstep on one coordinator thread; each logical step writes N JSONL lines sharing `timeline_step`.
- **`meta.json`:** `agent_count`, `agents[]` (`agent_id`, `label`, `recorded_cycles`), `timeline_mode: aligned`, `recording_layout: single_jsonl`. `meta.cycles` / `recorded_cycles` = total JSONL lines (agents × cycles-per-agent).
- **`--check`:** per-agent `cycle_id` contiguous `0..n-1` (not global line index). Single-agent sessions keep the legacy global contiguous check.
- **Dashboard:** transport agent selector; scrub/replay cursor indexes the **selected agent's** timeline. Banners note camera/bulk memory are per-agent live-only.
- **Reports:** `session_report.json` includes `agents: {"0": {...}, "1": {...}}` for multi-agent sessions; top-level scalars mirror agent 0 for backward compatibility. `compare_session_reports(..., agent_id=0)` selects a bucket; `compare_all_agents()` compares every agent.

#### `action_rationale` (Phase 14 extension — no schema_version bump)

Nested object per cycle. Legacy keys remain; new keys are optional.

| Field | Type | Description |
|-------|------|-------------|
| `explored`, `eps`, `goal_id`, `continuous`, `best_score`, `k_candidates`, `chosen_idx` | various | Pre-Phase-14 selection metadata |
| `score_components` | object | Discrete: `distance_gain`, `pga`, `confidence`, `alignment`, …; continuous MPC: `confidence`, `ref_align`, `pga` |
| `relevant_fact_ids` | list[str] | M4 fact ids used during selection |
| `drive_id` | int | Active drive at selection (usually equals `goal_id`) |
| `mechanism` | str | `explore`, `prediction`, `greedy_fallback`, `stay`, `continuous`, `rbta_safe`, `other` |
| `decision_reason` | str | Finer grain: `explore`, `d5_stay`, `rbta_safe`, `greedy_fallback`, `prediction`, `continuous_mpc`, `continuous_explore` |
| `relevant_facts_summary` | list | `[{fact_id, confidence}]`, cap 5 |
| `chosen_label` | str | Human label: `MOVE_E #2` or `τ#3` |

Shared formatter: [`action_explain.py`](../python/phca/monitoring/action_explain.py).

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

**Seek / jump (JSONL replay):** `seek(i)` sets `scrubbing=True` and emits with `force_rebuild=True`. The controller passes a rolling prefix `frames[max(0, i-200) : i+1]` (up to 201 frames) to the visible tab's `rebuild_histories()`; other tabs rebuild lazily on tab switch.

**Seek / jump (post-run review):** when `ObservatoryWindow._review_mode` and `PlaybackClock.review_mode` are set, scrub uses the full prefix `frames[0 : i+1]` (decimated to 2000 points) and `DashboardController.rebuild_all_histories()` refreshes **all seven tabs** on every seek so Phase Space, Retention, Memory, and Goals stay coherent.

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

All tab views implement `rebuild_histories(frames: List[ObservabilityFrame])`. On seek or jump, `DashboardController` invokes rebuild on **10 canvas views** across 7 tabs: Overview, Cognitive Flow, Action Selection, Phase Space (trajectory, radar, per-dim), Retention (+ embedded RBTA bounds), Memory & Belief, Goals & Motivation.

Live sequential ticks call `update(frame)` only; history grows by append unless a jump is detected.

## Replay modes

```bash
# PyQt dashboard from JSONL (canonical Observatory path — full fidelity + scrub)
PYTHONPATH=python python scripts/phca_replay.py logs/sessions/<ts>/ --qt

# Offline session report
PYTHONPATH=python python scripts/phca_replay.py logs/sessions/<ts>/ --report

# Multi-session comparison (Phase 12)
PYTHONPATH=python python scripts/phca_replay.py logs/sessions/<new>/ --compare logs/sessions/<prev>/
PYTHONPATH=python python scripts/phca_replay.py logs/sessions/<new>/ --compare logs/sessions/<prev>/ --compare-output .tmp/compare.json

# Matplotlib legacy reconstruct (deprecated for full review)
PYTHONPATH=python python scripts/phca_replay.py logs/sessions/<ts>/ --from-jsonl

# Multi-agent aligned run (Phase 17 — local, 2 agents default via wrapper)
PYTHONPATH=python python scripts/phca_observatory.py --agents 2 --cycles 100 --env gridworld
PYTHONPATH=python python scripts/phca_multi_observatory.py --cycles 100 --no-record

# Play recorded video only
PYTHONPATH=python python scripts/phca_replay.py logs/sessions/<ts>/
```

## Interactive query (Phase 18)

Query recorded sessions for cognitive moments (spike, violation, explore, near-bound module, drive, etc.) without opening the dashboard.

```bash
# List cycle_ids with prediction-error spikes
PYTHONPATH=python python scripts/phca_query.py logs/sessions/<ts>/ --spike

# Structured JSON + count
PYTHONPATH=python python scripts/phca_query.py logs/sessions/<ts>/ --violation --json

# Near-bound prediction module, cycles 100–500
PYTHONPATH=python python scripts/phca_query.py logs/sessions/<ts>/ \
  --near-bound prediction --cycle-range 100:500

# Export matching frames to JSONL
PYTHONPATH=python python scripts/phca_query.py logs/sessions/<ts>/ --explore --export /tmp/explore.jsonl

# Multi-agent: filter one agent
PYTHONPATH=python python scripts/phca_query.py logs/sessions/<ts>/ --spike --agent-id 1
```

**Filter semantics:** enabled flags are **AND**ed (`--spike --violation` = both). With no moment flags, `--cycle-range` alone matches all cycles in range. Uses `build_moment_series()` / `cognitive_moment()` — same logic as dashboard badges and `session_report` counts.

**Output modes:** plain `cycle_id` lines (or `agent_id:cycle_id` for multi-agent), `--json`, `--count-only`, `--export PATH`.

**Dashboard:** transport bar filter combo + `◀ moment` / `moment ▶` in replay/review (paused or scrubbing). Keyboard: `[` / `]` jump prev/next match. Disabled during live production follow.

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
3. `cycle_id` contiguous `0..N-1` per agent (multi-agent) or globally (single-agent)
4. All lines parse; `frame_from_json` smoke on first/mid/last
5. Video (if present) is non-zero; `ffprobe` validates stream when available
6. `schema_version` is supported; mixed versions fail unless `--allow-incomplete`

When `recorded_cycles` is present, `--check` also reports whether JSONL line count matches it. A mismatch fails unless `--allow-incomplete` is explicitly used; empty JSONL always fails.

Use `--allow-incomplete` to skip count/contiguity checks (still fails on empty JSONL).

When integrity checks pass, `--check` also runs session anomaly detection and prints:

```text
  anomalies  : spike=PASS drift=PASS leak=PASS goal_instability=PASS
  Anomaly overall: PASS
```

Anomaly flags are **informational** by default (exit code follows JSONL integrity only). Use `--anomaly-strict` to exit 1 when a **critical** flag is active (currently `leak` only).

When integrity checks pass, `--check` also reports explain field presence:

```text
  explain    : PASS (decision_reason in first/mid/last)
  explain    : WARN (legacy rationale only)
```

PASS requires Phase 14 `decision_reason` or `mechanism` on first/mid/last frames; WARN for legacy sessions — does not change exit code.

## Action explainability (Phase 14)

The Action Selection tab includes a replay-safe **explain band** (drive → reason → score → chosen → facts) built from JSONL fields only via `build_explain_chain()`. `candidate_rollouts` remain live-only (existing replay banner).

`session_report.json` adds `action_metrics.explain_metrics`:

- `decision_reason_counts` — histogram over the session
- `anchor_explain` — causal chains at anchor cycles (`0`, `mid`, `last`)

## Anomaly detection (Phase 13)

Shared module: [`session_anomalies.py`](../python/phca/monitoring/session_anomalies.py). RSS slope helpers: [`retention_slope.py`](../python/phca/monitoring/retention_slope.py) (same phase-aware thresholds as `nightly_stress.py`, D-112/D-113).

| Flag | Session rule | Severity |
|------|--------------|----------|
| `spike` | Spike rate > 20% when cycles ≥ 30, or ≥ 10 spikes when cycles ≥ 50 | info |
| `drift` | Late error median > 1.20× early and absolute Δ > 0.5 (distance drift uses 1.15× when kinematics present) | warn |
| `leak` | JSONL `rss_bytes` late slope ≥ phase threshold (5000 B/cyc fill / 1600 B/cyc post-cap); needs ≥ 50 RSS samples and ≥ 200 cycles for post-cap gate | **critical** |
| `goal_instability` | Combined drive-switch rate > 25%, or ≥ 3 goal/drive switches in any 20-cycle window | warn |

Cycle-level markers (cap 50) appear in `session_report.json` under `anomalies.cycle_markers` (e.g. per-cycle `spike` from `build_moment_series()`).

```bash
# Integrity + anomaly summary
PYTHONPATH=python python scripts/phca_replay.py --check logs/sessions/<ts>/

# Fail on critical leak flag
PYTHONPATH=python python scripts/phca_replay.py --check --anomaly-strict logs/sessions/<ts>/
```

**Nightly:** `make nightly` step `[7/7]` runs `scripts/nightly_anomaly_gate.py` (synthetic positive/negative checks; writes `logs/nightly_anomaly_gate.json`).

**Optional longer soak (pre-release, not in default nightly):**

```bash
QT_QPA_PLATFORM=offscreen PYTHONPATH=python \
  python scripts/phca_observatory.py --cycles=200 --mlp --close-at-end
PYTHONPATH=python python scripts/phca_replay.py --check --anomaly-strict logs/sessions/<latest>/
```

## Production operations / crash recovery (Phase 16)

### Subprocess supervisor (recommended for CI)

Run the Observatory in an isolated child process. On crash or incomplete session, the supervisor finalizes `meta.json`, writes `session_report.json`, and optionally runs `phca_replay --check` with `--allow-incomplete`.

```bash
QT_QPA_PLATFORM=offscreen PYTHONPATH=python \
  python scripts/phca_observatory_supervisor.py \
  --record-dir logs/sessions \
  -- python scripts/phca_observatory.py --cycles=200 --mlp --close-at-end
```

Supervisor flags:

| Flag | Purpose |
|------|---------|
| `--supervisor-log PATH` | JSONL event log (default `logs/supervisor/<ts>.jsonl`) |
| `--record-dir DIR` | Session root (default: parse from child `--record-dir` or `logs/sessions`) |
| `--no-recover` | Do not finalize partial sessions |
| `--no-verify` | Skip `phca_replay --check` after recovery |
| `--strict-verify` | Fail if strict `--check` fails (default recovery uses `--allow-incomplete`) |

Structured log events: `supervisor_start`, `child_spawn`, `child_exit`, `recover_start`, `recover_done`, `verify_result`.

### Manual recovery

```bash
PYTHONPATH=python python scripts/phca_replay.py logs/sessions/<ts>/ --recover
PYTHONPATH=python python scripts/phca_replay.py logs/sessions/<ts>/ --recover --allow-incomplete
```

### Abnormal exit runbook

| Situation | Expected artifacts | Strict `--check` | Forensic `--check --allow-incomplete` |
|-----------|-------------------|------------------|---------------------------------------|
| Normal completion | `status=complete`, full JSONL | PASS | PASS |
| User closes window early | `status=incomplete`, partial JSONL + report | FAIL | PASS (if lines parse) |
| SIGKILL / segfault | partial JSONL, no `close()` | FAIL | PASS after `--recover` |
| Empty JSONL | `status=empty` | FAIL | FAIL |

Early window close and SIGTERM (graceful) trigger `stop_flag`; the cycle thread still owns M3 SQLite close. SIGTERM does not close M3 from the signal handler.

### Headless CI vs desktop

| Environment | Recommendation |
|-------------|----------------|
| CI / headless | `QT_QPA_PLATFORM=offscreen` + supervisor + `--close-at-end` |
| Desktop Linux | Interactive review (default); optional `--close-at-end` |
| Wayland (KDE/GNOME) | Cosmetic `libdecor-gtk` noise — see [SETUP.md](../SETUP.md); use `QT_QPA_PLATFORM=xcb` to silence |

Implementation: [`session_recovery.py`](../python/phca/monitoring/session_recovery.py), [`phca_observatory_supervisor.py`](../scripts/phca_observatory_supervisor.py).

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

**311 monitoring tests** (656 total with MuJoCo — 2026-07-04). Key modules:

| Module | Coverage |
|--------|----------|
| `test_playback_store.py` | Seek/rebuild semantics, lazy tab rebuild, 3000-frame scrub budget |
| `test_observatory_launcher.py` | Post-run verify + session_report pipeline |
| `test_session_recovery.py` | Session recovery, supervisor crash simulation, strict vs `--allow-incomplete` |
| `test_observability_integrity.py` | RBTA units, `--check`, schema governance, report parity, JSON immutability |
| `test_*_dashboard.py` | Per-panel smoke, replay banners, scrub rebuild |
| `test_cognitive_panels.py` | Shared helper contracts |
| `test_session_report.py` | Offline report from JSONL |
| `test_session_anomalies.py` | Phase 13 spike/drift/leak/goal flags |
| `test_action_explain.py` | Phase 14 action rationale explain chain |
| `test_r2_extensions.py` | Moment parity, retention/memory/goals anchors |

## Known gaps (remaining)

| Gap | Status |
|-----|--------|
| Offline report does not cover every dashboard subview (camera, full rollouts) | By design — live-only fields marked in replay banners |
| Legacy matplotlib replay (`render.py`, `phca_visualise.py`) not full-fidelity | Deprecated — use PyQt `--qt` |
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
