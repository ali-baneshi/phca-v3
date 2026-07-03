# PHCA Cognitive Observatory — Observability Architecture

The Cognitive Observatory is a **PyQt5 live dashboard** plus **JSONL session recording**, **offline replay**, and **session reports**. One `ObservabilityFrame` is produced per cognitive cycle and is the ground truth for dashboard, JSONL, and reports.

## Thread model

```
CognitiveCycle.step()          [cycle thread]
    └─ ObservabilityFrame.from_cycle()
           └─ ObservabilityStore.push()     [ring buffer]

phca_observatory._tick()       [Qt main thread]
    ├─ SessionRecorder.record() → timeseries.jsonl
    └─ PlaybackClock.push() → DashboardController.update() → 7 tabs
```

Dashboard and report code **must not mutate** `ObservabilityFrame` instances after capture. Replay reconstructs frames via `frame_from_json()` with deep-copied dict/list fields.

## Session directory layout

| File | Purpose |
|------|---------|
| `meta.json` | Run metadata; `cycles` = **requested** run length; `recorded_cycles` written on recorder close |
| `timeseries.jsonl` | One compact JSON object per cycle (~2 KB) |
| `session_report.json` | Offline aggregate (rebuild with `write_session_report()` — do not trust stale files) |
| `session.mp4` / `.gif` | Optional dashboard video (not in JSONL) |

## JSONL schema

### Recorded fields (replay/report)

Scalars, `grid`, `obs_vector`, `predicted_state`, `goal_ref`, `gprime_uncertainty`, `attention_indices`, `attention_saliences`, `candidate_scores`, `action_rationale`, `module_timings` (**ms**), `rbta_bounds` (**time in seconds**), `rbta_violations`, `runtime_log`, `memory_log`, `energy_log`, `drive_*`, `goal_stack`, `goal_history`, `pareto_front`, `meta_stable`, `m3_top_error`, `dim_names`, `action_names`, retention caps, etc.

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
- `rbta_bounds[*].time` → **seconds**
- All near-bound ratios use `flow_timing_ratio()` in `cognitive_panels.py` (ms/ms).

## Replay modes

```bash
# PyQt dashboard from JSONL (full fidelity)
PYTHONPATH=python python scripts/phca_replay.py logs/sessions/<ts>/ --qt

# Matplotlib legacy reconstruct
PYTHONPATH=python python scripts/phca_replay.py logs/sessions/<ts>/ --from-jsonl

# Play recorded video only
PYTHONPATH=python python scripts/phca_replay.py logs/sessions/<ts>/
```

Replay banners appear when `PlaybackClock.mode == "replay"`. Live observatory scrubbing keeps live-only fields in the ring buffer and does **not** show replay banners (intentional).

## Session integrity (`--check`)

```bash
PYTHONPATH=python python scripts/phca_replay.py --check logs/sessions/<ts>/
```

Fails unless:

1. JSONL is non-empty
2. Line count equals `meta.cycles`
3. `cycle_id` is contiguous `0..N-1`
4. All lines parse; `frame_from_json` smoke on first/mid/last
5. Video (if present) is non-zero; `ffprobe` validates stream when available

Use `--allow-incomplete` to skip count/contiguity checks (still fails on empty JSONL).

## Dashboard / report parity

Shared helpers in `python/phca/monitoring/cognitive_panels.py`:

- `rbta_time_bound_ms`, `flow_timing_ratio`, `flow_near_bound_modules`
- `build_moment_series` for offline moment flags

`session_report.py` and Flow/Retention/RBTA panels use the same normalization.

## Manual smoke checklist

1. **Live 50-cycle run**
   ```bash
   QT_QPA_PLATFORM=offscreen PYTHONPATH=python python scripts/phca_observatory.py --cycles=50 --mlp
   ```
2. **Replay scrub all 7 tabs**
   ```bash
   PYTHONPATH=python python scripts/phca_replay.py logs/sessions/<ts>/ --qt
   ```
3. **Write session report**
   ```bash
   PYTHONPATH=python python scripts/phca_replay.py logs/sessions/<ts>/ --report
   ```
4. **Large replay** (1500+ / 3000+ cycles) — scrub seek, watch for lag on Retention rebuild
5. **Integrity check** — complete session PASS; interrupted session FAIL
6. **Video export** — `--record-video` with tab cycling if enabled

## Automated tests

```bash
mkdir -p .tmp
TMPDIR=.tmp QT_QPA_PLATFORM=offscreen PYTHONPATH=python \
  python -m pytest python/phca/monitoring/tests/ -q
```

## Known gaps

- `meta.cycles` is requested count; compare with JSONL via `--check` or `recorded_cycles`
- Legacy matplotlib replay (`render.py`, `phca_visualise.py`) may diverge from PyQt dashboard
- `RetentionView.rebuild_histories` is O(n) per seek (noticeable on 3000+ cycles)
