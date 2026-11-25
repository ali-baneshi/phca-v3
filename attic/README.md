# attic/

Quarantined code, scripts, tests, and documents removed during minimization.
Everything here is tracked so history and context remain browsable, but none of
it is part of the product.

**Inert by configuration:**

- `pytest.ini` — `norecursedirs` includes `attic`.
- `pyproject.toml` — packages are discovered only under `python/`; ruff
  `extend-exclude` includes `attic`.
- `.dockerignore` — `attic/` is excluded from images.

Nothing outside `attic/` may import from it. Restoring an item means moving it
back with `git mv` and re-adding its dependencies and tests.

## Ledger

| Path | Why removed | Replacement | Commit |
|---|---|---|---|
| `python/phca/monitoring/` (164 files, 22,171 LOC incl. tests) | PyQt5 Observatory, JSONL session store, dashboards. 49% of production code; not on any cognitive code path | `CycleObserver` protocol in `phca.core.cycle` (`on_cycle(cycle, metrics)`) | see `git log -- attic/python/phca/monitoring` |
| `scripts/phca_observatory.py`, `phca_observatory_supervisor.py`, `phca_multi_observatory.py`, `phca_replay.py`, `phca_query.py`, `phca_visualise.py`, `camera_probe.py`, `camera_probe_qt.py` | GUI, replay, and camera front-ends for monitoring | `phca` CLI (planned) | same |
| `scripts/nightly_stress.py`, `scripts/nightly_anomaly_gate.py` | Depend on monitoring retention/anomaly modules | Removed from `make nightly` | same |
| `python/tests/test_observatory_export.py` | Tested GUI frame JSON serialization, not cycle behavior | — | same |
