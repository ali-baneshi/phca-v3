# Dead and rarely-reached code

## Zero production fan-in

Modules imported by nothing outside tests:

| Module | Imported by |
|---|---|
| `phca.environments.eval_protocol` | nothing |
| `phca.environments.slow_wrapper` | nothing |
| `phca.evaluation.continual` | nothing |
| `phca.evaluation.continual.pendulum_tasks` | nothing |
| `phca.evaluation.metrics.forgetting` | nothing |
| `phca.evaluation.metrics.interaction` | 1 test |
| `phca.evaluation.runner` | 1 test |
| `phca.memory.migrations` | nothing |
| `phca.monitoring.render` | 11 tests |
| `phca.monitoring.session_recovery` | 2 tests |

`monitoring.render` having 11 importers and zero production callers is the
clearest example of a subsystem kept alive by its own test suite.

## Duplicate abstractions

- **Five world-model modules**: `mlp.py` (982 LOC), `graph.py`, `gaussian.py`,
  `ensemble.py`, `hybrid.py`. The cycle imports MLP, ensemble, and hybrid;
  the discrete default uses `WorldModelGPrime` (graph), and the MLP path is the
  continuous/opt-in one. Only one is needed for a reference implementation.
- **Two observability stacks**: `monitoring/observability.py` (JSONL frames) and
  `evaluation/trace.py` (in-memory records). The trace has 6 production users and
  is what the CLI needs; the frame is only consumed by the GUI.
- **Three memory-ablation paths** (`minimal_cycle`, `enable_m2`, `enable_m3_write`)
  plus `prediction_mode` string modes, plus three selector modes.

## The flag maze

`InterventionConfig` has 30 fields. Sixteen combinations are exercised by tests;
the rest are reachable only by direct construction. Modes that change control
flow:

- `disable_blended_scorer`, `confidence_gated_selector`,
  `adaptive_confidence_gating`, `agreement_gating`, `before_blended_warmup_cycles`
- `prediction_mode` ∈ {normal, zero, scramble, ensemble, deep_ensemble}
- `resource_policy` ∈ {time, energy, memory}
- `minimal_cycle`

The default path (`disable_blended_scorer=True`) makes the blended-scorer code
dead in the shipped configuration. It runs only under `--confidence-gated` or an
explicit `InterventionConfig`.

## Self-referential guarantees

`python/tests/test_static_contracts.py` greps source text for patterns such as
"M3 and G′ both used in learn" and "B4 recovery calls on forgetting detected".
These assert that code *contains* a construct, not that the construct has an
effect. Several will block legitimate removal and must be rewritten or quarantined
with the claim they encode.

## Generated / duplicated artifacts

- 227 git-tracked files under `logs/` (43 MB), overwhelmingly single-run JSON dumps.
- `results/validation/` duplicates `results/validation_smoke/` with different
  horizons.
- `Diagrams/` contains 9 mermaid sources and 9 PNG renders, several of which describe removed
  phases.
- Empty JS toolchain files (`package.json`, `pnpm-lock.yaml`, `yarn.lock`,
  `.yarnrc`, `.yarnrc.yml`, `bunfig.toml`, `.npmrc`, `package-lock.json`),
  `.gitmodules`, an empty `node_modules/`, and eight zero-byte `.env*` files.
  **None of these are tracked by git.** They all have the same mtime and look like
  local tool scaffolding, not project content. Nothing to remove from the
  repository; they are out of scope.

## Estimation

Removing monitoring, the non-MLP world models, evaluation baselines/continual,
consolidation, HPM, and the non-default selector paths reaches **~8k lines**
without touching the causal core. The remaining cycle is still the largest single
file at ~3.5k lines because it inlines orchestration that belongs in the modules
it calls.
