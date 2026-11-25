# Critique

## The central claim

The architecture's thesis is that action selection should be driven by a learned
predictive model rather than hand-written geometry. The repository's own record
shows this was tested, appeared to fail catastrophically, and then the failure was
retracted as a measurement artifact (D-159: 0.03% → 77.8% after fixing bound
scaling). The default is nevertheless still geometry.

So the honest current state is not "the hypothesis failed". It is **"the
comparison has not been run cleanly at adequate power"**. A minimal version should
say exactly that and ship the geometric default with the learned scorer as a
measurable opt-in — not inherit either the triumphalist or the defeatist framing.

## What is genuinely here

Little of the machinery is novel in isolation. Prediction-error-driven learning,
homeostatic intrinsic motivation, resource-bounded control, and model-based action
scoring are all standard. The distinguishing idea is the **coupling**: one
enforcement mechanism (RBTA) simultaneously bounds time, memory, energy, and belief
entropy, and the agent's action changes when any of them is violated.

That coupling is testable and interesting. Everything else is scaffolding around
it.

## Where rigor is simulated

- **The headline benchmark is mostly saturated.** On 5×5 the start is 2-4 steps
  from the goal, the goal does not move, and "goal rate" counts cycles spent on
  the goal. The agent reaches it at cycle 1-3 and then sits there, so 196/200 is
  structural, not learned. Graph and MLP models give identical goal counts because
  neither selects the action.
- **Learning curves depend on an undeclared package.** Default-path prediction
  error and confidence differ cycle by cycle depending on whether `pgmpy` is
  installed, because `graph.py` silently falls back without it. Any PE number in the
  repository's tables is uninterpretable unless its environment is known.
- **Φ-IQ** is a composite of internal counters with level thresholds, described as
  a cognitive metric without external validation. It is a dashboard number.
- **21 numbered steps mapped onto 12 stages**, plus per-stage sub-labels, reads as
  formal specification but corresponds to sequential code.
- **`test_static_contracts.py`** asserts that source text contains constructs. That
  is architecture cosplay: it cannot fail for a scientific reason.
- **The RBTA bounds themselves** are partly formulaic (`energy = runtime × 50`,
  hardcoded entropies for eight modules). A "resource bound" derived from the same
  clock it polices is a tautology unless the constants are justified.
- **Two RBTA checks per step** (preflight and post-action) build different
  composition trees, so enforcement semantics depend on which one fires. This, plus
  the pgmpy/torch import cost, produced the cold-start defect.

## What is over-engineered in a way that obscures the science

1. **15,162 lines of GUI** for a system whose research question is about action
   selection. 49% of the codebase is an observatory.
2. **Five world-model modules** with no single comparison at matched budget.
3. **30 intervention fields**, of which the default path reaches a handful. Ablation
   configurability has replaced ablation reporting.
4. **A 300 KB decision log** where the retraction of the central finding is buried
   as an addendum.

## Smallest interesting, checkable artifact

One cycle (`core`), one grid env, **one** world model, M1/M2/M3, MDIM drives, RBTA,
and TSPL. That is enough to show: prediction error falls; an energy bound changes
the action; the geometric and learned selectors can be compared at matched seeds on
an honest metric (cycles to first goal, across larger grids with resets).
Everything else is optional.

### World-model decision

The default model is the pgmpy graph model, but its useful behavior depends on an
undeclared, ~2 GB dependency chain (pgmpy → torch), and it only models 10 of 84
state dims on 5×5 (`coverage_ratio=0.119`, logged as a warning every run). The MLP
model is pure numpy, covers all dims, and is what the README calls "G′".

**Decision: make the MLP the only world model and the default.** This changes
default-path prediction metrics (not actions, which geometry selects). It is a
behavior change, so it gets its own commit with a test that pins the new default
and a golden trace regenerated after it.

## Cut ledger

### Keep (target ≤8k LOC)

| Item | LOC | Justification |
|---|---|---|
| `core/cycle.py` | 3,678 | The subject of study |
| `config.py` + `logging.py` | 446 | Types and a stdlib logger |
| `memory/` (M1/M2/M3) | 1,282 | Working memory is load-bearing for the cycle |
| `world_model/mlp.py` | 982 | The learned predictor |
| `motivation/mdim.py` | 773 | The intrinsic-drive claim |
| `regulation/` | 684 | RBTA is the thesis |
| `environments/grid_world.py` + `protocol` | 512 | The demo environment |
| `learning/tspl.py` | 318 | Skill compilation used by the learn step |
| `attention/` | 272 | Cheap, and modulates confidence |
| `prediction/` | 190 | Prediction-error computation |
| `asi/sanitizer.py` | 153 | Step 0 of the cycle |
| `resilience/` (detector + fallback) | 371 | Only the symbols the cycle observes |
| `evaluation/trace.py` | 132 | CLI output format |
| new `cli.py` | ~250 | The supported surface |

Sum: **~10.4k** before deduplication inside `cycle.py`. Reaching 8k requires
extracting or deleting ~2.4k of inlined orchestration and helper flags. That is
the realistic target; if it cannot be reached without changing behavior, the
exception goes in `MINIMIZATION_REPORT.md`.

### Attic

`monitoring/` (15,162), `world_model/{graph,gaussian,ensemble,hybrid}`,
`evaluation/{baselines,continual,metrics,runner,interventions*}`, `consolidation/`
(516), `hpm/` (188), `asi/{adapter,noise_injector,grounding}`, MuJoCo/CartPole/Reacher
environments, 9 monitoring scripts, `research/`, `docs/lessons/`, `Diagrams/`,
`experiments/`, 124 docs, unreferenced tracked logs.

`evaluation/interventions.py` is a special case: `InterventionConfig` is imported by
the cycle and by tests, so a **minimized** copy (the fields that change control flow)
stays in `phca/config.py`; the 30-field version is attic'd.

### Delete outright

Duplicated tracked result trees and stale tracked logs. The empty JS/`.env*` files
and the cache directories (`build/`, `.pytest_cache/`, `.ruff_cache/`,
`.pytest_tmp/`) are untracked local files, so they are not part of the repository
and are left alone.

## Known risks

- `core/cycle.py` imports every package; cuts must proceed leaf-first with the
  golden trace diffed after each step.
- `test_static_contracts.py` will fail on legitimate removals.
- The demo must remain deterministic; the warm-up fix is a prerequisite, not a
  convenience.
