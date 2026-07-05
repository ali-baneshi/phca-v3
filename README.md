# PHCA v3.0 — Predictive Hierarchical Cognitive Architecture

PHCA is an experimental research prototype for resource-bounded cognitive architectures. It investigates how predictive processing, hierarchical memory, intrinsic motivation, and resource-aware control can be integrated into a formally specified, reproducible architecture for continual learning and autonomous decision-making.

The architecture is organised as a deterministic cognitive cycle in which perception, prediction, memory, planning, adaptation, and action are executed under explicit computational constraints. Every cycle is supervised by the Resource-Bounded Turing Supervisor (RBTA), which enforces time, memory, energy, and entropy budgets while preserving architectural invariants.

Rather than optimising for a single benchmark, PHCA is designed as a research platform for studying bounded intelligence under controlled experimental conditions. The project emphasises reproducibility, formal architectural assumptions, causal evaluation, ablation studies, scaling experiments, out-of-distribution testing, and long-horizon validation.

PHCA currently supports discrete GridWorld environments together with continuous-control environments, enabling the same architectural principles to be evaluated across different task domains. The repository includes the complete implementation, reproducible experimental pipelines, scientific validation suite, and documentation describing the architectural assumptions, design rationale, and empirical results.

---

## Overview

PHCA (Predictive Hierarchical Cognitive Architecture) is a research
architecture built around five verified invariants (A1–A5, see
[research/outputs/07-rigorous-whitepaper.md](research/outputs/07-rigorous-whitepaper.md)):

- **A1 Resource Boundedness** — every module has time/memory/energy/entropy
  budgets, enforced every cycle by the RBTA.
- **A2 Temporal Causality** — module outputs are consumed only after they
  are produced (pipeline ordering).
- **A3 Incomplete Knowledge** — belief entropy is floored at ε > 0.
- **A4 Prediction as Primary** — every cycle computes ŝₜ₊₁ from sₜ.
- **A5 Feedback-Driven Adaptation** — prediction error drives TSPL learning.

The agent runs in a GridWorld (discrete, 5×5 with walls) or a MuJoCo physics
environment (Cartpole, Pendulum, Reacher), perceives its state, predicts the
next state, selects a goal-directed action, learns from the prediction error,
and self-regulates its exploration/criticality via a PID controller and six
homeostatic intrinsic drives (MDIM).

It exists to study bounded, autonomous, prediction-first agents that learn
continually without reward hacking — the formal argument for why each
component is necessary is in the whitepaper.

---

## Quick Start

```bash
# Install dependencies
make setup
# Optional MuJoCo (Cartpole/Pendulum/Reacher):
pip install -r requirements-mujoco.txt

# Run all tests (766 in python/ + 36 MuJoCo optional; set MUJOCO_GL=disabled for headless)
MUJOCO_GL=disabled make test-all
# MuJoCo integration tests (make test-all does not include them):
MUJOCO_GL=disabled make test-mujoco

# Canonical benchmark (4 levels, 200 cycles each, MLP mode)
MUJOCO_GL=disabled PYTHONPATH=python python scripts/benchmark.py --use-mlp --cycles=200

# Quick smoke test (Level 0 only, 20 cycles)
PYTHONPATH=python python scripts/benchmark.py --quick

# Dynamic-goal curriculum (L2 relocates the goal every 75 cycles — validated)
MUJOCO_GL=disabled PYTHONPATH=python python scripts/benchmark.py --use-mlp --cycles=200 --dynamic-goals --dynamic-goals-every 75

# MuJoCo environments — Pendulum + Reacher CONTINUOUS (Phase 6/7); Cartpole discrete
MUJOCO_GL=disabled PYTHONPATH=python python scripts/benchmark.py --env pendulum --use-mlp --cycles=100
MUJOCO_GL=disabled PYTHONPATH=python python scripts/benchmark.py --env cartpole --use-mlp --cycles=100
MUJOCO_GL=disabled PYTHONPATH=python python scripts/benchmark.py --env reacher  --use-mlp --cycles=100

# gprime_learn per-module profile (Phase 5 perf target)
MUJOCO_GL=disabled PYTHONPATH=python python scripts/profile_mlp_learn.py --cycles=200

# Phase 6 scientific hardening: OOD calibration + assumption validation
MUJOCO_GL=disabled PYTHONPATH=python python scripts/ood_calibration.py --output=logs/ood_calibration.json
MUJOCO_GL=disabled PYTHONPATH=python:scripts python scripts/assumption_validation.py --ci

# Phase 6 CI hardening: nightly stress + MuJoCo gate (one command, exit 0 = all green)
make nightly NIGHTLY_CYCLES=1000          # CI; use NIGHTLY_CYCLES=10000 for a true soak

# Phase 19: one-command scientific reproduction (see docs/reproducibility.md)
make reproduce-quick                      # ~10-15 min CI-science subset
make reproduce                            # ~45-90 min full nightly-equivalent suite

# CI Φ-IQ regression gate (static + MuJoCo modes)
python scripts/check_benchmark_gate.py logs/benchmark_report.json logs/benchmark_ci_baseline.json
python scripts/check_benchmark_gate.py --mujoco logs/nightly_mujoco_pendulum.json logs/nightly_mujoco_cartpole.json
python scripts/check_benchmark_gate.py --neg-test   # proves the gate catches violations

# Causal behavior gate: PHCA vs non-PHCA GridWorld controls, levels 1-3
PYTHONPATH=python python scripts/phca_causal_eval.py --levels all --cycles 200 --seeds 5 --output .tmp/phca_causal_eval_levels_200x5.json
```

---

## Architecture

PHCA executes a **12-step cognitive cycle** at ~95 Hz on consumer hardware
(~10.6 ms mean latency, MLP path, post-Phase-5). The cycle connects ASI
(input sanitisation) → working memory → world-model prediction → prediction
error → learning → action selection → intrinsic motivation → regulation →
attention → RBTA enforcement → consolidation.

### Cognitive Cycle (12 active steps)

```mermaid
flowchart TD
    ENV["Environment (GridWorld / MuJoCo)"] -- "raw_obs" --> ASI
    subgraph cycle ["Cognitive Cycle (12 active steps)"]
      ASI["Step 0: ASI Sanitize<br/>sanitize(raw_obs) -> clean_state"]
      WM["Step 1: WM Write<br/>m2.write(state) + m1.write(state)"]
      PE["Steps 2-4: G' Prediction<br/>engine.predict(state) -> predicted, confidence"]
      PEU["Steps 5-6: PEU Error<br/>peu.compute(next, prediction) -> error"]
      TSPL["Step 7: TSPL P-Stream<br/>tspl.update(error, state, prediction)"]
      LEARN["LEARN: gprime.learn(transition)<br/>(replay mini-batch, batched)"]
      ACT["Step 9: Action Selection<br/>branch on ActionSpace<br/>discrete: argmax(goal_align+conf)<br/>continuous: MPC sample K, pick best ŝ'→ref"]
      REG["Steps 10-13: MDIM + APC + ATTN + HPM<br/>generate_goal, regulate, attend, bounds"]
      RBTA["Step 14: RBTA Enforcement<br/>check_cycle(time, mem, energy, entropy)"]
      LOG["Step 15: Logging<br/>append metrics"]
      CONSOL["Steps 16-18: Consolidation<br/>episodic -> semantic transfer"]
      INC["Step 19: Increment<br/>cycle_count += 1"]
      ASI --> WM --> PE --> ACT
      ACT --> PEU --> TSPL --> LEARN
      PEU --> REG --> RBTA --> LOG --> CONSOL --> INC
    end
    ACT -- "action (int OR np.ndarray)" --> ENV
    ENV -- "step(action) -> obs, reward, terminal" --> PEU
```

### Module Map

| Module | File | Function |
| :--- | :--- | :--- |
| **ASI** | `phca/asi/sanitizer.py` | Input sanitization, NaN/Inf detection, finite checks. |
| **M1 (Sensory)** | `phca/memory/m1_sensory.py` | Short-term sensory buffer (50-cycle FIFO). |
| **M2 (Working)** | `phca/memory/m2_working.py` | Ring-buffer working memory with salience tracking. |
| **G' (Engine)** | `phca/prediction/engine.py` | Prediction engine wrapping Gaussian / discrete / MLP G'. |
| **G' (MLP)** | `phca/world_model/mlp.py` | Pure-NumPy MLP world model (38,868 params, hidden_dim=128). |
| **PEU** | `phca/prediction/error_unit.py` | Precision-weighted prediction error. |
| **TSPL** | `phca/learning/tspl.py` | Single-stream predictive learning (P-Stream only). |
| **MDIM** | `phca/motivation/mdim.py` | Multi-Drive Intrinsic Motivation: 6 homeostatic drives with softmax goal selection. |
| **APC** | `phca/regulation/pid_controller.py` | Adaptive parameter control (PID on prediction-error volatility). |
| **Attention** | `phca/attention/attention.py` | Precision-weighted sparse attention with Gumbel noise. |
| **HPM** | `phca/hpm/parser.py` | Hierarchical procedure memory: composition operators + resource bound computation. |
| **RBTA** | `phca/regulation/rbta_enforcer.py` | Resource-Bounded Turing Supervisor: enforces time/memory/energy/entropy budgets. |
| **M3 (Episodic)** | `phca/memory/m3_episodic.py` | SQLite-backed episode store. |
| **Consolidation** | `phca/consolidation/scheduler.py` | Periodic episodic→statistical fact extraction. |
| **Cycle** | `phca/core/cycle.py` | 12-step cognitive cycle orchestrator; branches on ActionSpace (discrete argmax / continuous MPC). |
| **ActionSpace** | `phca/config.py` | `DiscreteSpace(n)` / `ContinuousSpace(low, high, dim)` union + helpers (Phase 6). |
| **GridWorld** | `phca/environments/grid_world.py` | Configurable grid environment with walls, obstacles, and goal. |
| **MuJoCoEnv** | `phca/environments/mujoco_env.py` | MuJoCo physics wrapper (Cartpole discrete; Pendulum + Reacher continuous). |

### Verified Invariants (A1–A5)

| Invariant | Enforcement |
| :--- | :--- |
| **A1** Resource Boundedness | RBTA time/memory/energy/entropy checks every cycle; TERMINATE→STAY enforcement (D-113). **Measured**: inject over-budget → ≥1 violation. |
| **A2** Temporal Causality | Pipeline ordering in the 12-step cycle. **Measured** (D-113): `experiment_a2_temporal_order()` in `--ci`. |
| **A3** Incomplete Knowledge | Belief entropy floor ≥ ε; semantic facts from consolidation wired into MDIM context. **Measured**: 100-cyc min entropy ≥ 0.01. |
| **A4** Prediction as Primary | Every cycle computes sₜ→ŝₜ₊₁; MLP hidden_dim=128 (38,868 params). **Measured**: continuous MPC selector calls predict per candidate. **OOD measured**: blended confidence drops 0.97→0.26 as σ rises 0→1.0. |
| **A5** Feedback-Driven Adaptation | PEU error drives TSPL updates; error-modulated learning rate with per-dimension attention weights. **Measured**: no-op learn → frozen weights (rel Δ 0.0000); active learn → weights update (rel Δ 0.043). |

---

## Benchmark & Results

The Φ-IQ metric measures overall cognitive performance as a weighted composite:

```
Φ-IQ = 0.20·PredictionAccuracy + 0.20·AdaptationSpeed + 0.15·GoalComplexity
     + 0.15·TransferEfficiency + 0.20·ResourceEfficiency - 0.10·FailureRate
```

### Benchmark Levels

| Level | Name | What It Measures |
| :--- | :--- | :--- |
| **L0** | Stationary Prediction | Prediction accuracy in a static environment. |
| **L1** | Reactive Control | Prediction accuracy under active control + action diversity. |
| **L2** | Goal Pursuit | Goal reaching rate in a maze with walls + obstacles. |
| **L3** | Self-Motivated Exploration | MDIM drive diversity + autonomy in an empty environment. |

### Latest Results (MLP G', 200 cycles/level, 2026-07-05)

```
  PHCA v3.0 — Φ-IQ Benchmark Report (this machine)
  Overall Φ-IQ (4 levels, MLP): 0.7323   (gate PASS, ≥ 0.5486 floor)
  L0 Stationary:   0.7750
  L1 Reactive:     0.6748
  L2 Goal Pursuit: 0.7924   (goal_rate 0.97)
  L3 Exploration:  0.6868

  Pass Criteria:
    [✓] Cycle latency < 500 ms        (mean ~17 ms, p95 ~31 ms)
    [✓] Failure rate < 10%            (0 violations)
    [✓] Overall Φ-IQ > 0.5
    [✓] L2 Φ-IQ ≥ 0.5

  Causal gate L1/L2/L3: PASS (200 cyc × 5 seeds)
  Assumption validation --ci: 5/5 PASS (A1–A5)
  Nightly 10k soak: PASS (late RSS ~1257 B/cyc ≤ 1600, D-113)

  Scientific validation (30 seeds, grid 5/10/20 scaling):
    full_system Φ-IQ:     0.631 ± 0.029
    scaling overall Φ-IQ: 0.700 ± 0.046 (5×5) | 0.325 ± 0.042 (10×10) | 0.147 ± 0.042 (20×20)
    Φ-IQ predictive r:    0.992 (H006 Validated)
    Hypotheses (30-seed full run): H006 Validated; H001–H003,H005 Refuted;
    H004 from interaction_test (see hypothesis_verdicts.json). Smoke (3 seeds) may differ.
```

### Causal Evidence Gate

`scripts/phca_causal_eval.py` compares PHCA against non-PHCA GridWorld controls
on three scenario levels: simple navigation, constrained partial observation,
and long-horizon goal switching/interruption. Current 200-cycle × 5-seed result:
Levels 1–3 **pass** versus gated controls (`random`; L2/L3 also vs `greedy_observed`).
Greedy full-info remains a ceiling on all levels. PHCA GridWorld policy uses
task-lock observed-greedy navigation with sparse L3 coverage probes; it does not
yet exploit memory/consolidation for action selection on grid tasks.

See [docs/phca_causal_evidence.md](docs/phca_causal_evidence.md).

### Performance (Phase 5, Workstream A — D-092)

`gprime_learn` (the MLP replay-backward step, ~95% of cycle time in Phase 4)
was vectorised from a per-sample Python loop (512 forward+backward passes/cycle)
to batched NumPy matmuls. Measured by `scripts/profile_mlp_learn.py`
(200 cyc, MLP, steady-state, warm-up discarded):

| Metric | Phase 4 | Phase 5 | Change |
| :--- | :--- | :--- | :--- |
| `gprime_learn` mean | 35.55 ms (this machine) | **5.09 ms** | **−85.7%** |
| `gprime_learn` p95  | 40.0 ms | 5.51 ms | −86% |
| Full cycle mean     | 38.8 ms | 10.6 ms | −73% |
| Overall Φ-IQ        | 0.7328 | 0.7419 | +1.2% (resource_efficiency up) |

Target was ≥20% reduction (≤25.4 ms) with Φ-IQ ≥0.73 — far exceeded with no
regression. A zero-trust check confirmed the dynamic-goal L2 is unchanged by
the vectorisation (see Limitations / D-092).

---

## MuJoCo

`MuJoCoSimpleEnv` wraps gymnasium MuJoCo environments into the
`EnvironmentProtocol` so the cognitive cycle drives them unchanged. As of
Phase 6/7, each env declares its true action space via `get_action_space()`:
Pendulum-v1 (`ContinuousSpace([-2,2], dim=1)`) and Reacher-v5 (`ContinuousSpace([-1,1]², dim=2)`)
use an MPC-style, prediction-driven sampler (no reward, no policy gradient).
Cartpole stays on a discrete 3-bin path. MuJoCo is opt-in (`requirements-mujoco.txt`);
run headless with `MUJOCO_GL=disabled`.

| Env | ID | Action space | State dim | 100-cyc result (D-107) |
| :--- | :--- | :--- | :--- | :--- |
| Cartpole | `InvertedPendulum-v5` | Discrete (3: push L / stay / push R) | 4 | PASS — discrete, 0 violations |
| Pendulum | `Pendulum-v1` | **Continuous** (torque ∈ [-2,2], dim 1) | 3 | PASS — 7.1 ms, 0 violations, error 29.6→0.68 |
| Reacher  | `Reacher-v5` | **Continuous** (actuator ∈ [-1,1]², dim 2) | 10 | PASS — 4.4 ms, 0 violations, error 105.7→8.4 |

CI exercises 36 MuJoCo tests with `MUJOCO_GL=disabled`; the `make nightly`
MuJoCo gate (`check_benchmark_gate.py --mujoco`) asserts 0 violations + error↓
per env, with a `--neg-test` proving the gate catches synthetic violations.

### Dynamic-Goal Curriculum (experimental)

`--dynamic-goals` relocates the L2 goal on a cadence to exercise continual
adaptation. `--dynamic-goals-every N` controls the cadence (Phase 5 / D-094).
Graduated curriculum results (200 cyc, MLP, this machine):

| Cadence | L2 Φ-IQ | Verdict |
| :--- | :--- | :--- |
| every-100 | 0.4316 | below 0.50 on this machine (D-090's 0.573 was machine-specific) |
| **every-75**  | **0.6444** | **validated** — L2 ≥ 0.50, L0/L1/L3 within noise of static |
| every-50 | 0.4324 | not achievable (consistent with D-087's rejection) |

**every-75 is the supported dynamic cadence.** every-50 is honestly documented
as not achievable. Dynamic mode is experimental and measured separately from
the canonical static benchmark.

---

## Phase 6 — Scientific & CI Hardening

Phase 6 turned the whitepaper's A1–A5 claims and the OOD-confidence story into
**measured, falsifiable** checks, and added a nightly hardening suite.

### OOD Calibration (B1)

`scripts/ood_calibration.py` sweeps Gaussian observation perturbation
σ ∈ {0, 0.05, 0.1, 0.25, 0.5, 1.0} and records the MLP confidence
decomposition. The blended confidence is **monotonically non-increasing**
(drop 0.97→0.26, σ=0→1.0): aleatoric ↓, epistemic ↑, MSE ↑ — matching the
MC-Dropout uncertainty story. Persisted to `logs/ood_calibration.json`.

### Assumption Validation (B2)

`scripts/assumption_validation.py --ci` runs one falsifiable experiment per
invariant and exits non-zero on any FAIL:

| Inv | Experiment | Result |
| :--- | :--- | :--- |
| A1 | inject over-budget G' timing → RBTA flags ≥1 violation | **PASS** (1 violation) |
| A2 | temporal order at action selection | **PASS** (D-113) |
| A3 | 100-cyc low-noise drive → belief entropy ≥ floor (0.01) | **PASS** (min 0.50) |
| A4 | continuous MPC selector calls predict per candidate | **PASS** (8 calls) |
| A5 | no-op learn → frozen weights; active learn → weights update | **PASS** (frozen Δ 0.0000, active Δ 0.043) |

The rejected first designs ("Φ-IQ collapse on zeroed prediction", "MC-dropout
probe for frozen weights") are documented honestly in DECISIONS.md D-101 —
they were bad tests, not masked failures.

### Nightly Hardening (C1–C3)

`make nightly` runs the full suite (override length with `NIGHTLY_CYCLES`):

1. Static Φ-IQ gate (200-cyc MLP, ≥ 5%-floor).
2. MuJoCo benchmark gate (Pendulum + Reacher continuous, Cartpole discrete) + `--neg-test`.
3. Assumption validation `--ci`.
4. OOD calibration (monotonic check).
5. Nightly stress (`scripts/nightly_stress.py` — RSS leak detector, latency
   p95/p99, Φ-IQ at 1k/5k/10k, RBTA violations).
6. Causal behavior gate (`phca_causal_eval.py --gate` on L2+L3 in nightly).
7. Session anomaly gate (`scripts/nightly_anomaly_gate.py` — synthetic fixtures).

This is a **script+gate target, not a cron job** — schedule it externally
(GitHub Actions `schedule:` nightly, systemd timer, or cron). 1000-cyc CI run
exits 0 in ~43 s; a true 10k soak takes ~3 min. The nightly stress test
**caught** a real (pre-existing) memory-growth finding — see Limitations.

---

## Phase 9–12 — Cognitive Observatory Completion

Phases 9–12 complete the Observatory data contract, performance story, and multi-session
compare on top of Phase 8 replay/scrub hardening.

### Phase 9 — Schema Governance (D-110)

- `OBSERVABILITY_SCHEMA_VERSION = 1` stamped on each JSONL frame and in `meta.json`.
- Legacy v0 sessions normalize on replay via `normalize_observability_json()`.
- Unknown or mixed schema versions fail `--check` (fail-closed).

### Phase 10 — Report Parity

- `session_report.json` shares `format_session_results_lines()` with the Overview panel.
- Parity test: `python/phca/monitoring/tests/test_session_report.py`.

### Phase 11 — Large-Session Scrub Performance

- Decimated rolling rebuild + lazy per-tab rebuild on seek (`cognitive_panels.py`).
- Scrub budget tests: ≤2s (live) / ≤4s (review) at 3000+ cycles.

### Phase 12 — Multi-Session Comparison (D-115)

- `phca_replay.py --compare` with `compare_session_reports()` and structured deltas.
- Optional `--compare-output` JSON export for regression tracking.

See [docs/observability.md](docs/observability.md) and
[docs/PHCA_Cognitive_Observatory_Architecture.md](docs/PHCA_Cognitive_Observatory_Architecture.md).

---

## Limitations

See [docs/limitations.md](docs/limitations.md) for the full list. Highlights:

- **Long-run memory growth (Phase 7 retention).** `make nightly` uses a
  **phase-aware** late-half RSS slope gate (D-112, D-113): ≤5000 B/cyc for runs under
  7000 cycles (M3 fill phase) and ≤1600 B/cyc for post-cap soaks (default
  `NIGHTLY_CYCLES=10000`). 10k soak PASS (~1257 B/cyc). See
  [docs/limitations.md](docs/limitations.md) and [STATUS.md](STATUS.md).
- **Observatory Phases 7–20 complete:** live PyQt dashboard, JSONL recording, seek/scrub replay, schema governance, report parity, scrub performance, multi-session `--compare`, anomaly detection, action explainability, stable API, supervisor/recovery, multi-agent timelines, cognitive-moment query, and scientific reproduction — see [docs/observability.md](docs/observability.md).
- **Discrete GridWorld selector uses goal geometry**, not pure prediction; the
  **continuous MPC path** (Pendulum, Reacher) is prediction-primary (A4, D-101).
- **No NLP, vision, multi-agent cognition (shared memory / coordination), or M5 procedural memory.**
  Observatory **display** supports multi-agent replay and per-agent scrub (Phase 17).
- **Dynamic goals** are experimental at every-75 only.
- **P-Stream only** (E/S streams removed D-020).

---

## Contributing

This is an internal research project. The workflow is audit-driven: every
change is logged as a `D-XXX` entry in [DECISIONS.md](DECISIONS.md) (kept AND
reverted attempts), validated by the benchmark suite + gate script + relevant
unit tests, and must respect the surgical-change mandate (≤50 lines/change,
≤3 files/change) and the A1–A5 invariants. See `STATUS.md` for the open issue
registry and [docs/archive/](docs/archive/) phase sign-offs.

```bash
# Run tests before any change
MUJOCO_GL=disabled make test-all
MUJOCO_GL=disabled make test-mujoco

# Validate with the canonical benchmark + gate
MUJOCO_GL=disabled PYTHONPATH=python python scripts/benchmark.py --use-mlp --cycles=200 --output=logs/benchmark_report.json
python scripts/check_benchmark_gate.py logs/benchmark_report.json logs/benchmark_ci_baseline.json

# Phase 6 hardening gate (full suite)
make nightly NIGHTLY_CYCLES=1000
```

---

## Key Documents

| Document | Description |
| :--- | :--- |
| [DOCUMENTATION_MAP.md](DOCUMENTATION_MAP.md) | Which docs are living vs historical vs aspirational. |
| [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md) | Whitepaper criteria × code × gates matrix. |
| [docs/architecture.md](docs/architecture.md) | Architecture overview — 12-step cycle, module map, invariants. |
| [docs/archive/phase6_completion_report.md](docs/archive/phase6_completion_report.md) | Phase 6 sign-off (historical). |
| [docs/archive/phase5_completion_report.md](docs/archive/phase5_completion_report.md) | Phase 5 sign-off (historical). |
| [docs/archive/phase4_readiness_report.md](docs/archive/phase4_readiness_report.md) | Phase 4 sign-off (historical). |
| [STATUS.md](STATUS.md) | Audit progress, issue registry, test/benchmark status. |
| [DECISIONS.md](DECISIONS.md) | Complete design decision log (D-001 through D-127). |
| [docs/limitations.md](docs/limitations.md) | What PHCA cannot do; open backlog items. |
| [docs/phca_causal_evidence.md](docs/phca_causal_evidence.md) | Three-level causal behavior evidence gate (L1–L3). |
| [docs/observability.md](docs/observability.md) | Cognitive Observatory JSONL, replay/scrub, integrity checks. |
| [docs/PHCA_Cognitive_Observatory_Architecture.md](docs/PHCA_Cognitive_Observatory_Architecture.md) | Full Observatory architecture and 20-phase roadmap. |
| [docs/archive/phase3.3_full_completion_report.md](docs/archive/phase3.3_full_completion_report.md) | Phase 3.3 gap-closure completion report (historical). |
| [docs/architectural_audit_report.md](docs/architectural_audit_report.md) | Full audit of 28 issues with resolution status. |
| [research/outputs/07-rigorous-whitepaper.md](research/outputs/07-rigorous-whitepaper.md) | Formal scientific whitepaper (A1–A5, RBTA, MDIM, failure modes). |
| [research/glossary.md](research/glossary.md) | Terminology reference. |

---

## Project Structure

```
├── python/
│   ├── phca/                    # Core cognitive architecture
│   │   ├── core/                # CognitiveCycle orchestrator (12-step cycle)
│   │   ├── asi/                 # ASI sanitizer
│   │   ├── memory/              # M1 (sensory), M2 (working), M3 (episodic)
│   │   ├── consolidation/       # Episodic → statistical fact extraction
│   │   ├── world_model/         # G' Gaussian / discrete graph / MLP
│   │   ├── prediction/          # Prediction engine + PEU
│   │   ├── learning/            # TSPL (P-Stream)
│   │   ├── attention/           # Goal-driven sparse attention
│   │   ├── motivation/          # MDIM (6 drives)
│   │   ├── regulation/          # RBTA enforcer + adaptive parameter control
│   │   ├── hpm/                 # HPM composition grammar
│   │   ├── environments/        # GridWorld + MuJoCo + EnvironmentProtocol
│   │   ├── monitoring/          # Cognitive Observatory (ObservabilityFrame, Qt dashboard)
│   │   └── config.py            # Shared types + resource bounds
│   ├── tests/                   # Integration tests (stress, chaos, edge cases, MuJoCo)
│   └── benchmarks/              # Legacy runner (use scripts/benchmark.py)
├── scripts/
│   ├── benchmark.py             # Φ-IQ benchmark suite (primary; --env gridworld/cartpole/pendulum/reacher)
│   ├── check_benchmark_gate.py  # CI gate: static Φ-IQ + --mujoco + --neg-test (Phase 6)
│   ├── ood_calibration.py       # OOD σ-sweep confidence curve (Phase 6 / B1)
│   ├── assumption_validation.py # A1–A5 falsifiable experiments + --ci (Phase 6 / D-113)
│   ├── nightly_stress.py        # RSS-leak + latency + Φ-IQ stress (Phase 6 / C1)
│   ├── profile_mlp_learn.py     # gprime_learn per-module profile (Phase 5 perf target)
│   ├── longrun_probe.py         # 1000-cycle stability probe (latency creep + RSS)
│   ├── phca-logs.py             # Structured log viewer
│   ├── phca_observatory.py      # PyQt Cognitive Observatory (live + JSONL) — canonical UI
│   ├── phca_observatory_supervisor.py  # Subprocess wrapper + crash recovery (Phase 16)
│   ├── phca_multi_observatory.py       # Multi-agent aligned runner wrapper
│   ├── phca_query.py            # Cognitive-moment query CLI (Phase 18)
│   ├── phca_replay.py           # Session replay + --check integrity gate
│   ├── reproduce.py             # One-command scientific reproduction (Phase 19)
│   ├── nightly_anomaly_gate.py  # Nightly session anomaly gate (Phase 13)
│   ├── phca_visualise.py        # Matplotlib legacy dashboard (deprecated for full review)
│   └── profile_cycle.py         # Per-cycle profiling (Gaussian path)
├── docs/                        # Architecture, decisions, completion reports
│   └── observability.md         # Observatory JSONL schema, replay, --check rules
├── logs/                        # Benchmark reports + phca.log + CI baseline
├── STATUS.md                    # Audit progress and issue registry
├── Makefile                     # setup, test-all, bench-* targets
└── README.md
```

---

## Cognitive Observatory

Live PyQt dashboard, per-cycle JSONL recording, **seek/scrub replay** (Phases 7–20), and
offline session reports with schema versioning and Overview parity. PyQt `--qt` replay is the canonical path; matplotlib
`--from-jsonl` is legacy.

See **[docs/observability.md](docs/observability.md)** for JSONL schema, playback/scrub
semantics, transport controls, replay banners, and `--check` integrity rules.
See **[docs/PHCA_Cognitive_Observatory_Architecture.md](docs/PHCA_Cognitive_Observatory_Architecture.md)**
for the full Observatory architecture and roadmap.

```bash
# Live run
QT_QPA_PLATFORM=offscreen PYTHONPATH=python python scripts/phca_observatory.py --cycles=50 --mlp

# Replay with scrub (canonical Observatory path)
PYTHONPATH=python python scripts/phca_replay.py logs/sessions/<ts>/ --qt

# Session integrity + offline report
PYTHONPATH=python python scripts/phca_replay.py --check logs/sessions/<ts>/
PYTHONPATH=python python scripts/phca_replay.py logs/sessions/<ts>/ --report
```

---

## Technical Notes

- **MLP world model.** 38,868 params (hidden_dim=128). Confidence blends
  aleatoric `exp(-MSE)` with epistemic MC-Dropout variance (D-080); empowerment
  `I(s';a|s)` is estimated via MC-Dropout mutual information (D-077, samples=4
  per D-091). Learning uses a hybrid online/replay schedule with a unified
  `lr*0.5` rate (D-081); the steady-state replay backward pass is batched
  (Phase 5 / D-092, 85.7% faster).
- **Goal-directed action selection.** Goal alignment blends geometric Manhattan
  distance gain with a predicted-goal-alignment (PGA) term that ramps in over
  cycles 50–150 so the learned model drives action selection during measurement
  (D-087).
- **L2 benchmark metric.** `adaptation_speed` for Goal Pursuit uses
  `max(improvement, maintenance)` aligned with L0/L1 (D-086).
- **Gaussian inference caching.** The Gaussian G' joint moments are computed
  once and cached across all `predict()` calls per cycle (~1000× speedup).
- **EnvironmentProtocol.** `CognitiveCycle.build_for_env()` accepts any object
  implementing `get_action_names()`, `get_possible_actions()`,
  `get_goal_position()`, and (Phase 6) `get_action_space()` — not just
  `GridWorld`. Envs without `get_action_space()` fall back to
  `DiscreteSpace(action_space_size)` via getattr.
- **Continuous action selection (Phase 6).** For `ContinuousSpace` envs the
  cycle's `_select_continuous_action()` samples K=8 candidate actions ~ U(low,
  high) (A1-capped K·dim ≤ 16 forward passes), predicts each via G', and picks
  the candidate whose predicted next state best matches `get_goal_reference()`
  (score = 0.4·confidence + 0.5·goal-ref alignment + 0.1·PGA, ε-greedy).
  Prediction/goal-driven — no reward, value function, or policy gradient.

---

## License

Internal research project. All rights reserved.
