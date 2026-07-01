# PHCA v3.0 — Predictive Hierarchical Cognitive Architecture

A formally specified, resource-bounded cognitive architecture for continual learning, intrinsic motivation, and self-regulated autonomous agents. **284 tests passing.**

---

## Quick Start

```bash
# Install dependencies
make setup

# Run all tests (284 core + 2 optional MuJoCo if gymnasium installed)
make test-all

# Run benchmark suite (all 4 levels, 500 cycles each, MLP mode)
PYTHONPATH=python python scripts/benchmark.py --cycles=500 --use-mlp

# Quick smoke test (Level 0 only, 20 cycles)
PYTHONPATH=python python scripts/benchmark.py --quick
```

---

## Architecture

PHCA implements a **12-step cognitive cycle** executed at ~40 Hz on consumer hardware (25ms avg latency). Each cycle transforms raw sensor input into a goal-directed action through a pipeline of specialized modules. The entire system is governed by a **Resource-Bounded Turing Supervisor (RBTA)** enforcing time, memory, energy, and entropy budgets per cycle.

### Cognitive Cycle (12 Active Steps)

```
Step  0: ASI Sanitize        ─ sanitize(raw_obs) → clean_state
Step  1: WM Write            ─ m2.write(state) + m1.write(state)
Steps 2-4: G' Prediction     ─ engine.predict(state) → predicted_state, confidence
Steps 5-6: PEU Error         ─ peu.compute(state, prediction) → error
Step  7: TSPL P-Stream       ─ tspl.update(prediction_error, state, prediction)
Step  8: (reserved)
Step  9: Action Selection    ─ argmax(goal_alignment + confidence)
Steps 10-13: MDIM + APC + ATTN + HPM
                               ─ generate_goal, regulate_params, attend, bounds
Step 14: RBTA Enforcement    ─ check_cycle(runtime, memory, energy, entropy)
Step 15: Logging             ─ append metrics to history
Steps 16-18: Consolidation   ─ periodic E→S episodic→semantic transfer
Step 19: Increment           ─ cycle_count += 1
```

### Module Map

| Module | File | Function |
| :--- | :--- | :--- |
| **ASI** | `phca/asi/sanitizer.py` | Input sanitization, NaN/Inf detection, finite checks. |
| **M1 (Sensory)** | `phca/memory/m1_sensory.py` | Short-term sensory buffer (50-cycle FIFO). |
| **M2 (Working)** | `phca/memory/m2_working.py` | Ring-buffer working memory with salience tracking. |
| **G' (Engine)** | `phca/prediction/engine.py` | Prediction engine wrapping Gaussian / discrete / MLP G'. |
| **PEU** | `phca/prediction/error_unit.py` | Precision-weighted prediction error. |
| **TSPL** | `phca/learning/tspl.py` | Single-stream predictive learning (P-Stream only). |
| **MDIM** | `phca/motivation/mdim.py` | Multi-Drive Intrinsic Motivation: 6 homeostatic drives with softmax goal selection. |
| **APC** | `phca/regulation/pid_controller.py` | Adaptive parameter control (PID on prediction-error volatility). |
| **Attention** | `phca/attention/attention.py` | Precision-weighted sparse attention with goal-driven biasing. |
| **HPM** | `phca/hpm/parser.py` | Hierarchical procedure memory: composition operators + resource bound computation. |
| **RBTA** | `phca/regulation/rbta_enforcer.py` | Resource-Bounded Turing Supervisor: enforces time/memory/energy/entropy budgets. |
| **M3 (Episodic)** | `phca/memory/m3_episodic.py` | SQLite-backed episode store. |
| **Consolidation** | `phca/consolidation/scheduler.py` | Periodic episodic→statistical fact extraction. |
| **Cycle** | `phca/core/cycle.py` | 12-step cognitive cycle orchestrator. |
| **GridWorld** | `phca/environments/grid_world.py` | Configurable grid environment with walls, obstacles, and goal. |
| **MuJoCoEnv** | `phca/environments/mujoco_env.py` | MuJoCo physics environment wrapper (optional `gymnasium`). |

### Verified Invariants (A1-A5)

| Invariant | Enforcement |
| :--- | :--- |
| **A1** Resource Boundedness | RBTA time/memory/energy/entropy checks every cycle (composition tree reads energy from `energy_log`) |
| **A2** Temporal Causality | Pipeline ordering in 12-step cycle |
| **A3** Incomplete Knowledge | Belief entropy floor ≥ ε; semantic facts from consolidation wired into MDIM context |
| **A4** Prediction as Primary | Every cycle computes sₜ→ŝₜ₊₁; MLP hidden_dim=128 (38,868 params) |
| **A5** Feedback-Driven Adaptation | PEU error drives TSPL updates; error-modulated learning rate with per-dimension attention weights |

---

## Benchmark Suite (Φ-IQ)

The Φ-IQ metric measures overall cognitive performance as a weighted composite of 6 sub-metrics:

```
Φ-IQ = 0.20·PredictionAccuracy + 0.20·AdaptationSpeed + 0.15·GoalComplexity
     + 0.15·TransferEfficiency + 0.20·ResourceEfficiency - 0.10·FailureRate
```

### Benchmark Levels

| Level | Name | What It Measures |
| :--- | :--- | :--- |
| **L0** | Stationary Prediction | Prediction accuracy in a static environment (no action needed) |
| **L1** | Reactive Control | Prediction accuracy under active control + action diversity |
| **L2** | Goal Pursuit | Goal reaching rate in a maze with walls + obstacles |
| **L3** | Self-Motivated Exploration | MDIM drive diversity + autonomy in an empty environment |

### Latest Results (MLP G', 500 cycles L2 / 200 cycles full suite)

```
  PHCA v3.0 — Φ-IQ Benchmark Report (post Phase 4 audit)
  Overall Φ-IQ (4 levels, MLP): 0.668
  L2 Goal Pursuit (500 cycles):   0.477  (goal_rate 0.94; target ≥ 0.5)

  Pass Criteria:
    [✓] Cycle latency < 500ms
    [✓] Failure rate < 10%
    [✓] Overall Φ-IQ > 0.5 (full suite)
    [~] L2 Φ-IQ ≥ 0.5 (bottleneck: adaptation_speed + transfer_efficiency)
```

See `logs/benchmark_phase4_fix3.json` (full suite) and `logs/benchmark_l2_ps1.json` (L2).

---

## Usage

### Running the Cognitive Cycle

```python
from phca.core.cycle import CognitiveCycle

# Build a cycle for 5×5 GridWorld with obstacles
cycle = CognitiveCycle.build_for_env(
    size=5,
    seed=42,
    use_continuous=True,       # Gaussian G' (fast) or MLP
    use_mlp=False,
    obstacles=[(0, 2), (1, 2),
               (2, 2), (3, 2)],  # wall barrier in column 2
)

# Run 100 cognitive cycles
history = [cycle.step() for _ in range(100)]
metrics = history[-1]
print(f"Latency: {metrics.latency_ms:.0f}ms, "
      f"Error: {metrics.prediction_error:.3f}")
```

### Running Benchmarks

```bash
# Quick mode (Level 0, 20 cycles)
python scripts/benchmark.py --quick

# Full suite (Levels 0-3, 100 cycles each)
python scripts/benchmark.py

# Custom configuration
python scripts/benchmark.py --levels=0,2 --cycles=200 --output=my_report.json
```

### Running Tests

```bash
# All 284 tests
make test-all

# Or directly:
PYTHONPATH=python python -m pytest python/tests/ python/phca/ -v --tb=short

# Single module
PYTHONPATH=python python -m pytest python/phca/motivation/tests/ -v
PYTHONPATH=python python -m pytest python/phca/regulation/tests/ -v
PYTHONPATH=python python -m pytest python/phca/hpm/tests/ -v

# Cycle integration tests
PYTHONPATH=python python -m pytest python/phca/core/tests/ -v
```

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
│   │   └── config.py            # Shared types + resource bounds
│   ├── tests/                   # Integration tests (stress, chaos, edge cases)
│   └── benchmarks/              # Package benchmark runner (scripts/benchmark.py preferred)
├── scripts/
│   ├── benchmark.py             # Φ-IQ benchmark suite (primary)
│   ├── check_benchmark_gate.py  # CI Φ-IQ regression gate
│   ├── phca-monitor.py          # Live terminal dashboard
│   ├── phca-logs.py             # Structured log viewer
│   └── profile_cycle.py         # Per-cycle profiling
├── docs/                        # Architecture, decisions, completion reports
├── logs/                        # Benchmark reports + phca.log + CI baseline
├── STATUS.md                    # Audit progress and issue registry
├── Makefile                     # setup, test-all, bench-* targets
└── README.md
```

---

## Key Documents

| Document | Description |
| :--- | :--- |
| `docs/architecture.md` | Current architecture overview — 12-step cycle, module map, invariants |
| `docs/phase3.3_full_completion_report.md` | Gap-closure completion report (post-gap analysis, all items resolved) |
| `docs/monitoring_completion_report.md` | Monitoring system — MetricsStore, dashboard, file logging |
| `STATUS.md` | Audit progress, issue registry, test/benchmark status |
| `DECISIONS.md` | Complete design decision log (D-001 through D-075+) |
| `docs/architectural_audit_report.md` | Full audit of 28 issues with resolution status |
| `research/outputs/07-rigorous-whitepaper.md` | Formal scientific whitepaper (historical) |
| `research/glossary.md` | Terminology reference |

---

## Technical Notes

- **Gaussian inference caching**: The G' Bayesian network's joint moments are computed once and cached across all `predict()` calls per cycle providing a **1000× speedup** (24s/cycle → 13ms/cycle).
- **Goal-directed action selection**: Goal alignment is computed directly from environment state (`agent_pos`, `goal_pos`, `grid`) rather than from predictions, since G' applies uniform weights to all state dimensions.
- **MLP mode**: The MLP world model (38,868 params, hidden_dim=128) replaces the Gaussian G' for environments that benefit from learned transition dynamics. Needs ≥200 cycles to stabilise.
- **EnvironmentProtocol**: `CognitiveCycle.build_for_env()` accepts any object implementing `get_action_names()`, `get_possible_actions()`, and `get_goal_position()` — not just `GridWorld`.

---

## License

Internal research project. All rights reserved.
