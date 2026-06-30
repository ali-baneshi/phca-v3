# PHCA v3.0 — Predictive Hierarchical Cognitive Architecture

A formally specified, resource-bounded cognitive architecture for continual learning, intrinsic motivation, and self-regulated autonomous agents. **283 tests passing.**

---

## Quick Start

```bash
# Install dependencies
make setup

# Run all tests (289 tests, ~5s)
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
Steps 10-13: MDIM + CR + ATTN + HPM
                               ─ generate_goal, regulate_criticality, attend, bounds
Step 14: RBTA Enforcement    ─ check_cycle(runtime, memory, energy, entropy)
Step 15: Logging             ─ append metrics to history
Steps 16-18: Consolidation   ─ periodic E→S episodic→semantic transfer
Step 19: Increment           ─ cycle_count += 1
```

### Module Map

| Module | File | Function |
| :--- | :--- | :--- |
| **ASI** | `phca/perception/asi.py` | Input sanitization, NaN/Inf detection, finite checks. |
| **M1 (Sensory)** | `phca/working_memory/m1_sensory.py` | Short-term sensory buffer (50-cycle FIFO). |
| **M2 (Working)** | `phca/working_memory/m2_working.py` | Ring-buffer working memory with salience tracking. |
| **G' (Engine)** | `phca/world_model/engine.py` | Gaussian G' / MLP prediction engine. 1000× speedup via cached joint moments for G'. |
| **PEU** | `phca/learning/peu.py` | Precision-weighted prediction error. |
| **TSPL** | `phca/learning/tspl.py` | Single-stream predictive learning (P-Stream only). |
| **MDIM** | `phca/motivation/mdim.py` | Multi-Drive Intrinsic Motivation: 6 homeostatic drives (exploration, competence, novelty, etc.) with softmax goal selection. |
| **CR (PID)** | `phca/governance/pid_controller.py` | PID loop on Φ (integrated information) with orthogonality constraint. |
| **Attention** | `phca/learning/attention.py` | Precision-weighted sparse attention. |
| **HPM** | `phca/hpm/parser.py` | Hierarchical procedure memory: composition operators + resource bound computation. |
| **RBTA** | `phca/governance/rbta_enforcer.py` | Resource-Bounded Turing Supervisor: enforces time/memory/energy/entropy budgets. |
| **M3 (Episodic)** | `phca/episodic_memory/m3_episodic.py` | SQLite-backed episode store. |
| **Consolidation** | `phca/episodic_memory/consolidation.py` | Periodic episodic→semantic transfer. |
| **Cycle** | `phca/core/cycle.py` | 12-step cognitive cycle orchestrator. |
| **GridWorld** | `phca/environments/grid_world.py` | Configurable grid environment with walls, obstacles, and goal. |
| **MuJoCoEnv** | `phca/environments/mujoco_env.py` | MuJoCo physics environment wrapper for continuous control. |

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

### Latest Results (Gaussian G', 100 cycles/level)

```
  PHCA v3.0 — Φ-IQ Benchmark Report
  Overall Φ-IQ: 0.484  ~ (just below threshold)

  Level  Φ-IQ     Pred    Adapt   Goals   Transfer Resource Fail
  ────────────────────────────────────────────────────────────────
  L0     0.469    0.647   0.200   1.000   0.155    0.951    0.010
  L1     0.548    0.650   0.618   0.600   0.291    0.950    0.030
  L2     0.280    0.584   0.000   0.114   0.000    0.927    0.020
  L3     0.637    0.695   0.782   0.000   0.689    0.944    0.010

  Pass Criteria:
    [✓] Cycle latency < 500ms           (actual: ~50ms p95)
    [✓] Failure rate < 10%              (actual: 1.8%)
    [✓] Goal autonomy achieved          (Level 3 drive diversity)
    [~] Φ-IQ > 0.5                      (actual: 0.484 — L2 bottleneck from discrete G')
```

**Note:** The MLP world model (38,868 params, hidden_dim=128) achieves Φ-IQ > 0.6 with 500 cycles.
See `logs/benchmark_full_final.json` for latest run data.

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
# All 283 tests
make test-all

# Or directly:
PYTHONPATH=python python -m pytest python/tests/ python/phca/ -v --tb=short

# Single module
PYTHONPATH=python python -m pytest python/phca/motivation/tests/ -v
PYTHONPATH=python python -m pytest python/phca/governance/tests/ -v
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
│   │   ├── perception/          # ASI sanitizer
│   │   ├── working_memory/      # M1 (sensory), M2 (working)
│   │   ├── episodic_memory/     # M3 (episodic) + consolidation
│   │   ├── world_model/         # G' + MLP prediction engine
│   │   ├── learning/            # PEU + TSPL (P-Stream) + attention
│   │   ├── motivation/          # MDIM (6 drives) + criticality
│   │   ├── governance/          # RBTA enforcer + PID controller
│   │   ├── hpm/                 # HPM composition grammar
│   │   ├── environments/        # GridWorld + MuJoCo + EnvironmentProtocol
│   │   └── config.py            # Shared types + resource bounds
│   ├── tests/                   # Integration tests
│   └── benchmarks/              # Benchmark runner *
├── scripts/
│   ├── benchmark.py             # Φ-IQ benchmark suite (primary)
│   ├── phca-monitor.py          # Live terminal dashboard
│   ├── phca-logs.py             # Structured log viewer
│   └── profile_cycle.py         # Per-cycle profiling
├── docs/                        # Architecture, decisions, completion reports
├── logs/                        # Benchmark reports + phca.log
├── Makefile                     # setup, test-all, bench-* targets
└── README.md

*Note: Use `python scripts/benchmark.py` for benchmarks, not `python -m phca.benchmarks.runner`.
```

---

## Key Documents

| Document | Description |
| :--- | :--- |
| `docs/architecture.md` | Current architecture overview — 12-step cycle, module map, invariants |
| `docs/phase3.3_full_completion_report.md` | Gap-closure completion report (post-gap analysis, all items resolved) |
| `docs/monitoring_completion_report.md` | Monitoring system — MetricsStore, dashboard, file logging |
| `DECISIONS.md` | Complete design decision log (D-001 through D-044) |
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
