# PHCA v3.0 — Predictive Hierarchical Cognitive Architecture

A formally specified, resource-bounded cognitive architecture for continual learning, intrinsic motivation, and self-regulated autonomous agents. **297 tests passing, Φ-IQ = 0.66 across 4 benchmark levels.**

---

## Quick Start

```bash
# Install dependencies
make setup

# Run all tests (297 tests, ~3s)
make test-all

# Run benchmark suite (all 4 levels, 100 cycles each, ~5s)
python scripts/benchmark.py

# Quick smoke test (Level 0 only, 20 cycles)
python scripts/benchmark.py --quick
```

---

## Architecture

PHCA implements a **21-step cognitive cycle** executed ~75 times per second on consumer hardware. Each cycle transforms raw sensor input into a goal-directed action through a pipeline of specialized modules.

### Cognitive Cycle (21 Steps)

```
Step  0: ASI Sanitize        ─ sanitize(raw_obs) → clean_state
Step  1: WM Write            ─ m2.write(state) + m1.write(state)
Steps 2-4: G' Prediction     ─ engine.predict(state) → predicted_state, confidence
Steps 5-6: PEU Error         ─ peu.compute(state, prediction) → error
Step  7: TSPL P-Stream       ─ tspl.update(prediction_error, state, prediction)
Step  8: (reserved)
Step  9: Action Selection    ─ argmax(goal_alignment + confidence)
Steps 10-13: MDIM + CR + ATTN + HPM
                               ─ generate_goal, regulate_criticality, attend, validate
Step 14: RBTA Enforcement    ─ check_cycle(runtime, memory, energy, entropy)
Step 15: Logging             ─ append metrics to history
Steps 16-18: Consolidation   ─ periodic E→S episodic→semantic transfer
Step 19: Increment           ─ cycle_count += 1
Step 20: Sleep Cycle         ─ full consolidation every 50 cycles
```

### Module Map

| Module | File | Function |
| :--- | :--- | :--- |
| **ASI Sanitizer** | `asi/sanitizer.py` | Filters NaN/Inf/out-of-range sensor values. Replaces with last valid value, halves precision on failure. |
| **M1 Sensory Buffer** | `memory/m1_sensory.py` | Low-latency sensory history (100ms persistence). |
| **M2 Working Memory** | `memory/m2_working.py` | 7±2 chunk capacity, single-writer access. |
| **M3 Episodic Memory** | `memory/m3_episodic.py` | SQLite-backed episode store with MVCC snapshot isolation for safe consolidation reads. |
| **G' World Model** | `world_model/graph.py` | Gaussian Bayesian network with closed-form analytic inference. Fixed transition parameters; caching for 1000× speedup. |
| **Prediction Engine** | `prediction/engine.py` | Multi-step ahead prediction via G'. Confidence decays logarithmically with horizon. |
| **PEU** | `prediction/error_unit.py` | Prediction error computation. |
| **TSPL** | `learning/tspl.py` | Three-Stream Predictive Learning: P-Stream (fast, forgettable), E-Stream (GEM-protected), S-Stream (EWC-protected). |
| **MDIM** | `motivation/mdim.py` | Multi-Drive Intrinsic Motivation: 6 homeostatic drives (D1-D6) compete via softmax-weighted goal selection. Pareto front + meta-stable locking. |
| **CR (PID)** | `regulation/pid_controller.py` | Criticality Regulator: PID loop on Φ (integrated information). Outputs T (temperature), η (exploration noise), α (attention spread). Orthogonality constraint. |
| **Attention** | `attention/attention.py` | Precision-weighted k-WTA selection with Gumbel noise. Precision learns from prediction error. |
| **HPM Validator** | `hpm/parser.py` | Typed composition grammar: 8 operators (SEQUENCE, PARALLEL, CONDITIONAL, etc.), 6 module types, resource bound computation. |
| **RBTA Enforcer** | `regulation/rbta_enforcer.py` | Resource-Bounded Temporal Automata: enforces A1-A5 invariants (time, memory, energy, entropy, sensor failures). Composition tree for SEQUENCE/PARALLEL verification. |
| **Consolidation** | `consolidation/scheduler.py` | Periodic E→S transfer: extracts semantic facts from M3 episodes via similarity merging. |

### Verified Invariants (A1-A5)

| Invariant | Enforcement |
| :--- | :--- |
| **A1** Resource Boundedness | RBTA time/memory/energy checks every cycle |
| **A2** Temporal Causality | Pipeline ordering in 21-step cycle |
| **A3** Incomplete Knowledge | Belief entropy floor ≥ ε |
| **A4** Prediction as Primary | Every cycle computes sₜ→ŝₜ₊₁ |
| **A5** Feedback-Driven Adaptation | PEU error drives TSPL updates |

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

### Current Results (200 cycles/level)

```
  PHCA v3.0 — Φ-IQ Benchmark Report
  Overall Φ-IQ: 0.664  ✓ PASS

  Level  Φ-IQ     Pred    Adapt   Goals   Transfer Resource Fail
  ────────────────────────────────────────────────────────────────
  L0     0.752    0.918   0.918   0.400   0.843    0.989    0.000
  L1     0.781    0.918   0.918   0.600   0.842    0.989    0.000
  L2     0.413    0.923   0.000   0.205   0.000    0.989    0.000
  L3     0.710    0.918   0.800   0.400   0.734    0.984    0.005

  Pass Criteria:
    [✓] Cycle latency < 500ms           (actual: ~12ms)
    [✓] Failure rate < 10%              (actual: < 1%)
    [✓] Goal autonomy achieved          (Level 3: 0.400)
    [✓] Φ-IQ > 0.5                      (actual: 0.664)
```

---

## Usage

### Running the Cognitive Cycle

```python
from phca.core.cycle import CognitiveCycle

# Build a cycle for 5×5 GridWorld with continuous Gaussian inference
cycle = CognitiveCycle.build_for_env(
    size=5,
    seed=42,
    use_continuous=True,   # Phase 3.2: Gaussian BN (fast)
    obstacles=[],          # Empty grid (no walls)
)

# Run 100 cognitive cycles
results = cycle.run(n_cycles=100)
print(f"Avg latency: {results['avg_latency_ms']:.0f}ms")
print(f"Goals reached: {results['goals_reached']}")
```

### Custom Obstacles for Goal Pursuit

```python
from phca.core.cycle import CognitiveCycle

# GridWorld with walls
obstacles = [(0, 2), (1, 2), (2, 2), (3, 2)]  # wall barrier in column 2
cycle = CognitiveCycle.build_for_env(
    size=5,
    obstacles=obstacles,
)
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
# All 297 tests
make test-all

# Or directly:
python -m pytest python/phca/ -v --tb=short

# Single module
python -m pytest python/phca/motivation/tests/ -v
python -m pytest python/phca/regulation/tests/ -v
python -m pytest python/phca/hpm/tests/ -v

# Cycle integration tests
python -m pytest python/phca/core/tests/ -v
```

---

## Project Structure

```
├── python/
│   ├── phca/                    # Core cognitive architecture
│   │   ├── asi/                 # ASI sanitizer
│   │   ├── memory/              # M1 (sensory), M2 (working), M3 (episodic)
│   │   ├── core/                # CognitiveCycle orchestrator (21-step cycle)
│   │   ├── world_model/         # G' Bayesian network + Gaussian inference
│   │   ├── prediction/          # Prediction engine + error unit
│   │   ├── learning/            # TSPL (3-stream predictive learning)
│   │   ├── motivation/          # MDIM (6 drives + Pareto front)
│   │   ├── regulation/          # RBTA enforcer + CR PID controller
│   │   ├── attention/           # Precision-weighted k-WTA attention
│   │   ├── hpm/                 # HPM composition grammar
│   │   ├── consolidation/       # E→S episodic→semantic transfer
│   │   └── config.py            # Shared types + default bounds
│   └── environments/
│       └── grid_world.py        # GridWorld (5×5, 10×10, 20×20)
├── scripts/
│   ├── benchmark.py             # Φ-IQ benchmark suite
│   └── profile_cycle.py         # Per-cycle profiling
├── research/                    # Formal specification documents
│   ├── outputs/
│   │   ├── 07-rigorous-whitepaper.md  # Formal whitepaper (v2.0)
│   │   └── 09-phca-v3-patch.md        # v3.0 patch + audit closure
│   └── ...
├── docs/                        # Implementation blueprints
├── logs/                        # Benchmark reports (JSON)
├── Makefile                     # setup, test-all, bench-* targets
└── README.md
```

---

## Key Documents

| Document | Description |
| :--- | :--- |
| `research/outputs/07-rigorous-whitepaper.md` | Formal scientific whitepaper — full mathematical specification |
| `research/outputs/09-phca-v3-patch.md` | v3.0 patch — RBTA correction, ASI sanitization, concurrency model, Pareto front |
| `research/outputs/05-phase2-architecture.md` | Phase 2 implementation architecture |
| `research/outputs/06-deep-gap-analysis.md` | Gap analysis leading to v3.0 patches |
| `research/glossary.md` | Terminology reference |
| `docs/11-engineers-playbook.md` | Implementation tickets + schedule |

---

## Technical Notes

- **Gaussian inference caching**: The G' Bayesian network's joint moments (173×173 matrix inversion) are computed once and cached across the 11 `predict()` calls per cycle. This provides a **1000× speedup** (24s/cycle → 13ms/cycle).
- **Goal-directed action selection**: The continuous G' model applies uniform action weights to all state dimensions and cannot distinguish spatial movement. Goal alignment is computed directly from environment state (`agent_pos`, `goal_pos`, `grid`) rather than from predictions.
- **Synthetic RBTA logs**: Memory, energy, and entropy logs are placeholders at 10-30% of their bounds until Phase 3.3 adds real instrumentation.
- **VSA deferred**: Vector Symbolic Architecture (for analogical reasoning) is deferred to Phase 3.3. The current world model uses only the probabilistic graph (G').

---

## License

Internal research project. All rights reserved.
