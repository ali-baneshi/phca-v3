# PHCA v3.0 — Architecture

## Overview

PHCA (Predictive Hierarchical Cognitive Architecture) implements a **21-step cognitive cycle**
that transforms raw sensor input into goal-directed action through a pipeline of specialized
modules. The cycle runs at ~40 Hz on consumer hardware.

Formal specification: v3.0 (PHCA-3.1-011). Resource-bounded via the **Resource Bounded
Turing Supervisor (RBTA)** which enforces cycle-time, memory, energy, and entropy budgets
per the v3.0 specification §3.2–§4.

---

## System Diagram

```
                     ┌──────────────────────────────────────────┐
                     │              GridWorld (env)             │
                     │  EnvironmentProtocol                     │
                     └──────────────────┬───────────────────────┘
                                        │ raw_obs
                                        ▼
  ┌─────────────────────────────────────────────────────────────────┐
  │                      21-Step Cognitive Cycle                    │
  │                                                                  │
  │  Step  0: ASI Sanitize       sanitize(raw_obs) → clean_state    │
  │  Step  1: WM Write           m2.write(state) + m1.write(state)  │
  │  Steps 2-4: G' Prediction    engine.predict(state)              │
  │  Steps 5-6: PEU Error        peu.compute(state, prediction)     │
  │  Step  7: TSPL P-Stream      tspl.update(error, state, pred)    │
  │  Step  8: (reserved)                                            │
  │  Step  9: Action Selection   argmax(goal + confidence)          │
  │  Steps 10-13: MDIM+CR+ATTN+HPM   goal, criticality, attend,    │
  │                                  validate                       │
  │  Step 14: RBTA Enforcement   check_cycle(runtime, mem, energy)  │
  │  Step 15: Logging            append to metrics_history           │
  │  Steps 16-18: Consolidation  episodic → semantic transfer       │
  │  Step 19: Increment          cycle_count += 1                   │
  │  Step 20: Sleep Cycle        full consolidation every 50 cycles │
  └─────────────────────────────────────────────────────────────────┘
                                        │ action
                                        ▼
                     ┌──────────────────────────────────────────┐
                     │              GridWorld (env)             │
                     │  step(action) → obs, reward, done        │
                     └──────────────────────────────────────────┘
```

---

## Module Map

| Module | File | Function |
|---|---|---|
| **ASI** | `phca/perception/asi.py` | Input sanitization, NaN detection, finite checks |
| **M1 (Sensory)** | `phca/working_memory/m1_sensory.py` | Short-term sensory buffer (50-cycle horizon) |
| **M2 (Working)** | `phca/working_memory/m2_working.py` | Ring-buffer working memory with salience tracking |
| **G' (Engine)** | `phca/world_model/engine.py` | Gaussian G' / MLP prediction engine |
| **PEU** | `phca/learning/peu.py` | Precision-weighted prediction error |
| **TSPL** | `phca/learning/tspl.py` | Skill learning via P-Stream predictions |
| **MDIM** | `phca/motivation/mdim.py` | Multi-drive intrinsic motivation (exploration, competence, novelty) |
| **CR** | `phca/motivation/criticality.py` | Criticality regulation (energy homeostasis) |
| **ATTN** | `phca/learning/attention.py` | Sparse attention mechanism |
| **HPM** | `phca/hpm/` | Hierarchical procedure memory (parser, compute_bounds) |
| **RBTA** | `phca/governance/rbta_enforcer.py` | Resource-bounded Turing supervisor (time, memory, energy, entropy) |
| **PID** | `phca/governance/pid_controller.py` | PID-inspired controller for orthogonality constraint |
| **M3 (Episodic)** | `phca/episodic_memory/m3_episodic.py` | Episodic memory buffer (SQLite-backed) |
| **Consolidation** | `phca/episodic_memory/consolidation.py` | Episodic → semantic transfer |
| **Cycle** | `phca/core/cycle.py` | 21-step cognitive cycle orchestrator |
| **Config** | `phca/config.py` | Global constants, enums, StreamID |

---

## Key Design Decisions

- **EnvironmentProtocol** decouples `CognitiveCycle` from the concrete `GridWorld` environment.
  Any environment implementing the protocol (get_action_names, get_possible_actions, get_goal_position)
  can drive the cycle.
- **Only P-Stream** is active. E-Stream and S-Stream (for episodic/semantic consolidation via
  TSPL) were removed as always-disabled in Phase 3.3. Consolidation runs on a fixed 50-cycle
  timer instead.
- **HPM validation layer** stripped. Only `compute_bounds()` (resource additivity per v3.0
  Theorem 2.1/3.1) is used by RBTA.
- **SQLite-backed M3** uses a single backend — an abstraction layer was deferred to Phase 3.4
  (no benefit for a single implementation).

## Resource Bounds (RBTA)

The RBTA enforces four budgets every cycle:

| Resource | Default Bound | Scope |
|---|---|---|
| Time | 500 ms | Per cycle |
| Memory | 1,000,000 units | Per cycle |
| Energy | 100.0 units | Per cycle |
| Entropy | 1.0 units | Per cycle |

Violations are counted and reported in `CycleMetrics.violations_count`. The agent continues
running but the violation is logged for analysis.
