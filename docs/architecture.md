# PHCA v3.0 — Architecture

## Overview

PHCA (Predictive Hierarchical Cognitive Architecture) implements a **12-step cognitive cycle**
that transforms raw sensor input into goal-directed action through a pipeline of specialized
modules. The cycle runs at ~40 Hz on consumer hardware (~25ms avg latency).

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
  │                      12-Step Cognitive Cycle                    │
  │                                                                  │
  │  Step  0: ASI Sanitize       sanitize(raw_obs) → clean_state    │
  │  Step  1: WM Write           m2.write(state) + m1.write(state)  │
  │  Steps 2-4: G' Prediction    engine.predict(state)              │
  │  Steps 5-6: PEU Error        peu.compute(state, prediction)     │
  │  Step  7: TSPL P-Stream      tspl.update(error, state, pred)    │
  │  Step  8: (reserved)                                            │
  │  Step  9: Action Selection   argmax(goal_alignment + confidence)│
  │  Steps 10-13: MDIM+CR+ATTN+HPM   generate_goal, regulate,      │
  │                                  attend, compute_bounds         │
  │  Step 14: RBTA Enforcement   check_cycle(runtime, mem, energy)  │
  │  Step 15: Logging            append metrics to history           │
  │  Steps 16-18: Consolidation  episodic → semantic transfer       │
  │  Step 19: Increment          cycle_count += 1                   │
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
| **ASI** | `phca/perception/asi.py` | Input sanitization, NaN/Inf detection, finite checks |
| **M1 (Sensory)** | `phca/working_memory/m1_sensory.py` | Short-term sensory buffer (50-cycle horizon) |
| **M2 (Working)** | `phca/working_memory/m2_working.py` | Ring-buffer working memory with salience tracking |
| **G' (Engine)** | `phca/world_model/engine.py` | Gaussian G' / MLP prediction engine; 1000× speedup via cached joint moments |
| **PEU** | `phca/learning/peu.py` | Precision-weighted prediction error |
| **TSPL** | `phca/learning/tspl.py` | Single-stream predictive learning (P-Stream only) |
| **MDIM** | `phca/motivation/mdim.py` | Multi-drive intrinsic motivation (6 drives) with softmax goal selection |
| **CR** | `phca/motivation/criticality.py` | Criticality regulation (energy homeostasis) |
| **ATTN** | `phca/learning/attention.py` | Precision-weighted sparse attention with Gumbel noise |
| **HPM** | `phca/hpm/parser.py` | Hierarchical procedure memory: composition operators + `compute_bounds()` |
| **RBTA** | `phca/governance/rbta_enforcer.py` | Resource-Bounded Turing Supervisor: time/memory/energy/entropy enforcement |
| **PID** | `phca/governance/pid_controller.py` | PID-inspired controller for Φ orthogonality constraint |
| **M3 (Episodic)** | `phca/episodic_memory/m3_episodic.py` | SQLite-backed episode store with batch commits |
| **Consolidation** | `phca/episodic_memory/consolidation.py` | Episodic → semantic transfer with periodic fact extraction |
| **Cycle** | `phca/core/cycle.py` | 12-step cognitive cycle orchestrator |
| **GridWorld** | `phca/environments/grid_world.py` | Configurable grid environment (5×5, obstacles) |
| **MuJoCoEnv** | `phca/environments/mujoco_env.py` | MuJoCo physics environment wrapper |
| **Config** | `phca/config.py` | Global constants, resource bounds, `StreamID` (P-Stream only) |

---

## Key Design Decisions

- **EnvironmentProtocol** (`phca/environments/protocol.py`) decouples `CognitiveCycle` from
  any concrete environment. Any object implementing `get_action_names()`, `get_possible_actions()`,
  and `get_goal_position()` can drive the cycle. Both `GridWorld` and `MuJoCoSimpleEnv` implement it.
- **P-Stream only.** E-Stream and S-Stream were removed in Phase 3.3 (D-020). Consolidation
  runs on a fixed 10-cycle timer (Steps 16-18) with batch SQLite commits (D-014).
- **HPM validation layer** stripped. Only `compute_bounds()` (resource additivity per v3.0
  Theorem 2.1/3.1) is used by RBTA.
- **SQLite-backed M3** uses a single backend — an abstraction layer was deferred to Phase 3.4
  (no benefit for a single implementation). Batch commits every 10 cycles reduce fsync overhead 10×.
- **Monitoring system** (`MetricsStore`, file logging, curses dashboard) is optional, zero-overhead
  when unused. See `scripts/phca-monitor.py` and `scripts/phca-logs.py`.

## Resource Bounds (RBTA)

The RBTA enforces four budgets every cycle:

| Resource | Default Bound | Scope |
|---|---|---|
| Time | 500 ms | Per cycle |
| Memory | 1,000,000 units | Per cycle |
| Energy | 100.0 units | Per cycle (from `energy_log`, composition tree) |
| Entropy | 1.0 units | Per cycle |

Violations are counted and reported in `CycleMetrics.violations_count`. The agent continues
running but the violation is logged for analysis.

**Note:** Energy bounds are enforced via the composition tree which reads from `energy_log`
(effective Joules estimated from module runtimes) rather than `runtime_log` (seconds).
This was fixed in gap-closure issue A-001/A-004 (D-036).

### Up-to-Date Reference

See `DECISIONS.md` (D-001 through D-044) for the complete design decision history,
and `docs/phase3.3_full_completion_report.md` for the gap-closure execution summary.
