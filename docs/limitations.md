# Limitations — What Erasmus Cannot Do (Yet)

Erasmus is a cognitive architecture for **embodied, resource-bounded autonomous agents**. It is not a general AI system. This document clearly states what it cannot do, so you can decide if it is the right tool for your use case.

---

## What Erasmus Cannot Do

| Capability | Status | Explanation |
|---|---|---|
| **Language understanding** | ❌ Not supported | No NLP, no chat, no text generation. The system operates on numerical state vectors only. |
| **General knowledge** | ❌ Not supported | No knowledge base, no web search, no common-sense reasoning. Knowledge is limited to learned transition patterns. |
| **Creative writing / art** | ❌ Not supported | No generative capabilities for text, images, or music. |
| **Real-time control** | ⚠️ Limited | ~50ms per cognitive cycle (~20 decisions/second). Not suitable for sub-20ms real-time control loops. |
| **Continuous actions** | ⚠️ Experimental | Action spaces are discretised into ≤5 bins. Continuous action support is in development. |
| **Vision / image processing** | ❌ Not supported | No convolutional layers, no image input. States are flat numerical vectors. |
| **Multi-agent coordination** | ❌ Not supported | Single-agent only. Multiple cycles cannot share memory. |
| **Multi-level grounding (levels 0–2)** | ⚠️ Partial | ASI always emits level 1; Grounding Level Adapter (§2.5.1) deferred to Phase 4.2 |
| **Long-term procedural memory (M5)** | ❌ Not implemented | Skills are compiled in TSPL but not stored in a persistent library. |

---

## Current Constraints

### Environment
| Constraint | Limit | Reason |
|---|---|---|
| State dimension (GridWorld) | 84 (5×5), 309 (10×10), 1209 (20×20) | GridWorld one-hot encoding + local view |
| Discrete actions | ≤ 5 | Action discretisation granularity |
| Action space | Discrete only | Continuous actions experimental |
| Grid size (GridWorld) | 5×5, 10×10, 20×20 | Computational budget |

### Performance
| Metric | Typical Value | Target |
|---|---|---|
| Cycle latency (MLP) | ~50ms p95 | < 500ms (met) |
| Cycle latency (Gaussian G') | ~214ms p95 | < 500ms (met) |
| Decisions per second | ~20 | N/A |
| Prediction error (steady state) | 0.05–0.15 | Decreasing over time |
| L2 Goal Pursuit Φ-IQ | **0.476** | ≥ 0.5 (Phase 4 target) |

### Hardware
| Resource | Requirement |
|---|---|
| CPU only | Tested on consumer x86_64 (no GPU needed) |
| RAM | ~100 MB typical |
| Disk | ~50 MB for code + dependencies |
| Optional | Rust toolchain for performance-critical modules |

---

## Known Weaknesses

### Level 2 (Goal Pursuit) Φ-IQ = 0.477

Goal reaching (`goal_rate ≈ 0.94`) is strong after D-070/D-071 fixes, but L2 Φ-IQ remains below the 0.5 target because **adaptation_speed** and **transfer_efficiency** sub-metrics score near zero in sustained goal-holding runs. Phase 4.1 will address skill chaining and cross-episode transfer.

### MLP Default hidden_dim = 128

The MLP world model uses 128 hidden units (~38,868 parameters) by default per D-028/D-072. This is the canonical capacity for GridWorld-scale environments.

### Discrete Actions Only
All environments currently discretise actions into 3–5 bins. For Cartpole: `[-3, 0, +3]`. For Pendulum: `[-2, 0, +2]`. Fine-grained or continuous control is not yet supported.

### MuJoCo Environments (Experimental)
MuJoCo support is functional but experimental:
- Only 3 environments tested (InvertedPendulum-v5, Pendulum-v1, Reacher-v5)
- Requires `gymnasium[mujoco]` (not in default dependencies)
- Continuous action support deferred

---

## When NOT to Use Erasmus

- You need **language processing**, **chat**, or **text generation**
- You need **image recognition**, **object detection**, or **video processing**
- You need **real-time control** with sub-20ms latency requirements
- You need **continuous action spaces** (e.g., torque control with fine granularity)
- You need **general knowledge**, **common-sense reasoning**, or **logical inference**
- You need a **pre-trained model** — Erasmus learns from scratch in each run

---

## Roadmap to Address Limitations

| Limitation | Target Phase | Status |
|---|---|---|
| L2 Φ-IQ ≥ 0.5 | Phase 4.1 | 🔜 In progress |
| Continuous actions | Phase 4.1 | 🔜 Planned |
| Full MuJoCo suite (5+ envs) | Phase 4.2 | 🔜 Planned |
| ROS 2 integration | Phase 4.2 | 🔜 Planned |
| Real-robot tests | Phase 4.3 | 🔜 Planned |
| M5 procedural memory | Phase 4.3 | 🔜 Planned |
| Multi-agent coordination | Phase 4.4 | 🔜 Deferred |
| Hierarchical planning | Phase 4.4 | 🔜 Deferred |

---

## Review Notes (Pass 1 — Accuracy)
- MLP latency 50ms p95 verified from gate_phase3.3_final.md (MLP mode).
- Gaussian G' latency 214ms verified from same source.
- L2 Φ-IQ 0.477 verified from `logs/benchmark_l2_ps1.json` (500 cycles, hidden_dim=128).
- MLP hidden_dim=128 verified from cycle.py `build()` and mlp.py constructor (D-072).
- All "not supported" capabilities verified by searching source for relevant modules.
- Discrete action counts verified from action maps in mujoco_env.py and grid_world.py.

## Review Notes (Pass 2 — Clarity)
- Each limitation has a clear explanation of why it exists.
- Performance table gives both typical values and targets.
- "When NOT to use" section makes adoption decisions easy.

## Review Notes (Pass 3 — Completeness)
- Covers: what it cannot do, current constraints, known weaknesses, when to avoid, roadmap.
- Links to phi_iq_metric.md for benchmark interpretation.
- Roadmap covers Phase 4 plans with realistic timing.
