# Limitations — What PHCA Cannot Do (Yet)

PHCA (Predictive Hierarchical Cognitive Architecture) is a cognitive architecture for **embodied, resource-bounded autonomous agents**. It is not a general AI system. (The project was briefly referred to as "Erasmus" in early docs; **PHCA** is the canonical name.)

This document states what PHCA cannot do, so you can decide if it is the right tool for your use case.

---

## What PHCA Cannot Do

| Capability | Status | Explanation |
|---|---|---|
| **Language understanding** | ❌ Not supported | No NLP, no chat, no text generation. The system operates on numerical state vectors only. |
| **General knowledge** | ❌ Not supported | No knowledge base, no web search, no common-sense reasoning. Knowledge is limited to learned transition patterns. |
| **Creative writing / art** | ❌ Not supported | No generative capabilities for text, images, or music. |
| **Real-time control** | ⚠️ Limited | ~10–17 ms mean cycle latency on the MLP path (~60–95 decisions/second). Not suitable for sub-5 ms hard real-time loops. |
| **Vision / image processing** | ❌ Not supported | No convolutional layers, no image input. States are flat numerical vectors (MuJoCo camera is observability-only). |
| **Multi-agent coordination** | ❌ Not supported | Single-agent only. Multiple cycles cannot share memory. |
| **Multi-level grounding (levels 0–2)** | ⚠️ Partial | ASI always emits grounding level 1; Grounding Level Adapter deferred. |
| **Long-term procedural memory (M5)** | ❌ Not implemented | Skills are compiled in TSPL but not stored in a persistent library. |
| **Dual G′+V ensemble / VSA** | ❌ Not implemented | Blueprint items; only G′ (Gaussian / discrete graph / MLP) is implemented. |
| **Resilience failure matrix** | ❌ Stub only | `phca/resilience/` is a placeholder package. |

---

## Current Constraints

### Environment

| Constraint | Limit | Notes |
|---|---|---|
| State dimension (GridWorld) | 84 (5×5), 309 (10×10), 1209 (20×20) | One-hot encoding + local view |
| GridWorld actions | 5 discrete (N/S/E/W/stay) | Discrete only |
| Cartpole | 3 discrete bins | `InvertedPendulum-v5` |
| Pendulum | **Continuous** torque ∈ [-2, 2], dim 1 | Phase 6 — MPC prediction-driven selector |
| Reacher | **Continuous** actuator ∈ [-1, 1]², dim 2 | Phase 7 / D-107 — same MPC path as Pendulum |
| MuJoCo | Opt-in (`requirements-mujoco.txt`) | 3 envs validated; headless: `MUJOCO_GL=disabled` |

### Performance (this machine, re-measured 2026-07-03)

| Metric | Typical value | Target / bound |
|---|---|---|
| Cycle latency (MLP) mean | ~10–17 ms | < 500 ms RBTA bound (met) |
| Cycle latency (MLP) p95 | ~14–31 ms | < 500 ms (met) |
| `gprime_learn` mean | ~5–9 ms (env-dependent) | Dominant per-module cost |
| Decisions per second | ~60–95 | N/A |
| Prediction error (steady state) | 0.05–0.15 | Decreasing over time |
| L2 Goal Pursuit Φ-IQ (static, 200 cyc) | **0.777** | ≥ 0.5 (met) |
| Overall Φ-IQ (4-level MLP, 200 cyc) | **0.740** | ≥ 0.5486 gate floor (met) |

### Hardware

| Resource | Requirement |
|---|---|
| CPU only | Tested on consumer x86_64 (no GPU needed) |
| RAM | ~100–270 MB typical (grows under long soaks — see below) |
| Disk | ~50 MB for code + dependencies |

Rust toolchain is **not** required (workspace removed D-084).

---

## Known Weaknesses

### Long-run memory growth (Phase 7 workstream)

The nightly stress test uses a **late-half RSS slope** gate (`LEAK_SLOPE_LATE = 500 B/cyc`). On this machine (2026-07-03), a 1000-cycle run reports late slope **~4650 B/cyc**, so `make nightly` **fails** the retention gate even though latency, violations, and Φ-IQ checkpoints pass. Root cause: M3 episodic fill phase, in-memory SQLite fragmentation, and M4 fact accumulation. M4 now has a 1000-fact cap with pruning (Phase 7 / B2 in code); the soak slope target remains open.

### Discrete GridWorld selector is not purely prediction-driven

The canonical discrete path blends Manhattan distance gain with prediction confidence and MDIM alignment. The **continuous MPC path** (Pendulum, Reacher) is the clean prediction-primary mechanism (A4 measured there, D-101).

### MLP default hidden_dim = 128

The MLP world model uses 128 hidden units (~38,868 parameters) by default per D-028/D-072. This is the canonical capacity for GridWorld-scale environments.

### Dynamic goals are experimental

`--dynamic-goals-every 75` is the validated cadence (L2 ≥ 0.50). every-50 and every-100 are not achievable on this machine. Dynamic L2 is SGD-trajectory-sensitive across machines (D-092).

### Consolidation "semantic" facts are statistical

`SemanticFact` in `consolidation/scheduler.py` is pattern matching over episodes, not true semantic memory.

---

## When NOT to Use PHCA

- You need **language processing**, **chat**, or **text generation**
- You need **image recognition**, **object detection**, or **video processing**
- You need **sub-5 ms hard real-time** control loops
- You need **general knowledge**, **common-sense reasoning**, or **logical inference**
- You need a **pre-trained model** — PHCA learns from scratch in each run
- You need **multi-agent** coordination or shared memory across agents

---

## Open Work (Phase 7+)

| Item | Status |
|---|---|
| M3/M4 retention soak (late slope ≤ 500 B/cyc) | In progress — `make nightly` fails today |
| Cognitive Observatory hardening | In progress — JSONL/replay/report parity |
| Grounding adapter (levels 0/2) | Deferred |
| M5 procedural memory | Not implemented |
| Full MuJoCo suite (5+ envs) | Planned |
| Multi-agent coordination | Deferred |

---

## Related Documents

- [phi_iq_metric.md](phi_iq_metric.md) — benchmark level definitions
- [architecture.md](architecture.md) — 12-step cycle and module map
- [observability.md](observability.md) — Cognitive Observatory contracts
- [STATUS.md](../STATUS.md) — live issue registry and test status
