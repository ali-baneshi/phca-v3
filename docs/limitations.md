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
| **Resilience failure matrix** | ⚠️ Partial (MVP) | B1, B4, B5, C1, F5 detect + recover in `phca/resilience/`; see [resilience.md](resilience.md) |

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

The nightly stress test uses a **phase-aware late RSS slope** gate (D-112, D-113, D-134): **≤5000 B/cyc** for runs under 7000 cycles (M3 fill phase) and **≤1600 B/cyc** for post-cap soaks (default `make nightly NIGHTLY_CYCLES=11000`). For runs beyond 10k cycles, the gate fits late slope on **post-M3 samples only** (cycles > 10000) so the tail-quarter M3 fill window does not inflate the measurement. M3 VACUUM (D-108) and M4 cap 1000/prune 500 (D-109) bound steady-state growth.

### Discrete GridWorld selector is not purely prediction-driven

When an extrinsic goal is present, **`task_lock`** routes action selection through an observed-greedy Manhattan controller **when G′ prediction confidence ≥ 0.6** (fair vs `greedy_observed` baseline); below that threshold the blended per-candidate G′ scorer is used instead. A sparse L3 coverage probe every 50 cycles on-goal (D-112). G′ prediction drives the **continuous MPC path** (Pendulum, Reacher) — the clean prediction-primary mechanism (A4 measured there, D-101).

### MLP default hidden_dim = 128

The MLP world model uses 128 hidden units (~38,868 parameters) by default per D-028/D-072. This is the canonical capacity for GridWorld-scale environments.

### Dynamic goals are experimental

`--dynamic-goals-every 75` is the validated cadence (L2 ≥ 0.50). every-50 and every-100 are not achievable on this machine. Dynamic L2 is SGD-trajectory-sensitive across machines (D-092).

### Consolidation "semantic" facts are statistical

`SemanticFact` in `consolidation/scheduler.py` is pattern matching over episodes, not true semantic memory.

### Observatory limitations (Phases 7–20 complete)

Phases 7–20 delivered live PyQt dashboard, JSONL recording, seek/scrub replay, schema governance, report parity, large-session scrub performance, multi-session compare, anomaly detection, action explainability, stable API, supervisor/recovery, multi-agent timelines, cognitive-moment query, and scientific reproduction. Remaining Observatory limits:

- **Live-only fields in replay** — camera frames, bulky rollouts, full M3/M4 lists are not in JSONL (by design; banners mark gaps).
- **`schema_version`** — shipped Phase 9 (`OBSERVABILITY_SCHEMA_VERSION = 1`, D-110); legacy v0 sessions normalize on replay.
- **Large-session scrub** — sessions of 3000+ cycles: scrub rebuild budget **≤2s** (live) / **≤4s** (review) enforced by tests (Phase 11).
- **Legacy matplotlib replay** — `phca_visualise.py` / `--from-jsonl` is not full-fidelity; use PyQt `--qt`.
- **Offline report scope** — `session_report.json` summarizes key metrics and shares `format_session_results_lines()` with the Overview panel (Phase 10 parity for mechanism/phase budget); it does not replicate every live-only subview (camera, full rollouts).

---

## When NOT to Use PHCA

- You need **language processing**, **chat**, or **text generation**
- You need **image recognition**, **object detection**, or **video processing**
- You need **sub-5 ms hard real-time** control loops
- You need **general knowledge**, **common-sense reasoning**, or **logical inference**
- You need a **pre-trained model** — PHCA learns from scratch in each run
- You need **multi-agent** coordination or shared memory across agents

---

## Open Work

| Item | Status |
|---|---|
| M3/M4 retention soak (post-M3 late slope ≤ 1600 B/cyc @ 11k) | **Done** (D-134; post-M3 window) |
| `schema_version` + JSONL migration | Done (Phase 9 / D-110) |
| Full offline report ↔ dashboard parity | Done (Phase 10) |
| Multi-session comparison (`--compare`) | Done (Phase 12) |
| Large-session scrub performance (3000+ cycles) | Done (Phase 11) |
| Grounding adapter (levels 0/2) | Deferred |
| M5 procedural memory | Not implemented |
| Full MuJoCo suite (5+ envs) | Planned |
| Multi-agent coordination | Deferred |

**Phases 7–20 complete:** Cognitive Observatory — frame schema, JSONL, 7-tab dashboard, session reports, seek/scrub replay, schema governance, report parity, scrub perf, multi-session compare, anomalies, explainability, API, supervisor, multi-agent, query, reproduce — **386 monitoring tests** (743 total with MuJoCo — [STATUS.md](../STATUS.md)).

---

## Related Documents

- [phi_iq_metric.md](phi_iq_metric.md) — benchmark level definitions
- [architecture.md](architecture.md) — 12-step cycle and module map
- [observability.md](observability.md) — Cognitive Observatory contracts (Phases 7–20)
- [PHCA_Cognitive_Observatory_Architecture.md](PHCA_Cognitive_Observatory_Architecture.md) — full Observatory roadmap
- [STATUS.md](../STATUS.md) — live issue registry and test status
