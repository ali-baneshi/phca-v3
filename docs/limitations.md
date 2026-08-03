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
| **Resilience failure matrix** | ⚠️ Partial (MVP) | B1, B4, B5, C1, F5 detect + recover in `phca/resilience/`; E1 FallbackController (dual-signal entropy + cascade); see [resilience.md](resilience.md) |
| **NoiseInjector (grounding simulation)** | ⚠️ Limited | `python/phca/asi/noise_injector.py` adds synthetic Gaussian noise — not a true multi-level grounding adapter (L0/L2). Real-world grounding needs sensor-specific models. |
| **Continual learning (AT-2-lite)** | ⚠️ Measured FAIL @ L4b (37.83% at 30 seeds) | Level-4-lite gate `<5%` forgetting at 10 tasks not met under P-Stream + M3 replay + D-168 PER fix (30-seed sample). Headline number reflects goal_rate variance from geometric planner at adequate statistical power. `eval_prediction_error` per-task metric now collected as a less-confounded baseline. See [docs/experiments/re-run_l4_and_d161_round14.md](docs/experiments/re-run_l4_and_d161_round14.md) |

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
| L2 Goal Pursuit Φ-IQ (static, 200 cyc) | **0.777** (historical / geometry-confounded under default) | ≥ 0.5 |
| Overall Φ-IQ (4-level MLP, 200 cyc) | **0.740** (historical; not default-geometry re-cert) | ≥ 0.5486 gate floor |

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

### Discrete GridWorld default is pure geometry (not confidence-gated hybrid)

**Current default** (`disable_blended_scorer=True`, D-156/D-161): discrete actions are
chosen by BFS/Manhattan (`selector_mode=pure_geometry_ablation`). There is **no**
confidence≥0.6 hybrid switch on the hot path anymore — that narrative is obsolete.

The opt-in blended scorer (`--enable-blended-scorer`) adds G′-scored candidates with
agreement/confidence gating (D-157/D-159). A4 is **environment-scoped (D-158)**:
geometry is an inductive bias on fully-observable GridWorld; continuous MPC
(Pendulum, Reacher) remains prediction-primary (D-101).

Session reporting flags `pure_geometry_ablation_dominant` /
`geometry_dominated_action_selection` so goal success is not mistaken for
prediction-path quality.

### Partial-observability viewport is wall/goal-only

The `partial_obs_radius` feature (D-160) only masks walls and goal outside the agent's viewport. Cell-type classification (EMPTY, WALL, GOAL, HAZARD) in the local 3×3 neighborhood is unaffected — the agent always sees the true cell type of its 8 neighbors regardless of viewport distance. This means hazards are always visible when adjacent even if outside the viewport.

### RBTA partial-obs violations reduced but not eliminated

The RBTA entropy-floor fix (dividing `entropy_floor` by the partial-obs scale factor) dropped
early short-run rates from ~60% toward single-digit means. **Authoritative 30-seed × 200-cycle
viewport rates (README / NEW-14):** ≈0.009 / 0.000 / 0.000 — the older 9–12% figures are
pre-fix / short-cycle. However, individual short-run seeds historically still reached
48–56% violation rates at small radii (viewport1/2). The remaining violations are primarily
G' ENTROPY on cycles where MC-dropout uncertainty is lowest. A non-linear scaling (e.g.,
dividing floor by `scale²`) could eliminate residual violations but risks hiding genuine
entropy anomalies.

### Frontier exploration is Manhattan-distance heuristic

`_find_frontier_cell()` uses Manhattan distance to the nearest unobserved cell as a proxy
goal. This is a simple heuristic that works well when the frontier is close and reachable,
but can send the agent toward unreachable cells (behind walls) or fail to balance exploration
across multiple frontiers. A coverage-maximization approach (number of new cells revealed per
action) would be more robust but computationally heavier. The current heuristic is sufficient
for 10×10 grids at current viewport radii (1–3).

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
| Level-4-lite forgetting gate (10-task L4b) | **Measured FAIL** — honest capacity limit; maturation 2026-07-07 |
| Full MuJoCo suite (5+ envs) | Planned |
| Multi-agent coordination | Deferred |

**Phases 7–20 complete:** Cognitive Observatory — frame schema, JSONL, 7-tab dashboard, session reports, seek/scrub replay, schema governance, report parity, scrub perf, multi-session compare, anomalies, explainability, API, supervisor, multi-agent, query, reproduce. Observatory maturity ≠ cognitive maturity. Test totals: run `make test-all` (do not cite frozen STATUS.md counts).

---

## Related Documents

- [phi_iq_metric.md](phi_iq_metric.md) — benchmark level definitions
- [architecture.md](architecture.md) — 12-step cycle and module map
- [observability.md](observability.md) — Cognitive Observatory contracts (Phases 7–20)
- [PHCA_Cognitive_Observatory_Architecture.md](PHCA_Cognitive_Observatory_Architecture.md) — full Observatory roadmap
- [docs/investigations/](investigations/) — open gap register (G2-INV-05 etc.); [DECISIONS.md](../DECISIONS.md) through D-198+
- [STATUS.md](../STATUS.md) — frozen through D-137 (historical only)
