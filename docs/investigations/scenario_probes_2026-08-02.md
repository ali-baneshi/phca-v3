# Scenario Probe Report — 2026-08-02

Runtime probes are short (structural / smoke). Behavioral gate numbers prefer existing **30-seed** artifacts. Probe JSON: `logs/investigation_probes_2026-08-02.json`, `logs/investigation_t3_t7_probes.json`, `logs/investigation_t2_analysis.json`.

## S1 — Default 5×5 L2 (geometry)

| Check | Result |
|-------|--------|
| Config | `InterventionConfig()` default, discrete G′, seed 42, 40 cycles |
| `selector_mode` | **100% `pure_geometry_ablation`** |
| `engine.predict` during `_select_action` | **0** (40 total = perception-only) |
| Learn-off ablation (80 cyc, seed 42) | goal_frac **0.9875 on = 0.9875 off** |
| Decision | Default success is planner-dominated; learning off does not change goal_rate |

Existing 30-seed causal (5×5 L2 pure geo): PHCA goal_rate **0.552** vs greedy_observed **0.598** → gate **FAIL** (`logs/phase1_redux_5x5_pure_geo_30s.json`).

## S2 — 10×10 + partial obs / viewport

| Check | Result |
|-------|--------|
| Default 10×10 short probe | 30/30 `pure_geometry_ablation`; predict-during-select **0**; state_dim **309** |
| Discrete G′ coverage | ~20 graph nodes (10 dims × t/t+1) → **~6.5%** of state_dim |
| Causal 10×10 L2 (30s artifact) | PHCA **0.199** vs greedy **0.187** → gate **PASS** |
| Viewport1–3 (`logs/causal_frontier_full.json`) | gates **PASS**; PHCA ≫ greedy_observed |
| Decision | Scale/viewport PASS ≠ prediction-primary; UNKNOWN G′ contribution under fog — needs blended-off vs learn-off ablation at 30 seeds (not run this pass) |

## S3 — Blended scorer opt-in

| Check | Result |
|-------|--------|
| Config | `disable_blended_scorer=False`, `before_blended_warmup_cycles=0`, 5×5, 40 cycles |
| `selector_mode` | **100% `prediction_scored`** |
| predict-during-select | **200** (5 actions × 40) |
| Historical 30-seed | D-161: neither mode universally wins; agreement-gated can underperform at 10×10 |
| Decision | Opt-in path is live and G′-consulting; default remains geometry |

## S4 — MuJoCo continuous (Pendulum)

| Check | Result |
|-------|--------|
| Modes | mostly `continuous_mpc` (13/15), some `continuous_explore` |
| predict-during-select | **104 / 15 ≈ 6.93** per select (MPC K≈8) |
| Decision | **A4 structurally holds** on continuous path |

## S5 — L4 continual

| Check | Result |
|-------|--------|
| Artifact | `logs/benchmark_level4_30s.json` |
| Aggregate | forgetting_rate **0.3783**, passes_gate **false** |
| Per-seed | 28/30 at FR=1.0; 1 at 0.15; **1 vacuous PASS** (seed 1542, 8/10 excluded) |
| Mean per-seed FR | **~0.938** (≠ aggregate 0.378 — G4 risk) |
| `eval_prediction_error` (tasks 0–9) | ~2.58–7.03 (better retention proxy) |
| MLP learn-on vs off (5×5, 50 cyc) | goal_frac both 0.98; PE late **2.09 vs 39.3** — model learns, action geometry masks it |
| Decision | L4 goal-forgetting gate fails and is confounded; PE shows learning exists under MLP |

## S6 — Async mode

| Check | Result |
|-------|--------|
| Default | `_async_mode=False` |
| Carry | Sync applies `_rbta_carry` before action; **action loop does not**; learning loop does |
| Staleness | `age=0` literal in `_action_loop`; check inert |
| Queue | Dead PerceptionFrame queue **removed** (D-194); carry now applied in action loop |
| Decision | Still experimental; do not claim dual-thread correctness |

## S7 — RBTA stress / entropy honesty

| Check | Result |
|-------|--------|
| Entropy after remediation (D-194) | Real: G′, MDIM, ATTN only; others **omitted** (`entropy_na`) |
| Runtime log | `entropy=G'_MDIM_ATTN_only_others_entropy_na` |
| Historical | D-159 RBTA bound bug explained D-156 0.03%; D-161 still notes high violation rates can confound |
| Decision | A3 “measured PASS” stays scoped to modules with real epistemic entropy |

## Cross-cutting ASI probe (T6)

| Check | Result |
|-------|--------|
| Forced `SENSOR_FAILURE` | `sensor_failure_count=1`, **state unchanged** (stale) |
| Decision | Silent continue-on-failure; hardening candidate H-12 |

## Evidence standard compliance

- New short probes used for **structural** claims only.
- Gate / superiority claims cite existing **30-seed** artifacts or DECISIONS.
- UNKNOWN left explicit for viewport G′ contribution without new 30-seed ablation.
