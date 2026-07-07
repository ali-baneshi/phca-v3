# PHCA Investigation Report — 2026-07-07

## Purpose
This report completes the investigation phase requested for PHCA. It does not propose code changes yet. Its job is to turn recent runs, logs, reports, benchmarks, and tests into an evidence-backed defect inventory that can drive a later fix-planning phase.

## Evidence Inventory
### Available evidence
- Observatory sessions under `logs/sessions/*/` with `meta.json`, `timeseries.jsonl`, and often `session_report.json`
- Structured logs in `logs/phca.log`
- Supervisor logs in `logs/supervisor/*.jsonl`
- Benchmark and nightly artifacts under `logs/*.json`
- Monitoring and system tests under `python/phca/monitoring/tests/` and `python/tests/`

### Inventory summary
- `176` session directories were present under `logs/sessions/`
- Environment mix from `meta.json`:
  - `Reacher-v5`: `100`
  - `gridworld`: `69`
  - `Pendulum-v1`: `4`
  - `InvertedPendulum-v5`: `3`
- Agent-count mix:
  - single-agent: `166`
  - 2-agent: `7`
  - 3-agent: `1`
  - 4-agent: `2`
- Profile mix:
  - `custom`: `55`
  - `near_real`: `12`
  - `readable`: `35`
  - legacy / unset: `74`
- Top-level log artifacts under `logs/*.json`: `134`
  - benchmarks: `74`
  - phase-history artifacts: `29`
  - nightly artifacts: `11`
  - causal-eval artifacts: `5`
  - profile artifacts: `6`

### Representative scenario matrix
| Scenario | Evidence | Notes |
|---|---|---|
| Multi-agent GridWorld, MLP on | `logs/sessions/20260707_104736/` | 4 agents, 2000 total lines, `custom`, `auto`, structurally complete |
| Multi-agent GridWorld, MLP off | `logs/sessions/20260707_105557/` | 4 agents, 2000 total lines, `custom`, `auto`, structurally complete |
| Long single-agent GridWorld | `logs/sessions/20260707_104025/` | 4000 cycles, `readable`, video present |
| Short live Pendulum | `logs/sessions/20260707_092235/` | 800 requested, 438 recorded, `incomplete`, user-closed |
| Live Reacher | `logs/sessions/20260707_105506/` | 1000 cycles, `near_real`, live camera |
| Long Reacher soak | `logs/sessions/20260707_102218/` | 9900 cycles, `near_real`, high anomaly pressure |
| Nightly long-run baseline | `logs/nightly_stress.json` | global pass at 1000 cycles |
| Recovery benchmark | `logs/benchmark_recovery.json` | all scenarios recovered |
| Continual-learning ablation | `logs/l4_ablation.json` | L4b still fails in all ablations |
| Causal smoke | `logs/phca_causal_eval_smoke.json` | PHCA equals `greedy_observed` on level 2 smoke |

## Trustworthiness Ledger
The replay/check contract in `docs/observability.md` was used as the trust gate.

### Trustworthy sessions
These sessions passed structure/integrity checks and are safe to interpret diagnostically:
- `logs/sessions/20260707_105557/`
  - `PASS`: line count, per-agent contiguity, step-major multi-agent order, explain fields
  - `FAIL` anomalies: leak
- `logs/sessions/20260707_104025/`
  - `PASS`: line count, contiguous `cycle_id`, video present and ffprobe-valid, explain fields
  - `FAIL` anomalies: spike, leak
- `logs/sessions/20260707_105506/`
  - `PASS`: line count, contiguous `cycle_id`, explain fields
  - `FAIL` anomalies: spike, leak, goal_instability

### Partially trustworthy / forensic sessions
- `logs/sessions/20260707_092235/`
  - `meta.status = "incomplete"`
  - `recorded_cycles = 438` vs `cycles = 800`
  - recovery reason is `user_close`
  - still valuable for behavior and action-path evidence, but not for whole-run trend claims

### Supervisor evidence
- `logs/supervisor/long_live.jsonl` shows a clean wrapped run:
  - child started
  - child exit code `0`
  - `recover_skip` due to no `.latest` pointer, which matches documented normal close behavior
- `logs/supervisor/20260707_105548.jsonl` shows a malformed supervisor invocation:
  - child command parsing is broken
  - child exits immediately with code `2`
  - not useful for PHCA behavioral diagnosis

## Confirmed Symptom Families
### 1. Discrete GridWorld behavioral stagnation
Observed in:
- `logs/sessions/20260707_104025/session_report.json`
- `logs/sessions/20260707_104736/session_report.json`
- `logs/sessions/20260707_105557/session_report.json`

Symptoms:
- `greedy_fallback` mechanism is `100%`
- action status converges to repeated `STAY`
- `cycles_with_scores = 0` in report for GridWorld runs
- runs can still reach goals at high rates (`goal_reached_count` near cycle count), meaning the system is succeeding through a fallback path rather than a prediction-primary path

Interpretation:
- This is a real and expected architectural limitation for discrete GridWorld with extrinsic goals, not merely a UI bug.
- The effect still matters operationally because the dashboard can make the system look “stuck,” and in fact it is stuck on a fallback policy path even when goals are being reached.

### 2. Multi-agent GridWorld mode split: MLP-on vs MLP-off
Observed in:
- `logs/sessions/20260707_104736/session_report.json` (MLP on)
- `logs/sessions/20260707_105557/session_report.json` (MLP off)

Symptoms:
- Both sessions show `greedy_fallback = 100%`
- MLP-on session: `error_improved = true` for all four agents, but many RBTA violations and `gprime_learn` dominates
- MLP-off session: `error_improved = false` for all four agents, few violations, bottleneck shifts to `mdim`

Interpretation:
- The discrete fallback masks path quality at the policy level, so the apparent “similar behavior” in the UI hides a major underlying model-path difference.
- Turning MLP off appears to remove the error-improvement behavior without changing the observed task-lock outcome, which makes the dashboard look deceptively similar while the underlying predictive system has degraded.

### 3. Long-run learning-path pressure in GridWorld
Observed in:
- `logs/sessions/20260707_104025/session_report.json`

Symptoms:
- `spike_count = 346`
- `learn_burst_count = 3937`
- `interrupt_cycle_count = 491`
- `violation_cycle_count = 491`
- `phase_budget_pct.Learn = 75.52`
- `module_time_share_pct.gprime_learn = 71.34`
- replay `--check` flags `spike` and `leak`

Interpretation:
- The long-run GridWorld issue is not “no learning.” It is the opposite: the learning path (`gprime_learn`) dominates the cycle and causes recurrent interrupts/pressure.
- This is separate from the fallback-policy issue. One issue is policy path selection; the other is long-run cycle cost and resource pressure.

### 4. MuJoCo continuous path is real, but anomaly-heavy
Observed in:
- `logs/sessions/20260707_105506/session_report.json`
- `logs/sessions/20260707_092235/session_report.json`
- `logs/sessions/20260707_102218/session_report.json`

Symptoms:
- mechanism histogram is correctly continuous-dominated:
  - Reacher 1000-cycle run: `continuous = 96.4%`, `explore = 3.5%`
  - Pendulum incomplete run: `continuous = 94.75%`, `explore = 4.57%`
- prediction error improves strongly:
  - Reacher: `26.35 -> 1.24`
  - Pendulum incomplete: `1.81 -> 0.44`
- but anomaly pressure remains significant:
  - Reacher 1000-cycle run: spike, leak, goal instability
  - Reacher 9900-cycle run: `spike_rate = 0.2204`, `rss_late_slope_bytes_per_cycle = 52203`, `drive_switch_rate = 0.664`

Interpretation:
- The continuous MPC path is functioning; this is not a “wrong mechanism” issue.
- The core issue is stability under long live runs: leak pressure, frequent drive switching, and spiky behavior remain even while prediction error improves.

### 5. Observatory/report interpretation gaps
Observed in:
- `docs/observability.md`
- `docs/limitations.md`
- replay/check outputs above

Symptoms:
- missing video on some sessions despite structurally valid data
- live-only fields are absent in replay by design
- large-session review uses decimation, which can hide rare moments unless scrubbed near them
- multi-agent sessions are interleaved and agent-local, which can confuse casual reading of session lengths and counts

Interpretation:
- Some “display problems” are genuine Observatory limitations or contract misunderstandings, not system defects.
- They still need to be tracked because they can mislead diagnosis, but they should not be mixed into core-cognition bugs.

## Root-Cause Analysis
### Finding A: GridWorld fallback dominance is architectural, not incidental
Evidence:
- `docs/limitations.md` states that when an extrinsic goal is present, task lock routes discrete GridWorld action selection through an observed-greedy controller when confidence is high.
- `python/phca/core/cycle.py` explicitly does this:
  - task-lock gate at `cycle.py` around the `TASK_LOCK_CONFIDENCE_THRESHOLD`
  - immediate return with `decision_reason="greedy_fallback"` and `greedy_fallback=True`
  - `_select_greedy_grid_action()` can even route through BFS on larger grids

Why this matters:
- Reports showing `greedy_fallback = 100%` are not lying.
- In GridWorld, especially with extrinsic goals, “bad-looking” repetitive action patterns are often the fallback planner operating as designed.

Consequence:
- This is not just a UI issue.
- It is a product-level mismatch between what the dashboard suggests PHCA is doing and what the discrete GridWorld selector is actually allowed to do.

### Finding B: Long-run GridWorld pathologies are concentrated in `gprime_learn`
Evidence:
- `logs/sessions/20260707_104025/session_report.json`
  - `gprime_learn = 71.34%` of module time share
  - `phase_budget_pct.Learn = 75.52`
  - `interrupt_cycle_count = 491`
  - `violation_cycle_count = 491`
  - anomaly flags for `spike` and `leak`

Cross-check:
- `logs/nightly_stress.json` passes globally at 1000 cycles with `rss_late_slope_bytes_per_cycle = 1982.46`
- the long GridWorld session at 4000 cycles fails anomaly checks, which is consistent with pressure emerging outside the short nightly window

Most likely cause:
- Long-run pressure is accumulating through repeated learning bursts and memory growth rather than through action selection itself.

### Finding C: Multi-agent plus non-MLP can silently degrade predictive quality without changing task-lock behavior
Evidence:
- `logs/sessions/20260707_104736/session_report.json`: MLP on, all agents `error_improved = true`
- `logs/sessions/20260707_105557/session_report.json`: MLP off, all agents `error_improved = false`
- both still show nearly identical goal-reached counts and identical `greedy_fallback = 100%`

Most likely cause:
- the discrete task-lock path is dominating so strongly that the world-model quality difference is hidden at the visible behavior level

Consequence:
- Dashboard success on GridWorld is not a sufficient proxy for predictive-system health in these modes

### Finding D: MuJoCo issues are mostly stability/resource issues, not mechanism-selection issues
Evidence:
- `logs/sessions/20260707_105506/session_report.json`
  - continuous action path dominates
  - error improves strongly
  - anomalies remain active
- `logs/sessions/20260707_102218/session_report.json`
  - very high leak slope
  - very high drive-switch rate
  - still continuous path and improved error

Cross-check:
- `logs/benchmark_recovery.json` shows cognitive resilience scenarios recovering successfully
- nightly MuJoCo gate artifacts exist and pass in the current CI slice

Most likely cause:
- the continuous policy path itself is functioning
- the remaining issues cluster around long-run resource behavior and unstable high-level goal dynamics, not the basic action mechanism

### Finding E: Some suspected UI defects are actually contract limitations
Evidence:
- `docs/observability.md` and `docs/limitations.md`
- replay/check warnings for missing video do not invalidate JSONL/report
- replay intentionally lacks live-only camera/rollout/full-memory fields

Consequence:
- any later fix plan must separate:
  - core-system defects
  - report/anomaly-threshold defects
  - dashboard contract limitations
  - user-expectation / documentation issues

## Cross-Artifact Consistency
### What agrees across artifacts
- Continuous MuJoCo path is real and explainable:
  - session reports show `continuous_mpc`
  - explain metrics are present
  - benchmark and nightly MuJoCo artifacts pass
- L4 continual learning remains a true failure:
  - `logs/l4_ablation.json` shows `forgetting_rate = 1.0` in all tested ablations
- Resilience injectables are not the current bottleneck:
  - `logs/benchmark_recovery.json` shows recovery rate `1.0`

### What disagrees across artifacts
- Short nightly stress passes, but longer live runs can still fail leak/spike anomaly checks
- GridWorld can look successful in goal-reaching terms while being architecturally fallback-dominated and therefore not prediction-primary

## Ranked Findings
### 1. High priority: Discrete GridWorld is dominated by task-lock fallback, masking predictive behavior
- Severity: high
- Confidence: high
- Type: core cognition / policy-path design
- Evidence:
  - `logs/sessions/20260707_104025/session_report.json`
  - `logs/sessions/20260707_104736/session_report.json`
  - `logs/sessions/20260707_105557/session_report.json`
  - `docs/limitations.md`
  - `python/phca/core/cycle.py`
- Why high:
  - it distorts interpretation of most GridWorld live runs
  - it can hide world-model degradation

### 2. High priority: Long-run learning/resource pressure is producing spikes, interrupts, and leak failures
- Severity: high
- Confidence: high
- Type: cross-cutting performance/resource issue
- Evidence:
  - `logs/sessions/20260707_104025/session_report.json`
  - `logs/sessions/20260707_102218/session_report.json`
  - `logs/nightly_stress.json`
  - `python/phca/monitoring/session_anomalies.py`

### 3. Medium-high priority: MuJoCo long runs remain unstable at the anomaly layer despite good error improvement
- Severity: medium-high
- Confidence: high
- Type: core runtime stability / regulation / motivation issue
- Evidence:
  - `logs/sessions/20260707_105506/session_report.json`
  - `logs/sessions/20260707_102218/session_report.json`

### 4. Medium priority: Multi-agent and non-MLP modes can hide degraded predictive quality
- Severity: medium
- Confidence: medium-high
- Type: evaluation and observability interpretation issue with a real model-health component
- Evidence:
  - `logs/sessions/20260707_104736/session_report.json`
  - `logs/sessions/20260707_105557/session_report.json`

### 5. Medium priority: Observatory limitations can be mistaken for defects
- Severity: medium
- Confidence: high
- Type: observability/replay contract issue
- Evidence:
  - `docs/observability.md`
  - `docs/limitations.md`
  - replay outputs for valid sessions

### 6. Low priority for the current fix phase: user-closed incomplete sessions
- Severity: low
- Confidence: high
- Type: forensic-only evidence handling
- Evidence:
  - `logs/sessions/20260707_092235/meta.json`
  - `logs/sessions/20260707_092235/session_report.json`

## Fix-Readiness Brief
### Actionable now
- discrete GridWorld fallback dominance and visibility mismatch
- long-run `gprime_learn` / leak / spike pressure
- MuJoCo long-run goal-instability and leak pressure
- separation of trustworthy vs forensic evidence in Observatory workflows

### Actionable, but requires careful scoping
- multi-agent interpretation fixes
- anomaly/report threshold tuning versus true runtime fixes

### Do not treat as a bug by default
- missing replay camera/full rollouts/full M3/M4 lists
- sessions without video when JSONL/report still validate
- GridWorld not being prediction-primary in extrinsic-goal task-lock modes, unless the desired product behavior is to remove or weaken that fallback path

## English Follow-up Prompt
Use this in a later Plan-mode turn to request a fix plan only.

```text
I have completed the investigation phase and now need a fix-planning phase only.

Context:
- The PHCA system was investigated across Observatory sessions, structured logs, benchmark/nightly JSON artifacts, and test/CI evidence.
- Use the investigation findings already available in the conversation, especially the ranked findings, evidence inventory, trustworthiness ledger, root-cause analysis, and fix-readiness brief.
- Do not repeat the broad investigation. Assume the findings are already available and valid unless you need to reference them for sequencing.

Your task:
1. Build a precise implementation plan to fix the confirmed issues.
2. Group fixes by subsystem and dependency order.
3. Separate quick/local fixes from cross-cutting or architectural fixes.
4. For each fix, identify the likely files, tests, benchmarks, and validation commands that should be updated.
5. Call out risky fixes, fallback options, and any issues that should not be fixed yet.
6. Keep investigation work and implementation work separate: this plan is for fixing only, not for rediscovering problems.

Constraints:
- Do not implement anything yet.
- Produce a detailed but actionable plan.
- Prioritize by user impact, correctness risk, and verification cost.
```
