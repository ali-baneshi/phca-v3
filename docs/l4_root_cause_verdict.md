# Level-4-lite Root-Cause Verdict — 2026-07-07

Evidence from ablation matrix and diagnostic runs. See [`logs/l4_ablation.json`](../logs/l4_ablation.json).

## Verdict: **B + C (partial capacity limit)**

| Class | Finding |
|-------|---------|
| **B — Unwired (resolved)** | M3 `sample_prior_task_episodes` now wired via `cycle._replay_m3_prior_tasks` → `gprime.learn_m3_episodes`. Observability exports `task_id`, `failure_events`, `recovery_active`. |
| **A — Protocol (partial)** | Max-rolling baseline + invalid baseline exclusion reduce G1 noise; eval warmup alone does not fix collapse on tasks 0/3/7/9. |
| **C — Capacity (open)** | After wiring + R6 combined knobs, 10-task `forgetting_rate` may remain ≥0.05. G′ FIFO replay (500) + P-Stream-only path cannot retain all early layouts under AT-2-lite budget. |

## Ablation results (2026-07-07, 10 tasks × 80 train, 1 seed)

| Run | forgetting_rate | passes_gate |
|-----|-----------------|-------------|
| R0 default | 1.0000 | false |
| R2 max_rolling | 1.0000 | false |
| R3 m3 replay | 1.0000 | false |
| R6 combined | 1.0000 | false |

Protocol + wiring improvements do not close L4b under current P-Stream + 500-slot G′ FIFO architecture → **Verdict C confirmed**.

| Run | Knob | Expected if A | Expected if B | Expected if C |
|-----|------|---------------|---------------|---------------|
| R0 | Legacy last-window, no M3 budget | — | baseline fail | — |
| R2 | Max-rolling baseline | Δ stable | — | — |
| R3 | M3 replay budget 4 | — | replay_total > 0 | may still fail |
| R6 | Combined warmup + replay + interleaved | best protocol | best wiring | still fail → C |

**Action taken:** Protocol hardened in `benchmark_level4.py`; M3 replay wired; diagnostic mode added.

**Honest gate status:** L4b (10×80, 3 seeds, <5%) remains **T3 local FAIL** until measured PASS or documented limitation in [`limitations.md`](limitations.md).

## B4 integration note

B4 detector fires when `record_task_eval` history shows >2× baseline drop. Interleaved eval (`--interleaved-eval`) enables mid-sequence B4 + `on_forgetting_detected` during training, not only final eval pass.

## Do not

- Report `transfer_efficiency` as forgetting (G4).
- Treat injectable `recovery_rate=1.0` as whitepaper §1.3 full matrix (G3).
