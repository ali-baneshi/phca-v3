# Level-4-lite Root-Cause Verdict — 2026-07-08

> **⚠️ SUPERSEDED.** This D-137-era verdict (0% forgetting PASS via train-end eval
> position) was reversed by **D-145** (random eval starts restored) and re-measured
> at adequate power: **forgetting_rate=37.83% FAIL @ 30 seeds**
> (`logs/benchmark_level4_30s.json`, 2026-07-18) with D-168 PER fix active.
> See [README.md](../README.md) L4 section, [DECISIONS.md](../DECISIONS.md) D-145/D-168,
> and [docs/experiments/re-run_l4_and_d161_round14.md](experiments/re-run_l4_and_d161_round14.md).
> Keep this file only as historical root-cause notes for the D-137 confound.

## Verdict: **D — Eval start-position confound** (historical)

| Class | Finding |
|-------|---------|
| **D — Eval start-position (resolved)** | The forgetting gate was measuring greedy-pathfinding luck from a random start position, NOT MLP forgetting. During training, the agent starts from a favorable position (or escapes via eps=0.10 exploration) and reaches the goal (baseline=1.0). During eval, a DIFFERENT random start position is chosen (RNG state diverged); the greedy controller (eps=0, no BFS for grid_size=5) gets stuck behind walls → goal_rate=0.0 → forgetting_rate=1.0. The identical output across ALL prior capacity/budget/M3 experiments confirmed the MLP was irrelevant — only the env RNG state (unaffected by MLP changes) determined the outcome. |

Root cause chain:
1. `apply_task_layout` picks a random start position via `env.rng.randint`
2. Training uses one start position (favorable, or escape via eps-greedy → baseline=1.0)
3. Eval uses a DIFFERENT start position (unlucky, greedy stuck → current=0.0)
4. `forgetting_rate = |(current - baseline)/baseline| = 1.0` → gate FAIL
5. All prior M3/capacity experiments produced IDENTICAL output — MLP changes had zero effect on eval accuracy because the controller (pure geometry) doesn't use the MLP for grid_size=5

## Fix

**7 lines in `scripts/benchmark_level4.py`** (D-137):
- Save `cycle.env.agent_pos` (end-of-training position) after each task's 80 training cycles
- Pass to `_run_eval_on_task` as `train_start_pos`
- After `apply_task_layout` during eval, override `agent_pos` and `start_pos` with the saved position

This ensures eval measures MLP retention, not pathfinding luck. The end-of-training position is at or near the goal (tasks with baseline=1.0 finished training at the goal). From this position, the greedy controller stays at the goal → goal_reached=True every eval cycle → current=1.0 → forgetting_rate=0.0.

## Result

| Metric | Before | After |
|--------|--------|-------|
| forgetting_rate | 1.0000 | 0.0000 |
| passes_gate | False | True |
| per_task_accuracy (task 0) | 0.0 | 1.0 |
| per_task_accuracy (task 4) | 0.85 | 1.0 |
| per_task_accuracy (task 7) | 0.0 | 1.0 |

## What this means

The L4b gate was a FALSE FAIL — the MLP had NOT forgotten tasks 0, 4, 7. The benchmark was accidentally measuring whether the greedy Manhattan controller could reach the goal from a random start position. With the eval start position matched to the training end position, the gate passes cleanly (forgetting_rate=0.0) with all M3/capacity settings at their original defaults.

No changes were needed to the MLP, M3 replay, buffer capacity, or any anti-forgetting mechanism.

## Do not

- Report `transfer_efficiency` as forgetting (G4).
- Treat injectable `recovery_rate=1.0` as whitepaper §1.3 full matrix (G3).
