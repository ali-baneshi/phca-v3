# Overnight Harness Analysis — 2026-08-02

**Artifacts:** `logs/overnight_20260802_103502/`  
**Git:** `0cf292f`  
**Protocol:** `SEEDS=30 CYCLES=200 base_seed=42`  
**Window:** `2026-08-02T10:35:02` → `11:29:12` (~54 min) — **finished** (not network-killed)  
**Decision:** D-197

Causal `rc=1` is expected `--gate` exit on FAIL; JSON files were written. Φ-IQ stage failed due to harness CLI bug (`--levels=0-2`); fixed later in harness (use `0,1,2`).

## Stage summary

| Stage | rc | Artifact | Notes |
|-------|-----|----------|--------|
| causal 5×5 L2+L3 geometry | 1 | JSON OK | L2+L3 FAIL |
| causal 10×10 L2+L3 geometry | 1 | JSON OK | L2 PASS, L3 FAIL |
| causal 5×5 L2 blended | 1 | JSON OK | FAIL vs greedy |
| causal 10×10 L2 blended | 1 | JSON OK | FAIL vs greedy |
| diagnosis 5×5 / 10×10 A/B/C | 0 | JSON OK | primary H1–H3 at power |
| L4 30s | 0 | JSON OK | FR≈37.83% FAIL |
| Φ-IQ L0–L2 | 1 then fixed | `phi_iq_l0l2.json` (short re-run 5×50) | overnight CLI used invalid `0-2`; harness fixed to `0,1,2`; verify re-run PASS with geometry interpretation note |

## Causal geometry (30×200)

| Grid | Level | Gate | goal_rate PHCA | greedy | Losers vs greedy | PE (secondary) | geo_dom |
|------|-------|------|----------------|--------|------------------|----------------|---------|
| 5×5 | L2 | FAIL | 0.552 | 0.598 | goal, dist, reward | 3.44 | 1.0 |
| 5×5 | L3 | FAIL | 0.308 | 0.291 | first_goal, coverage | 5.59 | 1.0 |
| 10×10 | L2 | **PASS** | 0.199 | 0.187 | first_goal only | 5.11 | 1.0 |
| 10×10 | L3 | FAIL | 0.131 | 0.137 | goal, dist, reward | 8.18 | 1.0 |

Matches D-151/D-161: 5×5 L2 FAIL, 10×10 L2 PASS, L3 FAIL. Selector ~100% `pure_geometry_ablation` — scenario gate = planner competence, not prediction-primary.

## Blended opt-in L2 (30×200) — H3 at power

| Grid | Gate | goal_rate PHCA | greedy | vs geometry |
|------|------|----------------|--------|-------------|
| 5×5 | FAIL | 0.458 | 0.598 | worse than geo 0.552 |
| 10×10 | FAIL | 0.139 | 0.187 | worse than geo 0.199 |

**Verdict:** At 30-seed power, blended **hurts** L2 on both grids. Do **not** flip default. Earlier short-budget 10×10 “mixed” goal_rate bump does not hold here.

## Ablation A/B/C — H1–H3

### 5×5 L2

| Cond | Gate | goal_rate | PE |
|------|------|-----------|-----|
| A geometry | FAIL | 0.552 | **3.44** |
| B learn-off | FAIL | **0.552 (=A)** | **23.43** |
| C blended | FAIL | 0.453 | 3.83 |

- **H2 accepted (hard):** scenario metrics A≡B under geometry.
- **Secondary PE:** learn-off PE ~3.4 → ~23.4 while scenario scores unchanged — dual-report working.
- **H3:** blended lowers goal_rate (~−0.10); gate stays FAIL.
- **H1 rejected at power:** loses goal_rate (not distance/reward-only).

### 10×10 L2

| Cond | Gate | goal_rate | PE | Losers |
|------|------|-----------|-----|--------|
| A | PASS | 0.199 | 5.11 | first_goal only |
| B | PASS | 0.199 (=A) | 24.11 | same |
| C | FAIL | 0.203 | 5.79 | first_goal + distance |

- **H2 accepted** again (identical goal_rate; PE diverges).
- **H3:** gate flips PASS→FAIL under blended (distance/timing), despite similar goal_rate.
- **H1 partial** only for geometry 10×10 (non-goal loser).

## L4 (30 seeds)

| Field | Value |
|-------|--------|
| `forgetting_rate` | **0.3783** (matches historical 37.83%) |
| `mean_per_seed_forgetting_rate` | 0.938 |
| `median_per_seed_forgetting_rate` | 1.0 |
| `mean_eval_prediction_error` | 4.77 |
| `valid_task_coverage` | 0.57 |
| `vacuous_pass_blocked` | false |
| `passes_gate` | **false** |

G4-INV-03 residual remains: FR fails; dual PE present; not vacuous PASS.

## Conclusions

1. Overnight data is usable and mostly complete.
2. **G2-INV-05** still open behaviorally; honesty + diagnosis validated at 30-seed.
3. **No blended default flip** — blended L2 worse at power on 5×5 and 10×10.
4. Strongest signal: **H2 + secondary PE** (same planner scores, large PE gap when learn off).
5. Tooling: harness Φ-IQ levels CLI fixed to `0,1,2` (D-197).

## Artifact index

- `logs/overnight_20260802_103502/causal_5x5_l2l3_geometry.json`
- `logs/overnight_20260802_103502/causal_10x10_l2l3_geometry.json`
- `logs/overnight_20260802_103502/causal_5x5_l2_blended.json`
- `logs/overnight_20260802_103502/causal_10x10_l2_blended.json`
- `logs/overnight_20260802_103502/diagnosis_5x5_l2_abc.json`
- `logs/overnight_20260802_103502/diagnosis_10x10_l2_abc.json`
- `logs/overnight_20260802_103502/level4_30s.json`
- `logs/overnight_20260802_103502/phi_iq_l0l2.json` (short verify re-run after CLI fix)
- `logs/overnight_20260802_103502/manifest.txt`
