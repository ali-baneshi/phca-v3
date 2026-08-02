# Causal Diagnosis — G2-INV-05 (2026-08-02)

**Stance:** Diagnose with ablations only. Do **not** flip `disable_blended_scorer` default. No capability claim.

**Driver:** `scripts/diagnose_causal_ablation.py`  
**Artifacts:**
- `logs/diagnosis_causal_g2inv05_5x5_l2.json` (primary)
- `logs/diagnosis_causal_g2inv05_10x10_l2.json` (smoke)

**Budget:** 5 seeds × 100 cycles × MLP, shared seeds across conditions A/B/C. Underpowered vs D-151’s 30-seed causal gate — diagnostic only.

## Conditions

| ID | Config |
|----|--------|
| A | Default geometry (`disable_blended_scorer=True`) |
| B | Geometry + `enable_gprime_learn=False` + `enable_tspl=False` |
| C | Blended opt-in (`disable_blended_scorer=False`, warmup=0) |

## Results (5×5 L2)

| Cond | Gate | goal_rate PHCA | greedy_observed | Losers | geo_dom / modes |
|------|------|----------------|-----------------|--------|-----------------|
| A | PASS | 0.644 | 0.642 | `first_goal_cycle` only | 1.0 / 100% pure_geometry |
| B | PASS | 0.644 (=A) | 0.642 | same as A | 1.0 / 100% pure_geometry |
| C | FAIL | 0.388 | 0.642 | all 4 metrics | 0.8; ~48% prediction_scored |

Exact A≡B on goal_rate, distance, reward, first_goal_cycle.

## Results (10×10 L2 smoke)

| Cond | Gate | goal_rate PHCA | greedy | Notes |
|------|------|----------------|--------|-------|
| A | FAIL | 0.288 | 0.288 | Identical to greedy on all four metrics (geometry ≡ planner baseline) |
| B | FAIL | 0.288 (=A) | 0.288 | Identical to A |
| C | FAIL | 0.374 | 0.288 | Goal_rate and reward **better** than greedy; loses first_goal_cycle + mean_distance |

## Hypotheses

### H1 — Losses are on distance/reward not goal_rate → scenario metric design

**Partially accepted (budget-dependent).**

- At **5×5 / n=5**: A loses only `first_goal_cycle` (tie-or-worse timing); goal_rate/distance/reward beat or match greedy → **non-goal-only** pattern.
- At **10×10 / n=5**: A ties greedy on every metric (gate FAIL on strict “not worse” semantics) → not a pure distance/reward confound; geometry is the planner.
- **Do not claim** historical 30-seed 5×5 L2 FAIL is “fixed”; this PASS is underpowered.

### H2 — Learn-off ≈ learn-on → G′ irrelevant under default → geometry confound

**Accepted.**

- A vs B `abs_delta(goal_rate)=0` on both grids; full metric vectors match on 5×5 and 10×10.
- Under default geometry, disabling G′/TSPL learn does not change causal L2 scores at this protocol → **planner-dominated / G′ off the action path**.

### H3 — Blended changes gate → prediction path matters when enabled

**Accepted (direction grid-dependent; no default flip).**

- **5×5:** Blended **hurts** badly (goal_rate 0.388 vs 0.644); gate flips PASS→FAIL; prediction path is live (~48% `prediction_scored`).
- **10×10:** Gate stays FAIL, but blended **raises** goal_rate (0.374 vs 0.288) and reward while worsening distance/first_goal — mixed, consistent with D-161 “blended not a free win.”
- Prediction path matters when enabled; evidence does **not** support flipping the default.

## Falsifiable conclusions

1. Default discrete causal L2 under geometry is a **geometry/planner comparison**, not a G′-learning comparison (H2).
2. Opt-in blended is an active differentiator (H3) but **not** a safe default on 5×5 at this budget.
3. Metric-only narrative (H1) is weak: at 10×10 geometry simply **is** greedy_observed; any remaining “mixed L2/L3” behavioral gap is not cured by learn toggles.

## Suggested follow-ons (backlog only; not this track)

| If | Then |
|----|------|
| H2 (confirmed) | Keep dual PE / secondary PE reporting; do not sell geometry PASS as prediction competence |
| H3 mixed | Optional **30-seed** blended opt-in experiment PR only if product wants that comparison; no default flip |
| H1 partial | Gate-control / metric-weight docs revision only if product redefines “beat greedy” semantics |

## Non-claims

- No claim that 5×5 L2 is solved at production seed budget.
- No default action-selection change.
- No discrete G′ expansion / blueprint P3.
