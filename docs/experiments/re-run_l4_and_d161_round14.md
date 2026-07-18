# Round 14 re-run results — L4 forgetting (TASK 1) and D-161 reconciliation (TASK 2)

Run: 2026-07-18 on HEAD 15790d8, mucio_GL=disabled, all 30 seeds

---

## TASK 1 — Level-4-lite continual-learning benchmark

### Command
```
python scripts/benchmark_level4.py \
    --tasks 10 --task-cycles 80 --seeds 30 --m3-replay-budget 16 \
    --use-mlp --grid-size 5
```

### Result

| Metric | Pre-D-168 (3 seeds, from README.md) | Post-D-168 (30 seeds, this run) | Change |
|--------|:---:|:---:|:---:|
| forgetting_rate | 0.00% | **37.83%** | — |
| passes_gate | True | **False** | — |
| m3_replay_total | 23,040 | 23,040 | Unchanged |
| eval_prediction_error | (not recorded) | 2.6–7.0 across tasks | — |

### Per-task breakdown (30-seed aggregate)

| Task | delta_perf | eval_pred_error | Note |
|:---:|:---:|:---:|:---|
| 0 | −0.345 | 5.28 | |
| 1 | −0.123 | 6.21 | |
| 2 | −0.282 | 6.14 | |
| 3 | −0.332 | 4.68 | |
| 4 | −0.222 | 3.20 | Best-retained task |
| 5 | −0.245 | 3.89 | |
| 6 | −0.318 | 7.03 | Worst prediction error |
| 7 | −0.378 | 5.32 | Worst forgetting |
| 8 | −0.257 | 4.57 | |
| 9 | −0.227 | 2.58 | |

### Per-seed distribution

28/30 seeds have forgetting_rate = 1.000 (complete forgetting). 1 seed (742) has 0.150. 1 seed (1542) has 0.000.

### Interpretation

The forgetting rate jumped from 0.00% to 37.83%. The gate now fails. There are two confounded explanations:

1. **Seed-count effect.** The original 3-seed result was underpowered — 3/3 seeds happened to show zero forgetting by chance. The 30-seed distribution (28/1/1) is heavily skewed and would not be captured at 3 seeds.

2. **D-168 PER fix.** D-168 made PER sample from all tasks instead of just the current task. This could increase cross-task interference during replay, which would be expected to *increase* forgetting. However, since the baseline uses pure geometric action selection (which is memoryless), this effect should be indirect (through MLP → MDIM → goal selection → RBTA → goal_rate).

Without a 30-seed pre-D-168 counterfactual, these cannot be separated. The most defensible statement: **the forgetting rate, when measured at an adequate seed count (30) on current HEAD with D-168 active, is 37.83%** — substantially above the 0.00% previously cited, and well past the < 5% gate threshold. This is consistent with the Round 1/F-06 hypothesis that the headline number was dominated by factors other than learned-model retention (seed underpowering, geometric-planner confound, and now the pre-D-168 PER bug), but the present data alone cannot apportion the effect among those three factors.

**eval_prediction_error** is high (2.6–7.0 range across tasks; the MLP predicts next-state with MSE-type errors in this range). No pre-D-168 baseline exists for this metric, so it cannot be used to assess whether the PER fix itself had a detectable effect on model-level retention. This is worth collecting in any future re-run where a comparison would be meaningful.

---

## TASK 2 — D-161 four-configuration reconciliation

### Commands
Four `phca_causal_eval.py --levels level2 --cycles 200 --seeds 30 --use-mlp --gate` runs
with `--grid-size` 5 or 10 × `--enable-blended-scorer` on/off.

### Comparison table

| Config | D-161 goal_rate (2026-07-15) | Current goal_rate (2026-07-18) | Δ | D-161 RBTA | Current RBTA | D-161 Gate | Current Gate |
|--------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 5×5 Pure geometry | 0.552 | **0.552** | **+0.000** | 0.030 | 0.030 | FAIL | FAIL |
| 5×5 Agreement-gated | 0.544 | **0.451** | **−0.094** | 0.032 | 0.025 | FAIL | FAIL |
| 10×10 Pure geometry | 0.199 | **0.199** | **+0.000** | 0.503 | 0.502 | PASS | PASS |
| 10×10 Agreement-gated | 0.134 | **0.139** | **+0.005** | 0.534 | 0.515 | FAIL | FAIL |

### Interpretation

**Pure geometry is stable across the ~25 intervening decisions.** Both 5×5 and 10×10 pure-geometry goal rates are unchanged within rounding error (< 0.001). No sign of D-182's Φ fix or any other change affecting the pure-geometry path.

**Agreement-gated blended at 5×5 dropped 17%** (0.544 → 0.451). This is a meaningful shift. Three likely candidates:
- **D-182's Φ fix** (most plausible): D-182 made `_cached_phi` vary dynamically instead of being frozen. This drives `error_volatility`, which feeds the PID controller (APC) and MDIM D2. The agreement-gated path routes through MDIM goal selection and the G′-scored goal-alignment metric, so an active Φ signal changes the decision landscape. Pure geometry bypasses this path entirely — consistent with it being unaffected.
- **D-172 (MDIM prediction_error timing):** moved the stale one-cycle-lag error to the current-cycle error. This also only affects paths that depend on MDIM drive computation.
- **Seed variance:** possible but less likely given that pure geometry (same seed sequence, same grid, same number of seeds) is stable to < 0.001.

**Agreement-gated at 10×10 is unchanged** (0.134 vs 0.139, Δ = +0.005). The RBTA violation rate at 10×10 is ~50% for both modes, which dominates the signal and likely swamps any Φ-driven or MDIM-driven effect. The insensitivity to D-182 at this scale is consistent with an RBTA-bound-limited regime where action quality is secondary to whether the agent acts at all.

### Verdict on D-161's claims

- **Finding 1 (neither mode wins universally):** Holds unchanged. Pure geometry still fails at 5×5 L2, passes at 10×10 L2.
- **Finding 2 (agreement-gated blended does not close the gap):** Still true, and slightly strengthened — the gap at 5×5 is now larger (0.552 vs 0.451 vs D-161's 0.552 vs 0.544).
- **Decision (pure geometry stays the default):** Unchanged. The agreement-gated path lost ground, not gained it.

---

## Observations (unrelated, noted per instructions)

1. **README.md L4 forgetting number needs updating.** It still cites 0.00% at 3 seeds. The new 30-seed number (37.83%) should replace or supplement it. The caveat paragraphs are still accurate qualitatively but the specific number is now stale.

2. **The `eval_prediction_error` field is present in the code but was missing from the old `benchmark_level4.json`** (the 3-seed run predated its addition alongside D-145). The new 30-seed output now contains it, providing a baseline for future comparisons.

3. **The 5×5 L2 pure-geometry gate failure** (0/3 metrics vs greedy_observed) is persistent across both D-161 and this re-run. D-161 flagged this as worth a separate investigation — still unresolved, and D-161's own note about this remains correct.

4. **No script bugs were encountered** that required fixing to run these experiments. Both `scripts/benchmark_level4.py` and `scripts/phca_causal_eval.py` ran without modification on HEAD 15790d8.
