# Cognitive Resilience (Failure Detection & Recovery)

PHCA distinguishes **two recovery layers**:

| Layer | Package | What it recovers |
|---|---|---|
| **Observatory session recovery** | `phca/monitoring/session_recovery.py` | JSONL, meta, reports after Qt/cycle crash |
| **Cognitive resilience** | `phca/resilience/` | In-cycle failure detection + bounded mitigation |

This document covers cognitive resilience only. For crash/session recovery see
[observability.md](observability.md) Phase 16.

---

## E1 — FallbackController

`python/phca/resilience/fallback_controller.py` provides a **dual-signal fail-closed**
layer between ASI sanitization and action selection. It accepts two independent
trigger signals:

1. **Entropy-band overlap** — MDIM belief entropy exceeding a threshold bypasses
   G' prediction and falls back to a safe action distribution.
2. **Failure cascade** — `FailureDetector.detect()` calls `notify_failure()` on
   the controller; `FailureEvent` objects are tracked with a decay factor so that
   transient glitches do not trigger full fail-closed while persistent degradation
   does.

When either signal fires, `FallbackController.get_action()` returns
`env.neutral_action()` instead of the normal action-selection path.

```
cycle._build_resilience_snapshot() -> FailureDetector.detect()
  -> notify_failure() -> FallbackController.register_failure()
  -> controller.get_action() during action selection
```

---

## NoiseInjector

`python/phca/asi/noise_injector.py` adds configurable additive Gaussian noise to
observations for robustness stress testing. Used by
`scripts/benchmark_noise_closedloop.py` to measure Φ-IQ degradation under
controlled input corruption.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `noise_std` | 0.0 | Standard deviation of additive Gaussian noise |
| `noise_decay` | 0.999 | Per-cycle multiplicative decay |
| `warmup_cycles` | 50 | Cycles before noise starts |

`scripts/benchmark_noise_closedloop.py` runs a noise ramp (0 → 0.7 → 0) and
reports Φ-IQ at each level; `scripts/benchmark_noise_robustness.py` provides
a multi-seed noise-sweep variant.

---

## MVP coverage (whitepaper §4.1 subset)

| Mode | Detection | Recovery protocol |
|---|---|---|
| **B1** Distribution shift | `prediction_error > 3×` rolling median for 3+ cycles | TSPL η × 2 |
| **B4** Catastrophic forgetting | Per-task goal rate drop > 2× baseline | Replay boost + α × 0.5 |
| **B5** Mode collapse | Confidence > 0.99 and < 2 unique actions / 20 cycles | MDIM temperature × 1.5 |
| **C1** Feedback instability | RBTA `TERMINATE` for 3+ consecutive cycles | PID `Kp × 0.5` + integral hold |
| **F5** Consolidation failure | M3 > 90% full + stagnant M4 facts 50+ cycles | Force consolidation |

Detector budget: < 1 ms per cycle. Recovery window: 10 cycles; mitigated after
3 stable cycles within threshold.

---

## Integration

Each cognitive cycle (`cycle.py` step 14+):

1. `_build_resilience_snapshot()` collects signals
2. `FailureDetector.detect()` returns `FailureEvent` list
3. `RecoveryManager.apply()` runs protocols
4. `CycleMetrics.failure_events` and `recovery_active` are populated

---

## Benchmarks

```bash
# Injectable scenarios (B1, C1, F5)
PYTHONPATH=python python scripts/benchmark_recovery.py \
  --scenarios b1,c1,f5 --output logs/benchmark_recovery.json
```

Target: `recovery_rate >= 0.80` (fraction of scenarios mitigated within 10 cycles).

Forgetting rate (separate workstream):

```bash
PYTHONPATH=python python scripts/benchmark_level4.py \
  --tasks 10 --output logs/benchmark_level4.json
```

---

## Tests

```bash
PYTHONPATH=python python -m pytest \
  python/tests/test_resilience.py python/tests/test_forgetting.py -q
```
