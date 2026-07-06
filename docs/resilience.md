# Cognitive Resilience (Failure Detection & Recovery)

PHCA distinguishes **two recovery layers**:

| Layer | Package | What it recovers |
|---|---|---|
| **Observatory session recovery** | `phca/monitoring/session_recovery.py` | JSONL, meta, reports after Qt/cycle crash |
| **Cognitive resilience** | `phca/resilience/` | In-cycle failure detection + bounded mitigation |

This document covers cognitive resilience only. For crash/session recovery see
[observability.md](observability.md) Phase 16.

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
