# Benchmark Artifacts Index

The `logs/` directory contains many JSON files from iterative development.
**Only the files below are canonical** for current claims. Others are historical
experiments — do not cite without checking dates and config blocks.

---

## Canonical Artifacts

| File | Purpose | Updated when |
|---|---|---|
| `reproduce_manifest.json` | One-command reproduction step manifest (Phase 19) | When gates or profiles change |
| `logs/reproduce_report.json` | Reproduce driver audit trail (per-step PASS/FAIL) | `make reproduce` / `make reproduce-quick` |
| `logs/benchmark_report.json` | **Primary Φ-IQ report** (MLP, 200 cyc, L0–L3) | Manual canonical run |
| `logs/benchmark_ci_baseline.json` | CI regression floor (L0 quick, Gaussian) | Intentionally pinned; bump via DECISIONS |
| `logs/phca_causal_eval.json` | Causal gate L1–L3 (200 cyc × 5 seeds) | After causal eval changes |
| `logs/nightly_static.json` | Nightly MLP Φ-IQ snapshot | `make nightly` step 1 |
| `logs/nightly_stress.json` | Long-run RSS/latency/Φ proxy | `make nightly` step 5 |
| `logs/nightly_ood.json` | OOD σ-sweep confidence | `make nightly` step 4 |
| `logs/nightly_assumptions.json` | A1/A3/A4/A5 falsification | `make nightly` step 3 |
| `logs/nightly_mujoco_*.json` | Per-env MuJoCo smoke | `make nightly-mujoco` |
| `logs/phca_causal_eval_nightly.json` | Nightly causal L2+L3 gate | `make nightly` step 6 |
| `logs/benchmark_multiseed.json` | Multi-seed Φ-IQ variance | `--seeds N` benchmark runs |

---

## Session Recordings (Observatory)

```
logs/sessions/<YYYYMMDD_HHMMSS>/
├── meta.json
├── timeseries.jsonl      # schema_version=1
└── session_report.json   # when verify/report enabled
```

Integrity check:

```bash
PYTHONPATH=python python scripts/phca_replay.py --check logs/sessions/<ts>/
```

---

## Historical / Non-Canonical

Files matching these patterns are **development snapshots**:

- `logs/benchmark_phase*.json`
- `logs/phase5_*.json`, `logs/phase6_*.json`
- `logs/benchmark_post_gap.json`, `logs/benchmark_l0.json` (unless regenerated for CI)

When in doubt, re-run:

```bash
PYTHONPATH=python python scripts/benchmark.py --use-mlp --cycles=200 \
  --output=logs/benchmark_report.json
```

---

## Gate Scripts

| Script | Compares |
|---|---|
| `scripts/reproduce.py` | Manifest-driven suite; writes `logs/reproduce_report.json` |
| `scripts/check_benchmark_gate.py` | Report vs baseline (5% tolerance) |
| `scripts/check_benchmark_gate.py --mujoco` | MuJoCo violations + error trend |
| `scripts/phca_causal_eval.py --gate` | PHCA vs controls (75% metric rule) |

---

## Related

- [reproducibility.md](reproducibility.md)
- [phi_iq_metric.md](phi_iq_metric.md)
- [STATUS.md](../STATUS.md)
