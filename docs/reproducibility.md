# Reproducibility Guide

How to reproduce PHCA benchmark numbers on a clean machine. For artifact paths
see [benchmark_artifacts.md](benchmark_artifacts.md).

---

## Environment

| Requirement | Value |
|---|---|
| Python | **3.11** (CI-pinned; 3.12 works locally) |
| OS | Linux x86_64 tested (GitHub Actions ubuntu-latest) |
| GPU | Not required |
| MuJoCo | Optional; `MUJOCO_GL=disabled` for headless |

```bash
make setup
source .venv/bin/activate
pip install -r requirements-mujoco.txt   # optional
```

---

## Canonical Commands

### Full test suite

```bash
MUJOCO_GL=disabled make test-all
MUJOCO_GL=disabled make test-mujoco
```

Expected: **524** core + **36** MuJoCo = **560** total ([STATUS.md](../STATUS.md)).

### Φ-IQ benchmark (canonical)

```bash
MUJOCO_GL=disabled PYTHONPATH=python python scripts/benchmark.py \
  --use-mlp --cycles=200 --output=logs/benchmark_report.json
```

Defaults: seed **42**, levels **0–1–2–3**, 5×5 GridWorld.

### Multi-seed Φ-IQ (variance report)

```bash
PYTHONPATH=python python scripts/benchmark.py \
  --use-mlp --cycles=200 --seeds=5 --output=logs/benchmark_multiseed.json
```

Runs seeds 42…46; JSON includes `multi_seed` block with mean ± std.

### CI-equivalent local check

```bash
make ci-local
```

Runs lint, pytest (timeout 30s), L0 quick gate, causal L2 smoke.

### Full nightly hardening

```bash
make nightly NIGHTLY_CYCLES=10000
```

~3 min for 10k stress; verifies post-cap retention gate (D-112).

---

## Expected Results (reference machine, 2026-07-04)

| Artifact | Key metric | Expected |
|---|---|---|
| `logs/benchmark_report.json` | overall_phi_iq | ~0.74 (±0.01 run variance) |
| `logs/benchmark_ci_baseline.json` | L0 quick floor | gate ≥ 0.5486 (5% tolerance) |
| `logs/phca_causal_eval.json` | L1–L3 gate | PASS vs gated controls |
| `logs/nightly_stress.json` | fill-phase slope @ 1k | ≤ 5000 B/cyc |

Re-measure on your machine before citing numbers externally (D-090 zero-trust policy).

---

## Seeds and Determinism

- Benchmark default seed: **42** (per-level offset: `seed + level`)
- Causal eval: **5 seeds** by default (`--seeds 5`)
- Tests use fixed seeds; stress tests marked `@pytest.mark.slow`

Dynamic-goal L2 (`--dynamic-goals-every 75`) is **machine-sensitive** (SGD trajectory).

---

## Scheduled CI

GitHub Actions runs lint, full pytest, and L0 Φ-IQ gate on every push/PR.

A **scheduled nightly workflow** (`.github/workflows/nightly.yml`) runs
`make nightly NIGHTLY_CYCLES=10000` daily. Failures do not block PR merges but
should be triaged via STATUS.md.

---

## Troubleshooting

| Issue | Fix |
|---|---|
| Φ-IQ below floor on quick gate | Ensure Gaussian quick mode; check `benchmark_ci_baseline.json` |
| MuJoCo import errors | `pip install -r requirements-mujoco.txt`; `MUJOCO_GL=disabled` |
| Observatory tests fail | `QT_QPA_PLATFORM=offscreen TMPDIR=.tmp` |
| Retention gate fails @ 10k | Expected during M3 fill; see D-112 phase-aware thresholds |

---

## Related

- [benchmark_artifacts.md](benchmark_artifacts.md)
- [phi_iq_metric.md](phi_iq_metric.md)
- [IMPLEMENTATION_STATUS.md](../IMPLEMENTATION_STATUS.md)
