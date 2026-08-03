# Reproducibility Guide

How to reproduce PHCA benchmark numbers on a clean machine. For artifact paths
see [benchmark_artifacts.md](benchmark_artifacts.md).

---

## One-command reproduction (Phase 19)

The canonical entry point for scientific validation on a clean machine:

```bash
make setup
source .venv/bin/activate
pip install -r requirements-mujoco.txt   # required for full profile
make reproduce-quick    # ~10–15 min: lint, pytest, quick Φ-IQ gate, scientific gates
make reproduce          # ~45–90 min: full nightly-equivalent artifact suite
```

| Command | Profile | What it runs |
|---|---|---|
| `make reproduce-quick` | `quick` | `ci-local` subset + assumption `--ci`, OOD calibration, anomaly gate |
| `make reproduce` | `full` | Canonical Φ-IQ + nightly MuJoCo/stress/causal gates (10k stress soak) |

**Manifest:** [`reproduce_manifest.json`](../reproduce_manifest.json) — step list, output paths, gate floors, runtime estimates.

**Report:** `logs/reproduce_report.json` — per-step PASS/FAIL, durations, stdout tail.

**Dry-run** (no execution):

```bash
python scripts/reproduce.py --dry-run --profile quick
```

**Environment pins** (in manifest): Python **3.11**, `MUJOCO_GL=disabled`. Gate floors use existing CI/nightly scripts only — no tightened Φ-IQ thresholds (D-122).

The quick profile runs pytest; the full profile does not (matches `make nightly` scope).

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

Expected: run `pytest` / `make test-all` for current totals (+ optional MuJoCo suite). Do not cite frozen STATUS.md counts.

### Two benchmark execution paths

| Path | Entry | Emergence / synergy |
|------|-------|---------------------|
| Scaling steps | `scripts/benchmark.py` | No — Φ-IQ sub-metrics only |
| Ablations / OOD | `scripts/run_experiment.py` → `evaluation/runner.py` | Yes — trace-based |

Smoke profile (`--profile smoke`) uses 3 seeds and is **CI sanity only**; hypothesis
verdicts (e.g. H003 synergy) may differ from full 30-seed runs.

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

### Scientific validation suite (`run_validation_suite.py`)

Full multi-seed ablation, scaling, OOD, and Φ-IQ validation (~3 h on reference machine):

```bash
make validate-science
# equivalent:
MUJOCO_GL=disabled PYTHONPATH=python python scripts/run_validation_suite.py \
  --profile full --base-seed 42 --output-root results/validation --resume
MUJOCO_GL=disabled PYTHONPATH=python python scripts/aggregate_validation.py \
  --input results/validation \
  --hypotheses experiments/hypotheses.yaml \
  --output results/validation/summary.json
```

| Option | Meaning |
|--------|---------|
| `--profile smoke` | 3 seeds, 1k horizon (~15 min smoke) |
| `--profile full` | 30 seeds, 100k horizon, 100 Φ-IQ validation seeds |
| `--resume` | Skip steps listed in `results/validation/manifest_index.json` |
| `--output-root` | Artifact directory (default `results/validation`) |

**Outputs:** per-experiment JSON under `ablations/`, `scaling/`, `ood/`; rollup
`summary.json`, `hypothesis_verdicts.json`, `failure_log.json`. See
[`results/validation/README.md`](../results/validation/README.md).

**Reference results (2026-07-05):** aggregate Φ-IQ **0.526±0.189**; scaling
5/10/20 overall Φ-IQ **0.700±0.046** / **0.325±0.042** / **0.147±0.042**;
Φ-IQ predictive r **0.992** (H006 Validated).

---

## Expected Results (reference machine, 2026-07-04)

| Artifact | Key metric | Expected |
|---|---|---|
| `logs/benchmark_report.json` | overall_phi_iq | ~0.74 (±0.01) — **historical**; L2 planner-contaminated under geometry default |
| `logs/benchmark_ci_baseline.json` | L0 quick floor | gate ≥ 0.5486 (5% tolerance) |
| `results/validation/baselines/causal_eval_round4.json` | L1–L3 gate (30 seeds MLP) | **L1 PASS, L2 FAIL, L3 FAIL** — see D-151 |
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
should be triaged via DECISIONS.md and [`docs/investigations/`](investigations/) (not STATUS.md).

---

## Troubleshooting

| Issue | Fix |
|---|---|
| Φ-IQ below floor on quick gate | Ensure Gaussian quick mode; check `benchmark_ci_baseline.json` |
| MuJoCo import errors | `pip install -r requirements-mujoco.txt`; `MUJOCO_GL=disabled` |
| Observatory tests fail | `QT_QPA_PLATFORM=offscreen TMPDIR=.tmp` |
| Retention gate fails @ 10k | Expected during M3 fill; see D-112 phase-aware thresholds |

---

## Overnight / diagnosis harness (30-seed power)

```bash
# Multi-hour collection (geometry + blended + A/B/C + L4 + Φ-IQ)
SEEDS=30 CYCLES=200 ./scripts/overnight_diagnosis_harness.sh

# Ablation-only
PYTHONPATH=python python scripts/diagnose_causal_ablation.py \
  --level level2 --cycles 200 --seeds 30 --grid-size 5 --use-mlp
```

Artifacts: `logs/overnight_*/`, `logs/diagnosis_causal_g2inv05_*.json`.  
Analysis: [`investigations/overnight_analysis_2026-08-02.md`](investigations/overnight_analysis_2026-08-02.md).

## Related

- [benchmark_artifacts.md](benchmark_artifacts.md)
- [phi_iq_metric.md](phi_iq_metric.md)
- [IMPLEMENTATION_STATUS.md](../IMPLEMENTATION_STATUS.md)
