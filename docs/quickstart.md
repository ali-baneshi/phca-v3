# Quickstart — Get PHCA Running in 5 Minutes

**PHCA v3.0** is a resource-bounded cognitive architecture for continual learning
and intrinsic motivation. This guide gets you from zero to a running benchmark
in about five minutes.

> **Canonical setup guide:** For full developer setup (venv, Observatory, Wayland
> troubleshooting), see [SETUP.md](../SETUP.md).

---

## Prerequisites

| Tool | Minimum Version | Check |
|---|---|---|
| Python | 3.11 or 3.12 (CI-pinned) | `python --version` |
| Git | 2.40+ | `git --version` |
| Make | any | `make --version` |

Rust is **not** required (removed D-084). Python is not the latency bottleneck
at ~10–17 ms mean MLP cycle time.

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/ali-baneshi/phca-v3.git
cd phca-v3

# 2. Install dependencies (creates venv via Makefile)
make setup
source .venv/bin/activate

# 3. Optional: MuJoCo (Cartpole, Pendulum, Reacher)
pip install -r requirements-mujoco.txt

# 4. Verify everything works
MUJOCO_GL=disabled make test-all
MUJOCO_GL=disabled make test-mujoco
```

Expected output: **707** core tests + **36** MuJoCo tests = **743** total (see
[STATUS.md](../STATUS.md) for current counts).

---

## Run a Simple Example

Create a file `demo.py`:

```python
from phca.core.cycle import CognitiveCycle

cycle = CognitiveCycle.build_for_env(size=5, seed=42, use_mlp=True)

for i in range(100):
    metrics = cycle.step()
    if i % 20 == 0:
        print(
            f"Cycle {i:3d}: latency={metrics.latency_ms:.0f}ms, "
            f"error={metrics.prediction_error:.3f}, "
            f"action={metrics.action_name}"
        )

print(f"\nDone. Final accuracy: {cycle.tspl.skill_accuracy:.3f}")
```

Run it:

```bash
source .venv/bin/activate
PYTHONPATH=python python demo.py
```

Each cognitive cycle observes the environment, predicts the next state, selects
an action, and learns from the prediction error.

---

## Run a Benchmark

```bash
# Quick smoke test (Level 0 only, 20 cycles, Gaussian G')
python scripts/benchmark.py --quick

# Canonical benchmark (Levels 0–3, MLP, 200 cycles, 5×5 — full pass criteria)
MUJOCO_GL=disabled python scripts/benchmark.py --use-mlp --cycles=200 --grid-size 5

# Multi-seed report (mean ± std)
python scripts/benchmark.py --use-mlp --cycles=200 --seeds=5
```

See [phi_iq_metric.md](phi_iq_metric.md) for interpretation. MLP mode needs at
least 200 cycles per level to stabilise.

---

## Cognitive Observatory (Live Dashboard)

The canonical monitor is the **Cognitive Observatory** (PyQt), not a terminal
curses dashboard:

```bash
# Live run (headless CI-style)
QT_QPA_PLATFORM=offscreen PYTHONPATH=python python scripts/phca_observatory.py --cycles=50 --mlp

# Replay a recorded session
PYTHONPATH=python python scripts/phca_replay.py logs/sessions/<timestamp>/ --qt
```

See [observability.md](observability.md) for JSONL schema, replay/scrub, and
integrity checks.

---

## Troubleshooting

| Problem | Solution |
|---|---|
| `ModuleNotFoundError: No module named 'phca'` | Run from repo root: `python scripts/<script>.py` (scripts bootstrap `python/` automatically). Legacy: `PYTHONPATH=python python scripts/...` |
| Benchmark Overall FAIL on 10×10 (`failure_rate_under_10pct`) | Expected on scaling configs. Fail column `1.26` = ~126% violations/cycle, not 12.6%. Use canonical `python scripts/benchmark.py --use-mlp --cycles=200 --grid-size 5` for full pass. See [phi_iq_metric.md](phi_iq_metric.md). |
| `ModuleNotFoundError: No module named 'structlog'` | `pip install -r requirements.txt` |
| `ModuleNotFoundError: No module named 'gymnasium'` | Optional — install `requirements-mujoco.txt` |
| `pytest: error: unrecognized arguments: --timeout` | `pip install -r requirements-dev.txt` |
| MuJoCo display errors | `export MUJOCO_GL=disabled` |
| `make: command not found` | `sudo apt install build-essential` (Linux) |

---

## Next Steps

- [DOCUMENTATION_MAP.md](../DOCUMENTATION_MAP.md) — which docs are living vs historical
- [Architecture Overview](architecture.md) — 12-step cognitive cycle
- [Φ-IQ Metric](phi_iq_metric.md) — benchmark levels and pass criteria
- [MuJoCo Integration](mujoco_integration.md) — physics environments
- [Limitations](limitations.md) — what PHCA cannot do
- [Contributing Guide](../CONTRIBUTING.md) — PR workflow and gates
