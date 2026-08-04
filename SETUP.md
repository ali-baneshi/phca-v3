# PHCA v3.0 — Developer Setup Guide

## Prerequisites

| Tool | Version | Check |
| :--- | :--- | :--- |
| Python | 3.11 or 3.12 (supported) | `python --version` |
| Git | 2.40+ | `git --version` |
| Make | — | `make --version` |

> **Rust toolchain:** not required. The Rust workspace was removed in
> D-084 (empty crates; Python is not a bottleneck at ~10–17 ms mean MLP cycle
> latency). Re-introduce Rust only if profiling shows Python as a bottleneck.

> **Python 3.13+:** is not currently supported. Package metadata rejects these
> versions because the pinned scientific stack and timing gates are certified
> only on Python 3.11 and 3.12.

## Quick Setup (5 minutes)

```bash
# 1. Clone
git clone https://github.com/ali-baneshi/phca-v3.git
cd phca-v3

# 2. Create the supported venv and install the package/dependencies
PYTHON=python3.11 make setup   # or ensure python3 resolves to Python 3.12
source .venv/bin/activate

# 3. Optional: MuJoCo (Cartpole, Pendulum, Reacher)
make install-mujoco-deps

# 4. Verify everything works
make test-all
MUJOCO_GL=disabled make test-mujoco

# 5. Open the project
code .
```

## Running Tests

```bash
make test-all         # Core + monitoring (current count via pytest); MuJoCo files ignored
make test-mujoco      # MuJoCo integration only (36 tests); needs gymnasium[mujoco]
make lint             # Linters (ruff)
make profile-cycle    # Profile cognitive cycle latency
```

**Headless Observatory tests** (PyQt offscreen — run `pytest python/phca/monitoring/tests/` for count):

```bash
mkdir -p .tmp
TMPDIR=.tmp QT_QPA_PLATFORM=offscreen PYTHONPATH=python \
  python -m pytest python/phca/monitoring/tests/ -q
```

## Cognitive Observatory (Phases 7–20)

```bash
# Live run
QT_QPA_PLATFORM=offscreen PYTHONPATH=python python scripts/phca_observatory.py --cycles=50 --mlp

# Replay with scrub (canonical Observatory path)
PYTHONPATH=python python scripts/phca_replay.py logs/sessions/<ts>/ --qt

# Session integrity + offline report
PYTHONPATH=python python scripts/phca_replay.py --check logs/sessions/<ts>/
PYTHONPATH=python python scripts/phca_replay.py logs/sessions/<ts>/ --report
```

See [docs/observability.md](docs/observability.md) for playback/scrub semantics and transport controls.

## Observatory troubleshooting

### `Failed to load plugin 'libdecor-gtk.so'`

Harmless on many **Wayland** desktops (KDE, GNOME). Qt tries to load GTK window
decorations; the run still succeeds (`Done. N cycles; N JSONL lines`).

Fix (pick one):

```bash
# Arch / Manjaro — install the GTK decoration plugin
sudo pacman -S libdecor-gtk

# Or force XWayland / X11 for Qt
QT_QPA_PLATFORM=xcb PYTHONPATH=python python scripts/phca_observatory.py --cycles=50 --mlp
```

### `pgmpy.estimators.StructureScore is deprecated`

With `--mlp`, PHCA no longer imports pgmpy on startup. If you still see this,
you are likely running outside the project venv with a global `pgmpy` pulled in
by another import path. Recreate the venv from `requirements.txt`.

### Session integrity after a live run

Post-run verify and `session_report.json` run **by default** when recording completes.
Use `--no-verify` to skip.

```bash
PYTHONPATH=python python scripts/phca_replay.py --check logs/sessions/<ts>/
```

The Observatory prints a structured `=== Session summary ===` block with verify
status and report path; a one-line `Session OK:` precedes it when recording health matches.

**By default the window stays open** after the run so you can scrub all tabs and
read the Overview session-results panel. Close the window when done, or pass
`--close-at-end` for CI/headless auto-exit.

Optional comparison context:

```bash
PYTHONPATH=python python scripts/phca_observatory.py \
  --env reacher --cycles=100 --mlp --camera live \
  --compare-report logs/sessions/<prev>/session_report.json
```

Φ-IQ benchmark context (read-only from `logs/benchmark_report.json` by default;
use `--no-benchmark-display` to hide):

```bash
PYTHONPATH=python python scripts/phca_observatory.py \
  --env reacher --cycles=100 --mlp --benchmark-report logs/benchmark_report.json
```

### Reacher live run (Manjaro / Wayland)

```bash
# Quiet Qt decoration noise on KDE/Wayland:
QT_QPA_PLATFORM=xcb PYTHONPATH=python python scripts/phca_observatory.py \
  --env reacher --cycles=3000 --mlp --camera live

# Default (may print harmless libdecor-gtk plugin noise):
PYTHONPATH=python python scripts/phca_observatory.py \
  --env reacher --cycles=3000 --mlp --camera live

# After run (automatic unless --no-verify):
ls logs/sessions/<ts>/session_report.json
PYTHONPATH=python python scripts/phca_replay.py --check logs/sessions/<ts>/
```

### Pendulum live camera troubleshooting (Manjaro / X11/Wayland)

```bash
# Recommended live camera run with startup diagnostics:
QT_QPA_PLATFORM=xcb MUJOCO_GL=glx PYTHONPATH=python \
  python scripts/phca_observatory.py \
  --env pendulum --cycles=10000 --mlp --camera live --profile near_real \
  --record-video --camera-debug --camera-selftest-attempts 5 --camera-diag-samples 8

# Qt-in-context probe (quickly classify backend vs app issue):
QT_QPA_PLATFORM=xcb MUJOCO_GL=glx PYTHONPATH=python \
  python scripts/camera_probe_qt.py --env pendulum --samples 6
```

Expected startup logs include:
- `MUJOCO_GL` and `QT_QPA_PLATFORM`
- self-test result (`PASS/FAIL`, attempt count, reason)
- startup frame diagnostics (`std`, `green_frac`, `glitchy`, `shape`)
- explicit fallback reason + probe command if live capture is disabled

If the Overview still shows `Camera unavailable`, inspect `/tmp/phca_obs_numpy_*.png`
and `/tmp/phca_obs_pixmap_*.png` to determine whether failure is at capture
or pixmap conversion.

## Benchmarks

Use the canonical CLI (not `python -m phca.benchmarks.runner`):

```bash
# Full pass criteria (5×5 MLP, 200 cycles)
MUJOCO_GL=disabled python scripts/benchmark.py --use-mlp --cycles=200 --grid-size 5

# Quick smoke (L0, 20 cycles, Gaussian)
python scripts/benchmark.py --quick
```

Scripts bootstrap `python/` automatically; `PYTHONPATH=python` is optional.
For 10×10 scaling runs, see [docs/phi_iq_metric.md](docs/phi_iq_metric.md) — the
violation gate (`failure_rate_under_10pct`) is calibrated for canonical 5×5.

## Project Layout

```
phca-v3/
├── Makefile              # Build automation
├── requirements.txt      # Python dependencies
├── requirements-mujoco.txt  # Optional MuJoCo stack
├── python/
│   ├── phca/             # Core implementation (incl. environments/, monitoring/)
│   │   └── environments/ # GridWorld + MuJoCo + EnvironmentProtocol
│   ├── benchmarks/       # Legacy runner (prefer scripts/benchmark.py)
│   └── tests/            # Integration + acceptance tests
├── scripts/
│   ├── benchmark.py      # Canonical Φ-IQ benchmark
│   ├── phca_observatory.py  # PyQt Cognitive Observatory
│   └── phca_replay.py    # Session replay + --check
└── docs/                 # Specifications + operational docs
```

## Related

- [README.md](README.md) — overview and quick start
- [CONTRIBUTING.md](CONTRIBUTING.md) — PR workflow and standards
- [docs/observability.md](docs/observability.md) — Cognitive Observatory setup and replay
- [docs/limitations.md](docs/limitations.md) — what PHCA cannot do
