# PHCA v3.0 — Developer Setup Guide

## Prerequisites

| Tool | Version | Check |
| :--- | :--- | :--- |
| Python | 3.11 or 3.12 (recommended) | `python --version` |
| Git | 2.40+ | `git --version` |
| Make | — | `make --version` |

> **Rust toolchain:** not required. The Rust workspace was removed in
> D-084 (empty crates; Python is not a bottleneck at ~10–17 ms mean MLP cycle
> latency). Re-introduce Rust only if profiling shows Python as a bottleneck.

> **Python 3.14+:** may work locally but is not CI-pinned. Use a venv with
> `requirements.txt` (see below) to avoid drift (e.g. unpinned `pgmpy` 1.1.x
> requiring numpy 2.x while PHCA pins numpy 1.26.4 — see D-114).

## Quick Setup (5 minutes)

```bash
# 1. Clone
git clone https://github.com/your-org/phca-v3.git
cd phca-v3

# 2. Create a venv (Python 3.11 or 3.12 recommended)
python3.11 -m venv .venv   # or python3.12
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# 3. Optional: MuJoCo (Cartpole, Pendulum, Reacher)
pip install -r requirements-mujoco.txt

# 4. Verify everything works
make test-all
MUJOCO_GL=disabled make test-mujoco

# 5. Open the project
code .
```

## Running Tests

```bash
make test-all         # Core + monitoring (587 tests); MuJoCo files ignored
make test-mujoco      # MuJoCo integration only (36 tests); needs gymnasium[mujoco]
make lint             # Linters (ruff)
make profile-cycle    # Profile cognitive cycle latency
```

**Headless Observatory tests** (PyQt offscreen, **232 tests** — see [STATUS.md](../STATUS.md)):

```bash
mkdir -p .tmp
TMPDIR=.tmp QT_QPA_PLATFORM=offscreen PYTHONPATH=python \
  python -m pytest python/phca/monitoring/tests/ -q
```

## Cognitive Observatory (Phases 8–12)

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

## Benchmarks

Use the canonical CLI (not `python -m phca.benchmarks.runner`):

```bash
MUJOCO_GL=disabled PYTHONPATH=python python scripts/benchmark.py --use-mlp --cycles=200
```

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
