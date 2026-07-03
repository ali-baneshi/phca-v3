# PHCA v3.0 — Developer Setup Guide

## Prerequisites

| Tool | Version | Check |
| :--- | :--- | :--- |
| Python | 3.11+ | `python --version` |
| Git | 2.40+ | `git --version` |
| Make | — | `make --version` |

> **Rust toolchain:** not required. The Rust workspace was removed in
> D-084 (empty crates; Python is not a bottleneck at ~10–17 ms mean MLP cycle
> latency). Re-introduce Rust only if profiling shows Python as a bottleneck.

## Quick Setup (5 minutes)

```bash
# 1. Clone
git clone https://github.com/your-org/phca-v3.git
cd phca-v3

# 2. Install Python dependencies
python3 -m venv .venv
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
make test-all         # Core + monitoring (524 tests); MuJoCo files ignored
make test-mujoco      # MuJoCo integration only (36 tests); needs gymnasium[mujoco]
make lint             # Linters (ruff)
make profile-cycle    # Profile cognitive cycle latency
```

**Headless Observatory tests** (PyQt offscreen, 225 tests):

```bash
mkdir -p .tmp
TMPDIR=.tmp QT_QPA_PLATFORM=offscreen PYTHONPATH=python \
  python -m pytest python/phca/monitoring/tests/ -q
```

## Cognitive Observatory (Phase 8)

```bash
# Live run
QT_QPA_PLATFORM=offscreen PYTHONPATH=python python scripts/phca_observatory.py --cycles=50 --mlp

# Replay with scrub (canonical Phase 8 path)
PYTHONPATH=python python scripts/phca_replay.py logs/sessions/<ts>/ --qt

# Session integrity + offline report
PYTHONPATH=python python scripts/phca_replay.py --check logs/sessions/<ts>/
PYTHONPATH=python python scripts/phca_replay.py logs/sessions/<ts>/ --report
```

See [docs/observability.md](docs/observability.md) for playback/scrub semantics and transport controls.

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
