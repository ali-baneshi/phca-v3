# PHCA v3.0 — Developer Setup Guide

## Prerequisites

| Tool | Version | Check |
| :--- | :--- | :--- |
| Python | 3.11+ | `python --version` |
| Git | 2.40+ | `git --version` |
| Make | — | `make --version` |

> **Rust toolchain:** not required. The Rust workspace was removed in
> D-084 (empty crates, Python is not a bottleneck at ~50 ms p95 cycle
> latency). Re-introduce Rust only if profiling shows Python as a
> bottleneck.

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

# 3. Verify everything works
make test-all

# 4. Open the project
code .
```

## Running Tests

```bash
make test-all         # All Python tests (MuJoCo auto-skipped without gymnasium)
make test-python      # Python tests only
make lint             # Linters (ruff + black)
make profile-cycle    # Profile cognitive cycle latency
```

## Project Layout

```
phca-v3/
├── Makefile              # Build automation
├── requirements.txt      # Python dependencies
├── python/
│   ├── phca/             # Core implementation (incl. environments/)
│   │   └── environments/ # GridWorld + MuJoCo + EnvironmentProtocol
│   ├── benchmarks/       # Φ-IQ benchmark suite *
│   └── tests/            # Integration + acceptance tests
│
*Note: Use `python scripts/benchmark.py` (not `python -m phca.benchmarks.runner`).
└── docs/                 # Specifications + playbook
```

## Troubleshooting

| Problem | Solution |
| :--- | :--- |
| `ModuleNotFoundError: No module named 'structlog'` | `pip install structlog` or the code falls back to stdlib logging automatically |
| `pytest: error: unrecognized arguments: --timeout` | Install `pytest-timeout`: `pip install pytest-timeout` |
| `fixture 'mocker' not found` in `test_engine.py` | Install `pytest-mock`: `pip install pytest-mock` (tracked as TC-6 — not yet in `requirements-dev.txt`) |
