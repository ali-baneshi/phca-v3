# PHCA v3.0 — Developer Setup Guide

## Prerequisites

| Tool | Version | Check |
| :--- | :--- | :--- |
| Python | 3.11+ | `python --version` |
| Rust | 1.75+ | `rustc --version` |
| Git | 2.40+ | `git --version` |
| Make | — | `make --version` |

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

# 3. Build Rust workspace
cd rust && cargo build --release && cd ..

# 4. Verify everything works
make test-all

# 5. Open the project
code .
```

## Running Tests

```bash
make test-all         # All Python + Rust tests
make test-python      # Python tests only
make test-rust        # Rust tests only
make lint             # Linters (ruff + clippy)
make profile-cycle    # Profile cognitive cycle latency
```

## Project Layout

```
phca-v3/
├── Makefile              # Build automation
├── requirements.txt      # Python dependencies
├── Cargo.toml            # Rust workspace
├── python/
│   ├── phca/             # Core implementation
│   ├── environments/     # Simulated environments
│   ├── benchmarks/       # Φ-IQ benchmark suite
│   └── tests/            # Integration + acceptance tests
├── rust/
│   ├── common/           # Shared data types
│   ├── rpta/             # RBTA Constraint Enforcer
│   └── hpm-runtime/      # HPM Grammar Runtime
└── docs/                 # Specifications + playbook
```

## Troubleshooting

| Problem | Solution |
| :--- | :--- |
| `ModuleNotFoundError: No module named 'structlog'` | `pip install structlog` or the code falls back to stdlib logging automatically |
| `error: the configured Python interpreter version (3.14) is newer than PyO3's maximum supported version` | Run `export PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1` |
| `pytest: error: unrecognized arguments: --timeout` | Install `pytest-timeout`: `pip install pytest-timeout` |
