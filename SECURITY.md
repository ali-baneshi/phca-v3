# Security & Research Disclaimer

PHCA v3.0 is **internal research software**, not a production product.

## Scope

- Experimental cognitive architecture for bounded autonomous agents
- Known bugs, incomplete features, and environment-specific benchmarks
- See [docs/limitations.md](docs/limitations.md) and [STATUS.md](STATUS.md)

## Reporting issues

Open a GitHub issue with:

- Steps to reproduce (commands, env flags)
- Expected vs actual behavior
- Relevant gate output (`make test-python`, `check_benchmark_gate.py`)

**Do not** attach:

- Local Observatory sessions (`logs/sessions/`)
- Runtime logs (`logs/phca.log*`)
- Cursor debug logs (`.cursor/`)
- Temp/pytest artifacts (`.tmp/`, `.pytest_tmp/`)
- Credentials, API keys, or private paths

## Secrets

Never commit `.env`, private keys, or machine-specific session data. The
repository `.gitignore` blocks common local artifacts; verify with
`git status` before pushing.

## Dependencies

Install only from pinned `requirements.txt` in a virtual environment.
Report supply-chain concerns via issue; do not paste secrets in reports.
