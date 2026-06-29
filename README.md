# PHCA v3.0 — Predictive Hierarchical Cognitive Architecture

A formally specified, resource-bounded cognitive architecture for continual learning, intrinsic motivation, and self-regulated autonomous agents.

## Quick Start

```bash
make setup      # Install dependencies
make test-all   # Run all tests
make bench-level-0  # Run Level 0 benchmark
```

## Documentation

- `docs/10-implementation-blueprint.md` — Strategic implementation plan
- `docs/11-engineers-playbook.md` — Day-zero execution plan (tickets, checklists, schedule)
- `docs/09-phca-v3-patch.md` — Formal v3.0 specification

## Project Structure

```
python/phca/           # Core Python implementation
python/environments/   # Simulated environments (grid-world, MuJoCo)
python/benchmarks/     # Φ-IQ benchmark suite
python/tests/          # Integration + acceptance tests
rust/                  # Rust components (RBTA, HPM runtime)
```

## License

Internal research project. All rights reserved.
