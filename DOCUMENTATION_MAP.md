# PHCA v3.0 — Documentation Map

This map tells you **which documents to trust** for what purpose, and which
are historical or aspirational. When documents disagree, prefer the tier order
below.

---

## Document Tiers

| Tier | Meaning | Update policy |
|---|---|---|
| **Living** | Current truth for runtime, benchmarks, and contribution | Update with every gated change |
| **Historical** | Accurate snapshot at a past phase; metrics may be stale | Archive; do not edit except typos |
| **Research** | Phase 1 literature review; no code | Frozen reference; annotate, don't rewrite |
| **Aspirational** | Design targets not fully implemented | Keep with prominent status banners |

---

## Living Documents (start here)

| Document | Purpose |
|---|---|
| [README.md](README.md) | Project overview, quick start, latest benchmark summary |
| [STATUS.md](STATUS.md) | **Source of truth** for test counts, gate outcomes, issue registry |
| [DECISIONS.md](DECISIONS.md) | Design decision log D-001–D-112+ |
| [SETUP.md](SETUP.md) | Developer setup, Observatory commands |
| [CONTRIBUTING.md](CONTRIBUTING.md) | PR workflow, lint/test gates |
| [docs/architecture.md](docs/architecture.md) | 12-step cycle, module map, invariants |
| [docs/limitations.md](docs/limitations.md) | Honest capability boundaries |
| [docs/phi_iq_metric.md](docs/phi_iq_metric.md) | Φ-IQ definition, levels, interpretation |
| [docs/phca_causal_evidence.md](docs/phca_causal_evidence.md) | Causal behavior gate |
| [docs/observability.md](docs/observability.md) | Cognitive Observatory JSONL, replay, integrity |
| [docs/observatory_phase_prompts/](docs/observatory_phase_prompts/) | Copy-paste English prompts for Observatory Phases 13–20 |
| [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md) | Whitepaper criteria × code × gates matrix |
| [docs/reproducibility.md](docs/reproducibility.md) | How to reproduce benchmark numbers |
| [docs/benchmark_artifacts.md](docs/benchmark_artifacts.md) | Canonical log files index |
| [docs/action_selection.md](docs/action_selection.md) | Discrete vs continuous action paths |
| [docs/quickstart.md](docs/quickstart.md) | 5-minute getting started (links to SETUP) |

---

## Historical Documents (phase snapshots)

Located in [docs/archive/](docs/archive/). See [docs/archive/README.md](docs/archive/README.md)
for the index. Examples:

- Phase 3.3 completion reports
- Phase 4–6 sign-off reports
- Gap closure and audit reports from earlier passes

**Do not cite benchmark numbers from these without cross-checking STATUS.md.**

---

## Research Corpus (theory only)

| Path | Purpose |
|---|---|
| [research/index.md](research/index.md) | Phase 1 knowledge base index |
| [research/glossary.md](research/glossary.md) | Cross-domain AI/cog-sci vocabulary |
| [research/outputs/03-axiom-candidates.md](research/outputs/03-axiom-candidates.md) | A1–A5 axiom candidates (feeds invariants) |
| [research/outputs/07-rigorous-whitepaper.md](research/outputs/07-rigorous-whitepaper.md) | Formal specification (**aspirational in parts**) |
| [research/outputs/10-implementation-blueprint.md](research/outputs/10-implementation-blueprint.md) | Engineering blueprint (**mostly superseded**) |

The research tree contains **no executable code**. It informed design but does
not describe the running system verbatim.

---

## Aspirational / Superseded (read with caution)

| Document | Issue |
|---|---|
| [research/outputs/10-implementation-blueprint.md](research/outputs/10-implementation-blueprint.md) | Lists 21 components, Rust, 3-stream TSPL — largely unbuilt |
| [docs/monitoring_plan.md](docs/monitoring_plan.md) | References removed `phca-monitor.py` |
| [docs/monitoring_completion_report.md](docs/monitoring_completion_report.md) | Pre-Observatory era |
| [gate_phase3.3_final.md](gate_phase3.3_final.md) | Phase 3.3 release gate (historical) |

---

## Recommended Reading Order

### First-time visitor (15 min)

1. README.md → limitations.md → IMPLEMENTATION_STATUS.md

### Contributor (30 min)

1. SETUP.md → CONTRIBUTING.md → architecture.md → STATUS.md

### Researcher evaluating claims (1 hr)

1. phi_iq_metric.md → phca_causal_evidence.md → reproducibility.md → DECISIONS.md (D-101, D-112)

### Systems reviewer

1. architecture.md → observability.md → benchmark_artifacts.md → assumption_validation scripts

---

## Canonical Numbers

Always cite from [STATUS.md](STATUS.md) and named log files in
[docs/benchmark_artifacts.md](docs/benchmark_artifacts.md):

- **699 tests** (663 core + 36 MuJoCo)
- **232 monitoring tests** (subset of phca monitoring suite)
- **Overall Φ-IQ 0.7403** (MLP, 200 cyc/level, seed 42)
