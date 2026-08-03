# PHCA v3.0 — Documentation Map

This map tells you **which documents to trust** for what purpose, and which
are historical or aspirational. When documents disagree, prefer the tier order
below.

**Trust order (D-198):** named `logs/` artifacts → [`DECISIONS.md`](DECISIONS.md)
(through D-198+) → [`README.md`](README.md) / [`IMPLEMENTATION_STATUS.md`](IMPLEMENTATION_STATUS.md)
→ [`docs/investigations/`](docs/investigations/) → demoted subsidiary docs.
Never cite [`STATUS.md`](STATUS.md) for post-D-137 gates.

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
| [DECISIONS.md](DECISIONS.md) | Design decision log (authoritative; through D-198+) |
| [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md) | Whitepaper criteria × code × gates matrix |
| [docs/investigations/](docs/investigations/) | 2026-08 gap register, overnight analysis, hardening backlog |
| [SETUP.md](SETUP.md) | Developer setup, Observatory commands |
| [CONTRIBUTING.md](CONTRIBUTING.md) | PR workflow, lint/test gates |
| [docs/limitations.md](docs/limitations.md) | Honest capability boundaries |
| [docs/phi_iq_metric.md](docs/phi_iq_metric.md) | Φ-IQ definition, levels, interpretation caveats |
| [docs/phca_causal_evidence.md](docs/phca_causal_evidence.md) | Causal behavior gate |
| [docs/observability.md](docs/observability.md) | Cognitive Observatory JSONL, replay, integrity |
| [docs/reproducibility.md](docs/reproducibility.md) | How to reproduce benchmark numbers |
| [docs/benchmark_artifacts.md](docs/benchmark_artifacts.md) | Canonical log files index (incl. overnight) |
| [docs/action_selection.md](docs/action_selection.md) | Discrete vs continuous action paths |
| [docs/quickstart.md](docs/quickstart.md) | 5-minute getting started (links to SETUP) |

Architecture notes live under Historical/demoted with a drift banner:
[docs/architecture.md](docs/architecture.md) — prefer README A4 row + DECISIONS D-156/D-158/D-197.

---

## Historical Documents (phase snapshots)

| Document | Purpose |
|---|---|
| [STATUS.md](STATUS.md) | **Frozen 2026-07-08 (through D-137)** — historical only; not gate SoT |
| [docs/archive/](docs/archive/) | Phase completion / gap-closure reports (see archive README) |
| [docs/observatory_phase_prompts/](docs/observatory_phase_prompts/) | Observatory Phases 13–20 prompts (archived) |
| [docs/l4_root_cause_verdict.md](docs/l4_root_cause_verdict.md) | Superseded D-137-era L4 notes (bannered) |
| [docs/maturity_audit_2026-07-07.md](docs/maturity_audit_2026-07-07.md) | Dated maturity audit (bannered) |

**Do not cite benchmark numbers from historical docs without cross-checking named
`logs/` files and DECISIONS (through D-198+).** Never use STATUS.md to validate
overnight / causal / L4 gates.

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

1. README.md → limitations.md → IMPLEMENTATION_STATUS.md → docs/investigations/executive_summary

### Contributor (30 min)

1. SETUP.md → CONTRIBUTING.md → DECISIONS.md (recent D-194+) → docs/investigations/ → IMPLEMENTATION_STATUS.md

### Researcher evaluating claims (1 hr)

1. phca_causal_evidence.md → overnight_analysis (investigations) → phi_iq_metric.md → reproducibility.md → DECISIONS.md (D-151, D-156/D-159, D-197)

### Systems reviewer

1. action_selection.md → observability.md → benchmark_artifacts.md → assumption_validation scripts

---

## Canonical Numbers

Cite **named log files** first ([docs/benchmark_artifacts.md](docs/benchmark_artifacts.md)),
then DECISIONS / IMPLEMENTATION_STATUS. Prefer current overnight SoT when present:

- Causal / diagnosis / L4 overnight: `logs/overnight_20260802_103502/` (D-197)
- L4 30-seed: `logs/benchmark_level4_30s.json` — `forgetting_rate≈0.3783`, FAIL
- Diagnosis ablations: `logs/diagnosis_causal_g2inv05_*.json`
- Φ-IQ MLP historical (pre-/not re-certified under geometry default): overall ~0.73 in older `logs/benchmark_report.json` — treat as historical; L2 GC is planner-contaminated under default
- Test counts: run `make test-all` / `pytest` for current totals (do not hardcode from STATUS)
