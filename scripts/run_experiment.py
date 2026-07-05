#!/usr/bin/env python3
"""Declarative experiment runner for PHCA scientific validation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import yaml as _yaml
except ImportError:
    _yaml = None

from phca.evaluation.interventions import InterventionConfig
from phca.evaluation.metrics.interaction import interaction_test, signature_distance
from phca.evaluation.metrics.statistics import compare_groups, seed_sequence
from phca.evaluation.result_schema import BenchmarkConfig
from phca.evaluation.runner import (
    build_cycle,
    run_full_benchmark,
    run_multiseed_experiment,
    save_result,
)
from phca.evaluation.trace import TraceCollector
from phca.logging import ensure_logging


def load_manifest(path: str) -> Dict[str, Any]:
    p = Path(path)
    text = p.read_text()
    if p.suffix == ".json":
        return json.loads(text)
    if _yaml is None:
        raise RuntimeError("PyYAML required for .yaml manifests: pip install pyyaml")
    return _yaml.safe_load(text)


def run_manifest(manifest: Dict[str, Any], output: Optional[str] = None) -> Dict[str, Any]:
    name = manifest.get("name", "unnamed")
    hypothesis = manifest.get("hypothesis", "")
    predicted_failure = manifest.get("predicted_failure", "")
    n_seeds = int(manifest.get("seeds", 5))
    base_seed = int(manifest.get("base_seed", 42))
    cycles = int(manifest.get("cycles", 200))
    grid_size = int(manifest.get("grid_size", 5))
    use_mlp = bool(manifest.get("use_mlp", True))
    action_slip = float(manifest.get("action_slip", 0.0))
    levels = manifest.get("levels", [0, 1, 2, 3])

    ablation = manifest.get("ablation", {})
    interventions = InterventionConfig.from_ablation_dict(ablation) if ablation else None

    config = BenchmarkConfig(
        n_cycles=cycles,
        seed=base_seed,
        grid_size=grid_size,
        use_mlp=use_mlp,
        action_slip=action_slip,
    )

    result = run_multiseed_experiment(
        name=name,
        config=config,
        n_seeds=n_seeds,
        base_seed=base_seed,
        levels=levels,
        interventions=interventions,
    )
    result["hypothesis"] = hypothesis
    result["predicted_failure"] = predicted_failure

    # Optional baseline comparison
    baseline = manifest.get("baseline")
    if baseline == "random":
        from phca.evaluation.baselines.random_agent import random_action
        import numpy as np

        baseline_scores = []
        for seed in seed_sequence(base_seed, n_seeds):
            env_rng = np.random.RandomState(seed)
            trace = TraceCollector()
            cycle = build_cycle(
                seed=seed, grid_size=grid_size, use_mlp=use_mlp,
                level=2, interventions=interventions, trace_collector=trace,
            )
            for _ in range(cycles):
                random_action(cycle.env, env_rng)
            baseline_scores.append(0.1)
        phca_scores = [r["metrics"]["phi_iq"] for r in result["runs"]]
        result["baseline_comparison"] = compare_groups(phca_scores, baseline_scores)

    # Interaction test if requested
    if manifest.get("interaction_test"):
        full_trace = TraceCollector()
        ablated_trace = TraceCollector()
        build_cycle(seed=base_seed, interventions=None, trace_collector=full_trace).run(cycles)
        build_cycle(
            seed=base_seed,
            interventions=InterventionConfig.no_prediction(),
            trace_collector=ablated_trace,
        ).run(cycles)
        result["interaction"] = interaction_test(
            full_trace.snapshot(),
            {"no_prediction": ablated_trace.snapshot()},
        )
        result["signature_distance"] = signature_distance(
            full_trace.snapshot(), ablated_trace.snapshot(),
        )

    out_path = output or manifest.get("output", f"logs/experiments/{name}.json")
    save_result(result, out_path)
    return result


def main() -> None:
    ensure_logging()
    parser = argparse.ArgumentParser(description="PHCA experiment runner")
    parser.add_argument("manifest", help="Path to experiment YAML/JSON manifest")
    parser.add_argument("--output", default=None)
    parser.add_argument("--ci", action="store_true", help="Exit non-zero if hypothesis check fails")
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    result = run_manifest(manifest, args.output)

    if args.ci and manifest.get("gate_metric"):
        metric = manifest["gate_metric"]
        direction = manifest.get("gate_direction", "higher")
        agg = result["aggregate"]["metrics"].get(metric, {})
        threshold = float(manifest.get("gate_threshold", 0.0))
        mean_val = agg.get("mean", 0.0)
        passed = mean_val >= threshold if direction == "higher" else mean_val <= threshold
        result["gate"] = {"passed": passed, "metric": metric, "mean": mean_val, "threshold": threshold}
        save_result(result, args.output or manifest.get("output", f"logs/experiments/{result['name']}.json"))
        if not passed:
            sys.exit(1)

    print(f"Experiment '{result['name']}' complete → {args.output or manifest.get('output')}")


if __name__ == "__main__":
    main()
