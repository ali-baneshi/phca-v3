#!/usr/bin/env python3
"""Declarative experiment runner for PHCA scientific validation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import yaml as _yaml
except ImportError:
    _yaml = None

from phca.evaluation.interventions import InterventionConfig
from phca.evaluation.metrics.interaction import interaction_test, signature_distance
from phca.evaluation.metrics.statistics import aggregate_runs, compare_groups, seed_sequence
from phca.evaluation.result_schema import BenchmarkConfig
from phca.evaluation.runner import (
    build_cycle,
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


def run_manifest(
    manifest: Dict[str, Any],
    output: Optional[str] = None,
    *,
    n_seeds: Optional[int] = None,
    base_seed: Optional[int] = None,
) -> Dict[str, Any]:
    name = manifest.get("name", "unnamed")
    hypothesis = manifest.get("hypothesis", "")
    predicted_failure = manifest.get("predicted_failure", "")
    seeds = int(n_seeds if n_seeds is not None else manifest.get("seeds", 5))
    base = int(base_seed if base_seed is not None else manifest.get("base_seed", 42))
    cycles = int(manifest.get("cycles", 200))
    grid_size = int(manifest.get("grid_size", 5))
    use_mlp = bool(manifest.get("use_mlp", True))
    action_slip = float(manifest.get("action_slip", 0.0))
    levels = manifest.get("levels", [0, 1, 2, 3])
    environment = manifest.get("environment", "gridworld")
    mujoco_env = manifest.get("mujoco_env", "Pendulum-v1")
    causal_fair = bool(manifest.get("causal_fair", False))

    ablation = manifest.get("ablation", {})
    if ablation or causal_fair:
        interventions = InterventionConfig.from_ablation_dict(ablation or {}, causal_fair=causal_fair)
    else:
        interventions = None

    config = BenchmarkConfig(
        n_cycles=cycles,
        seed=base,
        grid_size=grid_size,
        use_mlp=use_mlp,
        action_slip=action_slip,
        dynamic_goals=bool(manifest.get("dynamic_goals", False)),
        dynamic_goals_every=int(manifest.get("dynamic_goals_every", 100)),
        environment=environment,
        mujoco_env=mujoco_env,
    )

    result = run_multiseed_experiment(
        name=name,
        config=config,
        n_seeds=seeds,
        base_seed=base,
        levels=levels,
        interventions=interventions,
    )
    result["hypothesis"] = hypothesis
    result["predicted_failure"] = predicted_failure
    result["config"] = {
        "cycles": cycles,
        "grid_size": grid_size,
        "use_mlp": use_mlp,
        "action_slip": action_slip,
        "environment": environment,
        "causal_fair": causal_fair,
        "levels": levels,
    }

    if manifest.get("interaction_test"):
        per_seed_interaction = []
        per_seed_distance = []
        for seed in seed_sequence(base, seeds):
            full_trace = TraceCollector()
            ablated_trace = TraceCollector()
            for level in levels:
                cycle_full = build_cycle(
                    seed=seed + level,
                    use_mlp=use_mlp,
                    grid_size=grid_size,
                    level=level,
                    trace_collector=full_trace,
                )
                for _ in range(cycles):
                    cycle_full.step()
                cycle_ab = build_cycle(
                    seed=seed + level,
                    use_mlp=use_mlp,
                    grid_size=grid_size,
                    level=level,
                    interventions=InterventionConfig.no_prediction(),
                    trace_collector=ablated_trace,
                )
                for _ in range(cycles):
                    cycle_ab.step()
            per_seed_interaction.append(
                interaction_test(
                    full_trace.snapshot(),
                    {"no_prediction": ablated_trace.snapshot()},
                )
            )
            per_seed_distance.append(
                signature_distance(full_trace.snapshot(), ablated_trace.snapshot())
            )
        keys = list(per_seed_interaction[0].keys()) if per_seed_interaction else []
        interaction_agg: Dict[str, float] = {}
        for key in keys:
            vals = [float(row[key]) for row in per_seed_interaction]
            interaction_agg[key] = float(sum(vals) / len(vals)) if vals else 0.0
        result["interaction"] = interaction_agg
        result["interaction_per_seed"] = per_seed_interaction
        result["signature_distance"] = float(
            sum(per_seed_distance) / len(per_seed_distance)
        ) if per_seed_distance else 0.0

    out_path = output or manifest.get("output", f"results/validation/ablations/{name}.json")
    save_result(result, out_path)
    return result


def main() -> None:
    ensure_logging()
    parser = argparse.ArgumentParser(description="PHCA experiment runner")
    parser.add_argument("manifest", help="Path to experiment YAML/JSON manifest")
    parser.add_argument("--output", default=None)
    parser.add_argument("--seeds", type=int, default=None)
    parser.add_argument("--base-seed", type=int, default=None)
    parser.add_argument("--ci", action="store_true")
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    result = run_manifest(
        manifest, args.output,
        n_seeds=args.seeds, base_seed=args.base_seed,
    )

    if args.ci and manifest.get("gate_metric"):
        metric = manifest["gate_metric"]
        direction = manifest.get("gate_direction", "higher")
        agg = result["aggregate"]["metrics"].get(metric, {})
        threshold = float(manifest.get("gate_threshold", 0.0))
        mean_val = agg.get("mean", 0.0)
        passed = mean_val >= threshold if direction == "higher" else mean_val <= threshold
        result["gate"] = {"passed": passed, "metric": metric, "mean": mean_val, "threshold": threshold}
        save_result(result, args.output or manifest.get("output"))
        if not passed:
            sys.exit(1)

    print(f"Experiment '{result['name']}' complete → {args.output or manifest.get('output')}")


if __name__ == "__main__":
    main()
