#!/usr/bin/env python3
"""Level-4-lite ablation matrix (R0–R6) for maturation Track E."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import _bootstrap  # noqa: F401

_scripts = Path(__file__).resolve().parent
if str(_scripts) not in sys.path:
    sys.path.insert(0, str(_scripts))
import benchmark_level4  # noqa: E402


ABLATIONS: Dict[str, Dict[str, Any]] = {
    "R0": {
        "label": "default",
        "eval_warmup_cycles": 0,
        "baseline_method": "last_window",
        "m3_replay_budget": 0,
        "interleaved_eval_cycles": 0,
        "mitigation": True,
    },
    "R1": {
        "label": "eval_warmup_10",
        "eval_warmup_cycles": 10,
        "baseline_method": "last_window",
        "m3_replay_budget": 0,
        "interleaved_eval_cycles": 0,
        "mitigation": True,
    },
    "R2": {
        "label": "max_rolling_baseline",
        "eval_warmup_cycles": 0,
        "baseline_method": "max_rolling",
        "m3_replay_budget": 0,
        "interleaved_eval_cycles": 0,
        "mitigation": True,
    },
    "R3": {
        "label": "m3_replay_wired",
        "eval_warmup_cycles": 0,
        "baseline_method": "max_rolling",
        "m3_replay_budget": 4,
        "interleaved_eval_cycles": 0,
        "mitigation": True,
    },
    "R4": {
        "label": "on_task_boundary_only",
        "eval_warmup_cycles": 0,
        "baseline_method": "max_rolling",
        "m3_replay_budget": 4,
        "interleaved_eval_cycles": 0,
        "mitigation": True,
    },
    "R5": {
        "label": "interleaved_eval",
        "eval_warmup_cycles": 0,
        "baseline_method": "max_rolling",
        "m3_replay_budget": 4,
        "interleaved_eval_cycles": 5,
        "interleaved_warmup_cycles": 2,
        "mitigation": True,
    },
    "R6": {
        "label": "combined",
        "eval_warmup_cycles": 10,
        "baseline_method": "max_rolling",
        "m3_replay_budget": 8,
        "interleaved_eval_cycles": 5,
        "interleaved_warmup_cycles": 2,
        "mitigation": True,
    },
}


def run_ablation(
    *,
    runs: List[str],
    n_tasks: int,
    task_cycles: int,
    eval_cycles: int,
    seeds: int,
) -> Dict[str, Any]:
    results: Dict[str, Any] = {}
    for run_id in runs:
        cfg = ABLATIONS[run_id]
        report = benchmark_level4.run_level4_benchmark(
            n_tasks=n_tasks,
            task_cycles=task_cycles,
            eval_cycles=eval_cycles,
            seeds=seeds,
            use_mlp=True,
            grid_size=5,
            mitigation=cfg.get("mitigation", True),
            baseline_method=cfg.get("baseline_method", "max_rolling"),
            eval_warmup_cycles=cfg.get("eval_warmup_cycles", 0),
            m3_replay_budget=cfg.get("m3_replay_budget", 4),
            interleaved_eval_cycles=cfg.get("interleaved_eval_cycles", 0),
            interleaved_warmup_cycles=cfg.get("interleaved_warmup_cycles", 0),
            diagnostic=True,
        )
        results[run_id] = {
            "label": cfg["label"],
            "forgetting_rate": report["forgetting_rate"],
            "passes_gate": report["passes_gate"],
            "per_task_accuracy": report["per_task_accuracy"],
            "config": cfg,
        }
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="L4 ablation matrix R0-R6")
    parser.add_argument("--runs", type=str, default="R0,R2,R3,R6")
    parser.add_argument("--tasks", type=int, default=10)
    parser.add_argument("--task-cycles", type=int, default=80)
    parser.add_argument("--eval-cycles", type=int, default=20)
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--output", type=str, default="logs/l4_ablation.json")
    args = parser.parse_args()

    run_ids = [r.strip() for r in args.runs.split(",") if r.strip()]
    t0 = time.perf_counter()
    results = run_ablation(
        runs=run_ids,
        n_tasks=args.tasks,
        task_cycles=args.task_cycles,
        eval_cycles=args.eval_cycles,
        seeds=args.seeds,
    )
    out = {
        "duration_s": time.perf_counter() - t0,
        "matrix": results,
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2, default=str))
    for rid, row in results.items():
        print(f"{rid} ({row['label']}): forgetting_rate={row['forgetting_rate']:.4f} passes={row['passes_gate']}")
    print(f"Saved {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
