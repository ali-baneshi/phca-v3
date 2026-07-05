#!/usr/bin/env python3
"""Φ-IQ validation studies: sensitivity, predictive validity, decomposition."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from phca.evaluation.interventions import InterventionConfig
from phca.evaluation.metrics.phi_iq import compute_level_metrics
from phca.evaluation.metrics.statistics import compare_groups, seed_sequence
from phca.evaluation.result_schema import BenchmarkConfig
from phca.evaluation.runner import build_cycle
from phca.logging import ensure_logging


def _early_late_phi(cycle, n_early: int, n_late: int, level: int = 2) -> tuple[float, float]:
    """Phi from early window vs goal rate late window."""
    history = list(cycle.metrics_history)
    if len(history) < n_early + n_late:
        return 0.0, 0.0
    early = compute_level_metrics(level, history[:n_early], cycle, n_early)
    late_goals = [m.goal_reached for m in history[n_early : n_early + n_late]]
    late_rate = float(np.mean(late_goals)) if late_goals else 0.0
    return early.phi_iq, late_rate


def run_validation(n_seeds: int, base_seed: int, cycles: int) -> Dict:
    config = BenchmarkConfig(n_cycles=cycles, use_mlp=True, seed=base_seed)
    n_early, n_late = 50, min(150, cycles - 50)

    full_phi, full_late = [], []
    minimal_phi, no_pred_phi = [], []

    for seed in seed_sequence(base_seed, n_seeds):
        for label, iv in [("full", None), ("minimal", InterventionConfig.minimal()), ("no_pred", InterventionConfig.no_prediction())]:
            cycle = build_cycle(seed=seed, use_mlp=True, level=2, interventions=iv)
            for _ in range(cycles):
                cycle.step()
            result = compute_level_metrics(2, cycle.metrics_history, cycle, cycles)
            if label == "full":
                full_phi.append(result.phi_iq)
                _, late = _early_late_phi(cycle, n_early, n_late)
                full_late.append(late)
            elif label == "minimal":
                minimal_phi.append(result.phi_iq)
            else:
                no_pred_phi.append(result.phi_iq)

    # Predictive validity: correlate early phi with late goal rate
    early_phis = []
    late_rates = []
    for seed in seed_sequence(base_seed, n_seeds):
        cycle = build_cycle(seed=seed, use_mlp=True, level=2)
        for _ in range(cycles):
            cycle.step()
        ep, lr = _early_late_phi(cycle, n_early, n_late)
        early_phis.append(ep)
        late_rates.append(lr)

    if len(early_phis) > 2 and np.std(early_phis) > 1e-6 and np.std(late_rates) > 1e-6:
        r = float(np.corrcoef(early_phis, late_rates)[0, 1])
    else:
        r = 0.0

    sensitivity_full_vs_minimal = compare_groups(full_phi, minimal_phi)
    sensitivity_full_vs_no_pred = compare_groups(full_phi, no_pred_phi)

    return {
        "n_seeds": n_seeds,
        "base_seed": base_seed,
        "cycles": cycles,
        "sensitivity": {
            "full_vs_minimal": sensitivity_full_vs_minimal,
            "full_vs_no_prediction": sensitivity_full_vs_no_pred,
            "full_mean": float(np.mean(full_phi)),
            "minimal_mean": float(np.mean(minimal_phi)),
            "no_pred_mean": float(np.mean(no_pred_phi)),
        },
        "predictive_validity": {
            "pearson_r": r,
            "pass_r_0_7": r >= 0.7,
            "n_early": n_early,
            "n_late": n_late,
        },
        "decomposition_note": "Run multivariate regression offline on per-run subindices if r fails",
    }


def main() -> None:
    ensure_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--cycles", type=int, default=200)
    parser.add_argument("--output", default="logs/phi_iq_validation.json")
    args = parser.parse_args()

    report = run_validation(args.seeds, args.base_seed, args.cycles)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2))
    print(f"Φ-IQ validation → {args.output}")
    print(f"  predictive r={report['predictive_validity']['pearson_r']:.3f}")
    print(f"  full vs minimal passed={report['sensitivity']['full_vs_minimal'].get('passed')}")


if __name__ == "__main__":
    main()
