#!/usr/bin/env python3
"""PHCA v3.0 Phase 6 / B1 — OOD calibration curve.

Sweeps observation perturbation magnitude (Gaussian noise σ) and records the
MLP world model's confidence decomposition (aleatoric, epistemic, blended) plus
prediction MSE vs σ. The whitepaper's A4 ("prediction as primary") and A3
("incomplete knowledge") claims imply confidence should DROP as the input
distribution shifts (σ rises) — this script turns that claim into a measured
monotonic curve.

Read-only diagnostic; does not modify any production code. Trains a GridWorld
MLP cycle briefly so the model has a learned distribution to be OOD relative
to, then perturbs the input state.

Usage:
    MUJOCO_GL=disabled PYTHONPATH=python python scripts/ood_calibration.py
    MUJOCO_GL=disabled PYTHONPATH=python python scripts/ood_calibration.py --output=logs/ood_calibration.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from phca.config import ResourceBounds, StateVector
from phca.core.cycle import CognitiveCycle
from phca.logging import ensure_logging


SIGMAS = [0.0, 0.05, 0.1, 0.25, 0.5, 1.0]
N_TRIALS = 50
TRAIN_CYCLES = 80  # warm the MLP so it has a learned distribution


def _build_trained_cycle(seed: int = 42, train_cycles: int = TRAIN_CYCLES) -> CognitiveCycle:
    """Build a GridWorld MLP cycle and train it briefly."""
    rng = np.random.RandomState(seed + 2)
    obstacles = []
    gap_row = rng.randint(0, 5)
    for r in range(5):
        if r != gap_row:
            obstacles.append((r, 2))
    for _ in range(2):
        r, c = rng.randint(0, 5, size=2)
        if (r, c) not in obstacles:
            obstacles.append((r, c))
    cycle = CognitiveCycle.build_for_env(
        size=5, seed=seed + 2, use_continuous=True, use_mlp=True, obstacles=obstacles,
    )
    cycle.rbta.update_bounds("G'", ResourceBounds(B_time=0.080, B_mem=500_000, B_energy=50.0))
    for _ in range(TRAIN_CYCLES):
        cycle.step()
    return cycle


def _measure_one(cycle: CognitiveCycle, state_vec: StateVector,
                 action: np.ndarray, target: np.ndarray, sigma: float,
                 rng: np.random.RandomState) -> dict:
    """One perturbed prediction trial. Returns aleatoric/epistemic/blended/mse."""
    mlp = cycle.gprime
    s = state_vec.values.astype(np.float32)
    if sigma > 0:
        s_pert = s + rng.normal(0.0, sigma, size=s.shape).astype(np.float32)
    else:
        s_pert = s.copy()
    sv = StateVector(values=s_pert, precision=state_vec.precision,
                     timestamp=state_vec.timestamp)

    # Blended confidence + prediction from the production predict() path.
    pred, blended = mlp.predict(sv, action)
    pred_vals = pred.values.astype(np.float32)

    # Aleatoric = exp(-MSE) of the mean prediction vs the un-perturbed target.
    mse = 0.5 * float(np.mean((pred_vals - target) ** 2))
    aleatoric = float(np.exp(-max(mse, 0.0)))

    # Epistemic = MC-Dropout mutual information log1p(var) over K passes.
    x = np.concatenate([s_pert, action.astype(np.float32)])
    outs = [mlp._forward_mc(x)[2] for _ in range(mlp.mc_samples)]
    var = float(np.var(np.stack(outs), axis=0).mean())
    epistemic = float(np.log1p(min(var, 2.0)))

    return {
        "aleatoric": aleatoric,
        "epistemic": epistemic,
        "blended": float(np.clip(blended, 0.0, 1.0)),
        "mse": float(mse),
    }


def main() -> None:
    ensure_logging()
    parser = argparse.ArgumentParser(description="PHCA OOD calibration curve (Phase 6 / B1)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trials", type=int, default=N_TRIALS)
    parser.add_argument("--train-cycles", type=int, default=TRAIN_CYCLES)
    parser.add_argument("--output", default="logs/ood_calibration.json")
    args = parser.parse_args()

    trials_per_sigma = args.trials
    cycle = _build_trained_cycle(args.seed, args.train_cycles)

    # Capture a reference (state, action, target) from the trained model's
    # current cycle context — the in-distribution point we perturb around.
    ref_state = cycle.current_state
    action = cycle.last_action.copy()
    # Ground-truth next state: step the env once without perturbation.
    obs, _, _, _ = cycle.env.step(int(np.argmax(action)) if action.size > 1 else 0)
    target = obs.astype(np.float32)

    rng = np.random.RandomState(args.seed + 100)
    curve = []
    for sigma in SIGMAS:
        trials = [_measure_one(cycle, ref_state, action, target, sigma, rng)
                  for _ in range(trials_per_sigma)]
        agg = {
            "sigma": sigma,
            "aleatoric_mean": float(np.mean([t["aleatoric"] for t in trials])),
            "epistemic_mean": float(np.mean([t["epistemic"] for t in trials])),
            "blended_mean": float(np.mean([t["blended"] for t in trials])),
            "mse_mean": float(np.mean([t["mse"] for t in trials])),
            "blended_std": float(np.std([t["blended"] for t in trials])),
        }
        curve.append(agg)
        print(f"  σ={sigma:<5}  aleatoric={agg['aleatoric_mean']:.4f}  "
              f"epistemic={agg['epistemic_mean']:.4f}  "
              f"blended={agg['blended_mean']:.4f}±{agg['blended_std']:.4f}  "
              f"mse={agg['mse_mean']:.4f}")

    # Monotonicity check (Gate B requirement): blended confidence should be
    # non-increasing as σ rises.
    blended = [c["blended_mean"] for c in curve]
    monotonic_non_increasing = all(blended[i] >= blended[i + 1] - 1e-3
                                   for i in range(len(blended) - 1))
    summary = {
        "sigmas": SIGMAS,
        "trials_per_sigma": trials_per_sigma,
        "train_cycles": args.train_cycles,
        "curve": curve,
        "blended_monotonic_non_increasing": bool(monotonic_non_increasing),
        "blended_drop": float(blended[0] - blended[-1]),
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(summary, indent=2))
    print(f"\nBlended confidence drop (σ=0 → σ=1.0): {summary['blended_drop']:.4f}")
    print(f"Monotonic non-increasing: {monotonic_non_increasing}")
    print(f"Saved to {args.output}")
    if hasattr(cycle.env, "close"):
        cycle.env.close()
    # Exit 0 if the curve shows the expected monotonic drop (A4-aligned).
    sys.exit(0 if monotonic_non_increasing and summary["blended_drop"] > 0 else 1)


if __name__ == "__main__":
    main()
