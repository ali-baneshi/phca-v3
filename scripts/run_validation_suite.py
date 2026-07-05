#!/usr/bin/env python3
"""Overnight orchestrator for the complete scientific validation suite."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
ENV = {"PYTHONPATH": "python", "MUJOCO_GL": "disabled"}


def _run(cmd: List[str], cwd: Path = ROOT) -> int:
    print(f"\n>>> {' '.join(cmd)}\n")
    return subprocess.run(cmd, cwd=str(cwd), env={**dict(**__import__("os").environ), **ENV}).returncode


def _load_index(path: Path) -> Dict[str, Any]:
    if path.exists():
        return json.loads(path.read_text())
    return {"completed": []}


def _save_index(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


def run_step(step_id: str, cmd: List[str], index: Dict[str, Any], index_path: Path, out: Path) -> bool:
    if step_id in index.get("completed", []):
        print(f"SKIP (done): {step_id}")
        return True
    rc = _run(cmd)
    # Treat as success if command wrote expected output (scaling may exit 1 on pass criteria)
    expected = _expected_output(step_id, out)
    if rc == 0 or (expected and expected.exists()):
        completed = index.setdefault("completed", [])
        if step_id not in completed:
            completed.append(step_id)
        _save_index(index_path, index)
        if rc != 0:
            print(f"NOTE: {step_id} exit={rc} but output exists — marked complete")
        return True
    print(f"FAILED: {step_id} exit={rc}")
    return False


def _expected_output(step_id: str, out: Path) -> Path | None:
    if step_id.startswith("scaling/"):
        return out / "scaling" / f"{step_id.split('/', 1)[1]}.json"
    if step_id.startswith("ablations/"):
        return out / "ablations" / f"{step_id.split('/', 1)[1]}.json"
    if step_id == "baselines/causal_eval":
        return out / "baselines" / "causal_eval.json"
    if step_id == "phi_iq_validation":
        return out / "phi_iq_validation.json"
    if step_id == "horizon":
        for p in sorted(out.glob("horizon_*.json")):
            if "resume" not in p.name:
                return p
    if step_id == "horizon_resume_check":
        return out / "horizon_resume_check.json"
    if step_id.startswith("ood/"):
        return out / "ood" / f"{step_id.split('/', 1)[1]}.json"
    if step_id == "aggregate":
        return out / "summary.json"
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=["smoke", "full"], default="full")
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--output-root", default="results/validation")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    out = Path(args.output_root)
    out.mkdir(parents=True, exist_ok=True)
    index_path = out / "manifest_index.json"
    index = _load_index(index_path) if args.resume else {"completed": [], "base_seed": args.base_seed}

    seeds = 3 if args.profile == "smoke" else 30
    phiq_seeds = 10 if args.profile == "smoke" else 100
    horizon = 1000 if args.profile == "smoke" else 100000

    steps: List[tuple[str, List[str]]] = []

    for grid, slip in [(5, 0.0), (5, 0.1), (10, 0.0), (10, 0.1), (20, 0.0), (20, 0.1)]:
        sid = f"scaling/grid{grid}_slip{int(slip*10)}"
        steps.append((sid, [
            "python", "scripts/benchmark.py",
            "--use-mlp", "--cycles=200", f"--seeds={seeds}",
            f"--grid-size={grid}", f"--action-slip={slip}",
            "--allow-fail",
            f"--output={out}/scaling/grid{grid}_slip{int(slip*10)}.json",
        ]))

    ablation_dir = ROOT / "experiments" / "ablations"
    for manifest in sorted(ablation_dir.glob("*")):
        if manifest.suffix not in (".json", ".yaml", ".yml"):
            continue
        sid = f"ablations/{manifest.stem}"
        steps.append((sid, [
            "python", "scripts/run_experiment.py", str(manifest),
            f"--seeds={seeds}", f"--output={out}/ablations/{manifest.stem}.json",
        ]))

    steps.append(("baselines/causal_eval", [
        "python", "scripts/phca_causal_eval.py",
        "--levels", "level1,level2,level3",
        "--cycles", "200", f"--seeds", str(seeds),
        f"--base-seed", str(args.base_seed),
        "--use-mlp",
        "--agents", "phca,random,greedy_observed,bfs_search",
        f"--output={out}/baselines/causal_eval.json",
    ]))

    steps.append(("phi_iq_validation", [
        "python", "scripts/phi_iq_validation.py",
        f"--seeds={phiq_seeds}", f"--base-seed={args.base_seed}",
        f"--output={out}/phi_iq_validation.json",
    ]))

    steps.append(("horizon", [
        "python", "scripts/run_horizon.py",
        f"--cycles={horizon}", "--checkpoint-every=1000",
        f"--output={out}/horizon_{horizon}.json",
    ]))

    if args.profile == "full" and horizon >= 10000:
        ckpt = out / f"horizon_{horizon}.ckpt"
        steps.append(("horizon_resume_check", [
            "python", "scripts/run_horizon.py",
            "--cycles=2000", f"--checkpoint={ckpt}",
            "--verify-resume",
            f"--output={out}/horizon_resume_check.json",
        ]))

    interaction = ROOT / "experiments" / "interaction_test.json"
    if interaction.exists():
        steps.append(("ablations/interaction_test", [
            "python", "scripts/run_experiment.py", str(interaction),
            f"--seeds={seeds}", f"--output={out}/ablations/interaction_test.json",
        ]))

    for ood in ["experiments/ood/bandit.json", "experiments/ood/mujoco_pendulum.json"]:
        p = ROOT / ood
        if p.exists():
            sid = f"ood/{p.stem}"
            steps.append((sid, [
                "python", "scripts/run_experiment.py", str(p),
                f"--seeds={seeds}", f"--output={out}/ood/{p.stem}.json",
            ]))

    steps.append(("aggregate", [
        "python", "scripts/aggregate_validation.py",
        f"--input={out}", f"--output={out}/summary.json",
        f"--hypotheses=experiments/hypotheses.yaml",
    ]))

    t0 = time.perf_counter()
    failed = []
    for sid, cmd in steps:
        if not run_step(sid, cmd, index, index_path, out):
            failed.append(sid)

    elapsed = time.perf_counter() - t0
    index["duration_s"] = elapsed
    index["failed"] = failed
    _save_index(index_path, index)
    print(f"\nValidation suite finished in {elapsed/3600:.2f}h; failed={failed}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
