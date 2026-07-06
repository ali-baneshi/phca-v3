#!/usr/bin/env python3
"""Long-horizon emergence runs with checkpoint/resume support."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

import _bootstrap  # noqa: F401

from phca.core.cycle import CognitiveCycle
from phca.world_model.mlp import gprime_stress_bounds
from phca.evaluation.metrics.emergence import compute_emergence_bundle
from phca.evaluation.metrics.phi_iq import generate_goal_pursuit_obstacles
from phca.evaluation.metrics.synergy import synergy_score
from phca.evaluation.trace import CycleTraceRecord, TraceCollector
from phca.logging import ensure_logging


class RollingTrace:
    """Keep only the latest window for emergence metrics (memory-safe)."""

    def __init__(self, window: int = 1000):
        self.window = window
        self._records: List[CycleTraceRecord] = []
        self._summaries: List[Dict[str, float]] = []
        self._last_goal_drive: int | None = None

    def record(self, **kwargs) -> None:
        drive = kwargs.get("goal_drive", 1)
        switched = self._last_goal_drive is not None and self._last_goal_drive != drive
        self._last_goal_drive = drive
        rec = CycleTraceRecord(
            cycle_id=kwargs["cycle_id"],
            action=kwargs["action"],
            reward=kwargs.get("reward", 0.0),
            prediction_error=kwargs.get("prediction_error", 0.0),
            prediction_confidence=kwargs.get("prediction_confidence", 0.0),
            goal_drive=drive,
            goal_switched=switched,
            rbta_action=kwargs.get("rbta_action", "CONTINUE"),
            violations=kwargs.get("violations", 0),
            latency_ms=kwargs.get("latency_ms", 0.0),
            goal_reached=kwargs.get("goal_reached", False),
            module_timings=kwargs.get("module_timings", {}),
        )
        self._records.append(rec)
        if len(self._records) > self.window:
            self._records = self._records[-self.window:]

    def flush_window_summary(self, window_start: int) -> None:
        if not self._records:
            return
        bundle = compute_emergence_bundle(self._records)
        bundle["synergy"] = synergy_score(self._records)
        bundle["window_start"] = window_start
        self._summaries.append(bundle)

    def snapshot(self) -> List[CycleTraceRecord]:
        return list(self._records)

    @property
    def summaries(self) -> List[Dict[str, float]]:
        return self._summaries


def save_checkpoint(path: str, cycle: CognitiveCycle, trace: RollingTrace, cycle_count: int) -> None:
    ckpt = {
        "cycle_count": cycle_count,
        "rng_state": np.random.get_state(),
        "window_summaries": trace.summaries,
        "m3_count": cycle.consolidation.m3.count() if hasattr(cycle, "consolidation") else 0,
        "recent_records": trace.snapshot(),
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_bytes(pickle.dumps(ckpt))


def load_checkpoint(path: str) -> Dict[str, Any]:
    return pickle.loads(Path(path).read_bytes())


def main() -> None:
    ensure_logging()
    parser = argparse.ArgumentParser(description="PHCA long-horizon runner")
    parser.add_argument("--cycles", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--grid-size", type=int, default=5, choices=[5, 10, 20])
    parser.add_argument("--checkpoint-every", type=int, default=1000)
    parser.add_argument("--checkpoint", default=None, help="Resume from checkpoint path")
    parser.add_argument("--verify-resume", action="store_true")
    parser.add_argument("--output", default="results/validation/horizon_100k.json")
    parser.add_argument("--window", type=int, default=1000)
    args = parser.parse_args()

    window = args.window
    trace = RollingTrace(window=window)
    start_cycle = 0
    pre_resume_composite: float | None = None

    obstacles = generate_goal_pursuit_obstacles(args.seed + 2, args.grid_size)
    cycle = CognitiveCycle.build_for_env(
        size=args.grid_size, seed=args.seed + 2,
        use_continuous=True, use_mlp=True, obstacles=obstacles,
    )
    cycle.rbta.update_bounds("G'", gprime_stress_bounds(cycle))

    if args.checkpoint and Path(args.checkpoint).exists():
        ckpt = load_checkpoint(args.checkpoint)
        start_cycle = ckpt["cycle_count"]
        np.random.set_state(ckpt["rng_state"])
        trace._summaries = list(ckpt.get("window_summaries", []))
        for rec in ckpt.get("recent_records", []):
            trace._records.append(rec)
        cycle.cycle_count = start_cycle
        if trace.summaries:
            pre_resume_composite = trace.summaries[-1].get("emergence_composite")
        print(f"Resumed from checkpoint at cycle {start_cycle}")

    target = args.cycles
    if args.verify_resume and not args.checkpoint:
        target = min(args.cycles, 2000)

    t0 = time.perf_counter()
    ckpt_path = Path(args.output).with_suffix(".ckpt")
    for i in range(start_cycle, target):
        m = cycle.step()
        trace.record(
            cycle_id=i,
            action=m.action_taken,
            reward=0.0,
            prediction_error=m.prediction_error,
            prediction_confidence=m.prediction_confidence,
            goal_drive=m.drive_id,
            rbta_action=m.rbta_action,
            violations=m.violations_count,
            latency_ms=m.latency_ms,
            goal_reached=m.goal_reached,
            module_timings=dict(m.module_timings),
        )
        if args.checkpoint_every and (i + 1) % args.checkpoint_every == 0:
            trace.flush_window_summary(i + 1 - window)
            save_checkpoint(str(ckpt_path), cycle, trace, i + 1)

    duration = time.perf_counter() - t0
    trace.flush_window_summary(max(0, target - window))
    final_emergence = compute_emergence_bundle(trace.snapshot())
    final_emergence["synergy"] = synergy_score(trace.snapshot())

    resume_drift = None
    if pre_resume_composite is not None:
        resume_drift = abs(
            final_emergence.get("emergence_composite", 0) - pre_resume_composite
        )

    report = {
        "cycles": target,
        "seed": args.seed,
        "grid_size": args.grid_size,
        "duration_s": duration,
        "final_emergence": final_emergence,
        "rolling_windows": trace.summaries,
        "checkpoint": str(ckpt_path) if ckpt_path.exists() else None,
        "resume_drift": resume_drift,
        "resume_ok": resume_drift is None or resume_drift < 0.01,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2, default=str))
    print(f"Horizon run complete: {target} cycles in {duration:.1f}s → {args.output}")
    print(f"  emergence_composite={final_emergence.get('emergence_composite', 0):.4f}")
    if resume_drift is not None:
        print(f"  resume_drift={resume_drift:.4f} ok={report['resume_ok']}")
        if args.verify_resume and not report["resume_ok"]:
            sys.exit(1)


if __name__ == "__main__":
    main()
