#!/usr/bin/env python3
"""Gate: TraceCollector overhead must stay below 5% at 500 cycles."""

from __future__ import annotations

import argparse
import sys
import time

from phca.core.cycle import CognitiveCycle
from phca.evaluation.trace import TraceCollector


def _run(n_cycles: int, *, with_trace: bool) -> float:
    trace = TraceCollector(verbose=False) if with_trace else None
    cycle = CognitiveCycle.build_for_env(size=5, seed=42, use_mlp=False, trace_collector=trace)
    t0 = time.perf_counter()
    for _ in range(n_cycles):
        cycle.step()
    return time.perf_counter() - t0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cycles", type=int, default=500)
    parser.add_argument("--threshold", type=float, default=0.05)
    args = parser.parse_args()

    base = _run(args.cycles, with_trace=False)
    traced = _run(args.cycles, with_trace=True)
    overhead = (traced - base) / max(base, 1e-9)
    print(f"base={base:.3f}s traced={traced:.3f}s overhead={overhead*100:.2f}%")
    if overhead > args.threshold:
        print(f"FAIL: trace overhead {overhead*100:.2f}% > {args.threshold*100:.0f}%", file=sys.stderr)
        sys.exit(1)
    print("PASS")


if __name__ == "__main__":
    main()
