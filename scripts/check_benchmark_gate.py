#!/usr/bin/env python3
"""Compare benchmark JSON against a committed baseline (Φ-IQ regression gate).

Two modes:
  1. Static Φ-IQ gate (default):
        check_benchmark_gate.py <report.json> <baseline.json> [tolerance]
     Fails if overall Φ-IQ drops more than tolerance (default 5%) vs baseline.

  2. MuJoCo gate (Phase 6 / C2):
        check_benchmark_gate.py --mujoco <pendulum.json> [cartpole.json ...]
     Per env asserts: no_errors AND violations==0 AND error_improved (early→late↓).

  3. Negative self-test (Phase 6 / C2):
        check_benchmark_gate.py --neg-test
     Synthesises a violating MuJoCo report and confirms the --mujoco gate flags
     it (proves the gate actually catches violations, not just passes vacuously).

Usage:
    PYTHONPATH=python python scripts/benchmark.py --quick --output=logs/benchmark_report.json
    python scripts/check_benchmark_gate.py logs/benchmark_report.json logs/benchmark_ci_baseline.json
    python scripts/check_benchmark_gate.py --mujoco logs/phase6_a3_pendulum.json logs/phase6_a3_cartpole.json
    python scripts/check_benchmark_gate.py --neg-test
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path


def load_phi_iq(path: Path) -> float:
    data = json.loads(path.read_text())
    value = data.get("overall_phi_iq")
    if value is None:
        raise KeyError(f"{path}: missing overall_phi_iq")
    return float(value)


def _mujoco_check(path: Path) -> tuple[bool, str]:
    """Check one MuJoCo benchmark JSON. Returns (passed, message)."""
    data = json.loads(path.read_text())
    env = data.get("env", path.stem)
    violations = int(data.get("violations", 0))
    improved = bool(data.get("error_improved", False))
    no_errors = bool(data.get("no_errors", True))
    early = float(data.get("early_error", 0.0))
    late = float(data.get("late_error", 0.0))
    ok = (violations == 0) and improved and no_errors
    msg = (f"{env}: violations={violations} error_improved={improved} "
           f"no_errors={no_errors} early={early:.3f}→late={late:.3f}")
    return ok, msg


def run_mujoco_gate(paths: list[Path]) -> int:
    print("MuJoCo benchmark gate (Phase 6 / C2):")
    all_ok = True
    for p in paths:
        ok, msg = _mujoco_check(p)
        print(f"  [{'PASS' if ok else 'FAIL'}] {msg}")
        all_ok = all_ok and ok
    print(f"Overall: {'PASS' if all_ok else 'FAIL'}")
    return 0 if all_ok else 1


def run_neg_test() -> int:
    """Synthesise a violating report; the gate MUST flag it (exit 1)."""
    violating = {
        "env": "SYNTHETIC-NEG-TEST", "n_cycles": 100,
        "mean_latency_ms": 5.0, "p95_latency_ms": 7.0, "max_latency_ms": 8.0,
        "mean_error": 10.0, "early_error": 5.0, "late_error": 15.0,
        "error_improved": False,          # error went UP — must be flagged
        "violations": 3, "violation_rate": 0.03, "no_errors": True,
    }
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(violating, f)
        tmp = Path(f.name)
    try:
        rc = run_mujoco_gate([tmp])
    finally:
        tmp.unlink(missing_ok=True)
    if rc == 1:
        print("Neg-test PASS: gate correctly flagged the synthetic violation.")
        return 0
    print("Neg-test FAIL: gate did NOT flag the synthetic violation.", file=sys.stderr)
    return 1


def main() -> None:
    args = sys.argv[1:]
    if args and args[0] == "--mujoco":
        paths = [Path(a) for a in args[1:]]
        if not paths:
            print("Usage: check_benchmark_gate.py --mujoco <json> [<json> ...]", file=sys.stderr)
            sys.exit(2)
        sys.exit(run_mujoco_gate(paths))
    if args and args[0] == "--neg-test":
        sys.exit(run_neg_test())

    if len(args) < 2:
        print(
            "Usage: check_benchmark_gate.py <report.json> <baseline.json> [tolerance]\n"
            "       check_benchmark_gate.py --mujoco <json> [<json> ...]\n"
            "       check_benchmark_gate.py --neg-test",
            file=sys.stderr,
        )
        sys.exit(2)

    report_path = Path(args[0])
    baseline_path = Path(args[1])
    tolerance = float(args[2]) if len(args) > 2 else 0.05

    report_phi = load_phi_iq(report_path)
    baseline_phi = load_phi_iq(baseline_path)
    floor = baseline_phi * (1.0 - tolerance)

    print(f"Baseline Φ-IQ: {baseline_phi:.4f}")
    print(f"Report Φ-IQ:   {report_phi:.4f}")
    print(f"Floor ({tolerance:.0%} drop): {floor:.4f}")

    if report_phi < floor:
        print(f"FAIL: Φ-IQ {report_phi:.4f} below floor {floor:.4f}", file=sys.stderr)
        sys.exit(1)
    print("PASS: Φ-IQ within tolerance")
    sys.exit(0)


if __name__ == "__main__":
    main()
