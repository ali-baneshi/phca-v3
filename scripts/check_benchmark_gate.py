#!/usr/bin/env python3
"""Compare benchmark JSON against a committed baseline (Φ-IQ regression gate).

Fails if overall Φ-IQ drops more than tolerance (default 5%) relative to baseline.

Usage:
    PYTHONPATH=python python scripts/benchmark.py --quick --output=logs/benchmark_report.json
    python scripts/check_benchmark_gate.py logs/benchmark_report.json logs/benchmark_ci_baseline.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def load_phi_iq(path: Path) -> float:
    data = json.loads(path.read_text())
    value = data.get("overall_phi_iq")
    if value is None:
        raise KeyError(f"{path}: missing overall_phi_iq")
    return float(value)


def main() -> None:
    if len(sys.argv) < 3:
        print(
            "Usage: check_benchmark_gate.py <report.json> <baseline.json> [tolerance]",
            file=sys.stderr,
        )
        sys.exit(2)

    report_path = Path(sys.argv[1])
    baseline_path = Path(sys.argv[2])
    tolerance = float(sys.argv[3]) if len(sys.argv) > 3 else 0.05

    report_phi = load_phi_iq(report_path)
    baseline_phi = load_phi_iq(baseline_path)
    floor = baseline_phi * (1.0 - tolerance)

    print(f"Baseline Φ-IQ: {baseline_phi:.4f}")
    print(f"Report Φ-IQ:   {report_phi:.4f}")
    print(f"Floor ({tolerance:.0%} drop): {floor:.4f}")

    if report_phi < floor:
        print(
            f"FAIL: Φ-IQ {report_phi:.4f} below floor {floor:.4f}",
            file=sys.stderr,
        )
        sys.exit(1)

    print("PASS: Φ-IQ within tolerance")
    sys.exit(0)


if __name__ == "__main__":
    main()
