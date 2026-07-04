#!/usr/bin/env python3
"""PHCA v3.0 — Session anomaly gate (Phase 13 / Observatory nightly step 7).

Fast deterministic checks for detect_session_anomalies() without a live Observatory run.
Writes a JSON summary and exits non-zero if any check fails.

Usage:
    PYTHONPATH=python python scripts/nightly_anomaly_gate.py
    PYTHONPATH=python python scripts/nightly_anomaly_gate.py --output=logs/nightly_anomaly_gate.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_PKG = _REPO / "python"
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from phca.monitoring.observability import ObservabilityFrame
from phca.monitoring.render import frame_from_json
from phca.monitoring.session_anomalies import anomaly_overall_pass, detect_session_anomalies

_FIXTURE = (
    _PKG / "phca" / "monitoring" / "tests" / "fixtures" / "reacher_short.jsonl"
)


def _frame(**kw) -> ObservabilityFrame:
    f = ObservabilityFrame()
    for k, v in kw.items():
        setattr(f, k, v)
    return f


def _check_positive_reacher() -> dict:
    lines = [ln for ln in _FIXTURE.read_text().splitlines() if ln.strip()]
    frames = [frame_from_json(json.loads(ln)) for ln in lines]
    result = detect_session_anomalies(frames)
    passed = anomaly_overall_pass(result)
    return {
        "name": "positive_reacher_fixture",
        "passed": passed,
        "active": result.get("active") or [],
    }


def _check_negative_leak() -> dict:
    n = 250
    base = 200_000_000.0
    frames = [
        _frame(
            cycle_id=i,
            prediction_error=1.0,
            rss_bytes=int(base + i * 5500),
            action_rationale={"goal_id": 1},
            active_drive_id=1,
        )
        for i in range(n)
    ]
    result = detect_session_anomalies(frames)
    passed = result["flags"].get("leak") is True
    return {
        "name": "negative_leak",
        "passed": passed,
        "late_slope": (result.get("metrics") or {}).get("rss_late_slope_bytes_per_cycle"),
    }


def _check_negative_drift() -> dict:
    frames = [
        _frame(
            cycle_id=i,
            prediction_error=1.0 + i * 0.15,
            action_rationale={"goal_id": 1},
            active_drive_id=1,
        )
        for i in range(100)
    ]
    result = detect_session_anomalies(frames)
    passed = result["flags"].get("drift") is True
    return {"name": "negative_drift", "passed": passed}


def _check_negative_goal() -> dict:
    frames = [
        _frame(
            cycle_id=i,
            prediction_error=1.0,
            action_rationale={"goal_id": 1 + (i % 4)},
            active_drive_id=1 + (i % 4),
        )
        for i in range(10)
    ]
    result = detect_session_anomalies(frames)
    passed = result["flags"].get("goal_instability") is True
    return {"name": "negative_goal_instability", "passed": passed}


def _check_negative_spike() -> dict:
    frames = [
        _frame(
            cycle_id=i,
            prediction_error=50.0 if i % 2 == 0 else 1.0,
            env_kind="grid",
            action_rationale={"goal_id": 1},
            active_drive_id=1,
        )
        for i in range(50)
    ]
    result = detect_session_anomalies(frames)
    passed = result["flags"].get("spike") is True
    return {"name": "negative_spike_density", "passed": passed}


def main() -> None:
    parser = argparse.ArgumentParser(description="PHCA session anomaly gate (Phase 13)")
    parser.add_argument("--output", default="logs/nightly_anomaly_gate.json")
    args = parser.parse_args()

    checks = [
        _check_positive_reacher(),
        _check_negative_leak(),
        _check_negative_drift(),
        _check_negative_goal(),
        _check_negative_spike(),
    ]
    all_pass = all(c["passed"] for c in checks)

    report = {"checks": checks, "all_pass": bool(all_pass)}
    out_p = Path(args.output)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(report, indent=2))

    print("=" * 60)
    print("  PHCA v3.0 — Session Anomaly Gate (Phase 13)")
    print("=" * 60)
    for c in checks:
        status = "PASS" if c["passed"] else "FAIL"
        print(f"  [{status}] {c['name']}")
    print(f"  Overall: {'PASS' if all_pass else 'FAIL'}  → {out_p}")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
