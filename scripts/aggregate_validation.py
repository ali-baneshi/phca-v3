#!/usr/bin/env python3
"""Aggregate validation results and evaluate hypotheses."""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import yaml as _yaml
except ImportError:
    _yaml = None

import numpy as np

from phca.evaluation.metrics.statistics import aggregate_runs, compare_groups


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


def _load_yaml(path: Path) -> Any:
    if _yaml is None:
        raise RuntimeError("PyYAML required")
    return _yaml.safe_load(path.read_text())


def _find_results(root: Path) -> List[Path]:
    return sorted(root.rglob("*.json"))


def _extract_phi_iq(data: Dict[str, Any]) -> Optional[float]:
    if "overall_phi_iq" in data:
        return float(data["overall_phi_iq"])
    agg = data.get("aggregate", {})
    metrics = agg.get("metrics", {})
    if "phi_iq" in metrics and isinstance(metrics["phi_iq"], dict):
        return float(metrics["phi_iq"].get("mean", 0))
    if "phi_iq" in metrics:
        return float(metrics["phi_iq"])
    return None


def _extract_metric(data: Dict[str, Any], key: str) -> Optional[float]:
    agg = data.get("aggregate", {})
    emergence = agg.get("emergence", {})
    if key in emergence and isinstance(emergence[key], dict):
        return float(emergence[key].get("mean", 0))
    metrics = agg.get("metrics", {})
    if key in metrics and isinstance(metrics[key], dict):
        return float(metrics[key].get("mean", 0))
    return None


def evaluate_hypotheses(
    hypotheses: List[Dict[str, Any]],
    results: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    full = results.get("ablations/full_system.json") or results.get("ablations/full_system")
    no_pred = results.get("ablations/no_prediction.json") or results.get("ablations/no_prediction")
    no_mem = results.get("ablations/no_memory.json") or results.get("ablations/no_memory")
    desync = results.get("ablations/desync.json") or results.get("ablations/desync")
    minimal = results.get("ablations/minimal_cycle.json") or results.get("ablations/minimal_cycle")
    phiq = results.get("phi_iq_validation.json") or results.get("phi_iq_validation")

    verdicts = []
    for h in hypotheses:
        hid = h["id"]
        status = "Partially_supported"
        observed: Dict[str, Any] = {}

        if hid == "H001" and no_mem and full:
            reuse_ab = _extract_metric(no_mem, "cross_context_reuse") or 0
            reuse_full = _extract_metric(full, "cross_context_reuse") or 0
            observed = {"cross_context_reuse_ablated": reuse_ab, "cross_context_reuse_full": reuse_full}
            status = "Validated" if reuse_ab < 0.2 and reuse_ab < reuse_full * 0.7 else "Refuted"

        elif hid == "H002" and no_pred and full:
            t_ab = _extract_metric(no_pred, "transfer_efficiency") or 0
            t_full = _extract_metric(full, "transfer_efficiency") or 0
            drop = (t_full - t_ab) / max(t_full, 1e-6)
            observed = {"transfer_drop_fraction": drop}
            status = "Validated" if drop > 0.25 else "Partially_supported" if drop > 0.1 else "Refuted"

        elif hid == "H003" and desync and full:
            s_ab = _extract_metric(desync, "synergy") or 0
            s_full = _extract_metric(full, "synergy") or 0
            observed = {"synergy_ablated": s_ab, "synergy_full": s_full}
            status = "Validated" if s_ab < 0.5 * max(s_full, 1e-6) else "Refuted"

        elif hid == "H005" and minimal and full:
            e_ab = _extract_metric(minimal, "emergence_composite") or 0
            e_full = _extract_metric(full, "emergence_composite") or 0
            ratio = e_ab / max(e_full, 1e-6)
            observed = {"emergence_ratio": ratio}
            status = "Validated" if ratio < 0.7 else "Refuted"

        elif hid == "H006" and phiq:
            r = phiq.get("predictive_validity", {}).get("pearson_r", 0)
            observed = {"pearson_r": r}
            status = "Validated" if r >= 0.7 else "Refuted"

        elif hid == "H004":
            observed = {"note": "Requires interaction_test.json comparison"}
            status = "Partially_supported"

        verdicts.append({
            **h,
            "status": status,
            "observed": observed,
            "verdict_date": str(date.today()),
        })
    return verdicts


def build_failure_log(results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    failures: List[Dict[str, Any]] = []
    for name, data in results.items():
        if "runs" not in data:
            continue
        for run in data["runs"]:
            f = run.get("failures", {})
            if f.get("rbta_terminate_max_streak", 0) > 3 or f.get("goal_thrash_events", 0) > 50:
                failures.append({"experiment": name, "seed": run.get("seed"), **f})
    return {"failure_events": failures, "count": len(failures)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="results/validation")
    parser.add_argument("--hypotheses", default="experiments/hypotheses.yaml")
    parser.add_argument("--output", default="results/validation/summary.json")
    args = parser.parse_args()

    root = Path(args.input)
    index: Dict[str, Dict[str, Any]] = {}
    for path in _find_results(root):
        if path.name in ("summary.json", "manifest_index.json", "hypothesis_verdicts.json", "failure_log.json"):
            continue
        rel = str(path.relative_to(root))
        try:
            index[rel] = _load_json(path)
        except json.JSONDecodeError:
            continue

    hypotheses_raw = _load_yaml(Path(args.hypotheses))
    hypotheses = hypotheses_raw.get("hypotheses", [])
    verdicts = evaluate_hypotheses(hypotheses, index)
    failure_log = build_failure_log(index)

    phi_values = [_extract_phi_iq(v) for v in index.values() if _extract_phi_iq(v) is not None]
    summary = {
        "date": str(date.today()),
        "n_result_files": len(index),
        "phi_iq_overall": aggregate_runs([{"phi_iq": v} for v in phi_values], ["phi_iq"]) if phi_values else {},
        "experiments": list(index.keys()),
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, default=str))
    (out.parent / "hypothesis_verdicts.json").write_text(json.dumps(verdicts, indent=2, default=str))
    (out.parent / "failure_log.json").write_text(json.dumps(failure_log, indent=2, default=str))
    print(f"Wrote {out}, hypothesis_verdicts.json, failure_log.json")


if __name__ == "__main__":
    main()
