#!/usr/bin/env python3
"""G2-INV-05 causal diagnosis: geometry vs learn-off vs blended (shared seeds).

Does **not** change defaults. Writes a combined JSON for investigation notes.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import _bootstrap  # noqa: F401

from phca.evaluation.interventions import InterventionConfig


def _load_causal():
    path = Path(__file__).resolve().parent / "phca_causal_eval.py"
    spec = importlib.util.spec_from_file_location("phca_causal_eval", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


causal = _load_causal()


CONDITIONS = (
    {
        "id": "A_geometry_default",
        "label": "default geometry (disable_blended_scorer=True)",
        "hypothesis": "H2 baseline",
        "build_iv": lambda: InterventionConfig(disable_blended_scorer=True),
    },
    {
        "id": "B_geometry_learn_off",
        "label": "geometry + G'/TSPL learn off",
        "hypothesis": "H2: if ≈ A then G′ irrelevant under default",
        "build_iv": lambda: InterventionConfig(
            disable_blended_scorer=True,
            enable_gprime_learn=False,
            enable_tspl=False,
        ),
    },
    {
        "id": "C_blended_opt_in",
        "label": "blended scorer opt-in",
        "hypothesis": "H3: if gate/metrics move vs A then prediction path matters",
        "build_iv": lambda: InterventionConfig(
            disable_blended_scorer=False,
            before_blended_warmup_cycles=0,
        ),
    },
)


def _phca_vs_greedy(summary: Dict[str, Any], metrics: List[str]) -> Dict[str, Any]:
    phca = summary.get("phca") or {}
    greedy = summary.get("greedy_observed") or {}
    out: Dict[str, Any] = {}
    for metric in metrics:
        key = f"{metric}_mean"
        if key not in phca or key not in greedy:
            continue
        pv = float(phca[key])
        gv = float(greedy[key])
        higher = causal.HIGHER_IS_BETTER.get(metric, True)
        delta = pv - gv
        out[metric] = {
            "phca": pv,
            "greedy_observed": gv,
            "delta_phca_minus_greedy": delta,
            "phca_better": bool(delta > 0 if higher else delta < 0),
        }
    return out


def run_condition(
    *,
    cond: Dict[str, Any],
    level: str,
    cycles: int,
    seeds: int,
    size: int,
    use_mlp: bool,
    base_seed: int,
) -> Dict[str, Any]:
    iv = cond["build_iv"]()
    t0 = time.perf_counter()
    report = causal.run_level(
        level=level,
        cycles=cycles,
        seeds=seeds,
        size=size,
        agents=["phca", "random", "greedy_observed", "greedy_full_info"],
        use_mlp=use_mlp,
        base_seed=base_seed,
        interventions=iv,
    )
    elapsed = time.perf_counter() - t0
    metrics = list(report["config"]["metrics"])
    gate = report["comparisons"].get("gate", {})
    summary = report["summary"]
    return {
        "id": cond["id"],
        "label": cond["label"],
        "hypothesis": cond["hypothesis"],
        "elapsed_s": round(elapsed, 2),
        "intervention": {
            "disable_blended_scorer": iv.disable_blended_scorer,
            "enable_gprime_learn": iv.enable_gprime_learn,
            "enable_tspl": iv.enable_tspl,
            "before_blended_warmup_cycles": iv.before_blended_warmup_cycles,
        },
        "config": report["config"],
        "summary_phca": summary.get("phca"),
        "summary_greedy_observed": summary.get("greedy_observed"),
        "summary_random": summary.get("random"),
        "gate": gate,
        "prediction_error_mean": gate.get("prediction_error_mean"),
        "secondary_prediction": gate.get("secondary_prediction"),
        "deltas_vs_greedy_observed": _phca_vs_greedy(summary, metrics),
        "metric_losers_vs_greedy": [
            m for m, d in _phca_vs_greedy(summary, metrics).items()
            if not d["phca_better"]
        ],
    }


def build_cross_condition(conditions: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_id = {c["id"]: c for c in conditions}
    a = by_id.get("A_geometry_default")
    b = by_id.get("B_geometry_learn_off")
    c = by_id.get("C_blended_opt_in")
    cross: Dict[str, Any] = {}
    if a and b:
        agr = float((a.get("summary_phca") or {}).get("goal_rate_mean", 0.0))
        bgr = float((b.get("summary_phca") or {}).get("goal_rate_mean", 0.0))
        cross["A_vs_B_goal_rate"] = {
            "A": agr,
            "B": bgr,
            "abs_delta": abs(agr - bgr),
            "approx_equal": abs(agr - bgr) < 0.02,
            "h2_supported": abs(agr - bgr) < 0.02,
        }
        ape = a.get("prediction_error_mean")
        if ape is None:
            ape = (a.get("summary_phca") or {}).get("prediction_error_mean")
        bpe = b.get("prediction_error_mean")
        if bpe is None:
            bpe = (b.get("summary_phca") or {}).get("prediction_error_mean")
        if ape is not None and bpe is not None:
            ape_f, bpe_f = float(ape), float(bpe)
            cross["A_vs_B_prediction_error_mean"] = {
                "A": ape_f,
                "B": bpe_f,
                "abs_delta": abs(ape_f - bpe_f),
                "note": "Secondary PE dual-report; not scenario-gated (D-195).",
            }
    for cond in (a, b, c):
        if not cond:
            continue
        cid = cond["id"]
        cross[f"{cid}_secondary_prediction"] = cond.get("secondary_prediction")
        cross[f"{cid}_prediction_error_mean"] = cond.get("prediction_error_mean")
    if a and c:
        agr = float((a.get("summary_phca") or {}).get("goal_rate_mean", 0.0))
        cgr = float((c.get("summary_phca") or {}).get("goal_rate_mean", 0.0))
        a_gate = bool((a.get("gate") or {}).get("passed"))
        c_gate = bool((c.get("gate") or {}).get("passed"))
        cross["A_vs_C_goal_rate"] = {
            "A": agr,
            "C": cgr,
            "delta_C_minus_A": cgr - agr,
            "A_gate": a_gate,
            "C_gate": c_gate,
            "h3_gate_changed": a_gate != c_gate,
            "h3_goal_rate_moved": abs(cgr - agr) >= 0.02,
        }
    if a:
        losers = a.get("metric_losers_vs_greedy") or []
        # H1: loses on distance/reward but not goal_rate
        cross["H1_metric_design"] = {
            "losers": losers,
            "loses_goal_rate": "goal_rate" in losers,
            "loses_non_goal_only": (
                "goal_rate" not in losers
                and any(m in losers for m in (
                    "mean_distance_to_goal", "cumulative_reward",
                    "first_goal_cycle", "coverage_rate",
                ))
            ),
        }
    return cross


def main() -> int:
    parser = argparse.ArgumentParser(description="G2-INV-05 causal ablation diagnosis")
    parser.add_argument("--level", default="level2", choices=sorted(causal.SCENARIOS))
    parser.add_argument("--cycles", type=int, default=100)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--grid-size", type=int, default=5, choices=[5, 10, 20])
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--use-mlp", action="store_true", default=True)
    parser.add_argument("--no-mlp", action="store_false", dest="use_mlp")
    parser.add_argument(
        "--conditions",
        default="A,B,C",
        help="Comma subset of A,B,C (default all)",
    )
    parser.add_argument(
        "--output",
        default="logs/diagnosis_causal_g2inv05_5x5_l2.json",
    )
    args = parser.parse_args()

    wanted = {p.strip().upper() for p in args.conditions.split(",") if p.strip()}
    selected = []
    for cond in CONDITIONS:
        letter = cond["id"][0]  # A/B/C
        if letter in wanted:
            selected.append(cond)

    results: List[Dict[str, Any]] = []
    for cond in selected:
        print(f"\n=== Running {cond['id']}: {cond['label']} ===")
        row = run_condition(
            cond=cond,
            level=args.level,
            cycles=args.cycles,
            seeds=args.seeds,
            size=args.grid_size,
            use_mlp=args.use_mlp,
            base_seed=args.base_seed,
        )
        results.append(row)
        gate = row["gate"]
        print(
            f"  gate={'PASS' if gate.get('passed') else 'FAIL'} "
            f"goal_rate={((row.get('summary_phca') or {}).get('goal_rate_mean'))} "
            f"vs greedy={((row.get('summary_greedy_observed') or {}).get('goal_rate_mean'))} "
            f"losers={row['metric_losers_vs_greedy']} "
            f"geo_dom={gate.get('geometry_dominated_frac')} "
            f"pe={row.get('prediction_error_mean')} "
            f"elapsed={row['elapsed_s']}s"
        )

    cross = build_cross_condition(results)
    payload = {
        "diagnosis": "G2-INV-05",
        "protocol": {
            "level": args.level,
            "cycles": args.cycles,
            "seeds": args.seeds,
            "grid_size": args.grid_size,
            "base_seed": args.base_seed,
            "use_mlp": args.use_mlp,
            "note": (
                "Diagnostic budget (not 30-seed gate). Shared base_seed sequence "
                "across conditions via phca_causal_eval.seed_sequence."
            ),
        },
        "conditions": results,
        "cross_condition": cross,
        "hypotheses": {
            "H1": "losses on distance/reward not goal_rate → metric design",
            "H2": "learn-off ≈ learn-on goal_rate → G′ irrelevant under geometry",
            "H3": "blended changes gate or goal_rate vs geometry → prediction path matters",
        },
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, default=str))
    print(f"\nWrote {out}")
    print(f"cross_condition={json.dumps(cross, indent=2)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
