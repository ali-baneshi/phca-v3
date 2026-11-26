"""Command line interface for the PHCA cognitive cycle.

    phca run     --env grid --cycles 100 --seed 0
    phca bench   --sizes 5,10 --seeds 0,1,2 --relocate-goal-every 25
    phca replay  --session result.json

``--out`` writes a JSON document. Its ``summary`` block is deterministic for a
fixed seed; its ``timing`` block is not.

Exit codes: 0 success, 1 unexpected failure, 2 invalid usage, 3 missing extra.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from phca.core.cycle import CognitiveCycle
from phca.evaluation.interventions import InterventionConfig

SCHEMA = "phca.result.v1"
GRID_SIZES = (5, 10, 20)
SELECTORS = ("geometry", "learned")

EXIT_OK = 0
EXIT_RUNTIME = 1
EXIT_USAGE = 2
EXIT_DEPENDENCY = 3


class UsageError(Exception):
    """Caller supplied something the CLI cannot honour."""


def _interventions(selector: str) -> InterventionConfig:
    if selector == "geometry":
        return InterventionConfig()
    return InterventionConfig(
        disable_blended_scorer=False,
        before_blended_warmup_cycles=0,
        adaptive_confidence_gating=False,
    )


def _build(env: str, size: int, seed: int, selector: str) -> CognitiveCycle:
    interventions = _interventions(selector)
    if env == "grid":
        return CognitiveCycle.build_for_env(size=size, seed=seed, interventions=interventions)
    try:
        import gymnasium  # noqa: F401
        import mujoco  # noqa: F401
    except ImportError as exc:
        raise UsageError(
            f"env '{env}' needs the MuJoCo extra (pip install 'phca[mujoco]'): {exc}"
        ) from exc
    return CognitiveCycle.build_for_mujoco(
        env_name="Pendulum-v1", seed=seed, interventions=interventions
    )


ACTION_NAMES = {0: "N", 1: "S", 2: "E", 3: "W", 4: "STAY"}


def _run_cycles(
    env: str,
    size: int,
    seed: int,
    selector: str,
    cycles: int,
    relocate_goal_every: int,
) -> Tuple[CognitiveCycle, List[Any], List[int]]:
    cycle = _build(env, size, seed, selector)
    history: List[Any] = []
    latencies: List[int] = []
    steps_since_goal = 0
    on_goal = False
    for index in range(1, cycles + 1):
        metrics = cycle.step()
        metrics.selector_mode = str(
            cycle.last_action_rationale.get("selector_mode") or "unknown"
        )
        steps_since_goal += 1
        metrics.goal_events = 0
        # Edge-triggered: a goal event is arriving at the goal, not every cycle
        # spent standing on it.
        if metrics.goal_reached and not on_goal:
            metrics.goal_events = 1
            latencies.append(steps_since_goal)
            steps_since_goal = 0
        on_goal = bool(metrics.goal_reached)
        history.append(metrics)
        if relocate_goal_every and index % relocate_goal_every == 0:
            if hasattr(cycle.env, "relocate_goal"):
                cycle.env.relocate_goal()
                on_goal = False
    return cycle, history, latencies


def _summarize(history: Sequence[Any], latencies: Sequence[int]) -> Dict[str, Any]:
    actions: Dict[str, int] = {}
    modes: Dict[str, int] = {}
    for m in history:
        name = ACTION_NAMES.get(m.action_taken, str(m.action_taken))
        actions[name] = actions.get(name, 0) + 1
        modes[m.selector_mode] = modes.get(m.selector_mode, 0) + 1

    errors = [float(m.prediction_error) for m in history]
    window = min(10, max(1, len(errors) // 3))
    first_goal = next((i + 1 for i, m in enumerate(history) if m.goal_events), None)
    return {
        "cycles": len(history),
        "action_counts": actions,
        "selector_mode_counts": modes,
        "rbta_non_continue": sum(1 for m in history if m.rbta_action != "CONTINUE"),
        "violations_total": sum(int(m.violations_count) for m in history),
        "cycles_on_goal": sum(1 for m in history if m.goal_reached),
        "goal_events": len(latencies),
        "first_goal_cycle": first_goal,
        "goal_latencies": list(latencies),
        "prediction_error_early": round(statistics.fmean(errors[:window]), 6),
        "prediction_error_late": round(statistics.fmean(errors[-window:]), 6),
    }


def _timing(history: Sequence[Any]) -> Dict[str, Any]:
    latencies = sorted(float(m.latency_ms) for m in history)
    if not latencies:
        return {}
    p95 = latencies[min(len(latencies) - 1, int(round(0.95 * (len(latencies) - 1))))]
    return {
        "mean_latency_ms": round(statistics.fmean(latencies), 3),
        "median_latency_ms": round(statistics.median(latencies), 3),
        "p95_latency_ms": round(p95, 3),
    }


def _emit(payload: Dict[str, Any], out: Optional[str]) -> None:
    if out:
        Path(out).write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )


def _print_run(result: Dict[str, Any]) -> None:
    cfg, s, t = result["config"], result["summary"], result["timing"]
    print(
        f"phca run  env={cfg['env']} size={cfg['size']} seed={cfg['seed']} "
        f"selector={cfg['selector']} cycles={cfg['cycles']}"
    )
    print(f"  actions           {s['action_counts']}")
    print(f"  selector modes    {s['selector_mode_counts']}")
    print(
        f"  first goal        cycle {s['first_goal_cycle']}"
        if s["first_goal_cycle"]
        else "  first goal        not reached"
    )
    if s["goal_latencies"]:
        lat = s["goal_latencies"]
        print(
            f"  goals             {len(lat)} reached, "
            f"mean {statistics.fmean(lat):.1f} cycles/goal, latencies {lat[:8]}"
        )
    else:
        print("  goals             0 reached")
    print(
        f"  prediction error  {s['prediction_error_early']:.3f} -> "
        f"{s['prediction_error_late']:.3f} (early -> late)"
    )
    print(
        f"  RBTA              {s['rbta_non_continue']} non-CONTINUE, "
        f"{s['violations_total']} violations"
    )
    if t:
        print(
            f"  latency           mean {t['mean_latency_ms']:.2f} ms, "
            f"p95 {t['p95_latency_ms']:.2f} ms"
        )


def _print_bench(result: Dict[str, Any]) -> None:
    print(
        f"{'size':>4} {'seed':>4} {'selector':<9} {'goals':>5} "
        f"{'mean steps':>10} {'first goal':>10} {'PE late':>8}"
    )
    for row in result["rows"]:
        mean = row["mean_goal_latency"]
        first = row["first_goal_cycle"]
        print(
            f"{row['size']:>4} {row['seed']:>4} {row['selector']:<9} "
            f"{row['goal_events']:>5} "
            f"{(f'{mean:.1f}' if mean is not None else '-'):>10} "
            f"{(str(first) if first is not None else '-'):>10} "
            f"{row['prediction_error_late']:>8.3f}"
        )
    print()
    for agg in result["aggregate"]:
        mean = agg["mean_goal_latency"]
        print(
            f"{agg['selector']:<9} size {agg['size']:>2}: goals {agg['goal_events']}, "
            f"mean steps/goal {(f'{mean:.2f}' if mean is not None else 'n/a')}"
        )


def _parse_int_list(raw: str, flag: str) -> List[int]:
    try:
        values = [int(p) for p in raw.split(",") if p.strip()]
    except ValueError as exc:
        raise UsageError(f"{flag} must be comma-separated integers, got {raw!r}") from exc
    if not values:
        raise UsageError(f"{flag} is empty")
    return values


def cmd_run(args: argparse.Namespace) -> int:
    if args.cycles < 1:
        raise UsageError("--cycles must be at least 1")
    if args.relocate_goal_every < 0:
        raise UsageError("--relocate-goal-every must be >= 0")
    _, history, latencies = _run_cycles(
        args.env, args.size, args.seed, args.selector,
        args.cycles, args.relocate_goal_every,
    )
    result = {
        "schema": SCHEMA,
        "command": "run",
        "config": {
            "env": args.env,
            "size": args.size,
            "seed": args.seed,
            "selector": args.selector,
            "cycles": args.cycles,
            "relocate_goal_every": args.relocate_goal_every,
        },
        "summary": _summarize(history, latencies),
        "timing": _timing(history),
    }
    _emit(result, args.out)
    if not args.quiet:
        _print_run(result)
    return EXIT_OK


def cmd_bench(args: argparse.Namespace) -> int:
    if args.cycles < 1:
        raise UsageError("--cycles must be at least 1")
    sizes = _parse_int_list(args.sizes, "--sizes")
    for size in sizes:
        if size not in GRID_SIZES:
            raise UsageError(f"--sizes entries must be one of {GRID_SIZES}, got {size}")
    seeds = _parse_int_list(args.seeds, "--seeds")

    rows = []
    pooled: Dict[Tuple[int, str], List[int]] = {}
    for size in sizes:
        for seed in seeds:
            for selector in SELECTORS:
                _, history, latencies = _run_cycles(
                    "grid", size, seed, selector, args.cycles,
                    args.relocate_goal_every,
                )
                summary = _summarize(history, latencies)
                pooled.setdefault((size, selector), []).extend(latencies)
                rows.append({
                    "size": size,
                    "seed": seed,
                    "selector": selector,
                    "cycles": summary["cycles"],
                    "goal_events": len(latencies),
                    "mean_goal_latency": (
                        round(statistics.fmean(latencies), 3) if latencies else None
                    ),
                    "first_goal_cycle": summary["first_goal_cycle"],
                    "prediction_error_early": summary["prediction_error_early"],
                    "prediction_error_late": summary["prediction_error_late"],
                    "selector_mode_counts": summary["selector_mode_counts"],
                })
    aggregate = [
        {
            "size": size,
            "selector": selector,
            "seeds": len(seeds),
            "goal_events": len(lats),
            "mean_goal_latency": round(statistics.fmean(lats), 3) if lats else None,
        }
        for (size, selector), lats in pooled.items()
    ]
    aggregate.sort(key=lambda a: (a["size"], a["selector"]))
    result = {
        "schema": SCHEMA,
        "command": "bench",
        "config": {
            "env": "grid",
            "sizes": sizes,
            "seeds": seeds,
            "cycles": args.cycles,
            "relocate_goal_every": args.relocate_goal_every,
        },
        "rows": rows,
        "aggregate": aggregate,
    }
    _emit(result, args.out)
    if not args.quiet:
        _print_bench(result)
    return EXIT_OK


def cmd_replay(args: argparse.Namespace) -> int:
    path = Path(args.session)
    if not path.is_file():
        raise UsageError(f"session file not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise UsageError(f"session file is not valid json: {exc}") from exc
    if payload.get("schema") != SCHEMA:
        raise UsageError(
            f"unexpected schema {payload.get('schema')!r}; expected {SCHEMA!r}"
        )
    command = payload.get("command")
    print(f"session   {path}  ({command}, {payload['schema']})")
    if command == "run":
        _print_run(payload)
        return EXIT_OK
    if command == "bench":
        _print_bench(payload)
        return EXIT_OK
    raise UsageError(f"cannot replay a result of command {command!r}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="phca",
        description="Run the PHCA cognitive cycle, benchmark selectors, or replay a result.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run one agent for N cycles")
    run.add_argument("--env", default="grid", choices=("grid", "pendulum"))
    run.add_argument("--size", type=int, default=5, choices=GRID_SIZES,
                     help="grid edge length (grid env only)")
    run.add_argument("--cycles", type=int, default=100)
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--selector", default="geometry", choices=SELECTORS,
                     help="geometry: BFS planner (default); learned: score actions by G'")
    run.add_argument("--relocate-goal-every", type=int, default=0,
                     help="move the goal every N cycles (0 = off)")
    run.add_argument("--out", default=None, help="write the result JSON here")
    run.add_argument("--quiet", action="store_true", help="suppress the human summary")
    run.set_defaults(func=cmd_run)

    bench = sub.add_parser("bench", help="compare selectors across sizes and seeds")
    bench.add_argument("--sizes", default="5,10", help="comma-separated grid sizes")
    bench.add_argument("--seeds", default="0,1,2", help="comma-separated seeds")
    bench.add_argument("--cycles", type=int, default=200)
    bench.add_argument("--relocate-goal-every", type=int, default=25,
                       help="move the goal every N cycles (0 = off)")
    bench.add_argument("--out", default=None, help="write the result JSON here")
    bench.add_argument("--quiet", action="store_true", help="suppress the table")
    bench.set_defaults(func=cmd_bench)

    replay = sub.add_parser("replay", help="re-print a result written by run or bench")
    replay.add_argument("--session", required=True, help="path to a result JSON")
    replay.set_defaults(func=cmd_replay)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except UsageError as exc:
        message = str(exc)
        if "phca[mujoco]" in message:
            print(f"phca: {message}", file=sys.stderr)
            return EXIT_DEPENDENCY
        print(f"phca: error: {message}", file=sys.stderr)
        return EXIT_USAGE
    except Exception as exc:  # pragma: no cover - unexpected failure path
        print(f"phca: unexpected failure: {type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_RUNTIME


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
