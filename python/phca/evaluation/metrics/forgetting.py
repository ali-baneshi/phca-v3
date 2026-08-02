"""Forgetting-rate metrics for Level-4-lite continual learning benchmarks."""

from __future__ import annotations

from typing import Dict, List, Mapping, Optional, Union

import numpy as np

from phca.core.cycle import CycleMetrics


HistoryInput = Union[Mapping[int, List[CycleMetrics]], List[CycleMetrics]]


def _metrics_for_task(history: HistoryInput, task_id: int) -> List[CycleMetrics]:
    if isinstance(history, Mapping):
        return list(history.get(task_id, []))
    return [m for m in history if getattr(m, "task_id", None) == task_id]


def task_accuracy(
    history: HistoryInput,
    *,
    task_id: int,
    metric: str = "goal_rate",
    window: Optional[int] = None,
) -> float:
    """Task accuracy proxy over history for a single task.

    GridWorld default (``goal_rate``): fraction of cycles with ``goal_reached``.
    """
    records = _metrics_for_task(history, task_id)
    if not records:
        return 0.0
    if window is not None and window > 0:
        records = records[-window:]
    if metric == "goal_rate":
        goals = [float(m.goal_reached) for m in records]
        return float(np.mean(goals)) if goals else 0.0
    if metric == "prediction_accuracy":
        errors = [m.prediction_error for m in records]
        mean_err = float(np.mean(errors)) if errors else 0.0
        return max(0.0, 1.0 - min(mean_err / 10.0, 1.0))
    raise ValueError(f"Unknown metric: {metric}")


def eval_window_accuracy(
    history: HistoryInput,
    *,
    task_id: int,
    eval_cycles: int = 20,
    metric: str = "goal_rate",
) -> float:
    """Accuracy over the last ``eval_cycles`` evaluation cycles for a task."""
    return task_accuracy(
        history, task_id=task_id, metric=metric, window=eval_cycles,
    )


def max_rolling_task_accuracy(
    history: HistoryInput,
    *,
    task_id: int,
    metric: str = "goal_rate",
    window: int = 20,
) -> float:
    """Best goal-rate over any contiguous ``window`` in train history."""
    records = _metrics_for_task(history, task_id)
    if not records:
        return 0.0
    if window <= 0 or len(records) <= window:
        return task_accuracy(history, task_id=task_id, metric=metric)
    best = 0.0
    for i in range(len(records) - window + 1):
        chunk = records[i : i + window]
        if metric == "goal_rate":
            acc = float(np.mean([float(m.goal_reached) for m in chunk]))
        elif metric == "prediction_accuracy":
            errors = [m.prediction_error for m in chunk]
            mean_err = float(np.mean(errors)) if errors else 0.0
            acc = max(0.0, 1.0 - min(mean_err / 10.0, 1.0))
        else:
            raise ValueError(f"Unknown metric: {metric}")
        best = max(best, acc)
    return best


DEFAULT_BASELINE_MIN_VALID = 0.2


def is_valid_baseline(goal_rate: float, *, min_valid: float = DEFAULT_BASELINE_MIN_VALID) -> bool:
    """Task was learned enough to measure forgetting."""
    return goal_rate >= min_valid


def delta_perf_valid_only(
    baseline: Dict[int, float],
    current: Dict[int, float],
    *,
    min_valid: float = DEFAULT_BASELINE_MIN_VALID,
) -> tuple[Dict[int, float], List[int]]:
    """Like ``delta_perf`` but excludes tasks with invalid baselines."""
    filtered = {
        tid: base
        for tid, base in baseline.items()
        if is_valid_baseline(base, min_valid=min_valid)
    }
    return delta_perf(filtered, current), [
        tid for tid, base in baseline.items()
        if not is_valid_baseline(base, min_valid=min_valid)
    ]


def delta_perf(
    baseline: Dict[int, float],
    current: Dict[int, float],
) -> Dict[int, float]:
    """Per-task relative performance change: (current - baseline) / baseline.

    Negative values indicate forgetting (performance drop).
    """
    delta: Dict[int, float] = {}
    for task_id, base_acc in baseline.items():
        if base_acc <= 1e-8:
            cur = current.get(task_id, 0.0)
            delta[task_id] = 0.0 if cur <= 1e-8 else 1.0
            continue
        cur_acc = current.get(task_id, 0.0)
        delta[task_id] = (cur_acc - base_acc) / base_acc
    return delta


def forgetting_rate(delta: Dict[int, float]) -> float:
    """Aggregate forgetting rate: max relative performance drop (AT-2-lite).

    Only counts drops (negative Δ_perf). Improvements do not inflate this metric.
    """
    if not delta:
        return 0.0
    return float(max(max(0.0, -v) for v in delta.values()))


def cycles_to_threshold(
    train_curve: List[float],
    threshold: float = 0.8,
    min_cycles: int = 5,
) -> int:
    """First cycle where a ``min_cycles``-rolling average crosses ``threshold``.

    Returns ``len(train_curve)`` if threshold is never reached (capped).
    """
    if len(train_curve) < min_cycles:
        return len(train_curve)
    for i in range(len(train_curve) - min_cycles + 1):
        avg = float(np.mean(train_curve[i : i + min_cycles]))
        if avg >= threshold:
            return i + min_cycles  # last cycle of the window
    return len(train_curve)


def forward_transfer(
    train_curves: Dict[int, List[float]],
    threshold: float = 0.8,
    min_cycles: int = 5,
) -> Dict[int, float]:
    """Speedup ratio per task relative to task 0 learning speed.

    Speedup = cycles(task_0) / cycles(task_N).
    Values > 1.0 indicate forward transfer (faster learning).
    Values < 1.0 indicate negative transfer (slower learning).
    """
    if 0 not in train_curves:
        return {}
    base_cycles = cycles_to_threshold(
        train_curves[0], threshold=threshold, min_cycles=min_cycles,
    )
    if base_cycles <= 0:
        base_cycles = 1  # avoid division by zero
    result: Dict[int, float] = {}
    for tid, curve in train_curves.items():
        c = cycles_to_threshold(curve, threshold=threshold, min_cycles=min_cycles)
        result[tid] = round(base_cycles / max(c, 1), 3)
    return result


def passes_forgetting_gate(
    delta: Dict[int, float],
    threshold: float = 0.05,
) -> bool:
    """True when max relative drop is within ``threshold`` (default 5%).

    Note: callers that exclude low-baseline tasks must also apply
    :func:`vacuous_pass_blocked` so sparsely-valid seeds cannot PASS.
    """
    if not delta:
        return True
    max_drop = max(-v for v in delta.values())
    return max_drop <= threshold


def valid_task_coverage(
    n_tasks: int,
    excluded_tasks: List[int],
) -> float:
    """Fraction of tasks retained after baseline exclusion (0..1)."""
    if n_tasks <= 0:
        return 0.0
    n_excl = len(excluded_tasks)
    return max(0.0, min(1.0, (n_tasks - n_excl) / float(n_tasks)))


def vacuous_pass_blocked(
    n_tasks: int,
    excluded_tasks: List[int],
    *,
    max_exclude_frac: float = 0.5,
    min_valid_tasks: int = 2,
) -> bool:
    """True when a forgetting PASS would be vacuous (too few valid tasks).

    Blocks the seed-1542 pattern: 8/10 tasks excluded → FR=0 on 2 leftovers.
    """
    if n_tasks <= 0:
        return True
    n_excl = len(excluded_tasks)
    n_valid = n_tasks - n_excl
    if n_valid < min_valid_tasks:
        return True
    if n_excl > max_exclude_frac * n_tasks:
        return True
    return False


def passes_forgetting_gate_with_coverage(
    delta: Dict[int, float],
    *,
    n_tasks: int,
    excluded_tasks: List[int],
    threshold: float = 0.05,
    max_exclude_frac: float = 0.5,
    min_valid_tasks: int = 2,
) -> tuple[bool, bool]:
    """Gate result plus whether a vacuous PASS was blocked.

    Returns ``(passes_gate, vacuous_pass_blocked)``.
    """
    blocked = vacuous_pass_blocked(
        n_tasks,
        excluded_tasks,
        max_exclude_frac=max_exclude_frac,
        min_valid_tasks=min_valid_tasks,
    )
    if blocked:
        return False, True
    return passes_forgetting_gate(delta, threshold=threshold), False
