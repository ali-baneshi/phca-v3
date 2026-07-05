"""Interaction / unpredictability tests for emergence."""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

from phca.evaluation.trace import CycleTraceRecord


def _features(trace: Sequence[CycleTraceRecord]) -> np.ndarray:
    """Simple feature matrix for surrogate predictability."""
    rows = []
    for i, t in enumerate(trace):
        prev_action = trace[i - 1].action if i > 0 else -1
        rows.append([
            float(t.prediction_error),
            float(t.prediction_confidence),
            float(t.goal_drive),
            float(prev_action),
            float(t.goal_reached),
        ])
    return np.asarray(rows, dtype=np.float64)


def _targets(trace: Sequence[CycleTraceRecord]) -> np.ndarray:
    return np.asarray([float(t.action) for t in trace if t.action >= 0], dtype=np.float64)


def surrogate_r2(trace: Sequence[CycleTraceRecord]) -> float:
    """R² of linear surrogate predicting next action from trace features."""
    if len(trace) < 30:
        return 0.0
    X = _features(trace[1:])
    y = _targets(trace)
    n = min(len(X), len(y))
    if n < 20:
        return 0.0
    X = X[:n]
    y = y[:n]
    split = int(n * 0.7)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    if len(X_test) < 5:
        return 0.0
    # Ridge regression via normal equations
    lam = 1e-3
    xt_x = X_train.T @ X_train + lam * np.eye(X_train.shape[1])
    try:
        w = np.linalg.solve(xt_x, X_train.T @ y_train)
    except np.linalg.LinAlgError:
        return 0.0
    y_pred = X_test @ w
    ss_res = float(np.sum((y_test - y_pred) ** 2))
    ss_tot = float(np.sum((y_test - np.mean(y_test)) ** 2))
    if ss_tot < 1e-12:
        return 1.0
    return float(np.clip(1.0 - ss_res / ss_tot, 0.0, 1.0))


def ngram_signature(trace: Sequence[CycleTraceRecord], n: int = 3) -> Dict[Tuple[int, ...], float]:
    """Normalized n-gram frequency profile."""
    actions = [t.action for t in trace if t.action >= 0]
    grams: Dict[Tuple[int, ...], int] = {}
    for i in range(len(actions) - n + 1):
        g = tuple(actions[i : i + n])
        grams[g] = grams.get(g, 0) + 1
    total = sum(grams.values()) or 1
    return {k: v / total for k, v in grams.items()}


def signature_distance(
    trace_a: Sequence[CycleTraceRecord],
    trace_b: Sequence[CycleTraceRecord],
    n: int = 3,
) -> float:
    """L1 distance between n-gram behavioral signatures."""
    sig_a = ngram_signature(trace_a, n)
    sig_b = ngram_signature(trace_b, n)
    keys = set(sig_a) | set(sig_b)
    if not keys:
        return 0.0
    dist = sum(abs(sig_a.get(k, 0.0) - sig_b.get(k, 0.0)) for k in keys)
    return float(np.clip(dist / 2.0, 0.0, 1.0))


def interaction_test(
    full_trace: Sequence[CycleTraceRecord],
    ablated_traces: Dict[str, Sequence[CycleTraceRecord]],
) -> Dict[str, float]:
    """Compare surrogate predictability full vs ablated."""
    r2_full = surrogate_r2(full_trace)
    r2_ablated = {name: surrogate_r2(tr) for name, tr in ablated_traces.items()}
    max_ablated = max(r2_ablated.values()) if r2_ablated else 0.0
    return {
        "r2_full": r2_full,
        "r2_max_ablated": max_ablated,
        "interaction_gain": float(np.clip(r2_full - max_ablated, 0.0, 1.0)),
        "unpredictable": float(r2_full < 0.9),
        "interaction_supported": float(r2_full > max_ablated and r2_full < 0.9),
        **{f"r2_{k}": v for k, v in r2_ablated.items()},
    }
