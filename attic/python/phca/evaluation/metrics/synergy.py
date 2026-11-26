"""Minimal information-theoretic synergy from traces."""

from __future__ import annotations

from typing import List, Sequence

import numpy as np

from phca.evaluation.trace import CycleTraceRecord


def _discretize_confidence(values: Sequence[float], n_bins: int = 5) -> List[int]:
    if not values:
        return []
    arr = np.asarray(values, dtype=np.float64)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    return list(np.digitize(arr, bins[1:-1]))


def mutual_information(x: Sequence[int], y: Sequence[int]) -> float:
    """Empirical MI between discrete sequences."""
    if len(x) != len(y) or len(x) == 0:
        return 0.0
    n = len(x)
    xy_counts: dict = {}
    x_counts: dict = {}
    y_counts: dict = {}
    for xi, yi in zip(x, y):
        xy_counts[(xi, yi)] = xy_counts.get((xi, yi), 0) + 1
        x_counts[xi] = x_counts.get(xi, 0) + 1
        y_counts[yi] = y_counts.get(yi, 0) + 1
    mi = 0.0
    for (xi, yi), c_xy in xy_counts.items():
        p_xy = c_xy / n
        p_x = x_counts[xi] / n
        p_y = y_counts[yi] / n
        if p_xy > 0 and p_x > 0 and p_y > 0:
            mi += p_xy * np.log(p_xy / (p_x * p_y))
    return float(max(0.0, mi))


def _entropy(values: Sequence[int]) -> float:
    if not values:
        return 0.0
    _, counts = np.unique(values, return_counts=True)
    probs = counts / counts.sum()
    return float(-np.sum(probs * np.log(probs + 1e-12)))


def prediction_action_synergy(trace: Sequence[CycleTraceRecord]) -> float:
    """Normalized MI between prediction confidence bucket and next action."""
    if len(trace) < 20:
        return 0.0
    pairs = [
        (trace[i].prediction_confidence, trace[i + 1].action)
        for i in range(len(trace) - 1)
        if trace[i + 1].action >= 0
    ]
    if len(pairs) < 10:
        return 0.0
    conf_bins = _discretize_confidence([c for c, _ in pairs])
    next_actions = [int(a) for _, a in pairs]
    if len(conf_bins) != len(next_actions) or not conf_bins:
        return 0.0
    mi = mutual_information(conf_bins, next_actions)
    h_conf = _entropy(conf_bins)
    h_act = _entropy(next_actions)
    denom = min(h_conf, h_act)
    if denom <= 1e-12:
        return 0.0
    return float(np.clip(mi / denom, 0.0, 1.0))


def synergy_score(trace: Sequence[CycleTraceRecord]) -> float:
    """Synergy proxy: MI minus marginal predictability floor."""
    mi_norm = prediction_action_synergy(trace)
    actions = [t.action for t in trace if t.action >= 0]
    if len(actions) < 10:
        return mi_norm
    _, counts = np.unique(actions, return_counts=True)
    max_marginal = float(np.max(counts) / len(actions))
    synergy = mi_norm - max_marginal * 0.5
    return float(np.clip(synergy, 0.0, 1.0))
