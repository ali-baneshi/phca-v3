"""Operational emergence metrics from cycle traces."""

from __future__ import annotations

import zlib
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from phca.evaluation.trace import CycleTraceRecord


def _actions(trace: Sequence[CycleTraceRecord]) -> List[int]:
    return [t.action for t in trace if t.action >= 0]


def _ngrams(actions: Sequence[int], n: int) -> List[Tuple[int, ...]]:
    if len(actions) < n:
        return []
    return [tuple(actions[i : i + n]) for i in range(len(actions) - n + 1)]


def novel_behaviour(trace: Sequence[CycleTraceRecord], window: int = 50) -> float:
    """Action subsequences in late window not seen in early window."""
    actions = _actions(trace)
    if len(actions) < window * 2:
        return 0.0
    early = set(_ngrams(actions[:window], 2))
    late = set(_ngrams(actions[-window:], 2))
    if not late:
        return 0.0
    novel = late - early
    return float(np.clip(len(novel) / len(late), 0.0, 1.0))


def strategy_diversity(trace: Sequence[CycleTraceRecord], n: int = 2) -> float:
    """Normalized entropy of action n-grams."""
    actions = _actions(trace)
    grams = _ngrams(actions, n)
    if not grams:
        return 0.0
    _, counts = np.unique(grams, axis=0, return_counts=True)
    probs = counts / counts.sum()
    h = -float(np.sum(probs * np.log(probs + 1e-12)))
    h_max = np.log(len(counts) + 1e-12)
    if h_max <= 0:
        return 0.0
    return float(np.clip(h / h_max, 0.0, 1.0))


def behavioral_compression(trace: Sequence[CycleTraceRecord]) -> float:
    """Compression ratio vs uniform random baseline."""
    actions = _actions(trace)
    if len(actions) < 10:
        return 0.0
    action_bytes = bytes(a % 256 for a in actions)
    random_bytes = bytes(np.random.randint(0, 5, size=len(actions), dtype=np.uint8))
    comp = len(zlib.compress(action_bytes))
    comp_rand = len(zlib.compress(random_bytes))
    if comp_rand <= 0:
        return 0.0
    ratio = comp / comp_rand
    return float(np.clip(1.0 - ratio, 0.0, 1.0))


def adaptation_stability(trace: Sequence[CycleTraceRecord], window: int = 20) -> float:
    """Low variance of rolling prediction error."""
    errors = [t.prediction_error for t in trace]
    if len(errors) < window * 2:
        return 0.0
    rolling = [
        float(np.mean(errors[i : i + window]))
        for i in range(0, len(errors) - window + 1, window)
    ]
    if len(rolling) < 2:
        return 0.0
    mean_r = float(np.mean(rolling))
    if mean_r < 1e-6:
        return 1.0
    cv = float(np.std(rolling)) / mean_r
    return float(np.clip(1.0 - cv, 0.0, 1.0))


def cross_context_reuse(trace: Sequence[CycleTraceRecord]) -> float:
    """Action policy similarity before/after goal switches."""
    switch_indices = [i for i, t in enumerate(trace) if t.goal_switched]
    if not switch_indices:
        return 0.0
    actions = _actions(trace)
    if len(actions) < 20:
        return 0.0
    before_profiles: List[np.ndarray] = []
    after_profiles: List[np.ndarray] = []
    for idx in switch_indices:
        lo = max(0, idx - 10)
        hi = min(len(trace), idx + 10)
        before = [t.action for t in trace[lo:idx] if t.action >= 0]
        after = [t.action for t in trace[idx:hi] if t.action >= 0]
        if len(before) < 3 or len(after) < 3:
            continue
        before_profiles.append(_action_profile(before, 5))
        after_profiles.append(_action_profile(after, 5))
    if not before_profiles:
        return 0.0
    similarities = [
        1.0 - _kl_div(b, a) for b, a in zip(before_profiles, after_profiles)
    ]
    return float(np.clip(np.mean(similarities), 0.0, 1.0))


def _action_profile(actions: Sequence[int], n_actions: int) -> np.ndarray:
    counts = np.zeros(n_actions, dtype=np.float64)
    for a in actions:
        if 0 <= a < n_actions:
            counts[a] += 1
    total = counts.sum()
    if total <= 0:
        return np.ones(n_actions) / n_actions
    return counts / total


def _kl_div(p: np.ndarray, q: np.ndarray) -> float:
    p = np.clip(p, 1e-12, 1.0)
    q = np.clip(q, 1e-12, 1.0)
    return float(np.sum(p * np.log(p / q)))


def self_generated_goals(trace: Sequence[CycleTraceRecord]) -> float:
    """Goal drive switches without extrinsic goal signal (proxy via drive changes)."""
    switches = sum(1 for t in trace if t.goal_switched)
    if len(trace) == 0:
        return 0.0
    rate = switches / len(trace)
    return float(np.clip(rate * 10.0, 0.0, 1.0))


def unexpected_skill_chains(
    full_trace: Sequence[CycleTraceRecord],
    ablated_trace: Sequence[CycleTraceRecord],
    pair_len: int = 2,
) -> float:
    """Action→goal-reached pairs in full but not ablated runs."""
    full_pairs = _action_goal_pairs(full_trace, pair_len)
    ablated_pairs = _action_goal_pairs(ablated_trace, pair_len)
    if not full_pairs:
        return 0.0
    novel = full_pairs - ablated_pairs
    return float(np.clip(len(novel) / len(full_pairs), 0.0, 1.0))


def _action_goal_pairs(trace: Sequence[CycleTraceRecord], n: int) -> Set[Tuple[int, ...]]:
    pairs: Set[Tuple[int, ...]] = set()
    for i in range(len(trace)):
        if trace[i].goal_reached and i >= n:
            key = tuple(t.action for t in trace[i - n : i] if t.action >= 0)
            if len(key) >= n:
                pairs.add(key)
    return pairs


def compute_emergence_bundle(
    trace: Sequence[CycleTraceRecord],
    ablated_trace: Optional[Sequence[CycleTraceRecord]] = None,
) -> Dict[str, float]:
    """Compute all emergence components."""
    bundle = {
        "novel_behaviour": novel_behaviour(trace),
        "strategy_diversity": strategy_diversity(trace),
        "behavioral_compression": behavioral_compression(trace),
        "adaptation_stability": adaptation_stability(trace),
        "cross_context_reuse": cross_context_reuse(trace),
        "self_generated_goals": self_generated_goals(trace),
    }
    if ablated_trace is not None:
        bundle["unexpected_skill_chains"] = unexpected_skill_chains(trace, ablated_trace)
    else:
        bundle["unexpected_skill_chains"] = 0.0
    bundle["emergence_composite"] = float(np.mean(list(bundle.values())))
    return bundle
