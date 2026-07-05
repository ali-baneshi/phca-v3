"""Statistical aggregation and significance tests for experiments."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


def seed_sequence(base_seed: int, n: int) -> List[int]:
    """Standardized seed list: base_seed, base_seed+1, ..."""
    return [base_seed + i for i in range(n)]


def mean_std(values: Sequence[float]) -> Dict[str, float]:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "n": 0}
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)) if arr.size > 1 else 0.0,
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "n": int(arr.size),
        "values": [float(v) for v in arr],
    }


def bootstrap_ci(
    values: Sequence[float],
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Tuple[float, float]:
    """95% bootstrap confidence interval for the mean."""
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return 0.0, 0.0
    if arr.size == 1:
        v = float(arr[0])
        return v, v
    rng = np.random.RandomState(seed)
    means = []
    for _ in range(n_bootstrap):
        sample = rng.choice(arr, size=arr.size, replace=True)
        means.append(float(np.mean(sample)))
    lo = float(np.percentile(means, 100 * alpha / 2))
    hi = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return lo, hi


def mann_whitney_u(
    sample_a: Sequence[float],
    sample_b: Sequence[float],
) -> Dict[str, Any]:
    """Two-sided Mann-Whitney U test (non-parametric)."""
    a = np.asarray(list(sample_a), dtype=np.float64)
    b = np.asarray(list(sample_b), dtype=np.float64)
    if a.size == 0 or b.size == 0:
        return {"u_statistic": None, "p_value": None, "significant_005": False}

    try:
        from scipy import stats
        u_stat, p_value = stats.mannwhitneyu(a, b, alternative="two-sided")
        return {
            "u_statistic": float(u_stat),
            "p_value": float(p_value),
            "significant_005": bool(p_value < 0.05),
            "a_mean": float(np.mean(a)),
            "b_mean": float(np.mean(b)),
        }
    except ImportError:
        # Fallback: rank-sum approximation without scipy
        combined = np.concatenate([a, b])
        ranks = _rankdata(combined)
        n_a = a.size
        r_a = float(np.sum(ranks[:n_a]))
        u_a = r_a - n_a * (n_a + 1) / 2
        u_b = n_a * b.size - u_a
        u_stat = min(u_a, u_b)
        mu = n_a * b.size / 2
        sigma = np.sqrt(n_a * b.size * (n_a + b.size + 1) / 12)
        if sigma < 1e-12:
            p_value = 1.0
        else:
            z = abs(u_stat - mu) / sigma
            p_value = 2 * (1 - _normal_cdf(z))
        return {
            "u_statistic": float(u_stat),
            "p_value": float(p_value),
            "significant_005": bool(p_value < 0.05),
            "a_mean": float(np.mean(a)),
            "b_mean": float(np.mean(b)),
        }


def _rankdata(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    ranks = np.empty(len(x), dtype=np.float64)
    ranks[order] = np.arange(1, len(x) + 1)
    return ranks


def _normal_cdf(z: float) -> float:
    return 0.5 * (1 + np.tanh(z * 0.7978845608))


def aggregate_runs(
    runs: List[Dict[str, float]],
    metric_keys: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Aggregate per-run metric dicts into mean/std/CI."""
    if not runs:
        return {}
    keys = metric_keys or list(runs[0].keys())
    out: Dict[str, Any] = {}
    for key in keys:
        values = [float(r[key]) for r in runs if key in r]
        stats = mean_std(values)
        lo, hi = bootstrap_ci(values)
        stats["ci95_lo"] = lo
        stats["ci95_hi"] = hi
        out[key] = stats
    return out


def compare_groups(
    group_a: Sequence[float],
    group_b: Sequence[float],
    higher_is_better: bool = True,
) -> Dict[str, Any]:
    """Compare two metric groups with Mann-Whitney and direction check."""
    mw = mann_whitney_u(group_a, group_b)
    a_mean = float(np.mean(group_a)) if len(group_a) else 0.0
    b_mean = float(np.mean(group_b)) if len(group_b) else 0.0
    if higher_is_better:
        direction_ok = a_mean > b_mean
    else:
        direction_ok = a_mean < b_mean
    return {
        **mw,
        "direction_ok": direction_ok,
        "passed": bool(mw.get("significant_005") and direction_ok),
    }
