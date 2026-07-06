"""Phase-aware RSS leak slope helpers (D-112/D-113; shared by nightly stress + session anomalies)."""
from __future__ import annotations

from typing import Dict, Sequence, Union

import numpy as np

# M3 FIFO caps at 10k episodes (one per cycle); post-M3 steady-state slope
# uses samples after this cycle when total_cycles > M3_CAP_CYCLES (D-134).
M3_CAP_CYCLES = 10_000
# M3 fills to cap ~10k episodes; M4 cap engages ~6700 cycles. Shorter soaks
# legitimately show ~4 KB/cyc fill-phase growth (D-112).
FILL_PHASE_CYCLES = 7000
# Post-cap steady-state tail slope (measured ~1390 B/cyc at 10k soak; D-113).
LEAK_SLOPE_LATE = 1600.0
# Fill-phase runs (<7000 cycles) use a higher threshold (D-112).
LEAK_SLOPE_FILL = 5000.0


def leak_threshold_for_cycles(total_cycles: int) -> tuple[float, str]:
    """Return (threshold B/cyc, gate mode) for a run of ``total_cycles``."""
    if total_cycles < FILL_PHASE_CYCLES:
        return LEAK_SLOPE_FILL, "fill_phase"
    return LEAK_SLOPE_LATE, "post_cap"


def compute_rss_slopes(
    cycle_ids: Sequence[int],
    rss_bytes: Sequence[float],
    *,
    total_cycles: int,
) -> Dict[str, Union[float, str, None]]:
    """Linear-fit RSS vs cycle; late segment uses post-M3 or tail-quarter.

    When ``total_cycles > M3_CAP_CYCLES`` and ≥2 samples exist after the M3
    FIFO cap, late slope is fit on post-M3 samples only (D-134). Otherwise
    post-cap runs use tail-quarter; shorter runs use late-half.

    Returns keys: ``full_slope``, ``late_slope``, ``late_slope_window``,
    ``leak_gate_mode``, ``leak_threshold``, ``sample_count``.
    """
    leak_threshold, leak_gate_mode = leak_threshold_for_cycles(total_cycles)
    xs = np.asarray(cycle_ids, dtype=np.float64)
    ys = np.asarray(rss_bytes, dtype=np.float64)
    sample_count = int(len(xs))
    if sample_count < 2:
        return {
            "full_slope": 0.0,
            "late_slope": 0.0,
            "late_slope_window": "insufficient",
            "leak_gate_mode": leak_gate_mode,
            "leak_threshold": leak_threshold,
            "sample_count": sample_count,
        }
    slope = float(np.polyfit(xs, ys, 1)[0])
    late_slope_window = "late_half"
    if total_cycles > M3_CAP_CYCLES:
        post_m3 = xs > M3_CAP_CYCLES
        if int(np.sum(post_m3)) >= 2:
            late_xs, late_ys = xs[post_m3], ys[post_m3]
            late_slope = float(np.polyfit(late_xs, late_ys, 1)[0])
            late_slope_window = "post_m3"
        elif total_cycles >= FILL_PHASE_CYCLES and len(xs) >= 4:
            tail_start = max(0, (len(xs) * 3) // 4)
            late_xs, late_ys = xs[tail_start:], ys[tail_start:]
            late_slope = (
                float(np.polyfit(late_xs, late_ys, 1)[0])
                if len(late_xs) >= 2 else slope
            )
            late_slope_window = "tail_quarter"
        else:
            half = len(xs) // 2
            late_xs, late_ys = xs[half:], ys[half:]
            late_slope = (
                float(np.polyfit(late_xs, late_ys, 1)[0])
                if len(late_xs) >= 2 else slope
            )
    elif total_cycles >= FILL_PHASE_CYCLES and len(xs) >= 4:
        tail_start = max(0, (len(xs) * 3) // 4)
        late_xs, late_ys = xs[tail_start:], ys[tail_start:]
        late_slope = (
            float(np.polyfit(late_xs, late_ys, 1)[0])
            if len(late_xs) >= 2 else slope
        )
        late_slope_window = "tail_quarter"
    else:
        half = len(xs) // 2
        late_xs, late_ys = xs[half:], ys[half:]
        late_slope = (
            float(np.polyfit(late_xs, late_ys, 1)[0])
            if len(late_xs) >= 2 else slope
        )
    return {
        "full_slope": slope,
        "late_slope": late_slope,
        "late_slope_window": late_slope_window,
        "leak_gate_mode": leak_gate_mode,
        "leak_threshold": leak_threshold,
        "sample_count": sample_count,
    }


def rss_leak_flagged(late_slope: float, total_cycles: int) -> bool:
    """True when late RSS slope exceeds the phase-aware threshold."""
    threshold, _ = leak_threshold_for_cycles(total_cycles)
    return late_slope >= threshold - 1e-3
