from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Dict, Any

import numpy as np


@dataclass
class TimeAnglePrediction:
    angle_deg: float
    bars_ahead: float
    timestamp: datetime


def _infer_bar_timedelta(timestamps: np.ndarray) -> float:
    """
    Infer bar duration from timestamps (int or datetime64).
    Returns seconds per bar.
    """
    if len(timestamps) < 2:
        return 0.0

    if np.issubdtype(timestamps.dtype, np.datetime64):
        dt = (timestamps[-1] - timestamps[-2]) / np.timedelta64(1, "s")
        return float(dt)

    # Numeric timestamps: try to guess unit.
    # Typical Unix epoch ranges:
    #   - seconds:     ~1e9
    #   - milliseconds:~1e12
    #   - nanoseconds: ~1e18
    t0, t1 = float(timestamps[-2]), float(timestamps[-1])
    delta = max(t1 - t0, 1.0)
    if t1 > 1e15:  # nanoseconds
        return delta / 1e9
    if t1 > 1e11:  # milliseconds
        return delta / 1e3
    # assume seconds
    return delta


def _ts_to_datetime(value: Any) -> datetime:
    """Convert numeric or datetime64 timestamp to UTC datetime."""
    if isinstance(value, datetime):
        return value
    arr = np.asarray([value])
    if np.issubdtype(arr.dtype, np.datetime64):
        ts = arr.astype("datetime64[ns]")[0]
        seconds = float(ts.astype("int64")) / 1e9
        return datetime.fromtimestamp(seconds, tz=timezone.utc)

    v = float(value)
    # Same magnitude heuristic as _infer_bar_timedelta
    if v > 1e15:  # nanoseconds
        return datetime.fromtimestamp(v / 1e9, tz=timezone.utc)
    if v > 1e11:  # milliseconds
        return datetime.fromtimestamp(v / 1e3, tz=timezone.utc)
    return datetime.fromtimestamp(v, tz=timezone.utc)


def compute_time_angle_predictions(
    trade_plan: Dict[str, Any],
    timestamps: np.ndarray,
    max_events: int = 4,
) -> List[TimeAnglePrediction]:
    """
    Compute forward-looking time-based cycle angle events
    (e.g., next 0°, 90°, 180°, 270° crossings) using the
    current Hilbert phase and dominant cycle length.
    """
    if timestamps is None or len(timestamps) == 0:
        return []

    phase_deg = float(trade_plan.get("phase_deg", 0.0))
    dominant = float(trade_plan.get("dominant_cycle_bars", 0.0) or 0.0)
    if dominant <= 0:
        # Fallback: try SSA or FFT period
        ssa_p = float(trade_plan.get("ssa_period", 0.0) or 0.0)
        if ssa_p > 0:
            dominant = ssa_p

    if dominant <= 0:
        return []

    bar_secs = _infer_bar_timedelta(timestamps)
    if bar_secs <= 0:
        return []

    current_ts = timestamps[-1]
    current_dt = _ts_to_datetime(current_ts)

    per_bar_deg = 360.0 / dominant
    if per_bar_deg <= 0:
        return []

    # Target cycle "angles" ahead of the current phase
    targets = [0.0, 90.0, 180.0, 270.0, 360.0]
    preds: List[TimeAnglePrediction] = []

    for tgt in targets:
        delta = (tgt - phase_deg) % 360.0
        if delta == 0.0:
            delta = 360.0
        bars_ahead = delta / per_bar_deg
        if bars_ahead <= 0:
            continue
        secs_ahead = bars_ahead * bar_secs
        ts_future = current_dt.timestamp() + secs_ahead
        dt_future = datetime.fromtimestamp(ts_future, tz=timezone.utc)
        preds.append(TimeAnglePrediction(angle_deg=tgt, bars_ahead=bars_ahead, timestamp=dt_future))

    preds.sort(key=lambda p: p.bars_ahead)
    return preds[:max_events]

