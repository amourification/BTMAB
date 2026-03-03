from __future__ import annotations

"""
Elliott Wave / Fibonacci helper engine.

This is intentionally lightweight and deterministic:
it does NOT attempt full subjective wave labelling,
but instead:

- Detects recent swing high/low using the same logic
  as the Gann module.
- Treats that last swing as the "impulse leg".
- Computes Fibonacci retracement levels on that leg.
- Projects simple Fibonacci extension targets in the
  direction of the impulse.

The result is used by a dedicated GUI chart to give
visually clear "headshot" target zones.
"""

from dataclasses import dataclass
from typing import Dict, Any

import numpy as np

from engine.gann_utils import find_swing_point
from engine.elliott_wave_utils import (
    find_primary_wave_count,
    compute_rsi,
    compute_macd,
    detect_divergences,
)


FIB_RETRACEMENTS = [0.236, 0.382, 0.5, 0.618, 0.786]
FIB_EXTENSIONS = [1.272, 1.618, 2.0]


@dataclass
class FibResult:
    direction: str
    swing_start_idx: int
    swing_end_idx: int
    swing_start_price: float
    swing_end_price: float
    retracements: Dict[str, float]
    extensions: Dict[str, float]


def _compute_fib_levels(
    start_price: float,
    end_price: float,
    direction: str,
) -> FibResult:
    """Compute retracement and extension levels for a single impulse leg."""
    direction = direction.lower()
    up = direction == "up"

    leg = end_price - start_price
    if abs(leg) < 1e-8:
        leg = 1e-8

    retr = {}
    for r in FIB_RETRACEMENTS:
        if up:
            level = end_price - leg * r
        else:
            level = end_price + leg * r
        retr[f"{int(r*100)}"] = round(level, 4)

    exts = {}
    for e in FIB_EXTENSIONS:
        if up:
            level = end_price + leg * (e - 1.0)
        else:
            level = end_price - leg * (e - 1.0)
        exts[f"{e:.3g}"] = round(level, 4)

    return FibResult(
        direction="up" if up else "down",
        swing_start_idx=0,  # filled by caller
        swing_end_idx=0,
        swing_start_price=start_price,
        swing_end_price=end_price,
        retracements=retr,
        extensions=exts,
    )


def run(prices: np.ndarray, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Core Elliott/Fibonacci helper.

    Parameters
    ----------
    prices : np.ndarray
        Close price series.
    cfg    : dict
        Config dict (uses GANN_LOOKBACK if present).
    """
    result: Dict[str, Any] = {
        "success": False,
        "swing_points": {},
        "fib_retracements": {},
        "fib_extensions": {},
        "direction": "neutral",
        "wave_count": None,
        "rsi": [],
        "macd_line": [],
        "macd_signal": [],
        "macd_hist": [],
        "rsi_divergences": [],
        "macd_divergences": [],
        "time_events": [],
        "error": None,
    }

    try:
        if prices is None or len(prices) < 10:
            result["error"] = "Not enough prices for Elliott/Fib analysis."
            return result

        lookback = int(cfg.get("GANN_LOOKBACK", 128))
        swing = find_swing_point(prices, lookback)
        start = swing["swing_low"]
        end = swing["swing_high"]

        # Decide impulse direction as last completed move between low and high
        if start["idx"] < end["idx"]:
            direction = "up"
            start_idx, end_idx = start["idx"], end["idx"]
        else:
            direction = "down"
            start_idx, end_idx = end["idx"], start["idx"]

        start_price = float(prices[start_idx])
        end_price = float(prices[end_idx])

        fib = _compute_fib_levels(start_price, end_price, direction)
        fib.swing_start_idx = start_idx
        fib.swing_end_idx = end_idx

        # Elliott wave counting (rules + guidelines)
        wave = find_primary_wave_count(prices)
        wave_payload: Dict[str, Any] | None = None
        if wave is not None and wave.valid:
            wave_payload = {
                "impulse_labels": wave.impulse_labels,
                "correction_labels": wave.correction_labels,
                "direction": wave.direction,
                "score": wave.score,
            }

        # Oscillators and divergences
        rsi = compute_rsi(prices, period=14)
        macd_line, macd_signal, macd_hist = compute_macd(prices)
        rsi_div = detect_divergences(prices, rsi)
        macd_div = detect_divergences(prices, macd_line)

        # Simple forward-looking time events in bars (to be mapped to
        # calendar timestamps by chart code using fetch timestamps).
        dominant = float(
            cfg.get("DOMINANT_CYCLE_BARS")
            or cfg.get("FFT_PRIMARY_PERIOD")
            or cfg.get("SSA_PERIOD")
            or 0.0
        )
        time_events = []
        if dominant > 0:
            time_events.append(
                {
                    "source": "elliott",
                    "kind": "wave_pivot",
                    "bars_ahead": float(dominant),
                    "label": "Next Elliott wave pivot",
                }
            )
            # Oscillator-based timing windows (approximate half-cycle extremes)
            half = max(dominant / 2.0, 1.0)
            time_events.append(
                {
                    "source": "rsi",
                    "kind": "cycle_extreme",
                    "bars_ahead": float(half),
                    "label": "Next RSI cycle extreme",
                }
            )
            time_events.append(
                {
                    "source": "macd",
                    "kind": "cycle_extreme",
                    "bars_ahead": float(half),
                    "label": "Next MACD cycle extreme",
                }
            )

        result.update(
            {
                "success": True,
                "direction": fib.direction,
                "swing_points": {
                    "start_idx": start_idx,
                    "end_idx": end_idx,
                    "start_price": round(start_price, 4),
                    "end_price": round(end_price, 4),
                },
                "fib_retracements": fib.retracements,
                "fib_extensions": fib.extensions,
                "wave_count": wave_payload,
                "rsi": rsi.tolist(),
                "macd_line": macd_line.tolist(),
                "macd_signal": macd_signal.tolist(),
                "macd_hist": macd_hist.tolist(),
                "rsi_divergences": rsi_div,
                "macd_divergences": macd_div,
                "time_events": time_events,
            }
        )
        return result

    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        return result

