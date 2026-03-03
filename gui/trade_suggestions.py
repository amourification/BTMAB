from __future__ import annotations

from typing import Dict, Any
import numpy as np

from risk.stops_utils import MIN_STOP_PCT


def compute_trade_suggestions(trade_plan: Dict[str, Any]) -> Dict[str, float]:
    """
    Derive a simple entry, stop-loss, and exit/target suggestion
    from the existing risk engine outputs.

    - Entry:     stops.entry_price, or last price if missing
    - Stop-loss: active_stop.stop_price
    - Exit:      entry +/- (risk_reward * distance_to_stop)

    All results are best-effort hints for humans, not hard signals.
    """
    engines = trade_plan.get("_engines", {}) or {}
    risk    = trade_plan.get("_risk",    {}) or {}

    stops   = risk.get("stops",     {}) or {}
    port    = risk.get("portfolio", {}) or {}

    detrend = engines.get("detrend", {}) or {}
    trend   = np.array(detrend.get("trend",     []), dtype=float)
    detr    = np.array(detrend.get("detrended", []), dtype=float)
    prices  = trend + detr if len(trend) == len(detr) and len(trend) > 0 else trend

    # --- Entry price ---
    entry = float(stops.get("entry_price", 0.0) or 0.0)
    if entry <= 0 and len(prices) > 0:
        entry = float(prices[-1])

    if entry <= 0:
        return {}

    direction = str(stops.get("direction", "long")).lower()

    # --- Choose a stop that is on the correct side of entry ---
    def _extract_stop(key: str) -> float:
        s = stops.get(key, {}) or {}
        return float(s.get("stop_price", 0.0) or 0.0)

    candidates: list[float] = []
    for key in ("active_stop", "initial_stop", "trailing_stop"):
        sp = _extract_stop(key)
        if sp <= 0:
            continue
        if direction == "short" and sp > entry:
            candidates.append(sp)
        elif direction != "short" and sp < entry:
            candidates.append(sp)

    if direction == "short":
        stop_price = min(candidates) if candidates else 0.0
    else:
        stop_price = max(candidates) if candidates else 0.0

    atr = float(stops.get("atr", 0.0) or 0.0)
    min_dist = abs(entry) * MIN_STOP_PCT

    if stop_price <= 0.0:
        # Fallback: derive a stop purely from ATR / min distance
        base_dist = max(atr, min_dist) if atr > 0 else min_dist
        if direction == "short":
            stop_price = entry + base_dist
        else:
            stop_price = entry - base_dist

    # Ensure final stop is still on the correct side
    if direction == "short" and stop_price <= entry:
        stop_price = entry + max(min_dist, abs(entry - stop_price))
    elif direction != "short" and stop_price >= entry:
        stop_price = entry - max(min_dist, abs(stop_price - entry))

    # --- Risk / reward estimate from portfolio engine ---
    risk_adj = port.get("risk_adjusted", {}) or {}
    rr_raw = float(risk_adj.get("risk_reward", 2.0) or 2.0)
    # Clamp risk/reward to a realistic band to avoid absurd targets
    rr_eff = min(max(rr_raw, 0.5), 3.0)

    # Distance from entry to stop (strictly positive)
    if direction == "short":
        dist = max(stop_price - entry, min_dist)
    else:
        dist = max(entry - stop_price, min_dist)

    # Take-profit ladder in R multiples
    if direction == "short":
        tp1 = entry - 1.0 * dist
        tp2 = entry - min(2.0, rr_eff / 1.5) * dist
        tp3 = entry - rr_eff * dist
    else:
        tp1 = entry + 1.0 * dist
        tp2 = entry + min(2.0, rr_eff / 1.5) * dist
        tp3 = entry + rr_eff * dist

    # Default single "exit" uses the full risk/reward multiple (TP3)
    exit_price = tp3

    return {
        "suggested_entry_price": round(entry, 4),
        "suggested_stop_price": round(stop_price, 4),
        "suggested_exit_price": round(exit_price, 4),
        "take_profit_1": round(tp1, 4),
        "take_profit_2": round(tp2, 4),
        "take_profit_3": round(tp3, 4),
    }

