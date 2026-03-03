# =============================================================================
#  engine/gann_utils.py — Gann Angle Internal Helpers
#  Split from gann.py to keep both modules under the 300-line limit.
# =============================================================================

import numpy as np

GANN_SCALES = [(1,8),(1,4),(1,2),(1,1),(2,1),(4,1),(8,1)]
GANN_SCALE_LABELS = {
    (1,8): "1×8 — Weakest Bull", (1,4): "1×4 — Moderate Bull",
    (1,2): "1×2 — Strong Bull",  (1,1): "1×1 — Master Angle (45°)",
    (2,1): "2×1 — Strong Bear",  (4,1): "4×1 — Rapid Bear",
    (8,1): "8×1 — Crash Angle",
}


def find_swing_point(prices: np.ndarray, lookback: int) -> dict:
    """Finds the most recent significant high/low as the Gann anchor."""
    window = prices[-lookback:] if len(prices) >= lookback else prices
    offset = len(prices) - len(window)
    hi_idx = int(np.argmax(window)) + offset
    lo_idx = int(np.argmin(window)) + offset
    if hi_idx >= lo_idx:
        return {"anchor_idx": hi_idx, "anchor_val": float(prices[hi_idx]),
                "anchor_type": "high",
                "swing_high": {"idx": hi_idx, "val": float(prices[hi_idx])},
                "swing_low":  {"idx": lo_idx, "val": float(prices[lo_idx])}}
    return {"anchor_idx": lo_idx, "anchor_val": float(prices[lo_idx]),
            "anchor_type": "low",
            "swing_high": {"idx": hi_idx, "val": float(prices[hi_idx])},
            "swing_low":  {"idx": lo_idx, "val": float(prices[lo_idx])}}


def compute_price_scale(prices: np.ndarray, lookback: int) -> float:
    """Price unit = 1/8th of lookback range for Gann angle calibration."""
    window = prices[-lookback:] if len(prices) >= lookback else prices
    rng    = float(window.max() - window.min())
    if rng < 1e-12:
        rng = float(window.mean()) * 0.01
    return rng / 8.0


def gann_angle_value(anchor_val, anchor_idx, bar_idx, price_unit, p, t, anchor_type):
    """Price value of a Gann angle at bar_idx."""
    bars = bar_idx - anchor_idx
    if bars < 0:
        return np.nan
    slope = (p / t) * price_unit
    return float(anchor_val - slope * bars) if anchor_type == "high" else float(anchor_val + slope * bars)


def build_angle_fan(prices: np.ndarray, anchor: dict, price_unit: float) -> list:
    """Builds the 7-angle Gann fan from the anchor point."""
    N, ai, av, at = len(prices), anchor["anchor_idx"], anchor["anchor_val"], anchor["anchor_type"]
    fan = []
    for (p, t) in GANN_SCALES:
        values  = np.array([gann_angle_value(av, ai, i, price_unit, p, t, at) for i in range(N)])
        current = float(values[-1]) if not np.isnan(values[-1]) else None
        fan.append({"scale": (p, t), "label": GANN_SCALE_LABELS[(p,t)],
                    "values": values, "current_value": round(current, 2) if current else None,
                    "angle_deg": round(float(np.degrees(np.arctan(p/t))), 1)})
    return fan


def find_nearest_angle(price: float, fan: list) -> dict:
    """Nearest angle above and below current price."""
    above, below = [], []
    for a in fan:
        v = a["current_value"]
        if v is None:
            continue
        (above if v > price else below).append((abs(v - price), a["label"], v))
    above.sort(); below.sort()
    return {
        "nearest_above": {"label": above[0][1], "price": above[0][2]} if above else None,
        "nearest_below": {"label": below[0][1], "price": below[0][2]} if below else None,
    }


def check_angle_breaks(prices: np.ndarray, fan: list, lookback_bars: int = 5) -> list:
    """Detects recent Gann angle crossings (breaks)."""
    breaks = []
    recent_prices = prices[-(lookback_bars + 1):]
    for angle in fan:
        vals = angle["values"]
        if vals is None or len(vals) < lookback_bars + 1:
            continue
        recent_angle = vals[-(lookback_bars + 1):]
        if any(np.isnan(recent_angle)):
            continue
        price_above = recent_prices > recent_angle
        for i in range(1, len(price_above)):
            if price_above[i] != price_above[i-1]:
                breaks.append({"angle": angle["label"],
                                "direction": "bullish_break" if price_above[i] else "bearish_break",
                                "bars_ago": lookback_bars - i})
                break
    return breaks
