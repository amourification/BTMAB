# =============================================================================
#  engine/murray_utils.py — Murray Math Internal Helpers
#  Split from murray.py to keep both modules under the 300-line limit.
# =============================================================================

import numpy as np

MURRAY_LABELS = [
    "0/8 — Ultimate Support (Major Low)",
    "1/8 — Weak, Sell/Cover Longs",
    "2/8 — Strong Support (Buy Here)",
    "3/8 — Lower Pivot (Buy Zone)",
    "4/8 — Major Pivot (Mid — Trend Decider)",
    "5/8 — Upper Pivot (Sell Zone)",
    "6/8 — Strong Resistance (Sell Here)",
    "7/8 — Weak, Dangerous to Buy",
    "8/8 — Ultimate Resistance (Major Top)",
]

MURRAY_ACTIONS = {
    0: "Strong Buy — Ultimate Support",
    1: "Buy — Oversold bounce likely",
    2: "Buy — Strong support zone",
    3: "Buy — Lower pivot support",
    4: "Neutral — Watch for breakout direction",
    5: "Sell — Upper pivot resistance",
    6: "Sell — Strong resistance zone",
    7: "Caution — Dangerous to buy here",
    8: "Sell / Hedge — Ultimate Resistance",
}


def find_octave_range(high: float, low: float) -> tuple[float, float]:
    """Finds the smallest power-of-2 range containing high and low."""
    R = high - low
    if R <= 0:
        R = high * 0.01
    power = 1.0
    while power < R:
        power *= 2
    floor_val   = np.floor(low / power) * power
    octave_high = floor_val + power
    while octave_high < high:
        octave_high += power
    return float(floor_val), float(octave_high)


def compute_levels(octave_low: float, octave_high: float) -> np.ndarray:
    """Returns 9 price levels from 0/8 to 8/8."""
    return np.linspace(octave_low, octave_high, 9)


def price_to_murray_index(price: float, levels: np.ndarray) -> float:
    """Returns fractional Murray index [0, 8] for current price."""
    lo, hi = levels[0], levels[-1]
    rng    = hi - lo
    if rng < 1e-12:
        return 4.0
    return float(np.clip((price - lo) / rng * 8, 0, 8))


def nearest_levels(price: float, levels: np.ndarray) -> dict:
    """Finds nearest Murray support and resistance levels."""
    below = [(i, lvl) for i, lvl in enumerate(levels) if lvl <= price]
    above = [(i, lvl) for i, lvl in enumerate(levels) if lvl > price]
    sup  = below[-1] if below else (0, levels[0])
    res  = above[0]  if above else (8, levels[-1])
    return {
        "support_index":       sup[0],
        "support_price":       round(sup[1], 4),
        "support_label":       MURRAY_LABELS[sup[0]],
        "support_pct_away":    round(abs(price - sup[1]) / (price + 1e-12) * 100, 2),
        "resistance_index":    res[0],
        "resistance_price":    round(res[1], 4),
        "resistance_label":    MURRAY_LABELS[res[0]],
        "resistance_pct_away": round(abs(res[1] - price) / (price + 1e-12) * 100, 2),
    }


def cycle_confluence(murray_index: float, turn_type: str) -> dict:
    """Checks Murray level + Hilbert turn type for confluence."""
    score, signal = 0.0, "neutral"
    if murray_index >= 6.5:
        if turn_type in ("distribution", "mid_expansion"):
            score, signal = 0.85, "high_confluence_sell"
        else:
            score, signal = 0.40, "overbought_watch"
    elif murray_index <= 1.5:
        if turn_type in ("accumulation", "early_bullish"):
            score, signal = 0.85, "high_confluence_buy"
        else:
            score, signal = 0.40, "oversold_watch"
    elif 3.5 <= murray_index <= 4.5:
        score, signal = 0.50, "pivot_breakout_watch"
    return {"confluence_score": round(score, 3), "signal": signal,
            "murray_index": round(murray_index, 2), "turn_type": turn_type}


def level_series(prices: np.ndarray, levels: np.ndarray) -> np.ndarray:
    """Maps each price bar to its fractional Murray index."""
    return np.array([price_to_murray_index(p, levels) for p in prices])
