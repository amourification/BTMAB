# =============================================================================
#  engine/murray.py — Equation 6: Murray Math Price Geometry
#  Divides price ranges into 1/8th increments acting as mathematical
#  support and resistance zones. Helpers live in murray_utils.py.
#
#  Standard interface:
#      result = run(prices, config) -> dict
# =============================================================================

import logging
import numpy as np
from engine.murray_utils import (
    find_octave_range, compute_levels, price_to_murray_index,
    nearest_levels, cycle_confluence, level_series,
    MURRAY_LABELS, MURRAY_ACTIONS,
)

logger = logging.getLogger("temporal_bot.engine.murray")
MURRAY_LOOKBACK: int = 64


def run(prices: np.ndarray, cfg: dict) -> dict:
    """
    Computes Murray Math levels and price position relative to them.

    Parameters
    ----------
    prices : np.ndarray — raw closing prices
    cfg    : dict — "MURRAY_LOOKBACK" (int), "HILBERT_TURN_TYPE" (str)

    Returns
    -------
    dict — success, octave_low/high, levels, level_labels, level_actions,
           current_price, murray_index, nearest_levels, level_series,
           confluence, recommended_action, confidence, metadata, error
    """
    _empty = {
        "success": False, "octave_low": 0.0, "octave_high": 0.0,
        "levels": np.array([]), "level_labels": MURRAY_LABELS,
        "level_actions": MURRAY_ACTIONS, "current_price": 0.0,
        "murray_index": 0.0, "nearest_levels": {}, "level_series": np.array([]),
        "confluence": {}, "recommended_action": "No action",
        "confidence": 0.0, "metadata": {}, "error": None,
    }

    if prices is None or len(prices) < 2:
        _empty["error"] = f"Murray: need >= 2 prices, got {len(prices) if prices is not None else 0}."
        logger.error(_empty["error"])
        return _empty

    lookback  = int(cfg.get("MURRAY_LOOKBACK", MURRAY_LOOKBACK))
    turn_type = str(cfg.get("HILBERT_TURN_TYPE", "unknown"))
    window    = prices[-lookback:] if len(prices) >= lookback else prices

    try:
        high  = float(window.max())
        low   = float(window.min())
        price = float(prices[-1])

        octave_low, octave_high = find_octave_range(high, low)
        levels       = compute_levels(octave_low, octave_high)
        murray_idx   = price_to_murray_index(price, levels)
        nearest      = nearest_levels(price, levels)
        lvl_series   = level_series(window, levels)
        confluence   = cycle_confluence(murray_idx, turn_type)

        level_int   = max(0, min(8, int(round(murray_idx))))
        rec_action  = MURRAY_ACTIONS[level_int]

        # Confidence: highest at extremes (0/8, 8/8); boost for confluence
        extreme_prox = max(murray_idx / 8, 1 - murray_idx / 8)
        confidence   = round(float(0.5 + 0.5 * extreme_prox), 4)
        if confluence["confluence_score"] > 0.7:
            confidence = round(min(1.0, confidence + 0.15), 4)

        logger.info(
            "Murray OK: price=%.2f index=%.2f/8 support=%.2f resist=%.2f "
            "confluence=%s confidence=%.3f",
            price, murray_idx, nearest["support_price"],
            nearest["resistance_price"], confluence["signal"], confidence,
        )

        return {
            "success": True, "octave_low": octave_low, "octave_high": octave_high,
            "levels": levels, "level_labels": MURRAY_LABELS,
            "level_actions": MURRAY_ACTIONS, "current_price": price,
            "murray_index": murray_idx, "nearest_levels": nearest,
            "level_series": lvl_series, "confluence": confluence,
            "recommended_action": rec_action, "confidence": confidence,
            "metadata": {"lookback": lookback, "high": high, "low": low},
            "error": None,
        }

    except Exception as exc:
        msg = f"{type(exc).__name__}: {exc}"
        logger.exception("Murray failed: %s", msg)
        _empty["error"] = msg
        return _empty


if __name__ == "__main__":
    np.random.seed(1)
    prices = np.linspace(40000, 70000, 100) + np.random.randn(100) * 2000
    r = run(prices, {"MURRAY_LOOKBACK": 64, "HILBERT_TURN_TYPE": "distribution"})
    if r["success"]:
        print(f"✅ Murray OK | index={r['murray_index']:.2f}/8 | {r['confluence']['signal']}")
    else:
        print(f"❌ {r['error']}")
