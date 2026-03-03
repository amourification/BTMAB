# =============================================================================
#  engine/gann.py — Equation 9: Law of Vibration & Gann Angle Fan
#  Builds the 7-angle Gann fan from the last major swing point.
#  Helpers live in gann_utils.py.
#
#  Standard interface:
#      result = run(prices, config) -> dict
# =============================================================================

import logging
import numpy as np
from engine.gann_utils import (
    find_swing_point, compute_price_scale, build_angle_fan,
    find_nearest_angle, check_angle_breaks,
)

logger = logging.getLogger("temporal_bot.engine.gann")
GANN_LOOKBACK: int = 128


def run(prices: np.ndarray, cfg: dict) -> dict:
    """
    Builds Gann fan and identifies angle support/resistance and breaks.

    Parameters
    ----------
    prices : np.ndarray — closing prices
    cfg    : dict — "GANN_LOOKBACK" (int), "HILBERT_TURN_TYPE" (str)

    Returns
    -------
    dict — success, anchor, price_unit, fan, nearest_angles, angle_breaks,
           master_angle_value, price_vs_master, confidence, metadata, error
    """
    _empty = {
        "success": False, "anchor": {}, "price_unit": 0.0,
        "fan": [], "nearest_angles": {}, "angle_breaks": [],
        "master_angle_value": 0.0, "price_vs_master": "unknown",
        "confidence": 0.0, "metadata": {}, "error": None,
    }

    if prices is None or len(prices) < 10:
        _empty["error"] = f"Gann: need >= 10 prices, got {len(prices) if prices is not None else 0}."
        logger.error(_empty["error"])
        return _empty

    lookback = int(cfg.get("GANN_LOOKBACK", GANN_LOOKBACK))

    try:
        anchor      = find_swing_point(prices, lookback)
        price_unit  = compute_price_scale(prices, lookback)
        fan         = build_angle_fan(prices, anchor, price_unit)
        current     = float(prices[-1])
        nearest     = find_nearest_angle(current, fan)
        breaks      = check_angle_breaks(prices, fan, lookback_bars=5)

        master      = next((a for a in fan if a["scale"] == (1, 1)), None)
        master_val  = master["current_value"] if master else None

        if master_val and not np.isnan(master_val):
            tol = price_unit * 0.5
            price_vs_master = "at" if abs(current - master_val) < tol \
                              else ("above" if current > master_val else "below")
        else:
            price_vs_master = "unknown"

        near_master = master_val and abs(current - master_val) < price_unit * 2
        confidence  = round(min(0.50 + (0.25 if near_master else 0) + (0.20 if breaks else 0), 1.0), 4)

        logger.info(
            "Gann OK: anchor=%s@%.2f master=%.2f breaks=%d confidence=%.3f",
            anchor["anchor_type"], anchor["anchor_val"],
            master_val or 0, len(breaks), confidence,
        )

        return {
            "success": True, "anchor": anchor, "price_unit": price_unit,
            "fan": fan, "nearest_angles": nearest, "angle_breaks": breaks,
            "master_angle_value": master_val, "price_vs_master": price_vs_master,
            "confidence": confidence,
            "metadata": {"lookback": lookback, "n_angles": len(fan)},
            "error": None,
        }

    except Exception as exc:
        msg = f"{type(exc).__name__}: {exc}"
        logger.exception("Gann failed: %s", msg)
        _empty["error"] = msg
        return _empty


if __name__ == "__main__":
    np.random.seed(9)
    t = np.arange(200)
    prices = 50000 + 200*t - 0.5*t**2 + np.random.randn(200)*500
    r = run(prices, {"GANN_LOOKBACK": 128})
    if r["success"]:
        print(f"✅ Gann OK | anchor={r['anchor']['anchor_type']} | vs_master={r['price_vs_master']} | breaks={len(r['angle_breaks'])}")
    else:
        print(f"❌ {r['error']}")
