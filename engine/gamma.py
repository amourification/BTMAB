# =============================================================================
#  engine/gamma.py — Equation 8: Gamma Exposure & Volatility Hedging
#  Detects Negative Gamma regimes and triggers hedge recommendations.
#  Helpers live in gamma_utils.py.
#
#  Standard interface:
#      result = run(prices, config) -> dict
# =============================================================================

import logging
import numpy as np
from engine.gamma_utils import (
    realized_volatility, gamma_proxy as compute_gamma_proxy,
    detect_regime, hedge_recommendation, vol_surface_stats,
    LOOKBACK_REALIZED_VOL, GAMMA_HEDGE_THRESHOLD, DEFAULT_HEDGE_RATIO,
)

logger = logging.getLogger("temporal_bot.engine.gamma")


def run(prices: np.ndarray, cfg: dict) -> dict:
    """
    Computes Gamma proxy, detects regime, and generates hedge recommendations.

    Parameters
    ----------
    prices : np.ndarray — closing prices
    cfg    : dict — "GAMMA_HEDGE_THRESHOLD", "DEFAULT_HEDGE_RATIO",
                    "HILBERT_TURN_TYPE"

    Returns
    -------
    dict — success, vol_series, gamma_proxy, regime, vol_stats,
           hedge_recommendation, confidence, metadata, error
    """
    _empty = {
        "success": False, "vol_series": np.array([]),
        "gamma_proxy": np.array([]), "regime": {}, "vol_stats": {},
        "hedge_recommendation": {}, "confidence": 0.0,
        "metadata": {}, "error": None,
    }

    if prices is None or len(prices) < LOOKBACK_REALIZED_VOL + 5:
        _empty["error"] = (
            f"Gamma: need >= {LOOKBACK_REALIZED_VOL + 5} prices, "
            f"got {len(prices) if prices is not None else 0}."
        )
        logger.error(_empty["error"])
        return _empty

    threshold   = float(cfg.get("GAMMA_HEDGE_THRESHOLD", GAMMA_HEDGE_THRESHOLD))
    hedge_ratio = float(cfg.get("DEFAULT_HEDGE_RATIO",   DEFAULT_HEDGE_RATIO))
    turn_type   = str(cfg.get("HILBERT_TURN_TYPE",       "unknown"))

    try:
        vol_series   = realized_volatility(prices, LOOKBACK_REALIZED_VOL)
        gamma_proxy  = compute_gamma_proxy(prices, vol_series)
        regime       = detect_regime(gamma_proxy, threshold)
        vol_stats    = vol_surface_stats(vol_series)
        hedge_rec    = hedge_recommendation(
            regime["regime"], regime["regime_strength"], turn_type, hedge_ratio
        )

        if regime["regime"] == "negative_gamma":
            confidence = round(0.60 + regime["regime_strength"] * 0.40, 4)
        elif regime["regime"] == "positive_gamma":
            confidence = round(0.55 + regime["regime_strength"] * 0.25, 4)
        else:
            confidence = 0.45
        confidence = max(0.0, min(1.0, confidence))

        logger.info(
            "Gamma OK: regime=%s strength=%.3f hedge=%.0f%% confidence=%.3f",
            regime["regime"], regime["regime_strength"],
            hedge_rec["hedge_pct"], confidence,
        )

        return {
            "success": True, "vol_series": vol_series,
            "gamma_proxy": gamma_proxy, "regime": regime,
            "vol_stats": vol_stats, "hedge_recommendation": hedge_rec,
            "confidence": confidence,
            "metadata": {"threshold": threshold, "hedge_ratio": hedge_ratio},
            "error": None,
        }

    except Exception as exc:
        msg = f"{type(exc).__name__}: {exc}"
        logger.exception("Gamma failed: %s", msg)
        _empty["error"] = msg
        return _empty


if __name__ == "__main__":
    np.random.seed(5)
    prices = np.cumsum(np.random.randn(200) * 500) + 50000
    r = run(prices, {"HILBERT_TURN_TYPE": "distribution"})
    if r["success"]:
        print(f"✅ Gamma OK | regime={r['regime']['regime']} | hedge={r['hedge_recommendation']['hedge_pct']}%")
    else:
        print(f"❌ {r['error']}")
