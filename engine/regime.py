from __future__ import annotations

from typing import Dict, Any

import logging
import numpy as np

logger = logging.getLogger("temporal_bot.engine.regime")


def _rolling_vol(prices: np.ndarray, window: int = 20) -> float:
    if prices is None or len(prices) < window + 1:
        return 0.0
    rets = np.diff(np.log(prices[-(window + 1) :]))
    if rets.size == 0:
        return 0.0
    return float(np.std(rets) * np.sqrt(252.0))


def _atr_like(high: np.ndarray, low: np.ndarray, close: np.ndarray, window: int = 14) -> float:
    n = min(len(high), len(low), len(close))
    if n < window + 1:
        return 0.0
    h = high[-n:]
    l = low[-n:]
    c = close[-n:]
    tr = np.maximum(h[1:], c[:-1]) - np.minimum(l[1:], c[:-1])
    return float(np.mean(tr[-window:]))


def run(prices: np.ndarray, detrended: np.ndarray, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Lightweight volatility / trend regime detector.

    Outputs:
      - vol_regime: 'low' | 'normal' | 'high' | 'very_high'
      - trend_regime: 'up' | 'down' | 'sideways'
      - stress_flag: bool
      - stress_score: 0..1
    """
    result: Dict[str, Any] = {
        "success": False,
        "vol_regime": "normal",
        "trend_regime": "sideways",
        "stress_flag": False,
        "stress_score": 0.0,
    }

    try:
        if prices is None or len(prices) < 32:
            logger.debug("Regime: insufficient data (len=%s)", 0 if prices is None else len(prices))
            return result

        prices = np.asarray(prices, dtype=float)
        detrended = np.asarray(detrended, dtype=float) if detrended is not None else prices - np.mean(prices)

        vol_20 = _rolling_vol(prices, window=20)
        vol_60 = _rolling_vol(prices, window=60)

        # Classify volatility regime based on recent vs longer-term vol
        ratio = vol_20 / (vol_60 + 1e-12)
        if ratio < 0.6:
            vol_regime = "low"
        elif ratio < 1.2:
            vol_regime = "normal"
        elif ratio < 1.8:
            vol_regime = "high"
        else:
            vol_regime = "very_high"

        # Trend regime via simple slope of trend component (or price proxy)
        x = np.arange(len(detrended), dtype=float)
        y = prices
        if len(x) >= 10:
            # Least-squares slope on the last 100 bars (or full series)
            w = min(100, len(x))
            xs = x[-w:]
            ys = y[-w:]
            xs = xs - xs.mean()
            denom = np.sum(xs ** 2) or 1.0
            slope = float(np.sum(xs * (ys - ys.mean())) / denom)
        else:
            slope = 0.0

        thr = np.std(prices[-min(len(prices), 100) :]) * 0.001
        if slope > thr:
            trend_regime = "up"
        elif slope < -thr:
            trend_regime = "down"
        else:
            trend_regime = "sideways"

        # Simple stress metric via ATR-like range vs price
        high = cfg.get("_REGIME_HIGH")  # optional pre-supplied arrays
        low = cfg.get("_REGIME_LOW")
        if isinstance(high, np.ndarray) and isinstance(low, np.ndarray):
            atr = _atr_like(high, low, prices)
        else:
            # Fallback: use high-low within price series as proxy
            atr = float(np.max(prices[-20:]) - np.min(prices[-20:])) if len(prices) >= 20 else 0.0

        level = float(prices[-1])
        atr_ratio = atr / (abs(level) + 1e-12)
        stress_score = max(0.0, min(1.0, atr_ratio * 50.0))  # heuristic scaling
        stress_flag = stress_score > 0.7 or vol_regime in ("high", "very_high")

        result.update(
            {
                "success": True,
                "vol_regime": vol_regime,
                "trend_regime": trend_regime,
                "stress_flag": stress_flag,
                "stress_score": round(stress_score, 3),
                "volatility_20d": vol_20,
                "volatility_60d": vol_60,
            }
        )
        logger.debug(
            "Regime: vol=%s trend=%s stress=%.3f flag=%s",
            vol_regime,
            trend_regime,
            stress_score,
            stress_flag,
        )
        return result
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        return result

