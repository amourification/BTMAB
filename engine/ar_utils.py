# =============================================================================
#  engine/ar_utils.py — AR Model Internal Helpers
#  Split from ar_model.py to keep both modules under the 300-line limit.
# =============================================================================

import logging
import numpy as np

logger = logging.getLogger("temporal_bot.engine.ar_utils")

NIGHT_SENTIMENT_WEIGHT: float = 1.35


def fit_ar_model(signal: np.ndarray, order: int):
    """Fits statsmodels AutoReg; returns result or None."""
    try:
        from statsmodels.tsa.ar_model import AutoReg
        return AutoReg(signal, lags=order, old_names=False).fit()
    except Exception as exc:
        logger.warning("statsmodels AR fit failed: %s — using numpy fallback.", exc)
        return None


def numpy_ar_fit(signal: np.ndarray, order: int) -> dict:
    """Fallback OLS AR estimation via numpy least squares."""
    n = len(signal)
    if n <= order + 1:
        return {"coeffs": np.zeros(order), "intercept": 0.0, "residuals": signal}
    Y = signal[order:]
    X = np.column_stack([np.ones(n - order),
                         *[signal[order-k-1:n-k-1] for k in range(order)]])
    try:
        coeffs, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
    except np.linalg.LinAlgError:
        return {"coeffs": np.zeros(order), "intercept": 0.0, "residuals": signal}
    residuals = Y - X @ coeffs
    return {"coeffs": coeffs[1:], "intercept": float(coeffs[0]), "residuals": residuals}


def forecast_ar(signal: np.ndarray, ar_params: dict, order: int, steps: int) -> np.ndarray:
    """Generates a multi-step ahead AR forecast via recursive substitution."""
    history   = list(signal[-order:])
    intercept = ar_params.get("intercept", 0.0)
    coeffs    = ar_params.get("coeffs", np.zeros(order))
    forecasts = []
    for _ in range(steps):
        nv = intercept + float(np.dot(coeffs, history[-order:][::-1]))
        forecasts.append(nv)
        history.append(nv)
    return np.array(forecasts)


def split_by_session(signal: np.ndarray, sessions: np.ndarray) -> dict:
    """Splits signal into day/night sub-series with summary stats."""
    def stats(arr):
        if len(arr) == 0:
            return {"mean": 0, "std": 0, "n": 0, "autocorr_lag1": 0}
        ac = float(np.corrcoef(arr[:-1], arr[1:])[0, 1]) if len(arr) > 2 else 0
        return {"mean": float(arr.mean()), "std": float(arr.std()),
                "n": int(len(arr)), "autocorr_lag1": round(ac, 4)}
    day_vals   = signal[sessions == "day"]
    night_vals = signal[sessions == "night"]
    return {"day":   {"values": day_vals,   "stats": stats(day_vals)},
            "night": {"values": night_vals, "stats": stats(night_vals)}}


def inertia_score(session_split: dict, weight: float) -> dict:
    """Computes day/night inertia asymmetry from lag-1 autocorrelation."""
    day_ac   = session_split["day"]["stats"]["autocorr_lag1"]
    night_ac = session_split["night"]["stats"]["autocorr_lag1"]
    wn       = night_ac * weight
    dominant = "night" if abs(wn) > abs(day_ac) else "day"
    return {"day_inertia": round(day_ac, 4), "night_inertia": round(night_ac, 4),
            "night_inertia_weighted": round(wn, 4), "dominant_session": dominant,
            "asymmetry": round(abs(wn) - abs(day_ac), 4)}


def blend_sentiment(forecast: np.ndarray, score: float, session: str, weight: float) -> np.ndarray:
    """Adjusts AR forecast by sentiment score; night session is upweighted."""
    if len(forecast) == 0:
        return forecast
    mag     = float(np.abs(forecast).mean())
    adj_wt  = weight if session == "night" else 1.0
    return forecast + score * mag * 0.10 * adj_wt
