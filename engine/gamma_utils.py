# =============================================================================
#  engine/gamma_utils.py — Gamma Engine Internal Helpers
#  Split from gamma.py to keep both modules under the 300-line limit.
# =============================================================================

import numpy as np
from scipy.signal import savgol_filter

LOOKBACK_REALIZED_VOL: int   = 20
GAMMA_HEDGE_THRESHOLD: float = -0.02
DEFAULT_HEDGE_RATIO:   float = 0.30


def realized_volatility(prices: np.ndarray, window: int) -> np.ndarray:
    """Rolling annualized realized vol from log returns."""
    log_ret    = np.diff(np.log(prices + 1e-12))
    vol_series = np.full(len(prices), np.nan)
    for i in range(window, len(log_ret) + 1):
        vol_series[i] = float(log_ret[i - window: i].std() * np.sqrt(252))
    return vol_series


def gamma_proxy(prices: np.ndarray, vol_series: np.ndarray) -> np.ndarray:
    """Gamma proxy = -(Δvol / vol). Negative = Negative Gamma regime."""
    gamma = np.full(len(prices), np.nan)
    vol_clean = vol_series.copy()
    vol_clean[np.isnan(vol_clean)] = 0
    d_vol = np.diff(vol_clean, prepend=vol_clean[0])
    with np.errstate(divide="ignore", invalid="ignore"):
        gamma_raw = -d_vol / (vol_clean + 1e-12)
    if len(gamma_raw) >= 7:
        try:
            gamma[:] = savgol_filter(gamma_raw, window_length=7, polyorder=2)
        except Exception:
            gamma[:] = gamma_raw
    else:
        gamma[:] = gamma_raw
    return gamma


def detect_regime(gamma_proxy_arr: np.ndarray, threshold: float) -> dict:
    """Classifies current Gamma regime and persistence."""
    current = float(gamma_proxy_arr[-1]) if not np.isnan(gamma_proxy_arr[-1]) else 0.0
    if current < threshold:
        regime = "negative_gamma"
    elif current > -threshold:
        regime = "positive_gamma"
    else:
        regime = "neutral"
    persistence = 0
    for val in reversed(gamma_proxy_arr):
        if np.isnan(val):
            break
        if (regime == "negative_gamma" and val < threshold) or \
           (regime == "positive_gamma" and val > -threshold) or \
           regime == "neutral":
            persistence += 1
        else:
            break
    strength = float(min(abs(current) / (abs(threshold) * 5 + 1e-12), 1.0))
    return {"regime": regime, "current_value": round(current, 5),
            "persistence": persistence, "regime_strength": round(strength, 4)}


def hedge_recommendation(regime: str, strength: float, turn_type: str, ratio: float) -> dict:
    """Generates hedge recommendation from regime + turn type."""
    if regime == "negative_gamma":
        if turn_type in ("distribution",):
            r, urg, act = min(0.5, ratio + 0.2), "immediate", f"Open {min(0.5, ratio+0.2):.0%} short hedge"
        elif turn_type in ("accumulation",):
            r, urg, act = ratio, "moderate", f"Open {ratio:.0%} short hedge"
        else:
            r, urg, act = ratio * 0.5, "low", f"Small {ratio*0.5:.0%} defensive hedge"
    elif regime == "positive_gamma":
        r, urg, act = 0.0, "none", "No hedge — Positive Gamma dampens volatility"
    else:
        if turn_type in ("distribution",):
            r, urg, act = ratio * 0.5, "low", f"Optional {ratio*0.5:.0%} hedge at resistance"
        else:
            r, urg, act = 0.0, "none", "No hedge — Neutral Gamma"
    return {"hedge_ratio": round(r, 3), "hedge_pct": round(r * 100, 1),
            "urgency": urg, "action": act, "trigger_active": r > 0}


def vol_surface_stats(vol_series: np.ndarray) -> dict:
    """Summary statistics of the realized volatility series."""
    clean = vol_series[~np.isnan(vol_series)]
    if len(clean) == 0:
        return {"current": 0.0, "mean": 0.0, "percentile": 0.5, "regime": "unknown"}
    current    = float(clean[-1])
    percentile = float(np.searchsorted(np.sort(clean), current) / len(clean))
    return {
        "current":    round(current, 4),
        "mean":       round(float(clean.mean()), 4),
        "percentile": round(percentile, 3),
        "regime":     "high" if percentile > 0.75 else ("low" if percentile < 0.25 else "normal"),
    }
