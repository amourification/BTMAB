# =============================================================================
#  engine/detrend.py — Equation 12: Least Squares Polynomial Detrending
#  MUST run first — all other engines consume the detrended signal.
#
#  Purpose: Remove the underlying growth trajectory from price data so that
#  cyclical analysis (SSA, FFT, Hilbert) operates on pure oscillation,
#  not a trending series. Distinguishes "cyclical turns" from "trend changes."
#
#  Standard interface:
#      result = run(prices, config) -> dict
# =============================================================================

import logging
import numpy as np
from numpy.polynomial.polynomial import polyfit, polyval

logger = logging.getLogger("temporal_bot.engine.detrend")

# Default polynomial degree if not in cfg
DEFAULT_POLY_DEGREE: int = 3


def _fit_polynomial(prices: np.ndarray, degree: int) -> np.ndarray:
    """
    Fits a least-squares polynomial of the given degree to `prices`.
    Returns the fitted trend line as a NumPy array (same length as prices).

    Uses numpy's polynomial module (numerically stable Chebyshev basis).
    x is normalised to [0, 1] to avoid ill-conditioning for large N.
    """
    n  = len(prices)
    x  = np.linspace(0.0, 1.0, n)
    coeffs = polyfit(x, prices, degree)
    trend  = polyval(x, coeffs)
    return trend, coeffs


def _compute_residuals(prices: np.ndarray, trend: np.ndarray) -> np.ndarray:
    """
    Subtracts the trend from raw prices to isolate the detrended oscillation.
    Result is centred near zero — suitable for FFT, SSA, Hilbert Transform.
    """
    return prices - trend


def _oscillator_stats(residuals: np.ndarray) -> dict:
    """
    Computes summary statistics of the detrended oscillator.
    Used by downstream engines to assess signal quality.
    """
    return {
        "mean":   float(residuals.mean()),
        "std":    float(residuals.std()),
        "min":    float(residuals.min()),
        "max":    float(residuals.max()),
        # Ratio of oscillator variance to original price variance
        # Lower = trend dominated; Higher = cycle dominated
        "cycle_dominance": float(residuals.std() / (residuals.std() + 1e-12)),
    }


def _trend_direction(trend: np.ndarray) -> str:
    """
    Classifies overall trend direction from the fitted polynomial.
    Compares first quarter mean to last quarter mean of the trend line.
    """
    n    = len(trend)
    q    = max(n // 4, 1)
    start_mean = trend[:q].mean()
    end_mean   = trend[-q:].mean()
    delta      = (end_mean - start_mean) / (abs(start_mean) + 1e-12)

    if delta > 0.02:
        return "uptrend"
    elif delta < -0.02:
        return "downtrend"
    else:
        return "sideways"


def _find_zero_crossings(residuals: np.ndarray) -> np.ndarray:
    """
    Returns indices where the detrended oscillator crosses zero.
    Zero crossings represent potential cycle boundaries — used by
    aggregator.py to cross-reference with Hilbert phase turns.
    """
    signs    = np.sign(residuals)
    crossings = np.where(np.diff(signs) != 0)[0]
    return crossings


# ── Public interface ──────────────────────────────────────────────────────────

def run(prices: np.ndarray, cfg: dict) -> dict:
    """
    Fits a polynomial trend and removes it from the price series.

    Parameters
    ----------
    prices : np.ndarray — raw closing prices (float64)
    cfg    : dict — supports key "POLY_DEGREE" (int, default 3)

    Returns
    -------
    dict with keys:
        "success"         : bool
        "trend"           : np.ndarray — fitted polynomial trend line
        "detrended"       : np.ndarray — prices minus trend (oscillator)
        "poly_degree"     : int        — degree used
        "poly_coeffs"     : np.ndarray — polynomial coefficients
        "trend_direction" : str        — "uptrend" | "downtrend" | "sideways"
        "zero_crossings"  : np.ndarray — bar indices of zero crossings
        "stats"           : dict       — mean, std, min, max, cycle_dominance
        "confidence"      : float      — 0.0–1.0 (based on R² of fit)
        "metadata"        : dict       — n_bars, poly_degree
        "error"           : str | None
    """
    _empty = {
        "success":         False,
        "trend":           np.array([]),
        "detrended":       np.array([]),
        "poly_degree":     0,
        "poly_coeffs":     np.array([]),
        "trend_direction": "unknown",
        "zero_crossings":  np.array([], dtype=int),
        "stats":           {},
        "confidence":      0.0,
        "metadata":        {},
        "error":           None,
    }

    # ── Validate ──────────────────────────────────────────────────────────────
    if prices is None or len(prices) < 10:
        _empty["error"] = f"Need at least 10 prices, got {len(prices) if prices is not None else 0}."
        logger.error(_empty["error"])
        return _empty

    if not np.isfinite(prices).all():
        _empty["error"] = "prices array contains NaN or Inf values."
        logger.error(_empty["error"])
        return _empty

    degree = int(cfg.get("POLY_DEGREE", DEFAULT_POLY_DEGREE))
    degree = max(1, min(degree, 10))   # clamp to [1, 10]

    # ── Fit ───────────────────────────────────────────────────────────────────
    try:
        trend, coeffs = _fit_polynomial(prices, degree)
        detrended     = _compute_residuals(prices, trend)

        # R² — goodness of fit of the trend
        ss_res = np.sum((prices - trend) ** 2)
        ss_tot = np.sum((prices - prices.mean()) ** 2)
        r2     = float(1.0 - ss_res / (ss_tot + 1e-12))
        r2     = max(0.0, min(1.0, r2))

        # Confidence: high R² means trend explains the data well
        # but for cycle analysis we actually want residuals to carry
        # meaningful variance, so confidence is balanced.
        # Rule: confidence = R² weighted toward moderate values (0.5–0.85)
        confidence = float(1.0 - abs(r2 - 0.70) / 0.70)
        confidence = max(0.0, min(1.0, confidence))

        stats          = _oscillator_stats(detrended)
        direction      = _trend_direction(trend)
        zero_crossings = _find_zero_crossings(detrended)

        logger.info(
            "Detrend OK: degree=%d R²=%.3f direction=%s crossings=%d confidence=%.2f",
            degree, r2, direction, len(zero_crossings), confidence,
        )

        return {
            "success":         True,
            "trend":           trend,
            "detrended":       detrended,
            "poly_degree":     degree,
            "poly_coeffs":     coeffs,
            "trend_direction": direction,
            "zero_crossings":  zero_crossings,
            "stats":           stats,
            "confidence":      confidence,
            "metadata":        {"n_bars": len(prices), "poly_degree": degree, "r2": r2},
            "error":           None,
        }

    except Exception as exc:
        msg = f"{type(exc).__name__}: {exc}"
        logger.exception("Detrend failed: %s", msg)
        _empty["error"] = msg
        return _empty


# ── Smoke test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Synthetic test: sine wave on top of a linear trend
    n      = 200
    x      = np.linspace(0, 4 * np.pi, n)
    trend  = np.linspace(100, 200, n)
    cycle  = 15 * np.sin(x)
    prices = trend + cycle + np.random.randn(n) * 2

    result = run(prices, {"POLY_DEGREE": 3})
    if result["success"]:
        print(f"✅ Detrend OK")
        print(f"   Direction  : {result['trend_direction']}")
        print(f"   Confidence : {result['confidence']:.3f}")
        print(f"   Crossings  : {len(result['zero_crossings'])}")
        print(f"   Stats      : {result['stats']}")
    else:
        print(f"❌ {result['error']}")
