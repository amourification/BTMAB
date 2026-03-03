# =============================================================================
#  engine/acf.py — Equation 2: Autocorrelation Function (ACF)
#  Measures "market memory" — the degree to which current price behaviour
#  is statistically linked to its own history at specific lag intervals.
#
#  Role in pipeline: Validates cycles discovered by SSA/FFT.
#  If SSA finds a 21-day cycle AND ACF shows ρ=0.75 at lag 21, confidence
#  is multiplied — providing the "secret" double-confirmation insight.
#
#  Standard interface:
#      result = run(prices, config) -> dict
# =============================================================================

import logging
import numpy as np
from scipy import stats

logger = logging.getLogger("temporal_bot.engine.acf")

DEFAULT_MAX_LAG: int = 100


# ── Internal helpers ──────────────────────────────────────────────────────────

def _compute_acf(signal: np.ndarray, max_lag: int) -> np.ndarray:
    """
    Computes the normalized ACF for lags 0 through max_lag.

    ρ_k = Σ(Y_t - Ȳ)(Y_{t+k} - Ȳ) / Σ(Y_t - Ȳ)²

    This is the unbiased estimator. ρ_0 = 1 by definition.
    """
    N    = len(signal)
    mean = signal.mean()
    var  = np.sum((signal - mean) ** 2)

    if var < 1e-12:
        logger.warning("ACF: signal variance is zero — returning zeros.")
        return np.zeros(max_lag + 1)

    acf_values = np.zeros(max_lag + 1)
    for k in range(max_lag + 1):
        if k == 0:
            acf_values[0] = 1.0
        else:
            cov = np.sum((signal[: N - k] - mean) * (signal[k:] - mean))
            acf_values[k] = cov / var
    return acf_values


def _confidence_bands(N: int, max_lag: int, alpha: float = 0.05) -> float:
    """
    Computes the 95% confidence band for the ACF under the null hypothesis
    that the time series is white noise (Bartlett's formula simplified):
        bound = ±z_{α/2} / sqrt(N)
    Correlations outside this band are statistically significant.
    """
    z = stats.norm.ppf(1 - alpha / 2)
    return float(z / np.sqrt(N))


def _find_significant_lags(
    acf_values: np.ndarray,
    conf_band: float,
    min_lag: int = 5,
) -> list[dict]:
    """
    Identifies lags where the ACF exceeds the confidence band.
    Filters out lags below `min_lag` to avoid trivial short-term correlations.

    Returns a sorted list of dicts: {lag, correlation, strength}
    """
    significant = []
    for k in range(min_lag, len(acf_values)):
        rho = acf_values[k]
        if abs(rho) > conf_band:
            significant.append({
                "lag":         k,
                "correlation": float(rho),
                "strength":    abs(float(rho)),
            })
    # Sort by absolute correlation strength descending
    significant.sort(key=lambda x: x["strength"], reverse=True)
    return significant


def _detect_cycle_lags(
    acf_values: np.ndarray,
    conf_band: float,
    max_lag: int,
) -> list[int]:
    """
    Finds periodic repetitions in the ACF by looking for local maxima
    that exceed the confidence band. These represent structural cycle lengths.

    Approach: peak detection on positive ACF values above conf_band.
    """
    cycle_lags = []
    for k in range(2, max_lag - 1):
        rho = acf_values[k]
        if rho > conf_band:
            # Local maximum check
            if rho >= acf_values[k - 1] and rho >= acf_values[k + 1]:
                cycle_lags.append(k)
    return cycle_lags


def _validate_ssa_cycle(
    acf_values: np.ndarray,
    ssa_period: float,
    conf_band: float,
    tolerance_pct: float = 0.15,
) -> dict:
    """
    Cross-validates the SSA dominant period against ACF peaks.
    If ACF shows significant correlation near the SSA period (±15%),
    confidence is boosted. This is the "double-confirmation" mechanism.

    Returns dict: {validated, ssa_period, nearest_acf_lag, acf_correlation, boost}
    """
    if ssa_period < 2:
        return {"validated": False, "ssa_period": ssa_period,
                "nearest_acf_lag": 0, "acf_correlation": 0.0, "boost": 0.0}

    # Search window around SSA period
    lo = max(2, int(ssa_period * (1 - tolerance_pct)))
    hi = min(len(acf_values) - 1, int(ssa_period * (1 + tolerance_pct)))

    # If the SSA period lies entirely beyond the available ACF lags,
    # or the window collapses, skip validation gracefully.
    if lo >= len(acf_values) or lo > hi:
        return {
            "validated":       False,
            "ssa_period":      ssa_period,
            "nearest_acf_lag": 0,
            "acf_correlation": 0.0,
            "boost":           0.0,
        }

    best_lag  = lo
    best_corr = acf_values[lo]
    for k in range(lo, hi + 1):
        if abs(acf_values[k]) > abs(best_corr):
            best_lag  = k
            best_corr = acf_values[k]

    validated = abs(best_corr) > conf_band
    boost     = float(abs(best_corr)) if validated else 0.0

    return {
        "validated":       validated,
        "ssa_period":      ssa_period,
        "nearest_acf_lag": best_lag,
        "acf_correlation": float(best_corr),
        "boost":           boost,
    }


# ── Public interface ──────────────────────────────────────────────────────────

def run(prices: np.ndarray, cfg: dict) -> dict:
    """
    Computes the ACF, identifies significant cycle lags, and cross-validates
    the SSA-identified dominant period.

    Parameters
    ----------
    prices : np.ndarray — closing prices or detrended signal (both work)
    cfg    : dict — supports keys:
                 "ACF_MAX_LAG"    (int,   default 100)
                 "SSA_PERIOD"     (float, optional — for cross-validation)

    Returns
    -------
    dict with keys:
        "success"          : bool
        "acf_values"       : np.ndarray  — ACF[0..max_lag]
        "conf_band"        : float       — 95% significance boundary
        "significant_lags" : list[dict]  — lags exceeding conf_band
        "cycle_lags"       : list[int]   — periodic ACF peaks
        "best_lag"         : int         — lag with highest |ACF|
        "best_correlation" : float       — ACF value at best_lag
        "ssa_validation"   : dict        — SSA cross-validation result
        "memory_score"     : float       — aggregate memory strength [0,1]
        "confidence"       : float       — confidence for aggregator [0,1]
        "metadata"         : dict        — N, max_lag
        "error"            : str | None
    """
    _empty = {
        "success":          False,
        "acf_values":       np.array([]),
        "conf_band":        0.0,
        "significant_lags": [],
        "cycle_lags":       [],
        "best_lag":         0,
        "best_correlation": 0.0,
        "ssa_validation":   {},
        "memory_score":     0.0,
        "confidence":       0.0,
        "metadata":         {},
        "error":            None,
    }

    # ── Validate ──────────────────────────────────────────────────────────────
    if prices is None or len(prices) < 20:
        _empty["error"] = f"ACF needs >= 20 samples, got {len(prices) if prices is not None else 0}."
        logger.error(_empty["error"])
        return _empty

    N       = len(prices)
    max_lag = int(cfg.get("ACF_MAX_LAG", DEFAULT_MAX_LAG))
    max_lag = min(max_lag, N // 2)   # can't lag more than half the series

    try:
        # ── Compute ACF ───────────────────────────────────────────────────────
        acf_values = _compute_acf(prices, max_lag)
        conf_band  = _confidence_bands(N, max_lag)

        # ── Find significant lags ─────────────────────────────────────────────
        sig_lags   = _find_significant_lags(acf_values, conf_band)
        cycle_lags = _detect_cycle_lags(acf_values, conf_band, max_lag)

        # ── Best single lag ───────────────────────────────────────────────────
        if sig_lags:
            best_lag  = sig_lags[0]["lag"]
            best_corr = sig_lags[0]["correlation"]
        else:
            # No significant lag — pick the max from lag 5 onward
            search = acf_values[5:] if len(acf_values) > 5 else acf_values
            best_lag  = int(np.argmax(np.abs(search))) + 5
            best_corr = float(acf_values[best_lag])

        # ── SSA cross-validation ──────────────────────────────────────────────
        ssa_period   = float(cfg.get("SSA_PERIOD", 0))
        ssa_val      = _validate_ssa_cycle(acf_values, ssa_period, conf_band)

        # ── Memory score: average absolute ACF of significant lags ────────────
        if sig_lags:
            memory_score = float(np.mean([l["strength"] for l in sig_lags]))
        else:
            memory_score = float(np.abs(acf_values[5:]).mean())
        memory_score = min(1.0, memory_score)

        # ── Confidence ────────────────────────────────────────────────────────
        # Based on: strength of best correlation + SSA validation boost
        confidence = float(abs(best_corr) * 0.7 + ssa_val["boost"] * 0.3)
        confidence = round(max(0.0, min(1.0, confidence)), 4)

        logger.info(
            "ACF OK: best_lag=%d ρ=%.3f sig_lags=%d cycle_lags=%s confidence=%.3f",
            best_lag, best_corr, len(sig_lags), cycle_lags[:5], confidence,
        )

        return {
            "success":          True,
            "acf_values":       acf_values,
            "conf_band":        conf_band,
            "significant_lags": sig_lags,
            "cycle_lags":       cycle_lags,
            "best_lag":         best_lag,
            "best_correlation": best_corr,
            "ssa_validation":   ssa_val,
            "memory_score":     memory_score,
            "confidence":       confidence,
            "metadata":         {"N": N, "max_lag": max_lag},
            "error":            None,
        }

    except Exception as exc:
        msg = f"{type(exc).__name__}: {exc}"
        logger.exception("ACF failed: %s", msg)
        _empty["error"] = msg
        return _empty


# ── Smoke test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    np.random.seed(7)
    n      = 300
    t      = np.arange(n)
    signal = np.sin(2 * np.pi * t / 21) + 0.5 * np.random.randn(n)

    result = run(signal, {"ACF_MAX_LAG": 80, "SSA_PERIOD": 21})
    if result["success"]:
        print(f"✅ ACF OK")
        print(f"   Best lag       : {result['best_lag']} bars")
        print(f"   Correlation    : {result['best_correlation']:.3f}")
        print(f"   Conf band      : ±{result['conf_band']:.3f}")
        print(f"   Cycle lags     : {result['cycle_lags'][:5]}")
        print(f"   SSA validation : {result['ssa_validation']}")
        print(f"   Confidence     : {result['confidence']:.3f}")
    else:
        print(f"❌ {result['error']}")
