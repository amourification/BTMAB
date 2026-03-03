# =============================================================================
#  data/preprocessor.py — Data Preprocessing & Enrichment
#  Transforms raw OHLCV from fetcher.py into an enriched format ready for
#  all 12 engine modules. Heavy helpers live in preprocessor_utils.py.
#
#  Standard interface:
#      result = run(fetch_result, config) -> dict
# =============================================================================

import logging
import numpy as np
import pandas as pd

from data.preprocessor_utils import (
    unix_ms_to_julian,
    julian_to_centurial_T,
    label_sessions,
    normalize_minmax,
    normalize_zscore,
    quality_check,
    clean,
)

logger = logging.getLogger("temporal_bot.preprocessor")


def run(fetch_result: dict, cfg: dict) -> dict:
    """
    Main entry point for the preprocessor module.

    Parameters
    ----------
    fetch_result : dict returned by data/fetcher.py run()
    cfg          : config dict from config.py

    Returns
    -------
    dict with keys:
        success, symbol, interval, bars, df,
        close, close_norm, close_z, high, low, volume,
        timestamps_ms, julian_dates, centurial_T,
        sessions, warnings, error
    """
    _empty = {
        "success": False,
        "symbol":  fetch_result.get("symbol", "UNKNOWN"),
        "interval": fetch_result.get("interval", "?"),
        "bars":    0,
        "df":      None,
        "close":   np.array([]),
        "close_norm": np.array([]),
        "close_z": np.array([]),
        "high":    np.array([]),
        "low":     np.array([]),
        "volume":  np.array([]),
        "timestamps_ms": np.array([], dtype=np.int64),
        "julian_dates":  np.array([]),
        "centurial_T":   np.array([]),
        "sessions":      np.array([]),
        "warnings":      [],
        "error":   None,
    }

    if not fetch_result.get("success"):
        _empty["error"] = (
            f"Preprocessor received failed fetch: "
            f"{fetch_result.get('error', 'unknown')}"
        )
        logger.error(_empty["error"])
        return _empty

    df = fetch_result["df"].copy()

    # ── Quality check & clean ─────────────────────────────────────────────────
    warnings = quality_check(df)
    for w in warnings:
        logger.warning("[preprocessor] %s", w)
    df = clean(df)

    if len(df) < 10:
        _empty["error"] = f"Only {len(df)} bars after cleaning — too few."
        logger.error(_empty["error"])
        return _empty

    logger.info(
        "Preprocessing OK: %d bars | %s → %s",
        len(df), df.index[0].isoformat(), df.index[-1].isoformat(),
    )

    # ── Extract raw arrays ────────────────────────────────────────────────────
    timestamps_ms = df["timestamp"].to_numpy(dtype=np.int64)
    close         = df["close"].to_numpy(dtype=np.float64)
    high          = df["high"].to_numpy(dtype=np.float64)
    low           = df["low"].to_numpy(dtype=np.float64)
    volume        = df["volume"].to_numpy(dtype=np.float64)

    # ── Derived arrays ────────────────────────────────────────────────────────
    julian_dates = unix_ms_to_julian(timestamps_ms)
    centurial_T  = julian_to_centurial_T(julian_dates)
    sessions     = label_sessions(timestamps_ms)
    close_norm   = normalize_minmax(close)
    close_z      = normalize_zscore(close)

    # ── Enrich DataFrame ──────────────────────────────────────────────────────
    df["julian_date"] = julian_dates
    df["centurial_T"] = centurial_T
    df["session"]     = sessions
    df["close_norm"]  = close_norm
    df["close_z"]     = close_z

    return {
        "success":       True,
        "symbol":        fetch_result["symbol"],
        "interval":      fetch_result["interval"],
        "bars":          len(df),
        "df":            df,
        "close":         close,
        "close_norm":    close_norm,
        "close_z":       close_z,
        "high":          high,
        "low":           low,
        "volume":        volume,
        "timestamps_ms": timestamps_ms,
        "julian_dates":  julian_dates,
        "centurial_T":   centurial_T,
        "sessions":      sessions,
        "warnings":      warnings,
        "error":         None,
    }


# ── Public aliases for direct use by engine modules ───────────────────────────
def norm_minmax(arr: np.ndarray) -> np.ndarray:
    return normalize_minmax(arr)

def norm_zscore(arr: np.ndarray) -> np.ndarray:
    return normalize_zscore(arr)

def to_julian(unix_ms: np.ndarray) -> np.ndarray:
    return unix_ms_to_julian(unix_ms)


# ── Smoke test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from pathlib import Path
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import config as cfg_module
    from data.fetcher import run as fetch

    test_cfg = {
        "BINANCE_API_KEY":    cfg_module.BINANCE_API_KEY,
        "BINANCE_API_SECRET": cfg_module.BINANCE_API_SECRET,
    }
    fetch_result = fetch("BTCUSDT", "1d", 20, test_cfg)
    result = run(fetch_result, test_cfg)

    if result["success"]:
        print(f"\n✅ Preprocessor OK — {result['bars']} bars")
        print(f"   Julian: {result['julian_dates'][:2]}")
        print(f"   Sessions: {result['sessions'][:5]}")
        print(f"   close_norm[-3:]: {result['close_norm'][-3:]}")
    else:
        print(f"\n❌ {result['error']}")
