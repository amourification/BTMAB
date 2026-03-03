# =============================================================================
#  data/fetcher.py — Binance OHLCV Data Fetcher
#  Public interface for all data ingestion. Heavy lifting is in fetcher_utils.py.
#
#  Standard interface:
#      result = run(symbol, interval, bars, config) -> dict
# =============================================================================

import logging
import numpy as np
from datetime import datetime, timezone

from data.fetcher_utils import (
    get_client,
    fetch_in_chunks,
    parse_klines,
    get_available_symbols,
)

logger = logging.getLogger("temporal_bot.fetcher")


def run(symbol: str, interval: str, bars: int, cfg: dict) -> dict:
    """
    Fetches historical OHLCV data from Binance.

    Parameters
    ----------
    symbol   : Trading pair e.g. "BTCUSDT"
    interval : Binance interval e.g. "1d"
    bars     : Number of bars to fetch (recommended: 2048)
    cfg      : Must contain "BINANCE_API_KEY" and "BINANCE_API_SECRET"

    Returns
    -------
    dict with keys:
        success, symbol, interval, bars, df, close, high, low,
        volume, timestamps, fetched_at, error
    """
    logger.info("Fetching %s | %s | %d bars", symbol, interval, bars)

    _empty = {
        "success":    False,
        "symbol":     symbol,
        "interval":   interval,
        "bars":       0,
        "df":         None,
        "close":      np.array([]),
        "high":       np.array([]),
        "low":        np.array([]),
        "volume":     np.array([]),
        "timestamps": np.array([], dtype=np.int64),
        "fetched_at": datetime.now(tz=timezone.utc).isoformat(),
        "error":      None,
    }

    api_key    = cfg.get("BINANCE_API_KEY", "")
    api_secret = cfg.get("BINANCE_API_SECRET", "")

    if not api_key or api_key == "your_binance_api_key_here":
        _empty["error"] = "Binance API key missing. Set BINANCE_API_KEY in .env"
        logger.error(_empty["error"])
        return _empty

    if bars < 1:
        _empty["error"] = f"'bars' must be >= 1, got {bars}."
        logger.error(_empty["error"])
        return _empty

    try:
        client = get_client(api_key, api_secret)
        raw    = fetch_in_chunks(client, symbol.upper(), interval, bars)

        if not raw:
            _empty["error"] = (
                f"No data returned for {symbol}/{interval}. "
                "Check symbol name and internet connection."
            )
            logger.error(_empty["error"])
            return _empty

        df = parse_klines(raw)
        logger.info(
            "Fetch OK: symbol=%s interval=%s bars=%d index=[0..%d] %s to %s",
            symbol.upper(),
            interval,
            len(df),
            len(df) - 1,
            df.index[0].isoformat(),
            df.index[-1].isoformat(),
        )

        return {
            "success":    True,
            "symbol":     symbol.upper(),
            "interval":   interval,
            "bars":       len(df),
            "df":         df,
            "close":      df["close"].to_numpy(dtype=np.float64),
            "high":       df["high"].to_numpy(dtype=np.float64),
            "low":        df["low"].to_numpy(dtype=np.float64),
            "volume":     df["volume"].to_numpy(dtype=np.float64),
            "timestamps": df["timestamp"].to_numpy(dtype=np.int64),
            "fetched_at": datetime.now(tz=timezone.utc).isoformat(),
            "error":      None,
        }

    except Exception as exc:
        msg = f"{type(exc).__name__}: {exc}"
        logger.exception("Unexpected fetch error: %s", msg)
        _empty["error"] = msg
        return _empty


def fetch_symbols(cfg: dict, quote_asset: str = "USDT") -> list[str]:
    """Returns sorted active Binance trading pairs. Used by the GUI dropdown."""
    try:
        client = get_client(
            cfg.get("BINANCE_API_KEY", ""),
            cfg.get("BINANCE_API_SECRET", ""),
        )
        return get_available_symbols(client, quote_asset)
    except Exception as exc:
        logger.error("Symbol list fetch failed: %s", exc)
        return [cfg.get("DEFAULT_SYMBOL", "BTCUSDT")]


# ── Smoke test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from pathlib import Path
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import config as cfg_module

    test_cfg = {
        "BINANCE_API_KEY":    cfg_module.BINANCE_API_KEY,
        "BINANCE_API_SECRET": cfg_module.BINANCE_API_SECRET,
        "DEFAULT_SYMBOL":     cfg_module.DEFAULT_SYMBOL,
    }
    result = run("BTCUSDT", "1d", 10, test_cfg)
    if result["success"]:
        print(f"\n✅ {result['bars']} bars | last close: {result['close'][-1]:.2f}")
    else:
        print(f"\n❌ {result['error']}")
