# =============================================================================
#  data/fetcher_utils.py — Binance Fetch Helpers
#  Internal utilities used by fetcher.py. Split to keep both files ≤ 300 lines.
#  Not intended to be called directly by other modules.
# =============================================================================

import time
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timezone

logger = logging.getLogger("temporal_bot.fetcher_utils")

COL_TIMESTAMP = "timestamp"
COL_OPEN      = "open"
COL_HIGH      = "high"
COL_LOW       = "low"
COL_CLOSE     = "close"
COL_VOLUME    = "volume"
OHLCV_COLUMNS = [COL_TIMESTAMP, COL_OPEN, COL_HIGH, COL_LOW, COL_CLOSE, COL_VOLUME]

INTERVAL_MS: dict[str, int] = {
    "1m":  60_000,        "3m":  180_000,       "5m":  300_000,
    "15m": 900_000,       "30m": 1_800_000,     "1h":  3_600_000,
    "2h":  7_200_000,     "4h":  14_400_000,    "6h":  21_600_000,
    "8h":  28_800_000,    "12h": 43_200_000,    "1d":  86_400_000,
    "3d":  259_200_000,   "1w":  604_800_000,   "1M":  2_592_000_000,
}


def get_client(api_key: str, api_secret: str):
    """Lazily imports and returns a Binance Client instance."""
    try:
        from binance.client import Client
        logger.info("Creating Binance client (key len=%d, secret len=%d)", len(api_key), len(api_secret))
        return Client(api_key, api_secret)
    except ImportError:
        raise ImportError(
            "python-binance is not installed. Run: pip install python-binance"
        )


def interval_to_ms(interval: str) -> int:
    """Converts a Binance interval string to milliseconds."""
    if interval not in INTERVAL_MS:
        raise ValueError(
            f"Unknown interval '{interval}'. Valid: {list(INTERVAL_MS.keys())}"
        )
    return INTERVAL_MS[interval]


def parse_klines(raw_klines: list) -> pd.DataFrame:
    """
    Converts raw Binance kline data into a clean OHLCV DataFrame.
    Binance kline format: [open_time, open, high, low, close, volume, close_time, ...]
    """
    records = [
        {
            COL_TIMESTAMP: int(k[0]),
            COL_OPEN:      float(k[1]),
            COL_HIGH:      float(k[2]),
            COL_LOW:       float(k[3]),
            COL_CLOSE:     float(k[4]),
            COL_VOLUME:    float(k[5]),
        }
        for k in raw_klines
    ]
    df = pd.DataFrame(records, columns=OHLCV_COLUMNS)
    df["datetime"] = pd.to_datetime(df[COL_TIMESTAMP], unit="ms", utc=True)
    df.set_index("datetime", inplace=True)
    df.sort_index(inplace=True)
    return df


def fetch_in_chunks(client, symbol: str, interval: str, total_bars: int) -> list:
    """
    Fetches up to `total_bars` klines from Binance by chunking into
    requests of 1000 bars (Binance hard limit per request).
    """
    interval_ms   = interval_to_ms(interval)
    now_ms        = int(time.time() * 1000)
    start_ms      = now_ms - (total_bars * interval_ms)
    all_klines:  list = []
    current_start    = start_ms
    CHUNK_SIZE       = 1000

    fetch_id = f"{symbol}-{interval}-{int(now_ms)}"
    logger.info(
        "[%s] API kline fetch START: symbol=%s interval=%s total_bars=%d "
        "start_time=%s",
        fetch_id,
        symbol,
        interval,
        total_bars,
        datetime.fromtimestamp(start_ms / 1000, tz=timezone.utc).isoformat(),
    )

    while len(all_klines) < total_bars:
        remaining = total_bars - len(all_klines)
        limit     = min(remaining, CHUNK_SIZE)

        logger.info(
            "[%s] API chunk request: symbol=%s interval=%s start=%s limit=%d "
            "(collected=%d/%d)",
            fetch_id,
            symbol,
            interval,
            datetime.fromtimestamp(current_start / 1000, tz=timezone.utc).isoformat(),
            limit,
            len(all_klines),
            total_bars,
        )

        chunk = client.get_klines(
            symbol    = symbol,
            interval  = interval,
            startTime = current_start,
            limit     = limit,
        )

        if not chunk:
            logger.warning("[%s] Binance returned empty chunk — stopping early (collected=%d).",
                           fetch_id, len(all_klines))
            break

        all_klines.extend(chunk)
        logger.info(
            "[%s] API chunk response: received=%d total_collected=%d",
            fetch_id,
            len(chunk),
            len(all_klines),
        )
        last_close_time = int(chunk[-1][6])
        current_start   = last_close_time + 1

        if current_start >= now_ms:
            logger.info(
                "[%s] Reached current time boundary; stopping chunked fetch.", fetch_id
            )
            break

        time.sleep(0.1)  # respect Binance rate limits

    logger.info(
        "[%s] API kline fetch DONE: collected=%d bars (requested=%d)",
        fetch_id,
        len(all_klines),
        total_bars,
    )
    return all_klines


def get_available_symbols(client, quote_asset: str = "USDT") -> list[str]:
    """Returns sorted list of active Binance trading pairs for a quote asset."""
    try:
        info = client.get_exchange_info()
        return sorted(
            s["symbol"]
            for s in info["symbols"]
            if s["quoteAsset"] == quote_asset and s["status"] == "TRADING"
        )
    except Exception as exc:
        logger.error("Could not fetch symbol list: %s", exc)
        return ["BTCUSDT"]
