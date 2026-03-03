from __future__ import annotations

"""
backtest/binance_range.py — Simple Binance backtest helper.

Shared logic between the pytest backtest and the GUI backtest button:
fetches BTCUSDT and XRPUSDT daily OHLCV between 2024-01-01 and 2025-12-31
and runs both classic and advanced aggregators.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Any, Callable, Iterable

from data.fetcher_utils import parse_klines
from consensus.aggregator import run as run_classic
from consensus.advanced_aggregator import run as run_advanced
from config import build_config


START_DATE = "2024-01-01"
END_DATE = "2026-01-01"  # inclusive of 2025-12-31
SYMBOLS = ("BTCUSDT", "XRPUSDT")
INTERVAL = "1d"


@dataclass
class BacktestResult:
    symbol: str
    bars: int
    start: datetime
    end: datetime
    classic_success: bool
    advanced_success: bool
    error: str | None = None


def run_backtest(
    symbols: Iterable[str] = SYMBOLS,
    interval: str = INTERVAL,
    start_date: str = START_DATE,
    end_date: str = END_DATE,
    cfg: Dict[str, Any] | None = None,
    progress_cb: Callable[[BacktestResult], None] | None = None,
) -> Dict[str, BacktestResult]:
    """
    Runs a simple backtest for the given symbols and returns a summary dict.
    Requires BINANCE_API_KEY / BINANCE_API_SECRET in the environment.
    """
    try:
        from binance.client import Client
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("python-binance is not installed") from exc

    if cfg is None:
        cfg = build_config()

    client = Client(cfg.get("BINANCE_API_KEY", ""), cfg.get("BINANCE_API_SECRET", ""))

    summaries: Dict[str, BacktestResult] = {}

    for symbol in symbols:
        try:
            raw = client.get_historical_klines(symbol, interval, start_date, end_date)
            if not raw:
                result = BacktestResult(
                    symbol=symbol,
                    bars=0,
                    start=datetime.now(timezone.utc),
                    end=datetime.now(timezone.utc),
                    classic_success=False,
                    advanced_success=False,
                    error="No data returned",
                )
                summaries[symbol] = result
                if progress_cb is not None:
                    progress_cb(result)
                continue

            df = parse_klines(raw)
            bars = len(df)
            start_dt = df.index[0].to_pydatetime()
            end_dt = df.index[-1].to_pydatetime()

            # Classic run
            plan_classic = run_classic(symbol, interval, bars, cfg)
            # Advanced run
            plan_advanced = run_advanced(symbol, interval, bars, cfg)

            result = BacktestResult(
                symbol=symbol,
                bars=bars,
                start=start_dt,
                end=end_dt,
                classic_success=bool(plan_classic.get("success")),
                advanced_success=bool(plan_advanced.get("success")),
                error=None,
            )
            summaries[symbol] = result
            if progress_cb is not None:
                progress_cb(result)
        except Exception as exc:
            result = BacktestResult(
                symbol=symbol,
                bars=0,
                start=datetime.now(timezone.utc),
                end=datetime.now(timezone.utc),
                classic_success=False,
                advanced_success=False,
                error=str(exc),
            )
            summaries[symbol] = result
            if progress_cb is not None:
                progress_cb(result)

    return summaries

