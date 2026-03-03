from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Iterable

import pytest


def _have_api_keys() -> bool:
    return bool(os.getenv("BINANCE_API_KEY") and os.getenv("BINANCE_API_SECRET"))


pytestmark = pytest.mark.skipif(
    not _have_api_keys(), reason="Binance API keys required for backtest integration test"
)


@pytest.mark.slow
def test_backtest_classic_and_advanced_on_daily_range():
    """
    Integration-style backtest: fetch real OHLCV data for BTCUSDT and XRPUSDT
    from 2024-01-01 to 2025-12-31 and ensure both the classic and advanced
    aggregators can process the full window without errors.
    """
    from backtest.binance_range import run_backtest, START_DATE, END_DATE, SYMBOLS, INTERVAL

    summaries = run_backtest(SYMBOLS, INTERVAL, START_DATE, END_DATE)
    for symbol, res in summaries.items():
        assert res.error is None, f"{symbol} backtest error: {res.error}"
        assert res.bars > 500
        assert res.classic_success is True
        assert res.advanced_success is True

