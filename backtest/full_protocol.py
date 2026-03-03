"""
backtest/full_protocol.py — Walk-forward backtesting for the Temporal Bot.

This module implements a "real" backtest protocol:
  - Fetches historical OHLCV for BTCUSDT / XRPUSDT (or any symbols)
  - Walks forward through time using a rolling window of past bars
  - At each step, runs the classic consensus engine OFFLINE on the history
  - Derives a directional trade decision (long / short / flat)
  - Compares that decision to realised price movement over a fixed horizon
  - Aggregates metrics like hit-rate and average return

It is intentionally read-only and does not modify the live engines.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Any, Callable, Iterable, List

import logging
import numpy as np

from data.fetcher_utils import get_client, fetch_in_chunks, parse_klines
from backtest.offline_aggregator import build_fetch_like, run_offline
from config import build_config

logger = logging.getLogger("temporal_bot.backtest.full_protocol")


START_DATE = "2024-01-01"
END_DATE = "2026-01-01"  # inclusive of 2025-12-31
SYMBOLS = ("BTCUSDT", "XRPUSDT")
INTERVAL = "1d"


@dataclass
class TradeDecision:
    symbol: str
    timestamp: datetime
    direction: str  # "long", "short", or "flat"
    entry_price: float
    horizon_bars: int
    realised_return: float  # signed percentage, e.g. 0.05 = +5%
    correct: bool
    bias: str
    kelly_pct: float
    confidence_pct: float


@dataclass
class SymbolBacktestReport:
    symbol: str
    bars: int
    start: datetime
    end: datetime
    n_decisions: int
    n_trades: int
    hit_rate: float
    avg_trade_return: float
    equity_curve: List[float]
    decisions: List[TradeDecision]
    error: str | None = None


def _decide_direction(plan: Dict[str, Any]) -> str:
    """
    Map a trade plan into a directional call.

    Simple rule:
      - if kelly_position_pct <= 0 → flat
      - else:
          bias bullish → long
          bias bearish → short
          otherwise    → flat
    """
    kelly_pct = float(plan.get("kelly_position_pct", 0.0) or 0.0)
    if kelly_pct <= 0.0:
        return "flat"
    bias = str(plan.get("market_bias", "neutral"))
    if bias == "bullish":
        return "long"
    if bias == "bearish":
        return "short"
    return "flat"


def _run_symbol_backtest(
    symbol: str,
    interval: str,
    start_date: str,
    end_date: str,
    cfg: Dict[str, Any],
    window_bars: int,
    horizon_bars: int,
    step_cb: Callable[[str, int, int], None] | None = None,
) -> SymbolBacktestReport:
    """
    Backtest a single symbol by walking through history and validating
    directional calls against realised returns.
    """
    client = get_client(
        cfg.get("BINANCE_API_KEY", ""),
        cfg.get("BINANCE_API_SECRET", ""),
    )
    logger.info(
        "Full backtest fetch START: symbol=%s interval=%s window=%d horizon=%d",
        symbol,
        interval,
        window_bars,
        horizon_bars,
    )
    raw = fetch_in_chunks(client, symbol.upper(), interval, window_bars * 4)
    # Fallback: if no data via chunks, try a direct historic fetch for the date range.
    if not raw:
        logger.info(
            "No data from fetch_in_chunks for %s/%s, falling back to get_historical_klines "
            "(start=%s end=%s).",
            symbol,
            interval,
            start_date,
            end_date,
        )
        raw = client.get_historical_klines(symbol.upper(), interval, start_date, end_date)

    if not raw:
        now = datetime.now(timezone.utc)
        logger.error(
            "Full backtest fetch FAILED: symbol=%s interval=%s (no data returned)",
            symbol,
            interval,
        )
        return SymbolBacktestReport(
            symbol=symbol,
            bars=0,
            start=now,
            end=now,
            n_decisions=0,
            n_trades=0,
            hit_rate=0.0,
            avg_trade_return=0.0,
            equity_curve=[],
            decisions=[],
            error="No data returned for backtest.",
        )

    logger.info(
        "Full backtest fetch OK: symbol=%s interval=%s raw_bars=%d",
        symbol,
        interval,
        len(raw),
    )
    df = parse_klines(raw)
    bars = len(df)
    if bars < window_bars + horizon_bars:
        now = datetime.now(timezone.utc)
        return SymbolBacktestReport(
            symbol=symbol,
            bars=bars,
            start=df.index[0].to_pydatetime() if bars else now,
            end=df.index[-1].to_pydatetime() if bars else now,
            n_decisions=0,
            n_trades=0,
            hit_rate=0.0,
            avg_trade_return=0.0,
            equity_curve=[],
            decisions=[],
            error=f"Not enough data for window={window_bars} and horizon={horizon_bars}.",
        )

    close = df["close"].to_numpy(dtype=np.float64)
    decisions: list[TradeDecision] = []

    # Walk forward: for each bar i, use a rolling window of length window_bars
    # ending at i as the "known history", then evaluate the next horizon_bars
    # as future.
    start_idx = window_bars - 1
    end_idx = bars - horizon_bars
    total_steps = max(0, end_idx - start_idx)
    step_idx = 0

    for i in range(start_idx, end_idx):
        step_idx += 1
        df_window = df.iloc[i + 1 - window_bars : i + 1]
        fetch_like = build_fetch_like(symbol, interval, df_window, cfg)
        if not fetch_like.get("success"):
            continue

        plan = run_offline(fetch_like, interval, cfg)
        if not plan.get("success"):
            continue

        direction = _decide_direction(plan)
        bias = str(plan.get("market_bias", "neutral"))
        kelly_pct = float(plan.get("kelly_position_pct", 0.0) or 0.0)
        conf_pct = float(plan.get("consensus_confidence", 0.0) or 0.0)

        entry_price = float(close[i])
        # Future horizon end index
        j = i + horizon_bars
        exit_price = float(close[j])
        ret = (exit_price / entry_price) - 1.0

        if direction == "flat":
            correct = abs(ret) < 0.01  # flat is "correct" if market moved < 1%
            realised = 0.0
        elif direction == "long":
            realised = ret
            correct = ret > 0.0
        else:  # short
            realised = -ret
            correct = ret < 0.0

        decisions.append(
            TradeDecision(
                symbol=symbol,
                timestamp=df.index[i].to_pydatetime(),
                direction=direction,
                entry_price=entry_price,
                horizon_bars=horizon_bars,
                realised_return=realised,
                correct=correct,
                bias=bias,
                kelly_pct=kelly_pct,
                confidence_pct=conf_pct,
            )
        )

        # Periodic progress callback so the GUI can show within-symbol progress.
        if step_cb is not None and total_steps > 0 and (step_idx % 10 == 0 or step_idx == total_steps):
            try:
                step_cb(symbol, step_idx, total_steps)
            except Exception:
                # Progress is best-effort only; never break the backtest.
                pass

    n_decisions = len(decisions)
    trades_only = [d for d in decisions if d.direction in ("long", "short")]
    n_trades = len(trades_only)

    if n_trades:
        hits = sum(1 for d in trades_only if d.correct)
        hit_rate = hits / n_trades
        avg_trade_return = float(np.mean([d.realised_return for d in trades_only]))
    else:
        hit_rate = 0.0
        avg_trade_return = 0.0

    # Simple equity curve: start at 1.0, compound trade returns only.
    equity = 1.0
    equity_curve: list[float] = [equity]
    for d in trades_only:
        equity *= 1.0 + d.realised_return
        equity_curve.append(equity)

    return SymbolBacktestReport(
        symbol=symbol,
        bars=bars,
        start=df.index[0].to_pydatetime(),
        end=df.index[-1].to_pydatetime(),
        n_decisions=n_decisions,
        n_trades=n_trades,
        hit_rate=round(float(hit_rate), 4),
        avg_trade_return=round(float(avg_trade_return), 6),
        equity_curve=equity_curve,
        decisions=decisions,
        error=None,
    )


def run_full_backtest(
    symbols: Iterable[str] = SYMBOLS,
    interval: str = INTERVAL,
    start_date: str = START_DATE,
    end_date: str = END_DATE,
    cfg: Dict[str, Any] | None = None,
    window_bars: int = 512,
    horizon_bars: int = 10,
    progress_cb: Callable[[SymbolBacktestReport], None] | None = None,
    step_cb: Callable[[str, int, int], None] | None = None,
) -> Dict[str, SymbolBacktestReport]:
    """
    High-level entry point for the full backtest protocol.

    Parameters
    ----------
    symbols : iterable[str]
        Symbols to backtest (default: BTCUSDT, XRPUSDT).
    interval : str
        Bar interval (default: "1d").
    start_date, end_date : str
        Date range; currently used only as a hint, since we pull a large
        enough history to cover the window.
    cfg : dict | None
        Config from config.py; if None, build_config() is used.
    window_bars : int
        Rolling lookback window for each decision.
    horizon_bars : int
        Number of future bars used to validate each decision.
    progress_cb : callable | None
        Optional callback invoked as each symbol report is ready.
    step_cb : callable | None
        Optional callback invoked periodically during each symbol's walk-forward
        loop with (symbol, current_step, total_steps).
    """
    if cfg is None:
        cfg = build_config()

    reports: Dict[str, SymbolBacktestReport] = {}
    for symbol in symbols:
        try:
            rep = _run_symbol_backtest(
                symbol=symbol,
                interval=interval,
                start_date=start_date,
                end_date=end_date,
                cfg=cfg,
                window_bars=window_bars,
                horizon_bars=horizon_bars,
                step_cb=step_cb,
            )
            reports[symbol] = rep
            if progress_cb is not None:
                progress_cb(rep)
        except Exception as exc:
            now = datetime.now(timezone.utc)
            rep = SymbolBacktestReport(
                symbol=symbol,
                bars=0,
                start=now,
                end=now,
                n_decisions=0,
                n_trades=0,
                hit_rate=0.0,
                avg_trade_return=0.0,
                equity_curve=[],
                decisions=[],
                error=str(exc),
            )
            reports[symbol] = rep
            if progress_cb is not None:
                progress_cb(rep)
    return reports

