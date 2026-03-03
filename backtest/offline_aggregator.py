"""
backtest/offline_aggregator.py — Run the consensus pipeline on pre-fetched data.

This module mirrors the classic consensus aggregator pipeline, but instead of
fetching from Binance it accepts a "fetch-like" dict built from an existing
DataFrame slice. It is used by the full backtest protocol to walk forward
through historical data without touching the live API.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, Any

import logging
import numpy as np

from consensus.aggregator_utils import (
    run_parallel,
    weighted_confidence,
    determine_market_bias,
    build_trade_plan,
)

from data.preprocessor import run as preprocess
from data.sentiment import run as get_sentiment

from engine.detrend import run as run_detrend
from engine.ssa_utils import run as run_ssa
from engine.acf import run as run_acf
from engine.fft import run as run_fft
from engine.hilbert import run as run_hilbert
from engine.solar import run as run_solar
from engine.murray import run as run_murray
from engine.kelly import run as run_kelly
from engine.gamma import run as run_gamma
from engine.gann import run as run_gann
from engine.elliott_fib import run as run_elliott_fib
from engine.ar_model import run as run_ar
from engine.walras import run as run_walras

from risk.hedging import run as run_hedging
from risk.stops import run as run_stops
from risk.portfolio import run as run_portfolio


logger = logging.getLogger("temporal_bot.backtest.offline_aggregator")


def build_fetch_like(symbol: str, interval: str, df, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Construct a fetcher.run-like dict from a pandas DataFrame slice.

    The structure matches data.fetcher.run so that downstream preprocessor
    and engines can be reused unchanged.
    """
    import pandas as pd  # local import to avoid hard dependency during import

    if df is None or len(df) == 0:
        now_iso = datetime.now(tz=timezone.utc).isoformat()
        return {
            "success": False,
            "symbol": symbol,
            "interval": interval,
            "bars": 0,
            "df": pd.DataFrame(),
            "close": np.array([]),
            "high": np.array([]),
            "low": np.array([]),
            "volume": np.array([]),
            "timestamps": np.array([], dtype=np.int64),
            "fetched_at": now_iso,
            "error": "Empty dataframe slice in offline backtest.",
        }

    # Ensure timestamp column exists; parse_klines already provides this.
    if "timestamp" not in df.columns:
        raise ValueError("DataFrame for offline backtest must contain a 'timestamp' column.")

    close = df["close"].to_numpy(dtype=np.float64)
    high = df["high"].to_numpy(dtype=np.float64)
    low = df["low"].to_numpy(dtype=np.float64)
    volume = df["volume"].to_numpy(dtype=np.float64)
    ts = df["timestamp"].to_numpy(dtype=np.int64)

    return {
        "success": True,
        "symbol": symbol,
        "interval": interval,
        "bars": len(df),
        "df": df,
        "close": close,
        "high": high,
        "low": low,
        "volume": volume,
        "timestamps": ts,
        "fetched_at": datetime.now(tz=timezone.utc).isoformat(),
        "error": None,
    }


def run_offline(fetch_r: Dict[str, Any], interval: str, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run the classic consensus pipeline on pre-fetched data.

    Parameters
    ----------
    fetch_r : dict
        A result dict in the shape returned by data.fetcher.run, but
        constructed from historical data slices.
    interval : str
        Interval label (e.g. "1d") for metadata only.
    cfg : dict
        Full config from config.py.
    """
    symbol = fetch_r.get("symbol", "UNKNOWN")
    bars = int(fetch_r.get("bars", 0))

    logger.info("Offline aggregator START: %s %s %d bars", symbol, interval, bars)

    err_base = {
        "success": False,
        "symbol": symbol,
        "interval": interval,
        "bars": bars,
        "error": None,
        "_engines": {},
        "_risk": {},
    }

    if not fetch_r.get("success"):
        return {**err_base, "error": f"Fetch: {fetch_r.get('error')}"}

    prep_r = preprocess(fetch_r, cfg)
    if not prep_r.get("success"):
        return {**err_base, "error": f"Preprocess: {prep_r.get('error')}"}

    prices = prep_r["close"]
    sessions = prep_r["sessions"]
    sym = fetch_r.get("symbol", symbol)
    sent_r = get_sentiment(sym, str(sessions[-1]), cfg)

    # Stage 2: Detrend
    det_r = run_detrend(prices, cfg)
    if not det_r.get("success"):
        return {**err_base, "error": f"Detrend: {det_r.get('error')}"}
    detrended = det_r["detrended"]

    # Stage 3: Core engines
    core = run_parallel(
        [
            ("ssa", run_ssa, (detrended, cfg)),
            ("fft", run_fft, (detrended, cfg)),
            ("hilbert", run_hilbert, (detrended, cfg)),
        ]
    )
    ssa_period = core.get("ssa", {}).get("dominant_period", 0.0)
    core["acf"] = run_acf(detrended, {**cfg, "SSA_PERIOD": ssa_period})

    turn_type = core.get("hilbert", {}).get("turn_type", "unknown")
    fft_period = core.get("fft", {}).get("primary_cycle", {}).get(
        "period",
        cfg.get("DOMINANT_CYCLE_BARS", 512),
    )
    cfg_sup = {
        **cfg,
        "HILBERT_TURN_TYPE": turn_type,
        "FFT_PRIMARY_PERIOD": fft_period,
        "SSA_PERIOD": ssa_period,
    }

    # Stage 4: Support engines
    sup = run_parallel(
        [
            ("solar", run_solar, (prep_r, cfg_sup)),
            ("murray", run_murray, (prices, cfg_sup)),
            ("gamma", run_gamma, (prices, cfg_sup)),
            ("gann", run_gann, (prices, cfg_sup)),
            ("elliott_fib", run_elliott_fib, (prices, cfg_sup)),
            ("walras", run_walras, (prices, cfg_sup)),
        ]
    )

    # Stage 5: Sequential
    mid_eng = {**core, **sup, "detrend": det_r}
    ar_r = run_ar(prep_r, sent_r, cfg_sup)
    kel_r = run_kelly({**mid_eng, "ar": ar_r, "prices": prices}, cfg_sup)

    fetch_extras = {
        "high": fetch_r.get("high", prices),
        "low": fetch_r.get("low", prices),
        "timestamps": fetch_r.get("timestamps", np.array([])),
    }
    engine_results = {**mid_eng, "ar": ar_r, "kelly": kel_r, "_fetch": fetch_extras}

    # Stage 6: Risk layer
    hed_r = run_hedging(engine_results, prices, cfg_sup)
    stp_r = run_stops(engine_results, prices, 0.0, cfg_sup)
    prt_r = run_portfolio(kel_r, hed_r, stp_r, prices, cfg_sup)
    risk_results = {"hedging": hed_r, "stops": stp_r, "portfolio": prt_r}

    # Stage 7: Consensus
    conf = weighted_confidence(engine_results)
    bias = determine_market_bias(engine_results)

    # For offline runs we approximate elapsed to 0; callers may override.
    elapsed = 0.0
    plan = build_trade_plan(sym, engine_results, risk_results, bias, conf, elapsed)
    plan["success"] = True
    plan.setdefault("symbol", sym)
    plan.setdefault("interval", interval)
    plan.setdefault("bars", bars)
    return plan

