"""
consensus/advanced_aggregator.py — Advanced Analysis Pipeline

Provides a second, more adaptive analysis path alongside the classic
aggregator. It reuses the same data / engine / risk layers wherever
possible and augments the trade plan with regime- and spectrum-aware
fields. The public interface intentionally mirrors consensus.aggregator:

    result = run(symbol, interval, bars, config) -> dict
"""

from __future__ import annotations

import logging
import time
from typing import Dict, Any

import numpy as np

from consensus.aggregator_utils import run_parallel

# Reuse the same data / engine / risk modules as the classic pipeline
from data.fetcher import run as fetch_data
from data.preprocessor import run as preprocess
from data.sentiment import run as get_sentiment

from engine.detrend import run as run_detrend
from engine.ssa_utils import run as run_ssa
from engine.fft import run as run_fft
from engine.hilbert import run as run_hilbert
from engine.acf import run as run_acf
from engine.solar import run as run_solar
from engine.murray import run as run_murray
from engine.kelly import run as run_kelly
from engine.gamma import run as run_gamma
from engine.gann import run as run_gann
from engine.elliott_fib import run as run_elliott_fib
from engine.ar_model import run as run_ar
from engine.walras import run as run_walras

from engine.multi_scale_cycles import run as run_multi_scale_cycles
from engine.regime import run as run_regime
from engine.gamma_adv import run as run_gamma_adv
from engine.walras_adv import run as run_walras_adv

from risk.hedging import run as run_hedging
from risk.stops import run as run_stops
from risk.portfolio import run as run_portfolio

from .advanced_consensus import build_advanced_trade_plan

logger = logging.getLogger("temporal_bot.consensus.advanced_aggregator")


def run(symbol: str, interval: str, bars: int, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Full advanced pipeline: fetch → classic engines → multi-scale / regime
    → risk layer → advanced trade plan.
    """
    t0 = time.perf_counter()
    logger.info("Advanced Aggregator START: %s %s %d bars", symbol, interval, bars)
    err_base: Dict[str, Any] = {
        "success": False,
        "mode": "advanced",
        "symbol": symbol,
        "interval": interval,
        "bars": bars,
        "error": None,
        "_engines": {},
        "_risk": {},
    }

    # --- Stage 1: Data --------------------------------------------------------
    fetch_r = fetch_data(symbol, interval, bars, cfg)
    if not fetch_r.get("success"):
        return {**err_base, "error": f"Fetch: {fetch_r.get('error')}"}

    prep_r = preprocess(fetch_r, cfg)
    if not prep_r.get("success"):
        return {**err_base, "error": f"Preprocess: {prep_r.get('error')}"}

    prices = prep_r["close"]
    sessions = prep_r["sessions"]
    sym = fetch_r["symbol"]
    sent_r = get_sentiment(sym, str(sessions[-1]), cfg)

    # --- Stage 2: Detrend -----------------------------------------------------
    det_r = run_detrend(prices, cfg)
    if not det_r.get("success"):
        return {**err_base, "error": f"Detrend: {det_r.get('error')}"}
    detrended = det_r["detrended"]

    # --- Stage 3: Core engines (parallel) -------------------------------------
    core = run_parallel(
        [
            ("ssa", run_ssa, (detrended, cfg)),
            ("fft", run_fft, (detrended, cfg)),
            ("hilbert", run_hilbert, (detrended, cfg)),
        ]
    )
    # ACF re-run with SSA period for cross-validation
    ssa_period = core.get("ssa", {}).get("dominant_period", 0.0)
    core["acf"] = run_acf(detrended, {**cfg, "SSA_PERIOD": ssa_period})

    turn_type = core.get("hilbert", {}).get("turn_type", "unknown")
    fft_period = (
        core.get("fft", {}).get("primary_cycle", {}).get("period", cfg.get("DOMINANT_CYCLE_BARS", 512))
    )
    cfg_sup = {
        **cfg,
        "HILBERT_TURN_TYPE": turn_type,
        "FFT_PRIMARY_PERIOD": fft_period,
        "SSA_PERIOD": ssa_period,
    }

    # --- Stage 4: Support engines (classic) -----------------------------------
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

    # --- Stage 5: Sequential engines (classic) --------------------------------
    mid_eng = {**core, **sup, "detrend": det_r}
    ar_r = run_ar(prep_r, sent_r, cfg_sup)
    kel_r = run_kelly({**mid_eng, "ar": ar_r, "prices": prices}, cfg_sup)

    fetch_extras = {
        "high": fetch_r.get("high", prices),
        "low": fetch_r.get("low", prices),
        "timestamps": fetch_r.get("timestamps", np.array([])),
    }
    engine_results: Dict[str, Any] = {
        **mid_eng,
        "ar": ar_r,
        "kelly": kel_r,
        "_fetch": fetch_extras,
    }

    # --- Stage 6: Advanced-only engines --------------------------------------
    ms_cycles = run_multi_scale_cycles(detrended, cfg_sup)
    cfg_regime = {
        **cfg_sup,
        "_REGIME_HIGH": fetch_r.get("high"),
        "_REGIME_LOW": fetch_r.get("low"),
    }
    regime = run_regime(prices, detrended, cfg_regime)
    gamma_adv = run_gamma_adv(engine_results.get("gamma", {}), regime, cfg_sup)
    walras_adv = run_walras_adv(engine_results.get("walras", {}), regime, cfg_sup)

    engine_results["multi_scale_cycles"] = ms_cycles
    engine_results["regime"] = regime
    engine_results["gamma_adv"] = gamma_adv
    engine_results["walras_adv"] = walras_adv

    # --- Stage 7: Risk layer (reuse classic) ---------------------------------
    hed_r = run_hedging(engine_results, prices, cfg_sup)
    stp_r = run_stops(engine_results, prices, 0.0, cfg_sup)
    prt_r = run_portfolio(kel_r, hed_r, stp_r, prices, cfg_sup)
    risk_results: Dict[str, Any] = {"hedging": hed_r, "stops": stp_r, "portfolio": prt_r}

    # --- Stage 8: Advanced consensus / trade plan ----------------------------
    elapsed = time.perf_counter() - t0
    plan = build_advanced_trade_plan(sym, engine_results, risk_results, regime, ms_cycles, elapsed)
    plan["success"] = True
    plan.setdefault("symbol", sym)
    plan.setdefault("interval", interval)
    plan.setdefault("bars", bars)
    logger.info(
        "Advanced Aggregator DONE: %s bias=%s conf=%.1f%% mode=%s %.2fs",
        sym,
        plan.get("market_bias"),
        plan.get("consensus_confidence", 0.0),
        plan.get("mode"),
        elapsed,
    )
    return plan


