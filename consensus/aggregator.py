# =============================================================================
#  consensus/aggregator.py — Master Consensus Engine
#  Orchestrates all 12 equations + risk modules and produces the final
#  trade plan dict consumed by main_gui.py and bot/telegram_bot.py.
#
#  Pipeline stages:
#    1. Data       : fetcher → preprocessor → sentiment
#    2. Detrend    : removes trend (required by all cycle engines)
#    3. Core       : ssa, acf, fft, hilbert  (parallel)
#    4. Support    : solar, murray, gamma, gann, walras  (parallel)
#    5. Sequential : ar_model → kelly  (need prior engine outputs)
#    6. Risk       : hedging → stops → portfolio
#    7. Consensus  : weighted confidence + bias vote → trade plan
#
#  Standard interface:
#      result = run(symbol, interval, bars, config) -> dict
# =============================================================================

import logging
import time
import numpy as np

from consensus.aggregator_utils import (
    run_parallel, weighted_confidence, determine_market_bias, build_trade_plan,
)

# Data layer
from data.fetcher      import run as fetch_data
from data.preprocessor import run as preprocess
from data.sentiment    import run as get_sentiment

# Core engines
from engine.detrend    import run as run_detrend
from engine.ssa_utils  import run as run_ssa
from engine.acf        import run as run_acf
from engine.fft        import run as run_fft
from engine.hilbert    import run as run_hilbert

# Support engines
from engine.solar      import run as run_solar
from engine.murray     import run as run_murray
from engine.kelly      import run as run_kelly
from engine.gamma      import run as run_gamma
from engine.gann       import run as run_gann
from engine.elliott_fib import run as run_elliott_fib
from engine.ar_model   import run as run_ar
from engine.walras     import run as run_walras

# Risk layer
from risk.hedging      import run as run_hedging
from risk.stops        import run as run_stops
from risk.portfolio    import run as run_portfolio

logger = logging.getLogger("temporal_bot.consensus.aggregator")


def run(symbol: str, interval: str, bars: int, cfg: dict) -> dict:
    """
    Full pipeline: fetch → 12 engines → risk layer → trade plan.

    Parameters
    ----------
    symbol   : str  — e.g. "BTCUSDT"
    interval : str  — e.g. "1d"
    bars     : int  — bars to fetch (recommended ≥ 512)
    cfg      : dict — full config from config.py

    Returns
    -------
    dict — all trade plan keys plus "success" and "error"
    """
    t0 = time.perf_counter()
    logger.info("Aggregator START: %s %s %d bars", symbol, interval, bars)
    err_base = {"success": False, "symbol": symbol, "error": None, "_engines": {}, "_risk": {}}

    # ── Stage 1: Data ─────────────────────────────────────────────────────────
    fetch_r = fetch_data(symbol, interval, bars, cfg)
    if not fetch_r["success"]:
        return {**err_base, "error": f"Fetch: {fetch_r['error']}"}

    prep_r = preprocess(fetch_r, cfg)
    if not prep_r["success"]:
        return {**err_base, "error": f"Preprocess: {prep_r['error']}"}

    prices   = prep_r["close"]
    sessions = prep_r["sessions"]
    sym      = fetch_r["symbol"]
    sent_r   = get_sentiment(sym, str(sessions[-1]), cfg)

    # ── Stage 2: Detrend ──────────────────────────────────────────────────────
    det_r = run_detrend(prices, cfg)
    if not det_r["success"]:
        return {**err_base, "error": f"Detrend: {det_r['error']}"}
    detrended = det_r["detrended"]

    # ── Stage 3: Core engines (parallel) ─────────────────────────────────────
    core = run_parallel([
        ("ssa",     run_ssa,     (detrended, cfg)),
        ("fft",     run_fft,     (detrended, cfg)),
        ("hilbert", run_hilbert, (detrended, cfg)),
    ])
    # ACF re-run with SSA period for cross-validation
    ssa_period = core.get("ssa", {}).get("dominant_period", 0.0)
    core["acf"] = run_acf(detrended, {**cfg, "SSA_PERIOD": ssa_period})

    turn_type  = core.get("hilbert", {}).get("turn_type", "unknown")
    fft_period = core.get("fft",     {}).get("primary_cycle", {}).get("period", 512.0)
    cfg_sup    = {**cfg, "HILBERT_TURN_TYPE": turn_type,
                  "FFT_PRIMARY_PERIOD": fft_period, "SSA_PERIOD": ssa_period}

    # ── Stage 4: Support engines (parallel) ──────────────────────────────────
    sup = run_parallel([
        ("solar",  run_solar,  (prep_r,   cfg_sup)),
        ("murray", run_murray, (prices,   cfg_sup)),
        ("gamma",  run_gamma,  (prices,   cfg_sup)),
        ("gann",   run_gann,   (prices,   cfg_sup)),
        ("elliott_fib", run_elliott_fib, (prices, cfg_sup)),
        ("walras", run_walras, (prices,   cfg_sup)),
    ])

    # ── Stage 5: Sequential engines ──────────────────────────────────────────
    mid_eng = {**core, **sup, "detrend": det_r}
    ar_r    = run_ar(prep_r, sent_r, cfg_sup)
    kel_r   = run_kelly({**mid_eng, "ar": ar_r, "prices": prices}, cfg_sup)

    fetch_extras = {
        "high":       fetch_r.get("high",       prices),
        "low":        fetch_r.get("low",        prices),
        "timestamps": fetch_r.get("timestamps", np.array([])),
    }
    engine_results = {**mid_eng, "ar": ar_r, "kelly": kel_r, "_fetch": fetch_extras}

    # ── Stage 6: Risk layer ───────────────────────────────────────────────────
    hed_r  = run_hedging(engine_results, prices, cfg)
    stp_r  = run_stops(engine_results,   prices, 0.0, cfg)
    prt_r  = run_portfolio(kel_r, hed_r, stp_r, prices, cfg)
    risk_results = {"hedging": hed_r, "stops": stp_r, "portfolio": prt_r}

    # ── Stage 7: Consensus ────────────────────────────────────────────────────
    conf   = weighted_confidence(engine_results)
    bias   = determine_market_bias(engine_results)
    elapsed = time.perf_counter() - t0

    logger.info("Aggregator DONE: %s bias=%s(%.0f%%) conf=%.1f%% %.2fs",
                sym, bias["bias"], bias["strength"] * 100, conf * 100, elapsed)

    plan = build_trade_plan(sym, engine_results, risk_results, bias, conf, elapsed)
    plan["success"] = True
    plan["error"]   = None
    # Attach core metadata used by log writer
    plan.setdefault("symbol", sym)
    plan.setdefault("interval", interval)
    plan.setdefault("bars", bars)
    return plan


# ── Smoke test (synthetic — no live API) ──────────────────────────────────────
if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    np.random.seed(42)
    n = 300; t = np.arange(n)
    prices = (50000 + 5000 * np.sin(2 * np.pi * t / 128)
              + np.cumsum(np.random.randn(n) * 200))

    cfg = {"DOMINANT_CYCLE_BARS": 128, "SSA_WINDOW_LENGTH": 50,
           "SSA_NUM_COMPONENTS": 6,    "ACF_MAX_LAG": 80}

    from engine.detrend   import run as _det
    from engine.ssa_utils import run as _ssa
    from engine.fft       import run as _fft
    from engine.hilbert   import run as _hil
    from engine.murray    import run as _mur
    from engine.kelly     import run as _kel
    from engine.gamma     import run as _gam
    from risk.hedging     import run as _hed
    from risk.stops       import run as _stp
    from risk.portfolio   import run as _prt

    det = _det(prices, cfg)
    ssa = _ssa(det["detrended"], cfg)
    fft = _fft(det["detrended"], cfg)
    hil = _hil(det["detrended"], {**cfg, "FFT_PRIMARY_PERIOD": fft["primary_cycle"].get("period", 128)})
    mur = _mur(prices, {**cfg, "HILBERT_TURN_TYPE": hil["turn_type"]})
    gam = _gam(prices, {**cfg, "HILBERT_TURN_TYPE": hil["turn_type"]})
    eng = {"detrend": det, "ssa": ssa, "fft": fft, "hilbert": hil, "murray": mur,
           "gamma": gam, "prices": prices, "_fetch": {"high": prices+200, "low": prices-200}}
    kel = _kel(eng, cfg); eng["kelly"] = kel
    hed = _hed(eng, prices, cfg)
    stp = _stp(eng, prices, 0.0, cfg)
    prt = _prt(kel, hed, stp, prices, {"TRADE_DIRECTION": "long", "PORTFOLIO_VALUE": 10000})

    conf = weighted_confidence(eng)
    bias = determine_market_bias(eng)
    print(f"✅ Aggregator smoke test OK")
    print(f"   Bias       : {bias['bias']} ({bias['strength']:.0%}) | {bias['votes']}")
    print(f"   Confidence : {conf:.3f}")
    print(f"   Phase      : {hil['phase_deg']:.1f}° — {hil['turn_type']}")
    print(f"   Kelly      : {kel['position_pct']:.1f}%")
    print(f"   Hedge      : {hed['unified_hedge']['pct']:.1f}% — {hed['unified_hedge']['urgency']}")
    print(f"   Stop       : {stp['active_stop']['stop_price']:.2f} ({stp['active_stop']['stop_type']})")
    print(f"   Portfolio  : {prt['summary']}")
