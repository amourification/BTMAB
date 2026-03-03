# =============================================================================
#  consensus/aggregator_utils.py — Aggregator Internal Helpers
#  Split from aggregator.py to keep both modules under 300 lines.
#  Contains: engine weighting, market bias voting, trade plan assembly.
# =============================================================================

import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from datetime import datetime, timezone

logger = logging.getLogger("temporal_bot.consensus.aggregator_utils")

ENGINE_WEIGHTS = {
    "detrend": 0.05, "ssa":  0.18, "acf": 0.15, "fft":    0.12,
    "hilbert": 0.18, "solar": 0.04, "murray": 0.08, "kelly": 0.06,
    "gamma":   0.07, "gann":  0.05, "ar":  0.06, "walras": 0.04,
}


def run_parallel(tasks: list) -> dict:
    """Runs (name, fn, args) tasks in parallel, returns {name: result}."""
    results = {}
    with ThreadPoolExecutor(max_workers=min(len(tasks), 6)) as ex:
        futures = {ex.submit(fn, *args): name for name, fn, args in tasks}
        for future in as_completed(futures):
            name = futures[future]
            try:
                results[name] = future.result()
            except Exception as exc:
                logger.error("Engine '%s' exception: %s", name, exc)
                results[name] = {"success": False, "error": str(exc), "confidence": 0.0}
    return results


def weighted_confidence(engine_results: dict) -> float:
    """Weighted average confidence across all engines (failed → 0.0)."""
    total_w = 0.0; weighted = 0.0
    for name, weight in ENGINE_WEIGHTS.items():
        r    = engine_results.get(name, {})
        conf = float(r.get("confidence", 0.0)) if r.get("success") else 0.0
        weighted += conf * weight
        total_w  += weight
    return round(weighted / (total_w + 1e-12), 4)


def determine_market_bias(engine_results: dict) -> dict:
    """Vote-based market bias from all engines."""
    votes = {"bullish": 0, "bearish": 0, "neutral": 0}
    reasons = []

    h = engine_results.get("hilbert", {})
    if h.get("success"):
        tt = h.get("turn_type", "")
        if tt in ("early_bullish", "accumulation"):
            votes["bullish"] += 2; reasons.append(f"Hilbert: {tt}")
        elif tt in ("distribution",):
            votes["bearish"] += 2; reasons.append(f"Hilbert: {tt}")
        # Treat mid_expansion as informational only (no directional vote)

    fft = engine_results.get("fft", {})
    if fft.get("success"):
        pct = fft.get("cycle_phase", {}).get("pct_complete", 0.5)
        if pct < 0.25:
            votes["bullish"] += 1; reasons.append(f"FFT: early cycle ({pct:.0%})")
        elif pct > 0.75:
            votes["bearish"] += 1; reasons.append(f"FFT: late cycle ({pct:.0%})")

    ar = engine_results.get("ar", {})
    if ar.get("success"):
        d = ar.get("forecast_direction", "flat")
        if d == "up":   votes["bullish"] += 1
        elif d == "down": votes["bearish"] += 1
        reasons.append(f"AR: {d}")

    m = engine_results.get("murray", {})
    if m.get("success"):
        sig = m.get("confluence", {}).get("signal", "neutral")
        if "buy" in sig:  votes["bullish"] += 1
        elif "sell" in sig: votes["bearish"] += 1

    g = engine_results.get("gann", {})
    if g.get("success"):
        pvm = g.get("price_vs_master", "")
        if pvm == "above": votes["bullish"] += 1
        elif pvm == "below": votes["bearish"] += 1

    total = sum(votes.values())
    if total == 0:
        return {"bias": "neutral", "strength": 0.0, "votes": votes, "reasons": reasons}
    if votes["bullish"] > votes["bearish"]:
        bias, strength = "bullish", votes["bullish"] / total
    elif votes["bearish"] > votes["bullish"]:
        bias, strength = "bearish", votes["bearish"] / total
    else:
        bias, strength = "neutral", 0.5
    return {"bias": bias, "strength": round(float(strength), 3),
            "votes": votes, "reasons": reasons}


def build_trade_plan(symbol, engine_results, risk_results, market_bias, consensus_conf, elapsed) -> dict:
    """Assembles the final trade plan dict for GUI and Telegram rendering."""
    h  = engine_results.get("hilbert",  {})
    fft= engine_results.get("fft",      {})
    ssa= engine_results.get("ssa",      {})
    k  = engine_results.get("kelly",    {})
    m  = engine_results.get("murray",   {})
    gm = engine_results.get("gamma",    {})
    w  = engine_results.get("walras",   {})
    sl = engine_results.get("solar",    {})
    hd = risk_results.get("hedging",    {})
    st = risk_results.get("stops",      {})
    pf = risk_results.get("portfolio",  {})

    active_stop = st.get("active_stop", {})
    sweep       = hd.get("sweep_detection", {})
    fft_pc      = fft.get("primary_cycle", {})
    fft_mc      = fft.get("macro_cycle",   {})
    fft_phase   = fft.get("cycle_phase",   {})
    cycle_pct   = fft_phase.get("pct_complete", 0.0)

    # Approximate ETA for next turn using timestamps and bars_to_next_turn
    bars_to_turn = h.get("bars_to_next_turn", {}).get("bars_to_boundary", 0)
    eta_str = ""
    try:
        if bars_to_turn and bars_to_turn > 0:
            fetch = engine_results.get("_fetch", {})
            ts_arr = np.asarray(fetch.get("timestamps", []), dtype=float)
            if ts_arr.size >= 2:
                v_prev, v_last = float(ts_arr[-2]), float(ts_arr[-1])
                delta = max(v_last - v_prev, 1.0)
                # Seconds per bar based on magnitude (ns / ms / s)
                if v_last > 1e15:      # nanoseconds
                    bar_secs = delta / 1e9
                    last_sec = v_last / 1e9
                elif v_last > 1e11:    # milliseconds
                    bar_secs = delta / 1e3
                    last_sec = v_last / 1e3
                else:                  # seconds
                    bar_secs = delta
                    last_sec = v_last
                if bar_secs > 0:
                    dt = datetime.fromtimestamp(
                        last_sec + bar_secs * float(bars_to_turn),
                        tz=timezone.utc,
                    )
                    eta_str = dt.strftime("%Y-%m-%d %H:%M UTC")
    except Exception:
        eta_str = ""

    return {
        # Identity
        "symbol":                 symbol,
        "analysis_time_sec":      round(elapsed, 2),
        "consensus_confidence":   round(consensus_conf * 100, 1),
        # Temporal cycle
        "dominant_cycle_bars":    fft_pc.get("period",       0.0),
        "macro_cycle_bars":       fft_mc.get("period",       0.0),
        "day_in_cycle":           round(cycle_pct * fft_pc.get("period", 0.0), 1),
        "cycle_pct_complete":     round(cycle_pct * 100, 1),
        "ssa_period":             ssa.get("dominant_period", 0.0),
        "ssa_confidence":         round(float(ssa.get("confidence", 0.0)) * 100, 1),
        "ssa_position":           ssa.get("cycle_position",  {}).get("position", "unknown"),
        # Phase & turn
        "phase_deg":              round(float(h.get("phase_deg",      0.0)), 1),
        "turn_type":              h.get("turn_type",        "unknown"),
        "turn_label":             h.get("turn_label",       "Unknown"),
        "turn_urgency":           h.get("turn_urgency",     "low"),
        "bars_to_next_turn":      bars_to_turn,
        "next_turn_eta_utc":      eta_str,
        "next_turn_type":         h.get("bars_to_next_turn",{}).get("next_turn_type",      ""),
        # Price geometry
        "murray_index":           round(float(m.get("murray_index",   0.0)), 2),
        "murray_resistance":      m.get("nearest_levels",   {}).get("resistance_price",   0.0),
        "murray_support":         m.get("nearest_levels",   {}).get("support_price",      0.0),
        "murray_action":          m.get("recommended_action",""),
        # Market context
        "market_bias":            market_bias["bias"],
        "bias_strength":          market_bias["strength"],
        "bias_votes":             market_bias["votes"],
        "bias_reasons":           market_bias["reasons"],
        "gamma_regime":           gm.get("regime",     {}).get("regime",      "unknown"),
        "vol_regime":             gm.get("vol_stats",  {}).get("regime",      "unknown"),
        "seasonal_bias":          sl.get("seasonal_bias",  "unknown"),
        "volatility_flag":        sl.get("volatility_flag", False),
        "walras_adjustment":      float(w.get("kelly_multiplier", 1.0)),
        # Trade plan
        "kelly_position_pct":     float(k.get("position_pct",   0.0)),
        "kelly_tier":             k.get("position_tier",  {}).get("label", ""),
        "kelly_ev":               round(float(k.get("expected_value", 0.0)), 5),
        "hedge_ratio":            hd.get("unified_hedge", {}).get("ratio",   0.0),
        "hedge_pct":              hd.get("unified_hedge", {}).get("pct",     0.0),
        "hedge_action":           hd.get("unified_hedge", {}).get("action",  ""),
        "hedge_urgency":          hd.get("unified_hedge", {}).get("urgency", "none"),
        "sweep_detected":         sweep.get("detected",  False),
        # Stop management
        "active_stop_price":      active_stop.get("stop_price",  0.0),
        "active_stop_type":       active_stop.get("stop_type",   ""),
        "active_stop_pct":        active_stop.get("stop_pct",    0.0),
        "stop_schedule":          st.get("stop_schedule",        []),
        # Portfolio risk
        "portfolio_summary":      pf.get("summary",              ""),
        "risk_reward_grade":      pf.get("risk_adjusted",   {}).get("grade",  "N/A"),
        "overall_risk_score":     float(pf.get("overall_risk_score", 0.5)),
        "drawdown":               pf.get("drawdown",             {}),
        # Raw results for charting
        "_engines":               engine_results,
        "_risk":                  risk_results,
    }
