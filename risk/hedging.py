# =============================================================================
#  risk/hedging.py — Hedge Ratio Logic & Trade Protection
#  Produces unified hedge recommendation. Helpers in hedging_utils.py.
#
#  Standard interface:
#      result = run(engine_results, prices, config) -> dict
# =============================================================================

import logging
import numpy as np
from risk.hedging_utils import (
    consolidate_hedge_signals, compute_unified_hedge,
    detect_sweep, pair_divergence_check,
)

logger = logging.getLogger("temporal_bot.risk.hedging")


def run(engine_results: dict, prices: np.ndarray, cfg: dict) -> dict:
    """
    Produces a unified hedge recommendation from all engine outputs.

    Parameters
    ----------
    engine_results : dict — gamma, hilbert, murray, walras, ar_model, _fetch
    prices         : np.ndarray — closing prices
    cfg            : dict — "DEFAULT_HEDGE_RATIO" (float, default 0.30)

    Returns
    -------
    dict — success, signals, unified_hedge, sweep_detection,
           pair_divergence, overall_confidence, metadata, error
    """
    _empty = {
        "success": False, "signals": {}, "unified_hedge": {},
        "sweep_detection": {}, "pair_divergence": {},
        "overall_confidence": 0.0, "metadata": {}, "error": None,
    }

    if prices is None or len(prices) < 5:
        _empty["error"] = f"Hedging: need >= 5 prices, got {len(prices) if prices is not None else 0}."
        logger.error(_empty["error"])
        return _empty

    base_ratio = float(cfg.get("DEFAULT_HEDGE_RATIO", 0.30))

    try:
        signals    = consolidate_hedge_signals(engine_results)
        unified    = compute_unified_hedge(signals, base_ratio)

        fetch  = engine_results.get("_fetch", {})
        high   = fetch.get("high", prices)
        low    = fetch.get("low",  prices)
        phase  = float(engine_results.get("hilbert", {}).get("phase_deg", 0.0))
        sweep  = detect_sweep(prices, high, low, phase)

        if sweep["detected"] and unified["ratio"] > 0:
            boost = round(min(unified["ratio"] + sweep["confidence"] * 0.15, 0.50), 3)
            unified["ratio"] = boost
            unified["pct"]   = round(boost * 100, 1)
            unified["action"] += f" | Sweep at {sweep['sweep_level']:.2f} — boosted to {boost:.0%}."

        divergence    = pair_divergence_check(engine_results)
        gamma_conf    = float(engine_results.get("gamma",   {}).get("confidence",  0.5))
        hilbert_conf  = float(engine_results.get("hilbert", {}).get("confidence",  0.5))
        sweep_bonus   = sweep["confidence"] * 0.10 if sweep["detected"] else 0.0
        div_factor    = divergence["convergence"]
        overall       = round(max(0.0, min(1.0,
            float((gamma_conf * 0.35 + hilbert_conf * 0.35 + sweep_bonus) * div_factor)
        )), 4)

        logger.info("Hedging OK: ratio=%.0f%% urgency=%s sweep=%s conf=%.3f",
                    unified["pct"], unified["urgency"], sweep["sweep_type"], overall)

        return {
            "success": True, "signals": signals, "unified_hedge": unified,
            "sweep_detection": sweep, "pair_divergence": divergence,
            "overall_confidence": overall,
            "metadata": {"base_ratio": base_ratio, "n_bars": len(prices)},
            "error": None,
        }

    except Exception as exc:
        msg = f"{type(exc).__name__}: {exc}"
        logger.exception("Hedging failed: %s", msg)
        _empty["error"] = msg
        return _empty


if __name__ == "__main__":
    np.random.seed(1)
    prices = np.cumsum(np.random.randn(100) * 500) + 60000
    fake = {
        "gamma":   {"success": True, "confidence": 0.72,
                    "hedge_recommendation": {"hedge_ratio": 0.30, "urgency": "immediate", "trigger_active": True}},
        "hilbert": {"success": True, "confidence": 0.68,
                    "turn_type": "distribution", "turn_urgency": "high", "phase_deg": 230.0},
        "murray":  {"success": True, "confluence": {"confluence_score": 0.85, "signal": "high_confluence_sell"}},
        "walras":  {"success": True, "kelly_multiplier": 0.85,
                    "risk_adjustment": {"emergency_stop": False}},
        "ar_model":{"success": True, "forecast_direction": "down"},
        "_fetch":  {"high": prices + 500, "low": prices - 500},
    }
    r = run(fake, prices, {})
    if r["success"]:
        print(f"✅ Hedging OK | ratio={r['unified_hedge']['ratio']:.0%} | conf={r['overall_confidence']:.3f}")
    else:
        print(f"❌ {r['error']}")
