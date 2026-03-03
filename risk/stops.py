# =============================================================================
#  risk/stops.py — Dynamic Stop Management
#  Phase-adaptive stops anchored to Murray Math pivot levels.
#  Helpers live in stops_utils.py.
#
#  Standard interface:
#      result = run(engine_results, prices, entry_price, config) -> dict
# =============================================================================

import logging
import numpy as np
from risk.stops_utils import (
    compute_atr, phase_stop_multiplier, initial_stop,
    trailing_stop, emergency_stop, stop_schedule, ATR_PERIOD, ATR_MULTIPLIER_BASE,
)

logger = logging.getLogger("temporal_bot.risk.stops")


def run(engine_results: dict, prices: np.ndarray, entry_price: float, cfg: dict) -> dict:
    """
    Computes initial, trailing, and emergency stops tied to cycle phase.

    Parameters
    ----------
    engine_results : dict — needs "hilbert", "murray", "walras", "_fetch"
    prices         : np.ndarray — closing prices
    entry_price    : float — 0 = use last close
    cfg            : dict — "TRADE_DIRECTION", "ATR_MULTIPLIER"

    Returns
    -------
    dict — success, entry_price, direction, atr, phase_deg,
           phase_multiplier, initial_stop, trailing_stop, emergency_stop,
           stop_schedule, active_stop, confidence, metadata, error
    """
    _empty = {
        "success": False, "entry_price": 0.0, "direction": "long",
        "atr": 0.0, "phase_deg": 0.0, "phase_multiplier": 2.0,
        "initial_stop": {}, "trailing_stop": {}, "emergency_stop": {},
        "stop_schedule": [], "active_stop": {}, "confidence": 0.0,
        "metadata": {}, "error": None,
    }

    if prices is None or len(prices) < ATR_PERIOD + 2:
        _empty["error"] = f"Stops: need >= {ATR_PERIOD + 2} prices."
        logger.error(_empty["error"])
        return _empty

    direction  = str(cfg.get("TRADE_DIRECTION", "long")).lower()
    atr_mult   = float(cfg.get("ATR_MULTIPLIER", ATR_MULTIPLIER_BASE))
    if entry_price <= 0:
        entry_price = float(prices[-1])

    try:
        fetch   = engine_results.get("_fetch", {})
        high    = fetch.get("high", prices + prices * 0.005)
        low     = fetch.get("low",  prices - prices * 0.005)
        atr     = compute_atr(high, low, prices, ATR_PERIOD)

        hilbert = engine_results.get("hilbert", {})
        phase   = float(hilbert.get("phase_deg", 180.0))
        t_type  = str(hilbert.get("turn_type",   "unknown"))
        p_mult  = phase_stop_multiplier(phase, t_type)

        murray  = engine_results.get("murray", {})
        levels  = murray.get("levels", np.array([]))
        walras  = engine_results.get("walras", {})

        init_s  = initial_stop(entry_price, atr, p_mult * atr_mult, levels, direction)
        trail_s = trailing_stop(prices, entry_price, atr, phase, direction, levels)
        emrg_s  = emergency_stop(prices, walras)
        sched   = stop_schedule(phase, entry_price, atr, direction)

        cands   = [init_s, trail_s]
        active  = max(cands, key=lambda s: s["stop_price"]) if direction == "long" \
                  else min(cands, key=lambda s: s["stop_price"])
        if emrg_s["triggered"]:
            active = emrg_s

        confidence = round(float(
            0.5 * float(hilbert.get("confidence", 0.5)) +
            0.5 * float(murray.get("confidence",  0.5))
        ), 4)

        logger.info("Stops OK: entry=%.2f %s atr=%.2f phase=%.1f° active=%s conf=%.3f",
                    entry_price, direction, atr, phase, active["stop_type"], confidence)

        return {
            "success": True, "entry_price": round(entry_price, 4),
            "direction": direction, "atr": round(atr, 4),
            "phase_deg": phase, "phase_multiplier": p_mult,
            "initial_stop": init_s, "trailing_stop": trail_s,
            "emergency_stop": emrg_s, "stop_schedule": sched,
            "active_stop": active, "confidence": confidence,
            "metadata": {"atr_period": ATR_PERIOD, "atr_mult": atr_mult},
            "error": None,
        }

    except Exception as exc:
        msg = f"{type(exc).__name__}: {exc}"
        logger.exception("Stops failed: %s", msg)
        _empty["error"] = msg
        return _empty


if __name__ == "__main__":
    np.random.seed(2)
    prices = np.cumsum(np.random.randn(100) * 300) + 60000
    fake = {
        "hilbert": {"success": True, "phase_deg": 230.0, "turn_type": "distribution", "confidence": 0.70},
        "murray":  {"success": True, "confidence": 0.65, "levels": np.linspace(55000, 70000, 9)},
        "walras":  {"liquidity_shock": {"shock_detected": False, "shock_magnitude": 0.5}},
        "_fetch":  {"high": prices + 400, "low": prices - 400},
    }
    r = run(fake, prices, 62000.0, {"TRADE_DIRECTION": "long"})
    if r["success"]:
        print(f"✅ Stops OK | active={r['active_stop']['stop_type']} @ {r['active_stop']['stop_price']:.2f}")
    else:
        print(f"❌ {r['error']}")
