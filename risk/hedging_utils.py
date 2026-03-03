# =============================================================================
#  risk/hedging_utils.py — Hedging Internal Helpers
#  Split from hedging.py to keep both modules under 300 lines.
# =============================================================================
import numpy as np

SWEEP_LOOKBACK_BARS: int   = 3
SWEEP_REVERSAL_PCT:  float = 0.0015


def consolidate_hedge_signals(engine_results: dict) -> dict:
    signals = {}
    gamma = engine_results.get("gamma", {})
    if gamma.get("success"):
        hr = gamma.get("hedge_recommendation", {})
        signals["gamma"] = {"ratio": float(hr.get("hedge_ratio", 0.0)),
                            "urgency": hr.get("urgency", "none"),
                            "trigger": hr.get("trigger_active", False), "weight": 0.40}
    hilbert = engine_results.get("hilbert", {})
    if hilbert.get("success"):
        um = {"high": 1.0, "medium": 0.5, "low": 0.2, "none": 0.0}
        signals["hilbert"] = {"turn_type": hilbert.get("turn_type", "unknown"),
                              "urgency": hilbert.get("turn_urgency", "none"),
                              "urg_score": um.get(hilbert.get("turn_urgency", "none"), 0.0),
                              "phase_deg": hilbert.get("phase_deg", 0.0), "weight": 0.35}
    murray = engine_results.get("murray", {})
    if murray.get("success"):
        conf = murray.get("confluence", {})
        signals["murray"] = {"confluence_score": float(conf.get("confluence_score", 0.0)),
                             "signal": conf.get("signal", "neutral"),
                             "is_extreme": float(conf.get("confluence_score", 0.0)) > 0.7,
                             "weight": 0.25}
    walras = engine_results.get("walras", {})
    if walras.get("success"):
        signals["walras"] = {"kelly_mult": float(walras.get("kelly_multiplier", 1.0)),
                             "emergency_stop": walras.get("risk_adjustment", {}).get("emergency_stop", False)}
    return signals


def compute_unified_hedge(signals: dict, base_ratio: float) -> dict:
    if signals.get("walras", {}).get("emergency_stop", False):
        return {"ratio": 0.0, "pct": 0.0, "urgency": "emergency", "is_emergency": True,
                "action": "EMERGENCY STOP — Extreme liquidity shock. Close all positions."}
    gamma_r   = signals.get("gamma",   {}).get("ratio",           base_ratio * 0.5)
    hil_urg   = signals.get("hilbert", {}).get("urg_score",       0.0)
    mur_conf  = signals.get("murray",  {}).get("confluence_score",0.0)
    walras_m  = signals.get("walras",  {}).get("kelly_mult",      1.0)
    turn_type = signals.get("hilbert", {}).get("turn_type",       "unknown")
    ratio = gamma_r
    if hil_urg > 0.5 and mur_conf > 0.7:
        ratio = min(ratio + 0.10, 0.50)
    elif hil_urg > 0.5 or mur_conf > 0.7:
        ratio = min(ratio + 0.05, 0.50)
    ratio = float(np.clip(ratio * max(walras_m, 0.5), 0.0, 0.50))
    if ratio == 0.0:
        urg, action = "none", "No hedge required."
    elif ratio < 0.15:
        urg, action = "low", f"Optional {ratio:.0%} hedge — mild signals."
    elif ratio < 0.30:
        urg, action = "moderate", f"Open {ratio:.0%} short hedge — {turn_type} approaching."
    else:
        urg, action = "high", f"Strong {ratio:.0%} hedge — cycle top + extreme level."
    return {"ratio": round(ratio, 3), "pct": round(ratio * 100, 1),
            "action": action, "urgency": urg, "is_emergency": False}


def detect_sweep(prices: np.ndarray, high: np.ndarray, low: np.ndarray, phase_deg: float) -> dict:
    if len(prices) < SWEEP_LOOKBACK_BARS + 2:
        return {"detected": False, "sweep_type": "none", "sweep_level": 0.0, "confidence": 0.0}
    yd_high = float(high[-2]); yd_low = float(low[-2]); cur = float(prices[-1])
    recent  = prices[-(SWEEP_LOOKBACK_BARS + 1):]
    sweep_type = "none"; sweep_level = 0.0
    if recent.max() > yd_high and cur < yd_high:
        if (recent.max() - cur) / (recent.max() + 1e-12) >= SWEEP_REVERSAL_PCT:
            sweep_type, sweep_level = "high_sweep", yd_high
    elif recent.min() < yd_low and cur > yd_low:
        if (cur - recent.min()) / (cur + 1e-12) >= SWEEP_REVERSAL_PCT:
            sweep_type, sweep_level = "low_sweep", yd_low
    if sweep_type == "none":
        return {"detected": False, "sweep_type": "none", "sweep_level": 0.0, "confidence": 0.0}
    phase_ok = (sweep_type == "high_sweep" and 180 <= phase_deg <= 360) or \
               (sweep_type == "low_sweep"  and 270 <= phase_deg <= 360)
    conf = round(min(0.60 + (0.25 if phase_ok else 0.0), 1.0), 3)
    return {"detected": True, "sweep_type": sweep_type,
            "sweep_level": round(sweep_level, 4), "confidence": conf, "phase_confluence": phase_ok}


def pair_divergence_check(engine_results: dict) -> dict:
    ar = engine_results.get("ar_model", {}); hilbert = engine_results.get("hilbert", {})
    if not ar.get("success") or not hilbert.get("success"):
        return {"convergence": 0.5, "note": "Insufficient data."}
    ar_dir = ar.get("forecast_direction", "flat"); h_turn = hilbert.get("turn_type", "unknown")
    bear_h = h_turn in ("distribution",); bull_h = h_turn in ("early_bullish", "accumulation")
    if (bear_h and ar_dir == "down") or (bull_h and ar_dir == "up"):
        return {"convergence": 0.85, "note": "Short/long-term signals aligned."}
    elif ar_dir == "flat":
        return {"convergence": 0.60, "note": "AR flat — neutral divergence."}
    return {"convergence": 0.35, "note": "Signals diverging — reduce size."}
