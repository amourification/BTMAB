from __future__ import annotations

from typing import Dict, Any


def run(walras_result: Dict[str, Any], regime: Dict[str, Any], cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Regime-aware wrapper around the Walras engine output.

    Computes a simple liquidity stress score and an emergency flag that
    can be used by the advanced pipeline, leaving the original Walras
    values untouched.
    """
    base = walras_result or {}
    reg = regime or {}

    adj = float(base.get("adjustment_factor", 0.0) or base.get("kelly_multiplier", 1.0) or 1.0)
    shock_flag = bool(base.get("liquidity_shock", False))
    pressure = str(base.get("directional_pressure", "neutral")).lower()

    stress_score = float(reg.get("stress_score", 0.0) or 0.0)
    regime_label = str(reg.get("vol_regime", "normal")).lower()

    # Map adjustment magnitude + regime into a 0..1 liquidity stress score
    mag = abs(adj - 1.0)
    base_stress = min(mag * 2.0, 1.0)  # large adjustment → high stress

    if regime_label in ("high", "very_high"):
        base_stress = max(base_stress, 0.6)

    combined = max(base_stress, stress_score)
    if shock_flag:
        combined = 1.0

    emergency = combined > 0.85

    return {
        "success": base.get("success", True),
        "directional_pressure": pressure,
        "adjustment_factor": adj,
        "regime_vol": regime_label,
        "regime_stress_score": stress_score,
        "liquidity_stress_score": round(combined, 3),
        "emergency_flag": emergency,
    }

