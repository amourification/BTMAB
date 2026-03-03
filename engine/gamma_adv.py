from __future__ import annotations

from typing import Dict, Any

import numpy as np


def _scalar_proxy(value: Any) -> float:
    """Convert gamma proxy (array/list/scalar) into a single float."""
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return 0.0
        return float(value[-1])
    if isinstance(value, (list, tuple)):
        if not value:
            return 0.0
        return float(value[-1])
    try:
        return float(value)
    except Exception:
        return 0.0


def run(gamma_result: Dict[str, Any], regime: Dict[str, Any], cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Regime-aware wrapper around the existing gamma engine output.

    It does not change the classic gamma values; instead it derives a
    smoothed hedge urgency / regime summary that the advanced pipeline
    can consume without affecting the original behaviour.
    """
    base = gamma_result or {}
    reg = regime or {}

    regime_label = str(reg.get("vol_regime", "normal")).lower()
    stress_flag = bool(reg.get("stress_flag", False))
    stress_score = float(reg.get("stress_score", 0.0) or 0.0)

    proxy = _scalar_proxy(base.get("gamma_proxy"))
    if proxy == 0.0:
        proxy = _scalar_proxy(base.get("gamma"))
    regime_gamma = base.get("regime", {}) or {}
    gamma_regime = str(regime_gamma.get("regime", "neutral")).lower()

    # Base urgency from gamma regime
    if gamma_regime == "negative":
        urgency = 0.6
    elif gamma_regime == "positive":
        urgency = 0.3
    else:
        urgency = 0.4

    # Volatility / stress adjustments
    if regime_label in ("high", "very_high"):
        urgency += 0.15
    if stress_flag:
        urgency = max(urgency, 0.8, stress_score)

    # Strength of hedge recommendation using a simple sigmoid on proxy
    scale = float(cfg.get("GAMMA_HEDGE_THRESHOLD", -0.02) or -0.02)
    # When proxy much lower than threshold, hedging is stronger
    x = (proxy - scale) / max(abs(scale), 1e-3)
    hedge_intensity = 1.0 / (1.0 + pow(2.718281828, x * 2.0))

    # Final combined urgency 0..1
    combined = max(0.0, min(1.0, 0.5 * urgency + 0.5 * hedge_intensity))

    return {
        "success": base.get("success", True),
        "gamma_proxy": proxy,
        "gamma_regime": gamma_regime,
        "vol_regime": regime_label,
        "stress_flag": stress_flag,
        "stress_score": stress_score,
        "hedge_urgency_score": round(combined, 3),
    }

