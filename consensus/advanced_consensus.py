import logging
from typing import Dict, Any

from .aggregator_utils import (
    weighted_confidence,
    determine_market_bias,
    build_trade_plan,
)

try:
    # Optional meta-model hook; safe to ignore if unavailable.
    from advanced.meta_model import predict_meta_signal
except Exception:  # pragma: no cover - defensive import
    def predict_meta_signal(_features):  # type: ignore[no-redef]
        return {}

logger = logging.getLogger("temporal_bot.consensus.advanced_consensus")


def compute_advanced_confidence(engine_results: Dict[str, Any]) -> float:
    """
    Wrapper around weighted_confidence() that leaves room for
    future regime- /performance-aware reweighting.
    """
    return weighted_confidence(engine_results)


def compute_advanced_bias(engine_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Wrapper around determine_market_bias() that can later inject
    regime- or ML-based adjustments to the raw bias vote.
    """
    return determine_market_bias(engine_results)


def compute_uncertainty(
    engine_results: Dict[str, Any],
    regime: Dict[str, Any],
    cycles: Dict[str, Any],
) -> float:
    """
    Derive a simple 0..1 uncertainty score from:
      - Engine disagreement (bull vs bear votes close together)
      - Volatility / stress regime
      - Cycle spectrum instability across windows

    This is intentionally lightweight and deterministic; it can be
    refined or replaced later without changing the public interface.
    """
    # --- Engine disagreement ---
    bias = determine_market_bias(engine_results)
    votes = bias.get("votes", {}) or {}
    bull = float(votes.get("bullish", 0))
    bear = float(votes.get("bearish", 0))
    total = max(bull + bear + float(votes.get("neutral", 0)), 1.0)
    disagreement = 1.0 - abs(bull - bear) / total  # 0 = clear, 1 = tied

    # --- Volatility / stress regime ---
    reg_label = str(regime.get("vol_regime", "normal")).lower()
    stress_flag = bool(regime.get("stress_flag", False))
    stress_score = float(regime.get("stress_score", 0.0))
    if reg_label in ("high", "very_high"):
        vol_term = 0.7
    elif reg_label in ("low",):
        vol_term = 0.2
    else:
        vol_term = 0.4
    if stress_flag:
        vol_term = max(vol_term, 0.8, min(1.0, stress_score))

    # --- Cycle spectrum stability across windows ---
    ms = cycles or {}
    windows = ms.get("windows", [])
    periods = [float(w.get("period", 0.0)) for w in windows if w.get("success")]
    if len(periods) >= 2:
        p_min, p_max = min(periods), max(periods)
        span = max(p_max - p_min, 0.0)
        denom = max(p_max, 1.0)
        spectrum_instability = min(span / denom, 1.0)
    else:
        spectrum_instability = 0.3  # neutral default when we don't know

    # Weighted combination (clamped)
    raw = 0.5 * disagreement + 0.3 * vol_term + 0.2 * spectrum_instability
    return round(max(0.0, min(1.0, raw)), 3)


def build_advanced_trade_plan(
    symbol: str,
    engine_results: Dict[str, Any],
    risk_results: Dict[str, Any],
    regime: Dict[str, Any],
    cycles: Dict[str, Any],
    elapsed: float,
) -> Dict[str, Any]:
    """
    Build a trade plan compatible with the classic one, then
    augment it with advanced-only fields. Existing GUI / bot
    code can continue to rely on the classic keys.
    """
    # Reuse the existing helpers for baseline values
    conf = compute_advanced_confidence(engine_results)
    bias = compute_advanced_bias(engine_results)

    base = build_trade_plan(symbol, engine_results, risk_results, bias, conf, elapsed)

    # Attach advanced-only fields in a dedicated namespace
    uncertainty = compute_uncertainty(engine_results, regime, cycles)

    # Apply a simple uncertainty/stress-aware adjustment to Kelly sizing so
    # advanced mode meaningfully differs from classic while remaining
    # deterministic and conservative.
    kelly_pct = float(base.get("kelly_position_pct", 0.0) or 0.0)
    stress_score = float(regime.get("stress_score", 0.0) or 0.0)
    if kelly_pct > 0.0:
        # 0 uncertainty & 0 stress → factor ~1.0
        # high uncertainty/stress → factor floored at 0.25
        factor = 1.0 - 0.7 * uncertainty - 0.3 * stress_score
        factor = max(0.25, min(1.0, factor))
        adj_kelly_pct = round(kelly_pct * factor, 4)
        base["kelly_position_pct"] = adj_kelly_pct
    else:
        adj_kelly_pct = kelly_pct

    # Optional meta-model adjustments (currently a no-op unless the user
    # plugs in a real implementation in advanced/meta_model.py).
    meta_features = {
        "symbol": symbol,
        "elapsed": elapsed,
        "regime": regime,
        "cycles": cycles,
        "engines": engine_results,
        "risk": risk_results,
        "base_confidence": conf,
        "base_bias": bias,
    }
    meta = predict_meta_signal(meta_features) or {}

    advanced_block = {
        "mode": "advanced",
        "advanced": {
            "vol_regime": regime.get("vol_regime"),
            "trend_regime": regime.get("trend_regime"),
            "stress_flag": bool(regime.get("stress_flag", False)),
            "stress_score": float(regime.get("stress_score", 0.0)),
            "uncertainty_score": uncertainty,
            "multi_scale_cycles": cycles,
             # Expose both raw and adjusted Kelly for transparency in
             # advanced mode UIs.
            "kelly_raw_pct": kelly_pct,
            "kelly_adj_pct": adj_kelly_pct,
            "meta": meta,
        },
    }

    plan = {**base, **advanced_block}
    logger.info(
        "Advanced plan: mode=%s vol_regime=%s trend_regime=%s uncertainty=%.3f",
        plan.get("mode"),
        regime.get("vol_regime"),
        regime.get("trend_regime"),
        uncertainty,
    )
    return plan

