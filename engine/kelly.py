# =============================================================================
#  engine/kelly.py — Equation 7: Kelly Criterion & Optimal Position Sizing
#  Determines the mathematically optimal fraction of capital to risk per
#  trade to maximise long-term compounded growth rate.
#
#  f* = (b·p − q) / b     where:
#    p = win probability (from ACF historical accuracy or SSA confidence)
#    q = 1 − p  (loss probability)
#    b = win/loss ratio (expected gain ÷ expected loss)
#
#  Uses "Half-Kelly" (f* × 0.5) to capture majority of returns while
#  significantly reducing catastrophic drawdown risk.
#
#  Standard interface:
#      result = run(engine_results, config) -> dict
# =============================================================================

import logging
import numpy as np

logger = logging.getLogger("temporal_bot.engine.kelly")

KELLY_FRACTION:    float = 0.5    # half-Kelly multiplier
MAX_POSITION_PCT:  float = 0.25   # hard cap: never risk more than 25%
MIN_CYCLES_REQD:   int   = 5      # min historical cycles before Kelly is trusted
DEFAULT_WIN_RATIO: float = 1.5    # default b if not computable from data


# ── Internal helpers ──────────────────────────────────────────────────────────

def _kelly_fraction(p: float, b: float) -> float:
    """
    Full Kelly fraction: f* = (b·p − q) / b
    Returns 0.0 if Kelly is negative (expected value is negative — don't trade).
    """
    q  = 1.0 - p
    f  = (b * p - q) / b
    return float(max(0.0, f))


def _half_kelly(f_full: float, multiplier: float = KELLY_FRACTION) -> float:
    """
    Applies the fractional Kelly multiplier and caps at MAX_POSITION_PCT.
    """
    return float(min(f_full * multiplier, MAX_POSITION_PCT))


def _derive_win_probability(engine_results: dict) -> float:
    """
    Derives the win probability p from a consensus of available engine outputs.

    Priority (highest to lowest confidence source):
    1. ACF best_correlation (direct historical repetition measure)
    2. SSA confidence (separability × cycle strength)
    3. FFT spectral reliability
    4. Hilbert confidence
    5. Fallback: 0.55 (slight edge)
    """
    scores = []

    acf = engine_results.get("acf", {})
    if acf.get("success"):
        corr = abs(float(acf.get("best_correlation", 0)))
        # Convert correlation [0,1] to win probability [0.5, 0.95]
        p_acf = 0.5 + corr * 0.45
        scores.append(("acf", p_acf, 0.35))   # (source, p, weight)

    ssa = engine_results.get("ssa", {})
    if ssa.get("success"):
        p_ssa = 0.5 + float(ssa.get("confidence", 0)) * 0.40
        scores.append(("ssa", p_ssa, 0.30))

    fft = engine_results.get("fft", {})
    if fft.get("success"):
        p_fft = 0.5 + float(fft.get("confidence", 0)) * 0.35
        scores.append(("fft", p_fft, 0.20))

    hilbert = engine_results.get("hilbert", {})
    if hilbert.get("success"):
        p_hil = 0.5 + float(hilbert.get("confidence", 0)) * 0.35
        scores.append(("hilbert", p_hil, 0.15))

    if not scores:
        logger.warning("Kelly: no engine results available — using fallback p=0.55")
        return 0.55

    total_weight = sum(s[2] for s in scores)
    p_weighted   = sum(s[1] * s[2] for s in scores) / total_weight
    return float(np.clip(p_weighted, 0.50, 0.95))


def _derive_win_ratio(engine_results: dict, prices: np.ndarray) -> float:
    """
    Estimates the win/loss ratio b from:
    - SSA amplitude (expected cycle move) vs recent ATR (expected loss)
    - Falls back to DEFAULT_WIN_RATIO if data is insufficient.

    b = expected_gain / expected_loss
    """
    if prices is None or len(prices) < 14:
        return DEFAULT_WIN_RATIO

    # Average True Range (simplified: uses close-to-close)
    returns      = np.abs(np.diff(prices))
    atr_14       = float(returns[-14:].mean()) if len(returns) >= 14 else float(returns.mean())

    # Expected gain: use SSA amplitude or a fraction of recent range
    ssa = engine_results.get("ssa", {})
    if ssa.get("success") and len(ssa.get("reconstruction", [])) > 0:
        recon      = np.array(ssa["reconstruction"])
        cycle_amp  = float(recon.max() - recon.min())
        # Expected gain = half amplitude (enter mid-cycle, exit at peak)
        exp_gain   = cycle_amp / 2.0
    else:
        # Fallback: use 5% of current price as expected gain
        exp_gain = float(prices[-1]) * 0.05

    if atr_14 < 1e-12:
        return DEFAULT_WIN_RATIO

    b = exp_gain / atr_14
    return float(np.clip(b, 0.5, 10.0))   # reasonable bounds


def _position_risk_tiers(f_half: float) -> dict:
    """
    Translates the Kelly fraction into tiered risk guidance for the GUI output.
    """
    if f_half <= 0:
        return {"tier": "no_trade", "label": "No Edge — Stand Aside", "color": "grey"}
    elif f_half < 0.05:
        return {"tier": "minimal",  "label": "Minimal (< 5%)",         "color": "yellow"}
    elif f_half < 0.10:
        return {"tier": "small",    "label": "Small (5–10%)",           "color": "lightgreen"}
    elif f_half < 0.15:
        return {"tier": "moderate", "label": "Moderate (10–15%)",       "color": "green"}
    elif f_half < 0.20:
        return {"tier": "large",    "label": "Large (15–20%)",          "color": "orange"}
    else:
        return {"tier": "maximum",  "label": "Maximum (20–25% cap)",    "color": "red"}


def _expected_value(p: float, b: float, f: float) -> float:
    """
    Expected value per unit risked: EV = p·b·f − q·f
    Normalised per $1 of portfolio.
    """
    q = 1.0 - p
    return float(p * b * f - q * f)


# ── Public interface ──────────────────────────────────────────────────────────

def run(engine_results: dict, cfg: dict) -> dict:
    """
    Computes optimal position size using the Kelly Criterion.

    Parameters
    ----------
    engine_results : dict — must contain sub-dicts keyed by engine name:
                     "acf", "ssa", "fft", "hilbert"
                     Also accepts "prices" key (np.ndarray) for ATR calc.
    cfg            : dict — keys:
                     "KELLY_FRACTION"    (float, default 0.5)
                     "MAX_POSITION_PCT"  (float, default 0.25)
                     "KELLY_MIN_CYCLES"  (int,   default 5)

    Returns
    -------
    dict with keys:
        "success"          : bool
        "p_win"            : float  — derived win probability
        "p_loss"           : float  — 1 − p_win
        "win_ratio_b"      : float  — expected gain / expected loss ratio
        "kelly_full"       : float  — full Kelly fraction f*
        "kelly_half"       : float  — half-Kelly (recommended)
        "position_pct"     : float  — final position size as % (capped)
        "position_tier"    : dict   — tier label + color for GUI
        "expected_value"   : float  — EV per $1 of portfolio
        "sources"          : dict   — p contributions per engine
        "confidence"       : float  — [0, 1]
        "warning"          : str | None — shown if min cycles not met
        "metadata"         : dict
        "error"            : str | None
    """
    _empty = {
        "success":        False,
        "p_win":          0.0,
        "p_loss":         1.0,
        "win_ratio_b":    0.0,
        "kelly_full":     0.0,
        "kelly_half":     0.0,
        "position_pct":   0.0,
        "position_tier":  {},
        "expected_value": 0.0,
        "sources":        {},
        "confidence":     0.0,
        "warning":        None,
        "metadata":       {},
        "error":          None,
    }

    kelly_mult   = float(cfg.get("KELLY_FRACTION",   KELLY_FRACTION))
    max_pos      = float(cfg.get("MAX_POSITION_PCT", MAX_POSITION_PCT))
    min_cycles   = int(cfg.get("KELLY_MIN_CYCLES",   MIN_CYCLES_REQD))

    try:
        prices = engine_results.get("prices")

        # ── Derive p and b ────────────────────────────────────────────────────
        p   = _derive_win_probability(engine_results)
        b   = _derive_win_ratio(engine_results, prices)
        q   = 1.0 - p

        # ── Kelly calculation ─────────────────────────────────────────────────
        f_full = _kelly_fraction(p, b)
        f_half = min(_half_kelly(f_full, kelly_mult), max_pos)
        pos_pct = f_half * 100   # as percentage

        # ── Supporting outputs ────────────────────────────────────────────────
        tier    = _position_risk_tiers(f_half)
        ev      = _expected_value(p, b, f_half)

        # ── Confidence ────────────────────────────────────────────────────────
        # Kelly confidence = how far above 0.5 the win probability is
        confidence = round(float((p - 0.5) * 2), 4)   # [0, 1]
        confidence = max(0.0, min(1.0, confidence))

        # ── Warning if not enough cycles ──────────────────────────────────────
        warning = None
        acf_cycles = len(engine_results.get("acf", {}).get("cycle_lags", []))
        if acf_cycles < min_cycles:
            warning = (
                f"Only {acf_cycles} confirmed cycles detected "
                f"(need {min_cycles}). Kelly estimate is less reliable."
            )

        logger.info(
            "Kelly OK: p=%.3f b=%.2f f*=%.3f half_kelly=%.3f "
            "pos_pct=%.1f%% EV=%.4f confidence=%.3f",
            p, b, f_full, f_half, pos_pct, ev, confidence,
        )

        return {
            "success":        True,
            "p_win":          round(p, 4),
            "p_loss":         round(q, 4),
            "win_ratio_b":    round(b, 3),
            "kelly_full":     round(f_full, 4),
            "kelly_half":     round(f_half, 4),
            "position_pct":   round(pos_pct, 2),
            "position_tier":  tier,
            "expected_value": round(ev, 5),
            "sources":        {"p_derived_from": "acf+ssa+fft+hilbert"},
            "confidence":     confidence,
            "warning":        warning,
            "metadata":       {
                "kelly_multiplier": kelly_mult,
                "max_position_pct": max_pos,
                "p": p, "b": b,
            },
            "error":          None,
        }

    except Exception as exc:
        msg = f"{type(exc).__name__}: {exc}"
        logger.exception("Kelly failed: %s", msg)
        _empty["error"] = msg
        return _empty


# ── Smoke test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    fake_engines = {
        "acf":     {"success": True, "best_correlation": 0.72, "cycle_lags": list(range(7))},
        "ssa":     {"success": True, "confidence": 0.80, "reconstruction": list(range(100))},
        "fft":     {"success": True, "confidence": 0.75},
        "hilbert": {"success": True, "confidence": 0.65},
        "prices":  np.linspace(50000, 65000, 100),
    }
    result = run(fake_engines, {})
    if result["success"]:
        print(f"✅ Kelly OK")
        print(f"   p_win       : {result['p_win']:.3f}")
        print(f"   win_ratio_b : {result['win_ratio_b']:.2f}")
        print(f"   Kelly full  : {result['kelly_full']:.3f} ({result['kelly_full']*100:.1f}%)")
        print(f"   Half-Kelly  : {result['kelly_half']:.3f} ({result['position_pct']:.1f}%)")
        print(f"   Tier        : {result['position_tier']}")
        print(f"   EV          : {result['expected_value']:.5f}")
        print(f"   Confidence  : {result['confidence']:.3f}")
        if result["warning"]:
            print(f"   ⚠️  {result['warning']}")
    else:
        print(f"❌ {result['error']}")
