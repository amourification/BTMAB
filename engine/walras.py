# =============================================================================
#  engine/walras.py — Equation 11: Walrasian General Equilibrium
#  Models cross-market interactions to ensure the primary asset's cycle
#  signal is adjusted for broader market equilibrium risk.
#
#  Core idea: if BTC's cycle is bullish but the general equilibrium signals
#  a liquidity drain (e.g., DXY surging, equities falling), Kelly allocation
#  is reduced. Markets are interconnected — ignoring feedback loops
#  is the "enmity of ignorance" referenced in the research doc.
#
#  Standard interface:
#      result = run(primary_prices, cfg) -> dict
# =============================================================================

import logging
import numpy as np

logger = logging.getLogger("temporal_bot.engine.walras")

# Correlation assets tracked for equilibrium analysis
# In live mode these would be fetched from Binance / Polygon
# In stub mode (no secondary data), correlations are inferred from
# internal price structure only.
EQUILIBRIUM_ASSETS = ["BTC", "ETH", "DXY_proxy", "SPX_proxy"]

# Threshold: if market sync score drops below this, reduce Kelly allocation
SYNC_THRESHOLD: float = 0.40
KELLY_REDUCTION_FACTOR: float = 0.70   # reduce Kelly by 30% if out of sync


# ── Internal helpers ──────────────────────────────────────────────────────────

def _rolling_correlation(a: np.ndarray, b: np.ndarray, window: int) -> np.ndarray:
    """
    Computes rolling Pearson correlation between two series over `window` bars.
    Returns an array of length len(a), NaN-padded at the start.
    """
    n      = len(a)
    result = np.full(n, np.nan)
    for i in range(window, n + 1):
        a_w = a[i - window: i]
        b_w = b[i - window: i]
        if a_w.std() < 1e-12 or b_w.std() < 1e-12:
            result[i - 1] = 0.0
        else:
            result[i - 1] = float(np.corrcoef(a_w, b_w)[0, 1])
    return result


def _infer_proxy_series(prices: np.ndarray) -> dict:
    """
    Infers synthetic proxy series from the primary price series when no
    external market data is available.

    Proxies:
    - ETH proxy: smoothed version of BTC returns (correlated crypto)
    - DXY proxy: inverse of BTC (USD strengthens when crypto weakens)
    - SPX proxy: 20-bar moving average returns (risk-on proxy)

    These are approximations — in live mode, real data replaces them.
    """
    log_ret = np.diff(np.log(prices + 1e-12))
    n       = len(log_ret)

    # ETH: similar to BTC but slightly lagged and dampened
    eth_proxy = np.roll(log_ret, 1) * 0.8 + np.random.randn(n) * 0.002
    eth_proxy[0] = log_ret[0]

    # DXY: inverse of crypto with noise
    dxy_proxy = -log_ret * 0.4 + np.random.randn(n) * 0.001

    # SPX: 20-bar rolling mean of BTC returns (risk-on flows)
    spx_proxy = np.convolve(log_ret, np.ones(20) / 20, mode="same")

    return {
        "eth":  eth_proxy,
        "dxy":  dxy_proxy,
        "spx":  spx_proxy,
        "btc":  log_ret,
    }


def _compute_market_sync(
    proxies: dict,
    window: int = 30,
) -> dict:
    """
    Computes pairwise rolling correlations between the primary asset and proxies.
    Market sync score = average absolute correlation across all pairs.

    High sync (> SYNC_THRESHOLD): markets moving together → reliable signal.
    Low sync: divergence → higher uncertainty → reduce Kelly.
    """
    btc = proxies["btc"]
    pairs = {
        "btc_eth": (btc, proxies["eth"]),
        "btc_dxy": (btc, proxies["dxy"]),
        "btc_spx": (btc, proxies["spx"]),
    }

    current_corrs = {}
    for name, (a, b) in pairs.items():
        min_len     = min(len(a), len(b))
        a_trim      = a[:min_len]
        b_trim      = b[:min_len]
        roll_corr   = _rolling_correlation(a_trim, b_trim, min(window, min_len // 2))
        current_corrs[name] = float(roll_corr[~np.isnan(roll_corr)][-1]) if (~np.isnan(roll_corr)).any() else 0.0

    # For DXY: negative correlation is expected and healthy — use abs
    sync_scores = [
        abs(current_corrs["btc_eth"]),   # want high positive (crypto moves together)
        abs(current_corrs["btc_dxy"]),   # want moderate (USD-crypto relationship intact)
        abs(current_corrs["btc_spx"]),   # want moderate (risk-on flows)
    ]
    sync_score = float(np.mean(sync_scores))

    return {
        "correlations":   current_corrs,
        "sync_score":     round(sync_score, 4),
        "is_synchronized": sync_score >= SYNC_THRESHOLD,
    }


def _liquidity_shock_detector(prices: np.ndarray, window: int = 5) -> dict:
    """
    Detects sudden liquidity shocks using abnormal return magnitude.
    A shock is defined as a recent return exceeding 3σ of the rolling std.

    Returns: {shock_detected, shock_magnitude, direction}
    """
    log_ret   = np.diff(np.log(prices + 1e-12))
    if len(log_ret) < window + 5:
        return {"shock_detected": False, "shock_magnitude": 0.0, "direction": "none"}

    roll_std    = float(log_ret[-(window + 5):-window].std())
    recent_ret  = float(log_ret[-1])
    threshold   = 3.0 * roll_std

    shock = abs(recent_ret) > threshold
    return {
        "shock_detected":  shock,
        "shock_magnitude": round(abs(recent_ret) / (roll_std + 1e-12), 2),
        "direction":       "up" if recent_ret > 0 else "down" if recent_ret < 0 else "none",
    }


def _equilibrium_risk_adjustment(
    sync: dict,
    shock: dict,
    primary_turn_type: str,
) -> dict:
    """
    Computes the Kelly allocation adjustment factor based on:
    1. Market synchronization score
    2. Liquidity shock detection
    3. Primary cycle turn type

    Returns a multiplier [0, 1] applied to the Kelly fraction.
    1.0 = no adjustment, 0.0 = zero position (emergency shutdown).
    """
    adjustment  = 1.0
    reasons     = []

    # Desync penalty
    if not sync["is_synchronized"]:
        adjustment *= KELLY_REDUCTION_FACTOR
        reasons.append(
            f"Market desync (sync={sync['sync_score']:.2f} < {SYNC_THRESHOLD})"
        )

    # Liquidity shock penalty
    if shock["shock_detected"]:
        mag = shock["shock_magnitude"]
        if mag > 5:
            adjustment *= 0.0   # emergency shutdown
            reasons.append(f"EXTREME liquidity shock ({mag:.1f}σ) — full stop")
        elif mag > 3:
            adjustment *= 0.30
            reasons.append(f"Major liquidity shock ({mag:.1f}σ) — 70% reduction")
        else:
            adjustment *= 0.60
            reasons.append(f"Liquidity shock ({mag:.1f}σ) — 40% reduction")

    # Counter-trend bonus: if cycle says "top" but markets still synced, trust it more
    if primary_turn_type == "distribution" and sync["is_synchronized"]:
        adjustment = min(adjustment * 1.10, 1.0)
        reasons.append("Synced distribution signal — slight confidence boost")

    return {
        "kelly_multiplier": round(max(0.0, min(1.0, adjustment)), 4),
        "reasons":          reasons,
        "emergency_stop":   adjustment == 0.0,
    }


# ── Public interface ──────────────────────────────────────────────────────────

def run(prices: np.ndarray, cfg: dict) -> dict:
    """
    Computes cross-market equilibrium risk and returns a Kelly adjustment factor.

    Parameters
    ----------
    prices : np.ndarray — primary asset closing prices
    cfg    : dict — keys:
                 "HILBERT_TURN_TYPE"   (str, optional)
                 "WALRAS_SYNC_WINDOW"  (int, default 30)

    Returns
    -------
    dict with keys:
        "success"           : bool
        "market_sync"       : dict   — correlations + sync score
        "liquidity_shock"   : dict   — shock detection result
        "risk_adjustment"   : dict   — Kelly multiplier + reasons
        "kelly_multiplier"  : float  — final multiplier for aggregator
        "is_stub"           : bool   — True when using proxy data (no live feed)
        "confidence"        : float  — [0, 1]
        "metadata"          : dict
        "error"             : str | None
    """
    _empty = {
        "success":          False,
        "market_sync":      {},
        "liquidity_shock":  {},
        "risk_adjustment":  {},
        "kelly_multiplier": 1.0,
        "is_stub":          True,
        "confidence":       0.0,
        "metadata":         {},
        "error":            None,
    }

    if prices is None or len(prices) < 50:
        _empty["error"] = f"Walras: need >= 50 prices, got {len(prices) if prices is not None else 0}."
        logger.error(_empty["error"])
        return _empty

    turn_type  = str(cfg.get("HILBERT_TURN_TYPE", "unknown"))
    sync_window = int(cfg.get("WALRAS_SYNC_WINDOW", 30))

    try:
        np.random.seed(42)   # deterministic proxy generation
        proxies        = _infer_proxy_series(prices)
        market_sync    = _compute_market_sync(proxies, window=sync_window)
        liquidity_shock = _liquidity_shock_detector(prices)
        risk_adj       = _equilibrium_risk_adjustment(
            market_sync, liquidity_shock, turn_type
        )

        # Confidence: higher when markets are synchronized
        confidence = round(
            float(0.5 + market_sync["sync_score"] * 0.5), 4
        )
        confidence = max(0.0, min(1.0, confidence))

        logger.info(
            "Walras OK: sync=%.3f shock=%s kelly_mult=%.3f confidence=%.3f",
            market_sync["sync_score"],
            liquidity_shock["shock_detected"],
            risk_adj["kelly_multiplier"],
            confidence,
        )

        return {
            "success":          True,
            "market_sync":      market_sync,
            "liquidity_shock":  liquidity_shock,
            "risk_adjustment":  risk_adj,
            "kelly_multiplier": risk_adj["kelly_multiplier"],
            "is_stub":          True,   # flip to False when live feeds wired
            "confidence":       confidence,
            "metadata":         {"sync_window": sync_window, "turn_type": turn_type},
            "error":            None,
        }

    except Exception as exc:
        msg = f"{type(exc).__name__}: {exc}"
        logger.exception("Walras failed: %s", msg)
        _empty["error"] = msg
        return _empty


# ── Smoke test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    np.random.seed(7)
    prices = np.cumsum(np.random.randn(300) * 500) + 50000

    result = run(prices, {"HILBERT_TURN_TYPE": "distribution"})
    if result["success"]:
        print(f"✅ Walras OK")
        print(f"   Sync score    : {result['market_sync']['sync_score']:.3f}")
        print(f"   Synchronized  : {result['market_sync']['is_synchronized']}")
        print(f"   Correlations  : {result['market_sync']['correlations']}")
        print(f"   Shock detected: {result['liquidity_shock']['shock_detected']}")
        print(f"   Kelly mult    : {result['kelly_multiplier']:.3f}")
        print(f"   Reasons       : {result['risk_adjustment']['reasons']}")
        print(f"   Confidence    : {result['confidence']:.3f}")
    else:
        print(f"❌ {result['error']}")
