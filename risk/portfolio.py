# =============================================================================
#  risk/portfolio.py — Portfolio Exposure & Cross-Position Risk
#  Aggregates total exposure across the primary trade + any open hedges.
#  Helpers live in portfolio_utils.py.
#
#  Standard interface:
#      result = run(kelly_result, hedging_result, stops_result, prices, cfg)
# =============================================================================

import logging
import numpy as np
from risk.portfolio_utils import (
    parse_positions, exposure_checks, compute_drawdown,
    risk_adjusted_return, dollar_values, exposure_summary,
    MAX_GROSS_EXPOSURE,
)

logger = logging.getLogger("temporal_bot.risk.portfolio")
PORTFOLIO_VALUE: float = 10000.0


def run(kelly_result, hedging_result, stops_result, prices, cfg):
    """
    Aggregates all risk module outputs into a unified portfolio view.

    Parameters
    ----------
    kelly_result   : dict — from engine/kelly.py
    hedging_result : dict — from risk/hedging.py
    stops_result   : dict — from risk/stops.py
    prices         : np.ndarray — closing prices (for drawdown calc)
    cfg            : dict — "TRADE_DIRECTION" (str), "PORTFOLIO_VALUE" (float)

    Returns
    -------
    dict — success, positions, dollar_values, drawdown, risk_adjusted,
           exposure_checks, summary, overall_risk_score, confidence,
           metadata, error
    """
    _empty = {
        "success": False, "positions": {}, "dollar_values": {},
        "drawdown": {}, "risk_adjusted": {}, "exposure_checks": [],
        "summary": "", "overall_risk_score": 0.5,
        "confidence": 0.0, "metadata": {}, "error": None,
    }

    if not kelly_result.get("success"):
        _empty["error"] = "Portfolio: kelly_result not successful."
        logger.error(_empty["error"])
        return _empty

    direction       = str(cfg.get("TRADE_DIRECTION", "long")).lower()
    portfolio_value = float(cfg.get("PORTFOLIO_VALUE", PORTFOLIO_VALUE))
    current_price   = float(prices[-1]) if prices is not None and len(prices) > 0 else 1.0

    try:
        positions = parse_positions(kelly_result, hedging_result, direction)
        checks    = exposure_checks(positions)
        dd        = compute_drawdown(prices) if prices is not None and len(prices) > 0 else {}
        risk_adj  = risk_adjusted_return(kelly_result, stops_result, positions)
        dollars   = dollar_values(positions, portfolio_value, current_price)
        summary   = exposure_summary(positions, risk_adj, dd)

        gross_f    = positions["gross_pct"] / MAX_GROSS_EXPOSURE
        dd_f       = dd.get("current_dd_pct", 0) / 20.0
        warn_f     = len(checks) / 3.0
        risk_score = round(float(np.clip(0.4*gross_f + 0.4*dd_f + 0.2*warn_f, 0, 1)), 4)
        confidence = round(float(
            (float(kelly_result.get("confidence", 0.5)) +
             float(hedging_result.get("overall_confidence", 0.5))) / 2
        ), 4)

        logger.info(
            "Portfolio OK: net=%+.1f%% gross=%.1f%% R/R=%.1f DD=%.1f%% risk=%.3f conf=%.3f",
            positions["net_pct"]*100, positions["gross_pct"]*100,
            risk_adj["risk_reward"], dd.get("current_dd_pct", 0), risk_score, confidence,
        )

        return {
            "success": True, "positions": positions, "dollar_values": dollars,
            "drawdown": dd, "risk_adjusted": risk_adj, "exposure_checks": checks,
            "summary": summary, "overall_risk_score": risk_score,
            "confidence": confidence,
            "metadata": {"direction": direction, "portfolio_value": portfolio_value,
                         "current_price": current_price},
            "error": None,
        }

    except Exception as exc:
        msg = f"{type(exc).__name__}: {exc}"
        logger.exception("Portfolio failed: %s", msg)
        _empty["error"] = msg
        return _empty


if __name__ == "__main__":
    np.random.seed(3)
    prices = np.cumsum(np.random.randn(100) * 300) + 60000
    r = run(
        {"success": True, "position_pct": 12.5, "confidence": 0.72, "expected_value": 0.045},
        {"success": True, "overall_confidence": 0.65, "unified_hedge": {"ratio": 0.30}},
        {"success": True, "active_stop": {"stop_price": 58500.0, "stop_pct": 2.5, "stop_type": "trailing"}},
        prices, {"TRADE_DIRECTION": "long", "PORTFOLIO_VALUE": 50000},
    )
    if r["success"]:
        print(f"✅ Portfolio OK | {r['summary']} | risk={r['overall_risk_score']:.3f}")
    else:
        print(f"❌ {r['error']}")
