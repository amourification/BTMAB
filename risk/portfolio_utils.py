# =============================================================================
#  risk/portfolio_utils.py — Portfolio Internal Helpers
#  Split from portfolio.py to keep both modules under 300 lines.
# =============================================================================
import numpy as np

MAX_GROSS_EXPOSURE: float = 2.0
MAX_NET_EXPOSURE:   float = 0.80


def parse_positions(kelly_result: dict, hedging_result: dict, direction: str) -> dict:
    primary_pct = float(kelly_result.get("position_pct", 0.0)) / 100.0
    hedge_pct   = float(hedging_result.get("unified_hedge", {}).get("ratio", 0.0))
    is_long     = direction.lower() == "long"
    net_pct     = primary_pct - hedge_pct if is_long else hedge_pct - primary_pct
    return {
        "primary":   {"direction": "long" if is_long else "short", "size_pct": round(primary_pct, 4)},
        "hedge":     {"direction": "short" if is_long else "long", "size_pct": round(hedge_pct, 4)},
        "net_pct":   round(net_pct, 4),
        "gross_pct": round(primary_pct + hedge_pct, 4),
        "is_long":   is_long,
    }


def exposure_checks(positions: dict) -> list:
    warnings = []
    g = positions["gross_pct"]; n = abs(positions["net_pct"])
    if g > MAX_GROSS_EXPOSURE:
        warnings.append(f"⚠️  Gross {g:.0%} exceeds limit {MAX_GROSS_EXPOSURE:.0%}. Reduce position.")
    if n > MAX_NET_EXPOSURE:
        warnings.append(f"⚠️  Net {n:.0%} exceeds limit {MAX_NET_EXPOSURE:.0%}. Increase hedge.")
    if positions["hedge"]["size_pct"] == 0.0 and positions["primary"]["size_pct"] > 0.20:
        warnings.append("ℹ️  No hedge on large position. Consider protective hedge.")
    return warnings


def compute_drawdown(prices: np.ndarray) -> dict:
    if len(prices) == 0:
        return {"current_dd_pct": 0.0, "max_dd_pct": 0.0, "bars_in_dd": 0}
    peak = float(prices[0]); max_dd = 0.0; bars_in_dd = 0
    for p in prices:
        if float(p) > peak:
            peak = float(p); bars_in_dd = 0
        dd = (peak - float(p)) / peak
        max_dd = max(max_dd, dd)
        if dd > 0:
            bars_in_dd += 1
    return {"current_dd_pct": round((peak - float(prices[-1])) / peak * 100, 3),
            "max_dd_pct": round(max_dd * 100, 3),
            "bars_in_dd": bars_in_dd, "peak_price": round(peak, 4)}


def risk_adjusted_return(kelly_result: dict, stops_result: dict, positions: dict) -> dict:
    ev       = float(kelly_result.get("expected_value", 0.0))
    stop_pct = float(stops_result.get("active_stop", {}).get("stop_pct", 2.0))
    primary  = positions["primary"]["size_pct"]
    if stop_pct < 1e-6 or primary < 1e-6:
        return {"risk_reward": 0.0, "ev_per_risk_unit": 0.0, "portfolio_at_risk": 0.0, "grade": "N/A"}
    risk_pct   = stop_pct / 100.0 * primary
    rr         = float((ev * primary) / (risk_pct + 1e-12))
    grade      = "A" if rr >= 3 else "B" if rr >= 2 else "C" if rr >= 1 else "D"
    return {"risk_reward": round(rr, 3),
            "ev_per_risk_unit": round(float(ev / (risk_pct + 1e-12)), 3),
            "portfolio_at_risk": round(risk_pct * 100, 3), "grade": grade}


def dollar_values(positions: dict, portfolio_value: float, current_price: float) -> dict:
    return {
        "portfolio_value": portfolio_value,
        "primary_dollars": round(positions["primary"]["size_pct"] * portfolio_value, 2),
        "hedge_dollars":   round(positions["hedge"]["size_pct"]   * portfolio_value, 2),
        "net_dollars":     round(positions["net_pct"]             * portfolio_value, 2),
        "units_primary":   round(positions["primary"]["size_pct"] * portfolio_value
                                 / (current_price + 1e-12), 6),
        "currency": "USD",
    }


def exposure_summary(positions: dict, risk_adj: dict, drawdown: dict) -> str:
    d = "LONG" if positions["net_pct"] >= 0 else "SHORT"
    return (f"Net {d} {abs(positions['net_pct']):.0%} | "
            f"Gross {positions['gross_pct']:.0%} | "
            f"R/R {risk_adj['risk_reward']:.1f}× ({risk_adj['grade']}) | "
            f"DD {drawdown.get('current_dd_pct', 0):.1f}%")
