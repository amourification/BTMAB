# =============================================================================
#  charts/exporter_utils.py — Exporter Internal Helpers
#  Split from exporter.py to keep both modules under 300 lines.
#  Contains: summary page renderer, trade plan flattener, smoke test data.
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from gui.theme import get_current_theme

_THEME    = get_current_theme()
BG_COLOR   = _THEME["BG"]
TEXT_COLOR = _THEME["FG"]
ACCENT_CLR = _THEME["ACCENT"]


def render_summary_page(trade_plan: dict) -> plt.Figure:
    """Renders the text-based summary page for the PDF report."""
    fig = plt.figure(figsize=(16, 9), facecolor=BG_COLOR)
    ax  = fig.add_subplot(111); ax.set_facecolor(BG_COLOR); ax.axis("off")

    sym   = trade_plan.get("symbol",              "?")
    ts    = datetime.now().strftime("%Y-%m-%d %H:%M UTC")
    conf  = trade_plan.get("consensus_confidence", 0.0)

    def g(k, d="?"): return trade_plan.get(k, d)

    bias_str   = str(g("market_bias","neutral")).upper()
    turn_urg   = str(g("turn_urgency","low")).upper()
    gamma_reg  = str(g("gamma_regime","?")).replace("_"," ").title()
    sol_bias   = str(g("seasonal_bias","?")).replace("_"," ").title()
    turn_type  = str(g("turn_type","?")).replace("_"," ").title()
    ssa_pos    = str(g("ssa_position","?")).replace("_"," ").title()

    fig.text(0.04, 0.96, f"TEMPORAL MARKET ANALYSIS — {sym}", fontsize=16,
             fontweight="bold", color=ACCENT_CLR, va="top")
    fig.text(0.04, 0.91, f"Generated: {ts}  |  Confidence: {conf:.1f}%",
             fontsize=10, color=TEXT_COLOR, va="top")

    sections = [
        ("── TEMPORAL CYCLE ──────────────────",  None,                        ACCENT_CLR),
        ("Dominant Cycle",   f"{g('dominant_cycle_bars',0):.0f} bars",         TEXT_COLOR),
        ("SSA Period",       f"{g('ssa_period',0):.1f} bars",                  TEXT_COLOR),
        ("SSA Position",     ssa_pos,                                           TEXT_COLOR),
        ("Phase",            f"{g('phase_deg',0):.1f}°",                       TEXT_COLOR),
        ("Turn Type",        turn_type,                                         TEXT_COLOR),
        ("Turn Urgency",     turn_urg,  "#ff7043" if turn_urg in ("HIGH","IMMEDIATE") else TEXT_COLOR),
        ("Bars to Next Turn",f"{g('bars_to_next_turn',0)}",                    TEXT_COLOR),
        ("── MARKET CONTEXT ─────────────────",  None,                        ACCENT_CLR),
        ("Market Bias",      f"{bias_str} ({g('bias_strength',0):.0%})",
             "#4caf50" if "BULL" in bias_str else "#ef5350"),
        ("Murray Level",     f"{g('murray_index',0):.2f}/8",                   TEXT_COLOR),
        ("Gamma Regime",     gamma_reg, "#ef5350" if "Negative" in gamma_reg else TEXT_COLOR),
        ("Seasonal Bias",    sol_bias,                                          TEXT_COLOR),
        ("Sweep Detected",   "⚡ YES" if g("sweep_detected") else "No",
             "#ffa726" if g("sweep_detected") else TEXT_COLOR),
        ("── TRADE PLAN ─────────────────────",  None,                        ACCENT_CLR),
        ("Kelly Position",   f"{g('kelly_position_pct',0):.1f}%  ({g('kelly_tier','')})", TEXT_COLOR),
        ("Hedge Ratio",      f"{g('hedge_pct',0):.1f}%  urgency: {str(g('hedge_urgency','')).upper()}", TEXT_COLOR),
        ("Active Stop",      f"{g('active_stop_price',0):,.2f}  ({g('active_stop_type','')} {g('active_stop_pct',0):.2f}%)", TEXT_COLOR),
        ("R/R Grade",        str(g("risk_reward_grade","?")),
             "#4caf50" if g("risk_reward_grade","?") in ("A","B") else "#ffa726"),
        ("Risk Score",       f"{g('overall_risk_score',0.5):.2f}",
             "#4caf50" if g("overall_risk_score",0.5)<0.4 else ("#ffa726" if g("overall_risk_score",0.5)<0.7 else "#ef5350")),
        ("── PORTFOLIO ──────────────────────",  None,                        ACCENT_CLR),
        (str(g("portfolio_summary",""))[:65], "",                              TEXT_COLOR),
    ]

    y = 0.86
    for item in sections:
        label, value, color = item
        if value is None:
            fig.text(0.04, y, label, fontsize=8.5, color=color, va="top", fontfamily="monospace")
            y -= 0.028
        else:
            fig.text(0.04, y, f"  {label:<28}", fontsize=9, color=TEXT_COLOR, va="top", fontfamily="monospace")
            fig.text(0.42, y, value, fontsize=9, color=color, va="top", fontfamily="monospace", fontweight="bold")
            y -= 0.033

    bar_ax = fig.add_axes([0.04, 0.04, 0.92, 0.025])
    bar_ax.set_facecolor("#1e2330")
    bar_ax.barh(0, conf/100, color=ACCENT_CLR, height=1.0)
    bar_ax.set_xlim(0, 1); bar_ax.set_ylim(-0.5, 0.5); bar_ax.axis("off")
    bar_ax.text(conf/100/2, 0, f"Consensus Confidence: {conf:.1f}%",
                ha="center", va="center", color="#0d0f14", fontsize=9, fontweight="bold")
    return fig


def flatten_trade_plan(trade_plan: dict) -> dict:
    """Flattens the trade plan to a single-level dict for CSV export."""
    flat = {}
    skip = {"_engines", "_risk", "stop_schedule", "bias_votes", "bias_reasons", "drawdown"}
    for k, v in trade_plan.items():
        if k in skip: continue
        if isinstance(v, (int, float, str, bool)) or v is None:
            flat[k] = v
        elif isinstance(v, dict):
            for sk, sv in v.items():
                if isinstance(sv, (int, float, str, bool)) or sv is None:
                    flat[f"{k}_{sk}"] = sv
    return flat


def make_smoke_test_plan() -> dict:
    """Builds a fully-populated fake trade plan for smoke testing."""
    np.random.seed(42)
    n = 200; t = np.arange(n)
    prices    = 50000 + 3000 * np.sin(2 * np.pi * t / 64) + np.cumsum(np.random.randn(n) * 100)
    detrended = 3000 * np.sin(2 * np.pi * t / 64) + np.random.randn(n) * 150
    phase_rad = 2 * np.pi * t / 64
    vol       = np.full(n, np.nan)
    for i in range(20, n):
        vol[i] = np.std(np.diff(np.log(prices[i-20:i+1]))) * np.sqrt(252)

    return {
        "symbol": "BTCUSDT", "consensus_confidence": 74.3,
        "market_bias": "bearish", "bias_strength": 0.62,
        "phase_deg": 215.0, "turn_type": "distribution", "turn_urgency": "high",
        "bars_to_next_turn": 12, "dominant_cycle_bars": 64.0,
        "ssa_period": 61.0, "ssa_position": "near_peak", "ssa_confidence": 78.0,
        "murray_index": 6.7, "murray_action": "Sell — Strong resistance",
        "murray_support": 58000.0, "murray_resistance": 67000.0,
        "gamma_regime": "negative_gamma", "vol_regime": "high",
        "seasonal_bias": "autumn_volatility_bias", "volatility_flag": True,
        "sweep_detected": True, "walras_adjustment": 0.85,
        "kelly_position_pct": 8.0, "kelly_tier": "Small (5–10%)", "kelly_ev": 0.031,
        "hedge_pct": 25.0, "hedge_urgency": "high",
        "hedge_action": "Open 25% short hedge",
        "active_stop_price": 59400.0, "active_stop_type": "trailing",
        "active_stop_pct": 2.1, "stop_schedule": [],
        "portfolio_summary": "Net SHORT 17% | Gross 33% | R/R 2.1× (B) | DD 0.8%",
        "risk_reward_grade": "B", "overall_risk_score": 0.41, "drawdown": {},
        "_engines": {
            "detrend":  {"trend": prices - detrended, "detrended": detrended, "zero_crossings": []},
            "ssa":      {"reconstruction": detrended*0.8, "dominant_period": 61.0,
                         "confidence": 0.78, "cycle_position": {"position": "near_peak"}},
            "fft":      {"primary_cycle": {"period": 64.0, "amplitude": 3000.0},
                         "macro_cycle":   {"period": 512.0},
                         "cycle_phase":   {"pct_complete": 0.60}},
            "hilbert":  {"phase_rad": phase_rad, "phase_deg": 215.0,
                         "turn_type": "distribution", "turn_urgency": "high",
                         "turn_label": "Distribution / Top", "confidence": 0.71,
                         "bars_to_next_turn": {"bars_to_boundary": 12, "next_turn_type": "accumulation"}},
            "acf":      {"acf_values": np.exp(-np.arange(80)*0.05)*np.cos(2*np.pi*np.arange(80)/21),
                         "conf_band": 0.12, "best_lag": 21, "memory_score": 0.55,
                         "significant_lags": [{"lag": 21, "correlation": 0.6}],
                         "cycle_lags": [21, 42]},
            "ar":       {"success": True, "forecast_direction": "down",
                         "forecast_raw": np.random.randn(10)*0.003,
                         "forecast_adjusted": np.random.randn(10)*0.003 - 0.001,
                         "sentiment_score": -0.2,
                         "session_split": {"day":   {"values": np.random.randn(80)*0.004},
                                           "night": {"values": np.random.randn(80)*0.004}},
                         "inertia": {"dominant_session": "night"}},
            "solar":    {"solar_sine": np.sin(2*np.pi*np.arange(n)/365),
                         "seasonal_bias": "autumn_volatility_bias", "volatility_flag": True,
                         "days_to_cardinals": {"days_to_cardinals": {}}},
            "murray":   {"levels": np.linspace(55000, 70000, 9), "murray_index": 6.7,
                         "confluence": {"signal": "high_confluence_sell"},
                         "recommended_action": "Sell — Strong resistance", "confidence": 0.80},
            "gann":     {"fan": [], "anchor": {"anchor_type": "high", "anchor_idx": 150,
                         "anchor_val": 65000.0}, "angle_breaks": [],
                         "price_unit": 1875.0, "price_vs_master": "below"},
            "gamma":    {"success": True, "vol_series": vol,
                         "gamma_proxy": np.random.randn(n)*0.01,
                         "regime": {"regime": "negative_gamma", "regime_strength": 0.65},
                         "vol_stats": {"current": 0.72, "percentile": 0.88, "regime": "high"}},
            "kelly":    {"success": True, "position_pct": 8.0, "confidence": 0.65,
                         "p_win": 0.60, "win_ratio_b": 1.7, "expected_value": 0.031,
                         "position_tier": {"tier": "small", "label": "Small (5–10%)"}},
            "_fetch":   {"high": prices+300, "low": prices-300, "timestamps": t},
        },
        "_risk": {
            "stops": {"success": True, "entry_price": 61000.0, "atr": 380.0,
                      "phase_multiplier": 1.5, "stop_schedule": [],
                      "initial_stop":  {"stop_price": 59300.0},
                      "trailing_stop": {"stop_price": 59400.0},
                      "active_stop":   {"stop_type": "trailing",
                                        "stop_price": 59400.0, "stop_pct": 2.1}},
            "portfolio": {"success": True,
                          "positions": {"net_pct": -0.17, "gross_pct": 0.33,
                                        "primary": {"size_pct": 0.08},
                                        "hedge":   {"size_pct": 0.25}},
                          "risk_adjusted": {"grade": "B", "risk_reward": 2.1,
                                            "portfolio_at_risk": 1.2},
                          "overall_risk_score": 0.41, "exposure_checks": [],
                          "summary": "Net SHORT 17% | Gross 33% | R/R 2.1× (B) | DD 0.8%",
                          "drawdown": {"current_dd_pct": 0.8}},
        },
    }
