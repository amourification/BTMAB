# =============================================================================
#  charts/risk_chart_utils.py — Risk Chart Internal Helpers
#  Split from risk_chart.py to keep both modules under 300 lines.
# =============================================================================
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Arc
from gui.theme import get_current_theme

_THEME      = get_current_theme()
BG_COLOR    = _THEME["BG"]
PANEL_COLOR = _THEME["PANEL"]
GRID_COLOR  = "#4e1f3d"
TEXT_COLOR  = _THEME["FG"]
ACCENT_CLR  = _THEME["ACCENT"]

TIER_COLORS = {
    "no_trade": "#546e7a", "minimal":  "#fdd835",
    "small":    "#66bb6a", "moderate": "#26a69a",
    "large":    "#ffa726", "maximum":  "#ef5350",
}
RR_COLORS = {"A": "#4caf50", "B": "#8bc34a", "C": "#ffa726", "D": "#ef5350", "N/A": "#546e7a"}


def draw_kelly_gauge(ax, kelly_result):
    """Semicircular gauge showing Kelly position %."""
    ax.set_xlim(-1.3, 1.3); ax.set_ylim(-0.2, 1.3)
    ax.set_aspect("equal"); ax.axis("off")
    pos_pct  = float(kelly_result.get("position_pct",  0.0))
    tier     = kelly_result.get("position_tier", {}).get("tier",  "no_trade")
    tier_lbl = kelly_result.get("position_tier", {}).get("label", "No trade")
    p_win    = float(kelly_result.get("p_win",          0.55))
    b_ratio  = float(kelly_result.get("win_ratio_b",    1.5))
    ev       = float(kelly_result.get("expected_value", 0.0))
    bg_arc   = Arc((0,0), 2, 2, angle=0, theta1=0, theta2=180, color="#1e2330", lw=18)
    ax.add_patch(bg_arc)
    filled_deg = min(pos_pct / 25.0, 1.0) * 180
    fill_color = TIER_COLORS.get(tier, "#546e7a")
    fill_arc   = Arc((0,0), 2, 2, angle=0, theta1=0, theta2=filled_deg, color=fill_color, lw=18)
    ax.add_patch(fill_arc)
    na = np.radians(filled_deg)
    ax.annotate("", xy=(np.cos(na)*0.85, np.sin(na)*0.85), xytext=(0,0),
                arrowprops=dict(arrowstyle="-|>", color="white", lw=1.5, mutation_scale=14))
    ax.text(0, 0.55, f"{pos_pct:.1f}%", ha="center", va="center", color="white", fontsize=20, fontweight="bold")
    ax.text(0, 0.25, tier_lbl, ha="center", va="center", color=fill_color, fontsize=8.5)
    ax.text(0, 0.05, f"p={p_win:.2f}  b={b_ratio:.2f}  EV={ev:+.4f}", ha="center", va="center", color=TEXT_COLOR, fontsize=7.5)
    for pct in [0, 5, 10, 15, 20, 25]:
        a = np.radians(pct/25*180)
        ax.text(np.cos(a)*1.12, np.sin(a)*1.12, f"{pct}%", ha="center", va="center", color=TEXT_COLOR, fontsize=6.5)
    ax.set_title("Kelly Position Gauge", color=ACCENT_CLR, fontsize=9, loc="center", pad=2)


def draw_stop_ladder(ax, stops_result, current_price):
    """Horizontal bar chart of stop levels."""
    if not stops_result.get("success"):
        ax.text(0.5, 0.5, "Stop data unavailable", transform=ax.transAxes, ha="center", color=TEXT_COLOR)
        return
    entry       = float(stops_result.get("entry_price",   current_price))
    init_s      = float(stops_result.get("initial_stop",  {}).get("stop_price", 0))
    trail_s     = float(stops_result.get("trailing_stop", {}).get("stop_price", 0))
    active_type = stops_result.get("active_stop", {}).get("stop_type", "")
    atr         = float(stops_result.get("atr", 0))
    phase_mult  = float(stops_result.get("phase_multiplier", 2.0))
    prices_ref  = [entry, trail_s, init_s]
    labels      = ["Entry", "Trailing Stop", "Initial Stop"]
    colors      = ["#ffd54f", "#4fc3f7", "#7986cb"]
    alphas      = [1.0, 1.0 if active_type=="trailing" else 0.5, 1.0 if active_type=="initial" else 0.5]
    for i, (p, lbl, col, alp) in enumerate(zip(prices_ref, labels, colors, alphas)):
        if p <= 0: continue
        ax.barh(i, p, color=col, alpha=alp, height=0.5, zorder=3)
        ax.text(p*1.001, i, f"{p:,.2f}", va="center", color=col, fontsize=8, alpha=alp)
    ax.axvline(current_price, color="white", lw=1.5, ls="--", alpha=0.8, zorder=5)
    ax.text(current_price, len(prices_ref)-0.5, f"Now: {current_price:,.2f}", ha="center", color="white", fontsize=7.5, va="bottom")
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, color=TEXT_COLOR, fontsize=8)
    ax.set_xlabel("Price", color=TEXT_COLOR, fontsize=8)
    ax.set_title(f"Stop Ladder — Phase mult: {phase_mult}× | ATR: {atr:.2f}", color=ACCENT_CLR, fontsize=9, loc="left", pad=4)


def draw_gamma_vol(ax, gamma_result):
    """Sparkline of realized vol + gamma proxy with regime shading."""
    vol_series  = np.array(gamma_result.get("vol_series",  []))
    gamma_proxy = np.array(gamma_result.get("gamma_proxy", []))
    regime      = gamma_result.get("regime",    {}).get("regime",    "neutral")
    vol_stats   = gamma_result.get("vol_stats", {})
    if len(vol_series) == 0:
        ax.text(0.5, 0.5, "Gamma: No data", transform=ax.transAxes, ha="center", color=TEXT_COLOR); return
    n = len(vol_series); x = np.arange(n); valid = ~np.isnan(vol_series)
    rc = {"negative_gamma": "#b71c1c", "positive_gamma": "#1b5e20", "neutral": "#37474f"}
    ax.axhspan(0, vol_series[valid].max()*1.3 if valid.any() else 1, color=rc.get(regime,"#37474f"), alpha=0.08)
    ax.plot(x[valid], vol_series[valid]*100, color="#ff7043", lw=1.2, alpha=0.9, label="Realized Vol %")
    gp_valid = ~np.isnan(gamma_proxy)
    if gp_valid.any() and valid.any():
        gr = np.nanmax(np.abs(gamma_proxy)); vr = vol_series[valid].max()
        if gr > 1e-12:
            gp_sc = gamma_proxy / gr * vr * 0.5 * 100
            ax.plot(x[gp_valid], gp_sc[gp_valid], color="#ce93d8", lw=0.9, ls="--", alpha=0.75, label="Gamma Proxy (scaled)")
    pct_rank = float(vol_stats.get("percentile", 0.5))*100
    ax.set_title(f"Gamma: {regime.replace('_',' ').title()}  |  Vol Rank: {pct_rank:.0f}th %ile",
                 color="#ff7043" if regime=="negative_gamma" else ACCENT_CLR, fontsize=9, loc="left", pad=4)
    ax.set_ylabel("Vol %", color=TEXT_COLOR, fontsize=8)
    ax.legend(loc="upper left", fontsize=7, facecolor=PANEL_COLOR, labelcolor=TEXT_COLOR, framealpha=0.6)
    ax.set_xlim(0, n-1)


def draw_portfolio_summary(ax, portfolio_result, kelly_result):
    """Exposure bars + R/R grade badge + risk score."""
    positions  = portfolio_result.get("positions",     {})
    risk_adj   = portfolio_result.get("risk_adjusted", {})
    risk_score = float(portfolio_result.get("overall_risk_score", 0.5))
    grade      = risk_adj.get("grade", "N/A")
    summary    = portfolio_result.get("summary", "")
    checks     = portfolio_result.get("exposure_checks", [])
    net_pct    = float(positions.get("net_pct",  0.0))*100
    gross_pct  = float(positions.get("gross_pct",0.0))*100
    primary_pct= float(positions.get("primary",  {}).get("size_pct",0.0))*100
    hedge_pct  = float(positions.get("hedge",    {}).get("size_pct",0.0))*100
    ax.axis("off")
    for (lbl, pct, col), y in zip(
        [("Primary",primary_pct,"#4fc3f7"),("Hedge",hedge_pct,"#ff7043"),("Net",abs(net_pct),ACCENT_CLR)],
        [0.80, 0.60, 0.40]
    ):
        w = min(pct/30, 1.0)
        ax.add_patch(mpatches.FancyBboxPatch((0.10,y-0.06), w*0.75, 0.10,
            boxstyle="round,pad=0.01", facecolor=col, alpha=0.75, transform=ax.transAxes, zorder=3))
        ax.text(0.10, y, f"{lbl}: {pct:.1f}%", va="center", color="white", fontsize=8.5,
                fontweight="bold", transform=ax.transAxes, zorder=4)
    grade_color = RR_COLORS.get(grade, "#546e7a")
    ax.add_patch(mpatches.FancyBboxPatch((0.72,0.55), 0.22, 0.30,
        boxstyle="round,pad=0.02", facecolor=grade_color, alpha=0.85, transform=ax.transAxes, zorder=3))
    ax.text(0.83, 0.70, grade, ha="center", va="center", color="white", fontsize=26, fontweight="bold", transform=ax.transAxes, zorder=4)
    ax.text(0.83, 0.57, "R/R Grade", ha="center", va="center", color="white", fontsize=7, transform=ax.transAxes, zorder=4)
    risk_col = "#4caf50" if risk_score<0.4 else ("#ffa726" if risk_score<0.7 else "#ef5350")
    ax.text(0.05, 0.20, f"Risk Score: {risk_score:.2f}", color=risk_col, fontsize=9, transform=ax.transAxes)
    ax.text(0.05, 0.10, summary[:70], color=TEXT_COLOR, fontsize=7.5, transform=ax.transAxes)
    if checks:
        ax.text(0.05, 0.01, checks[0][:80], color="#ffa726", fontsize=7, transform=ax.transAxes)
    ax.set_title("Portfolio Risk Summary", color=ACCENT_CLR, fontsize=9, loc="left", pad=4)
