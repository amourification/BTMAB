# =============================================================================
#  charts/gann_chart.py — Gann Fan + Square of Nine + Gamma Overlay (Eq 9, 8)
#  Renders the full 7-angle Gann fan alongside a volatility-coloured price
#  channel and a "Square of Nine" price level grid overlay.
#
#  Panel layout:
#    [0] Price + 7-angle Gann fan + anchor marker + break annotations
#    [1] Gann angle distance chart — how far price is from each angle
#
#  draw() interface: draws onto caller-supplied axes list.
#  render() interface: creates and returns a standalone Figure.
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from gui.theme import get_current_theme

_THEME      = get_current_theme()
BG_COLOR    = _THEME["BG"]
PANEL_COLOR = _THEME["PANEL"]
GRID_COLOR  = "#4e1f3d"
TEXT_COLOR  = _THEME["FG"]
ACCENT_CLR  = _THEME["ACCENT"]
PRICE_COLOR = "#f2f1f0"

GANN_COLORS = {
    (1, 8): "#4fc3f7", (1, 4): "#81d4fa", (1, 2): "#b3e5fc",
    (1, 1): "#ffd54f",
    (2, 1): "#ffab40", (4, 1): "#ff7043", (8, 1): "#d32f2f",
}
GANN_ALPHAS = {
    (1, 8): 0.35, (1, 4): 0.45, (1, 2): 0.55,
    (1, 1): 0.95,
    (2, 1): 0.55, (4, 1): 0.45, (8, 1): 0.35,
}
GANN_LW = {
    (1, 8): 0.7, (1, 4): 0.8, (1, 2): 0.9,
    (1, 1): 2.0,
    (2, 1): 0.9, (4, 1): 0.8, (8, 1): 0.7,
}


# ── Internal helpers ──────────────────────────────────────────────────────────

def _apply_style(fig, axes):
    fig.patch.set_facecolor(BG_COLOR)
    for ax in axes:
        ax.set_facecolor(PANEL_COLOR)
        ax.tick_params(colors=TEXT_COLOR, labelsize=8)
        ax.xaxis.label.set_color(TEXT_COLOR)
        ax.yaxis.label.set_color(TEXT_COLOR)
        for spine in ax.spines.values():
            spine.set_color(GRID_COLOR)
        ax.grid(True, color=GRID_COLOR, linewidth=0.4, alpha=0.5)


def _draw_fan_panel(ax, prices, high, low, fan, anchor, angle_breaks,
                    price_vs_master, trade_plan):
    """
    Panel 0: price bars + all 7 Gann angle lines + anchor + break markers.
    """
    n = len(prices)
    x = np.arange(n)

    # Price
    ax.plot(x, prices, color=PRICE_COLOR, lw=0.9, alpha=0.9, zorder=5, label="Price")

    # Wicks (sparse, for clarity)
    step = max(1, n // 80)
    for i in range(0, n, step):
        c = "#2e7d32" if i == 0 or prices[i] >= prices[i - 1] else "#c62828"
        ax.plot([i, i], [low[i], high[i]], color=c, lw=0.5, alpha=0.35, zorder=2)

    # Gann fan lines
    legend_handles = []
    for angle_dict in fan:
        scale  = angle_dict.get("scale", (1, 1))
        values = angle_dict.get("values", np.array([]))
        label  = angle_dict.get("label", "")
        deg    = angle_dict.get("angle_deg", 0.0)
        if values is None or len(values) == 0:
            continue
        xv    = np.arange(len(values))
        valid = ~np.isnan(values)
        col   = GANN_COLORS.get(scale, "#888")
        alp   = GANN_ALPHAS.get(scale, 0.4)
        lw    = GANN_LW.get(scale, 0.8)
        ls    = "-" if scale == (1, 1) else "--"
        ax.plot(xv[valid], values[valid], color=col, lw=lw,
                alpha=alp, linestyle=ls, zorder=3)
        if scale in ((1, 1), (1, 2), (2, 1)):
            legend_handles.append(
                Line2D([0], [0], color=col, lw=lw, label=f"{label[:18]} ({deg}°)")
            )

    # Anchor marker
    if anchor:
        ai  = anchor.get("anchor_idx", 0)
        av  = anchor.get("anchor_val", 0)
        at  = anchor.get("anchor_type", "low")
        col = "#4caf50" if at == "low" else "#ef5350"
        ax.scatter([ai], [av], color=col, s=70, zorder=8, marker="^" if at=="low" else "v")
        ax.text(ai + 1, av, f"Anchor\n({at})", color=col, fontsize=6.5,
                va="bottom" if at == "low" else "top", zorder=9)

    # Break annotations
    for brk in angle_breaks[:4]:
        direction = brk.get("direction", "")
        bars_ago  = int(brk.get("bars_ago", 0))
        angle_lbl = brk.get("angle", "")[:8]
        col       = "#4caf50" if "bullish" in direction else "#ef5350"
        sym       = "↑" if "bullish" in direction else "↓"
        ax.axvline(n - 1 - bars_ago, color=col, lw=1.0, ls=":", alpha=0.7)
        ax.text(n - 1 - bars_ago, ax.get_ylim()[0] if ax.get_ylim()[0] != 0 else prices.min(),
                f"{sym}{angle_lbl}", color=col, fontsize=6.5, va="bottom")

    title_str = (f"Gann Fan  |  Price vs 1×1: {price_vs_master.upper()}"
                 f"  |  Breaks: {len(angle_breaks)}")
    ax.set_title(title_str, color=ACCENT_CLR, fontsize=9, loc="left", pad=4)
    ax.set_ylabel("Price", color=TEXT_COLOR, fontsize=8)
    ax.set_xlim(0, n - 1)
    ax.legend(handles=legend_handles, loc="upper left", fontsize=6.5,
              facecolor=PANEL_COLOR, labelcolor=TEXT_COLOR, framealpha=0.6)


def _draw_distance_panel(ax, prices, fan, price_unit):
    """
    Panel 1: bar chart of how many price_units the current price sits
    above/below each Gann angle. Negative = below angle (bearish),
    positive = above angle (bullish).
    """
    if not fan or len(prices) == 0:
        ax.text(0.5, 0.5, "No Gann data", transform=ax.transAxes,
                ha="center", color=TEXT_COLOR)
        return

    current = float(prices[-1])
    labels  = []
    dists   = []
    colors  = []

    for angle_dict in fan:
        cv = angle_dict.get("current_value")
        if cv is None or (isinstance(cv, float) and np.isnan(cv)):
            continue
        dist_pu = (current - float(cv)) / (price_unit + 1e-12)
        labels.append(angle_dict.get("label", "?")[:12])
        dists.append(dist_pu)
        colors.append("#4caf50" if dist_pu >= 0 else "#ef5350")

    if not labels:
        ax.text(0.5, 0.5, "No valid angles", transform=ax.transAxes,
                ha="center", color=TEXT_COLOR)
        return

    y = np.arange(len(labels))
    bars = ax.barh(y, dists, color=colors, alpha=0.8, height=0.6, zorder=3)
    ax.axvline(0, color=TEXT_COLOR, lw=0.8, alpha=0.6)

    for i, (d, lbl) in enumerate(zip(dists, labels)):
        ax.text(d + (0.05 if d >= 0 else -0.05), i,
                f"{d:+.1f}u", va="center",
                ha="left" if d >= 0 else "right",
                color="white", fontsize=7)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, color=TEXT_COLOR, fontsize=7.5)
    ax.set_xlabel(f"Distance from Angle (price units = {price_unit:.0f})",
                  color=TEXT_COLOR, fontsize=8)
    ax.set_title("Distance from Each Gann Angle (+ = above)",
                 color=ACCENT_CLR, fontsize=9, loc="left", pad=4)


# ── Public interfaces ─────────────────────────────────────────────────────────

def draw(axes: list, trade_plan: dict) -> None:
    """
    Draws the Gann fan chart onto caller-supplied axes [ax0, ax1].

    Parameters
    ----------
    axes       : list of 2 matplotlib Axes
    trade_plan : dict — output of aggregator.run()
    """
    eng   = trade_plan.get("_engines", {})
    gann  = eng.get("gann",   {})
    det   = eng.get("detrend",{})
    fetch = eng.get("_fetch", {})

    trend     = np.array(det.get("trend",     []))
    detrended = np.array(det.get("detrended", []))
    prices    = trend + detrended if len(trend) == len(detrended) and len(trend) > 0 else trend
    high      = np.array(fetch.get("high", prices + prices * 0.003))
    low       = np.array(fetch.get("low",  prices - prices * 0.003))

    fan           = gann.get("fan",           [])
    anchor        = gann.get("anchor",        {})
    angle_breaks  = gann.get("angle_breaks",  [])
    price_unit    = float(gann.get("price_unit", 1.0))
    price_vs_mast = gann.get("price_vs_master", "unknown")

    if len(prices) == 0:
        for ax in axes:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                    ha="center", va="center", color=TEXT_COLOR)
        return

    _apply_style(axes[0].figure, axes)
    _draw_fan_panel(axes[0], prices, high, low, fan, anchor,
                    angle_breaks, price_vs_mast, trade_plan)
    _draw_distance_panel(axes[1], prices, fan, price_unit)


def render(trade_plan: dict, figsize: tuple = (14, 8)) -> plt.Figure:
    """Creates and returns a standalone Gann fan Figure."""
    fig = plt.figure(figsize=figsize, facecolor=BG_COLOR)
    fig.suptitle(
        f"Gann Angle Fan — {trade_plan.get('symbol', '?')}",
        color=ACCENT_CLR, fontsize=11, fontweight="bold", y=0.98,
    )
    gs   = GridSpec(2, 1, figure=fig, height_ratios=[4, 1], hspace=0.35,
                    top=0.93, bottom=0.08, left=0.10, right=0.97)
    axes = [fig.add_subplot(gs[i]) for i in range(2)]
    draw(axes, trade_plan)
    return fig


# ── Smoke test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    np.random.seed(9)
    n = 150; t = np.arange(n)
    prices = 50000 + 200 * t - 0.5 * t ** 2 + np.random.randn(n) * 500
    price_unit = (prices.max() - prices.min()) / 8

    fan_vals = []
    anchor_val = float(prices.max()); anchor_idx = int(np.argmax(prices))
    for (p, tt) in [(1,8),(1,4),(1,2),(1,1),(2,1),(4,1),(8,1)]:
        slope  = (p / tt) * price_unit
        values = np.array([anchor_val - slope * (i - anchor_idx) for i in range(n)])
        fan_vals.append({
            "scale": (p, tt),
            "label": f"{p}×{tt}",
            "values": values,
            "current_value": float(values[-1]),
            "angle_deg": round(float(np.degrees(np.arctan(p / tt))), 1),
        })

    fake_plan = {
        "symbol": "BTCUSDT",
        "_engines": {
            "gann": {"fan": fan_vals, "anchor": {"anchor_idx": anchor_idx,
                     "anchor_val": anchor_val, "anchor_type": "high"},
                     "angle_breaks": [{"angle": "1×1 — Master", "direction": "bearish_break", "bars_ago": 5}],
                     "price_unit": price_unit, "price_vs_master": "below"},
            "detrend": {"trend": prices, "detrended": np.zeros(n)},
            "_fetch":  {"high": prices + 300, "low": prices - 300},
        },
        "_risk": {},
    }
    fig = render(fake_plan)
    fig.savefig("/tmp/gann_chart_test.png", dpi=120, bbox_inches="tight")
    print("✅ Gann chart rendered → /tmp/gann_chart_test.png")
    plt.close(fig)
