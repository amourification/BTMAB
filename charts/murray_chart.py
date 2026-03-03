# =============================================================================
#  charts/murray_chart.py — Murray Math Levels + Gann Fan Chart (Eq 6, 9)
#  Renders the current price relative to all 9 Murray Math levels,
#  overlays the 7-angle Gann fan, and highlights confluence zones.
#
#  Panel layout:
#    [0] Candlestick-style OHLC + Murray levels + Gann fan
#    [1] Murray index heatbar (0/8 → 8/8 position indicator)
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

_THEME       = get_current_theme()
BG_COLOR     = _THEME["BG"]
PANEL_COLOR  = _THEME["PANEL"]
GRID_COLOR   = "#4e1f3d"
TEXT_COLOR   = _THEME["FG"]
ACCENT_COLOR = _THEME["ACCENT"]
PRICE_COLOR  = "#f2f1f0"

# Murray level colours — green (support) → neutral → red (resistance)
MURRAY_COLORS = [
    "#1b5e20", "#2e7d32", "#388e3c", "#558b2f",
    "#f9a825",
    "#e65100", "#c62828", "#b71c1c", "#880e4f",
]

GANN_COLORS = {
    (1, 8): "#4fc3f7", (1, 4): "#81d4fa", (1, 2): "#b3e5fc",
    (1, 1): "#ffd54f",   # master angle — gold
    (2, 1): "#ffab40", (4, 1): "#ff7043", (8, 1): "#d32f2f",
}

GANN_ALPHAS = {
    (1, 8): 0.35, (1, 4): 0.45, (1, 2): 0.55,
    (1, 1): 0.90,   # master angle most prominent
    (2, 1): 0.55, (4, 1): 0.45, (8, 1): 0.35,
}


# ── Internal helpers ──────────────────────────────────────────────────────────

def _apply_style(fig, axes):
    fig.patch.set_facecolor(BG_COLOR)
    for ax in axes:
        ax.set_facecolor(PANEL_COLOR)
        ax.tick_params(colors=TEXT_COLOR, labelsize=8)
        for spine in ax.spines.values():
            spine.set_color(GRID_COLOR)
        ax.grid(True, color=GRID_COLOR, linewidth=0.4, alpha=0.5)


def _draw_ohlc(ax, prices, high, low):
    """Simple close-only line chart when full OHLC is unavailable."""
    n = len(prices)
    x = np.arange(n)
    ax.plot(x, prices, color=PRICE_COLOR, lw=0.9, alpha=0.9, zorder=5, label="Close")

    # Simulate candle wicks using high/low if available
    if high is not None and low is not None and len(high) == n:
        for i in range(0, n, max(1, n // 60)):   # draw every nth wick for clarity
            color = "#2e7d32" if i == 0 or prices[i] >= prices[i - 1] else "#c62828"
            ax.plot([i, i], [low[i], high[i]], color=color, lw=0.6, alpha=0.4, zorder=3)


def _draw_murray_levels(ax, levels, murray_index, current_price, n_bars):
    """Draws horizontal Murray level lines across the full chart width."""
    if levels is None or len(levels) == 0:
        return

    for i, (level, color) in enumerate(zip(levels, MURRAY_COLORS)):
        lw   = 1.5 if i in (0, 4, 8) else 0.8   # emphasise 0/8, 4/8, 8/8
        dash = (4, 2) if i == 4 else (2, 4) if i in (0, 8) else (1, 3)
        ax.axhline(level, color=color, lw=lw, alpha=0.75,
                   linestyle=(0, dash), zorder=4)

        label_x = n_bars * 0.01
        ax.text(label_x, level, f" {i}/8  {level:,.0f}",
                color=color, fontsize=6.5, va="bottom", alpha=0.85, zorder=6)

    # Highlight current Murray index band
    lo_idx = max(0, int(murray_index) )
    hi_idx = min(8, lo_idx + 1)
    if lo_idx < len(levels) and hi_idx < len(levels):
        ax.axhspan(levels[lo_idx], levels[hi_idx],
                   color="#ffd54f", alpha=0.06, zorder=2, label="Current band")


def _draw_gann_fan(ax, fan, anchor_type, n_bars):
    """Draws all 7 Gann angle lines from anchor forward."""
    if not fan:
        return

    legend_elements = []
    for angle_dict in fan:
        scale  = angle_dict.get("scale", (1, 1))
        values = angle_dict.get("values", np.array([]))
        label  = angle_dict.get("label", "")

        if values is None or len(values) == 0:
            continue

        x      = np.arange(len(values))
        valid  = ~np.isnan(values)
        color  = GANN_COLORS.get(scale, "#888")
        alpha  = GANN_ALPHAS.get(scale, 0.4)
        lw     = 1.8 if scale == (1, 1) else 0.8

        ax.plot(x[valid], values[valid], color=color, lw=lw, alpha=alpha,
                linestyle="-" if scale == (1, 1) else "--", zorder=3)

        if scale == (1, 1):
            legend_elements.append(
                Line2D([0], [0], color=color, lw=1.8, label=f"1×1 Master ({angle_dict.get('angle_deg', 45)}°)")
            )

    ax.legend(handles=legend_elements, loc="upper left", fontsize=7,
              facecolor=PANEL_COLOR, labelcolor=TEXT_COLOR, framealpha=0.6)


def _draw_confluence_zone(ax, angle_breaks, murray_result):
    """Highlights convergence points (Gann break + Murray extreme)."""
    if not angle_breaks:
        return

    for brk in angle_breaks:
        label     = brk.get("angle", "")
        direction = brk.get("direction", "")
        bars_ago  = brk.get("bars_ago", 0)
        color     = "#4caf50" if direction == "bullish_break" else "#f44336"
        ax.axvline(x=-(bars_ago), color=color, lw=1.2, alpha=0.7,
                   linestyle=":", zorder=6)


def _draw_murray_heatbar(ax, murray_index, confluence_signal):
    """
    Draws a horizontal heatbar showing the Murray 0/8–8/8 position.
    The bar is coloured from green (0/8) through yellow (4/8) to red (8/8).
    """
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list(
        "murray", ["#1b5e20", "#f9a825", "#880e4f"], N=256
    )

    # Draw gradient bar
    gradient = np.linspace(0, 1, 256).reshape(1, -1)
    ax.imshow(gradient, aspect="auto", cmap=cmap, extent=[0, 8, 0, 1], zorder=2)

    # Current position needle
    ax.axvline(murray_index, color="white", lw=2.5, zorder=5)
    ax.scatter([murray_index], [0.5], color="white", s=80, zorder=6, marker="v")

    # Labels
    for i, lbl in enumerate(["0/8", "1/8", "2/8", "3/8", "4/8",
                               "5/8", "6/8", "7/8", "8/8"]):
        ax.text(i, -0.15, lbl, ha="center", va="top",
                color=TEXT_COLOR, fontsize=6.5, transform=ax.get_xaxis_transform())

    ax.set_xlim(0, 8); ax.set_ylim(0, 1)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel(f"Murray Position: {murray_index:.2f}/8  |  {confluence_signal}",
                  color=ACCENT_COLOR, fontsize=8)
    for spine in ax.spines.values():
        spine.set_visible(False)


# ── Public interfaces ─────────────────────────────────────────────────────────

def draw(axes: list, trade_plan: dict) -> None:
    """
    Draws Murray + Gann chart onto caller-supplied axes [ax0, ax1].

    Parameters
    ----------
    axes       : list of 2 matplotlib Axes
    trade_plan : dict — output of aggregator.run()
    """
    eng    = trade_plan.get("_engines", {})
    murray = eng.get("murray",  {})
    gann   = eng.get("gann",    {})
    detrend= eng.get("detrend", {})
    fetch  = eng.get("_fetch",  {})

    trend     = np.array(detrend.get("trend",     []))
    detrended = np.array(detrend.get("detrended", []))
    prices    = trend + detrended if len(trend) == len(detrended) and len(trend) > 0 else trend
    high      = np.array(fetch.get("high", prices + prices * 0.003))
    low       = np.array(fetch.get("low",  prices - prices * 0.003))
    n_bars    = len(prices)

    levels         = np.array(murray.get("levels",       []))
    murray_index   = float(murray.get("murray_index",    4.0))
    conf_signal    = murray.get("confluence",  {}).get("signal", "neutral")
    fan            = gann.get("fan",           [])
    anchor_type    = gann.get("anchor",        {}).get("anchor_type", "low")
    angle_breaks   = gann.get("angle_breaks",  [])

    if len(prices) == 0:
        for ax in axes:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                    ha="center", va="center", color=TEXT_COLOR)
        return

    _apply_style(axes[0].figure, axes)

    # Panel 0: OHLC + Murray + Gann
    _draw_ohlc(axes[0], prices, high, low)
    _draw_murray_levels(axes[0], levels, murray_index, float(prices[-1]), n_bars)
    _draw_gann_fan(axes[0], fan, anchor_type, n_bars)
    _draw_confluence_zone(axes[0], angle_breaks, murray)

    title_parts = [
        f"Murray {murray_index:.2f}/8",
        f"{trade_plan.get('murray_action', '')}",
        f"| Gann: {trade_plan.get('_engines',{}).get('gann',{}).get('price_vs_master','?')} master",
    ]
    axes[0].set_title("  ".join(title_parts), color=ACCENT_COLOR, fontsize=9, loc="left", pad=4)
    axes[0].set_ylabel("Price", color=TEXT_COLOR, fontsize=8)
    axes[0].set_xlim(0, n_bars - 1)

    # Panel 1: Murray heatbar
    _draw_murray_heatbar(axes[1], murray_index, conf_signal)


def render(trade_plan: dict, figsize: tuple = (14, 7)) -> plt.Figure:
    """Creates and returns a standalone Figure."""
    fig = plt.figure(figsize=figsize, facecolor=BG_COLOR)
    fig.suptitle(
        f"Murray Math & Gann Fan — {trade_plan.get('symbol', '?')}",
        color=ACCENT_COLOR, fontsize=11, fontweight="bold", y=0.98,
    )
    gs   = GridSpec(2, 1, figure=fig, height_ratios=[6, 1], hspace=0.25,
                    top=0.93, bottom=0.09, left=0.07, right=0.97)
    axes = [fig.add_subplot(gs[i]) for i in range(2)]
    draw(axes, trade_plan)
    return fig


# ── Smoke test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    np.random.seed(5)
    n = 150; prices = np.cumsum(np.random.randn(n) * 300) + 60000
    levels  = np.linspace(55000, 75000, 9)
    fan_vals = [{"scale": (p, t),
                 "label": f"{p}x{t}",
                 "values": np.linspace(60000, 60000 - (p/t) * 500 * n, n),
                 "current_value": 60000 - (p/t) * 500 * n,
                 "angle_deg": round(float(np.degrees(np.arctan(p/t))), 1)}
                for (p, t) in [(1,8),(1,4),(1,2),(1,1),(2,1),(4,1),(8,1)]]

    fake_plan = {
        "symbol": "BTCUSDT", "murray_action": "Sell — Strong resistance zone",
        "murray_index": 6.7,
        "_engines": {
            "murray": {"levels": levels, "murray_index": 6.7,
                       "confluence": {"signal": "high_confluence_sell"}},
            "gann":   {"fan": fan_vals, "angle_breaks": [],
                       "anchor": {"anchor_type": "high"}, "price_vs_master": "below"},
            "detrend":{"trend": prices, "detrended": np.zeros(n)},
            "_fetch": {"high": prices + 400, "low": prices - 400},
        },
        "_risk": {},
    }
    fig = render(fake_plan)
    fig.savefig("/tmp/murray_chart_test.png", dpi=120, bbox_inches="tight")
    print("✅ Murray chart rendered → /tmp/murray_chart_test.png")
    plt.close(fig)
