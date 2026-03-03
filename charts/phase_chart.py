# =============================================================================
#  charts/phase_chart.py — ACF / AR / Solar Dashboard (Eq 2, 5, 10)
#  A three-panel analytical dashboard showing market memory, short-term
#  momentum forecast, and solar cycle synchronisation.
#
#  Panel layout:
#    [0] ACF correlogram — significant lags highlighted, confidence bands
#    [1] AR model forecast — next 10 bars with sentiment shading
#    [2] Solar sine + seasonal bias band
#
#  draw() interface: draws onto caller-supplied axes list.
#  render() interface: creates and returns a standalone Figure.
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from gui.theme import get_current_theme

_THEME        = get_current_theme()
BG_COLOR      = _THEME["BG"]
PANEL_COLOR   = _THEME["PANEL"]
GRID_COLOR    = "#4e1f3d"
TEXT_COLOR    = _THEME["FG"]
ACCENT_COLOR  = _THEME["ACCENT"]
POS_COLOR     = "#4caf50"
NEG_COLOR     = "#f44336"
SOLAR_COLOR   = "#e95420"
CONF_BAND_CLR = "#37474f"
AR_COLOR      = "#ce93d8"
AR_ADJ_COLOR  = "#f48fb1"
SENT_POS_CLR  = "#1b5e20"
SENT_NEG_CLR  = "#b71c1c"

SEASONAL_COLORS = {
    "spring_rally_bias":      "#1b5e20",
    "summer_drift_bias":      "#f9a825",
    "autumn_volatility_bias": "#b71c1c",
    "yearend_rally_bias":     "#1565c0",
}
SEASONAL_LABELS = {
    "spring_rally_bias":      "Spring Rally Bias",
    "summer_drift_bias":      "Summer Drift",
    "autumn_volatility_bias": "Autumn Volatility Risk",
    "yearend_rally_bias":     "Year-End Rally Bias",
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


def _draw_acf_panel(ax, acf_result, cycle_lags):
    """
    Renders the ACF correlogram with:
    - Blue bars for all lags
    - Gold bars for statistically significant lags
    - Dashed ±confidence band lines
    - Vertical markers at detected cycle lags
    """
    acf_vals  = np.array(acf_result.get("acf_values",  []))
    conf_band = float(acf_result.get("conf_band",       0.10))
    sig_lags  = acf_result.get("significant_lags",      [])
    cycle_lgs = acf_result.get("cycle_lags",            [])
    best_lag  = acf_result.get("best_lag",              0)

    if len(acf_vals) == 0:
        ax.text(0.5, 0.5, "ACF: No data", transform=ax.transAxes,
                ha="center", color=TEXT_COLOR)
        return

    max_lag = len(acf_vals)
    x       = np.arange(max_lag)
    sig_set = {d["lag"] for d in sig_lags}

    # Draw bars
    colors = [ACCENT_COLOR if i in sig_set else "#37474f" for i in x]
    ax.bar(x, acf_vals, color=colors, width=0.8, alpha=0.85, zorder=3)

    # Confidence bands
    ax.axhline(+conf_band, color=CONF_BAND_CLR, lw=1.0, ls="--", alpha=0.8, zorder=4)
    ax.axhline(-conf_band, color=CONF_BAND_CLR, lw=1.0, ls="--", alpha=0.8, zorder=4)
    ax.axhline(0.0,        color=GRID_COLOR,    lw=0.6,           alpha=0.6, zorder=2)
    ax.fill_between(x, -conf_band, conf_band, color=CONF_BAND_CLR, alpha=0.08)

    # Cycle lag markers
    for lag in cycle_lgs[:8]:
        if lag < max_lag:
            ax.axvline(lag, color=POS_COLOR, lw=0.9, ls=":", alpha=0.7, zorder=5)
            ax.text(lag, conf_band + 0.03, f"{lag}", ha="center",
                    color=POS_COLOR, fontsize=6, va="bottom")

    # Best lag annotation
    if 0 < best_lag < max_lag:
        ax.axvline(best_lag, color=ACCENT_COLOR, lw=1.4, ls="-", alpha=0.9, zorder=6)
        ax.text(best_lag, ax.get_ylim()[1] * 0.9, f"Best: {best_lag}",
                ha="center", color=ACCENT_COLOR, fontsize=7.5, fontweight="bold")

    ax.set_ylabel("ACF ρ", color=TEXT_COLOR, fontsize=8)
    ax.set_xlabel("Lag (bars)", color=TEXT_COLOR, fontsize=8)
    ax.set_xlim(0, max_lag - 1)
    ax.set_ylim(-1.05, 1.05)

    mem_score = float(acf_result.get("memory_score", 0.0))
    ax.set_title(f"ACF — Market Memory Score: {mem_score:.2f}",
                 color=ACCENT_COLOR, fontsize=9, loc="left", pad=4)


def _draw_ar_panel(ax, ar_result, n_history=40):
    """
    Renders the AR model: recent log-returns history + multi-step forecast.
    Raw and sentiment-adjusted forecasts overlaid.
    """
    # Historical returns (last n_history bars)
    day_vals = ar_result.get("session_split", {}).get("day",   {}).get("values", np.array([]))
    all_vals = np.concatenate([
        ar_result.get("session_split", {}).get("day",   {}).get("values", np.array([])),
        ar_result.get("session_split", {}).get("night", {}).get("values", np.array([])),
    ]) if ar_result.get("success") else np.array([])

    fcast_raw = np.array(ar_result.get("forecast_raw",      []))
    fcast_adj = np.array(ar_result.get("forecast_adjusted", []))
    sent_score = float(ar_result.get("sentiment_score",     0.0))
    direction  = ar_result.get("forecast_direction", "flat")

    if len(all_vals) == 0 and len(fcast_raw) == 0:
        ax.text(0.5, 0.5, "AR: No data", transform=ax.transAxes,
                ha="center", color=TEXT_COLOR)
        return

    # History segment
    hist = all_vals[-n_history:] if len(all_vals) >= n_history else all_vals
    n_h  = len(hist)
    x_h  = np.arange(n_h)
    x_f  = np.arange(n_h, n_h + len(fcast_raw))

    # Colour history bars by positive/negative
    colors = [POS_COLOR if v >= 0 else NEG_COLOR for v in hist]
    ax.bar(x_h, hist, color=colors, width=0.8, alpha=0.7, zorder=3)

    # Forecast area
    if len(fcast_raw) > 0 and len(x_f) > 0:
        ax.plot(x_f, fcast_raw, color=AR_COLOR, lw=1.5,
                marker="o", ms=3, zorder=5, label="AR Forecast")
    if len(fcast_adj) > 0 and len(x_f) > 0:
        ax.plot(x_f, fcast_adj, color=AR_ADJ_COLOR, lw=1.5, ls="--",
                marker="s", ms=3, zorder=5, label=f"Sentiment-adj (s={sent_score:+.2f})")

    ax.axvline(n_h - 0.5, color=ACCENT_COLOR, lw=1.0, ls=":", alpha=0.8)
    ax.axhline(0, color=GRID_COLOR, lw=0.6)

    # Shade forecast background
    if len(x_f) > 0:
        fc   = SENT_POS_CLR if direction == "up" else (SENT_NEG_CLR if direction == "down" else GRID_COLOR)
        ax.axvspan(n_h - 0.5, n_h + len(fcast_raw) - 0.5, color=fc, alpha=0.07)

    inertia = ar_result.get("inertia", {})
    dom_ses = inertia.get("dominant_session", "?")
    ax.set_title(f"AR Forecast — {direction.upper()}  |  Dominant session: {dom_ses}",
                 color=ACCENT_COLOR, fontsize=9, loc="left", pad=4)
    ax.set_ylabel("Log Return", color=TEXT_COLOR, fontsize=8)
    ax.set_xlabel("Bars", color=TEXT_COLOR, fontsize=8)
    ax.legend(loc="upper left", fontsize=7, facecolor=PANEL_COLOR,
              labelcolor=TEXT_COLOR, framealpha=0.6)


def _draw_solar_panel(ax, solar_result):
    """
    Renders the solar sine wave and seasonal bias band across the bar series.
    Marks upcoming solstice/equinox with vertical dashed lines.
    """
    solar_sine = np.array(solar_result.get("solar_sine",    []))
    bias       = solar_result.get("seasonal_bias",           "unknown")
    vol_flag   = solar_result.get("volatility_flag",         False)
    cards      = solar_result.get("days_to_cardinals",       {}).get("days_to_cardinals", {})

    if len(solar_sine) == 0:
        ax.text(0.5, 0.5, "Solar: No data", transform=ax.transAxes,
                ha="center", color=TEXT_COLOR)
        return

    n = len(solar_sine); x = np.arange(n)

    # Seasonal band
    bias_color = SEASONAL_COLORS.get(bias, GRID_COLOR)
    ax.fill_between(x, -1, 1, color=bias_color, alpha=0.08)

    # Solar sine wave
    ax.fill_between(x, solar_sine, 0,
                    where=solar_sine >= 0, color=SOLAR_COLOR, alpha=0.35)
    ax.fill_between(x, solar_sine, 0,
                    where=solar_sine < 0,  color="#7986cb",   alpha=0.35)
    ax.plot(x, solar_sine, color=SOLAR_COLOR, lw=1.1, alpha=0.9)

    ax.axhline(0, color=GRID_COLOR, lw=0.6)
    ax.axhline(+0.9, color=ACCENT_COLOR, lw=0.5, ls="--", alpha=0.5)
    ax.axhline(-0.9, color=ACCENT_COLOR, lw=0.5, ls="--", alpha=0.5)

    ax.set_ylim(-1.1, 1.4)
    ax.set_yticks([-1, -0.5, 0, 0.5, 1])
    ax.set_yticklabels(["-1", "-0.5", "0", "+0.5", "+1"],
                       color=TEXT_COLOR, fontsize=7)

    bias_label = SEASONAL_LABELS.get(bias, bias)
    vol_str    = "  ⚡ Volatility zone!" if vol_flag else ""
    ax.set_title(f"Solar Cycle — {bias_label}{vol_str}",
                 color=SOLAR_COLOR if not vol_flag else "#ff7043",
                 fontsize=9, loc="left", pad=4)
    ax.set_ylabel("Solar Sine", color=TEXT_COLOR, fontsize=8)
    ax.set_xlabel("Bars", color=TEXT_COLOR, fontsize=8)
    ax.set_xlim(0, n - 1)


# ── Public interfaces ─────────────────────────────────────────────────────────

def draw(axes: list, trade_plan: dict) -> None:
    """
    Draws ACF / AR / Solar onto caller-supplied axes [ax0, ax1, ax2].

    Parameters
    ----------
    axes       : list of 3 matplotlib Axes
    trade_plan : dict — output of aggregator.run()
    """
    eng    = trade_plan.get("_engines", {})
    acf_r  = eng.get("acf",     {})
    ar_r   = eng.get("ar",      {})
    solar_r= eng.get("solar",   {})

    _apply_style(axes[0].figure, axes)
    _draw_acf_panel(axes[0], acf_r, acf_r.get("cycle_lags", []))
    _draw_ar_panel(axes[1],  ar_r)
    _draw_solar_panel(axes[2], solar_r)


def render(trade_plan: dict, figsize: tuple = (14, 9)) -> plt.Figure:
    """Creates and returns a standalone Figure."""
    fig = plt.figure(figsize=figsize, facecolor=BG_COLOR)
    fig.suptitle(
        f"ACF / AR Forecast / Solar Cycle — {trade_plan.get('symbol','?')}",
        color=ACCENT_COLOR, fontsize=11, fontweight="bold", y=0.98,
    )
    gs   = GridSpec(3, 1, figure=fig, hspace=0.40,
                    top=0.93, bottom=0.07, left=0.07, right=0.97)
    axes = [fig.add_subplot(gs[i]) for i in range(3)]
    draw(axes, trade_plan)
    return fig


# ── Smoke test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    np.random.seed(7)
    n = 100; t = np.arange(n)
    fake_acf  = {"acf_values": np.exp(-t * 0.05) * np.cos(2 * np.pi * t / 21),
                 "conf_band": 0.12, "best_lag": 21, "memory_score": 0.55,
                 "significant_lags": [{"lag": 21, "correlation": 0.6, "strength": 0.6}],
                 "cycle_lags": [21, 42, 63]}
    fake_ar   = {"success": True, "forecast_raw": np.random.randn(10) * 0.003,
                 "forecast_adjusted": np.random.randn(10) * 0.003 + 0.001,
                 "forecast_direction": "up", "sentiment_score": 0.15,
                 "session_split": {"day":   {"values": np.random.randn(50) * 0.005},
                                   "night": {"values": np.random.randn(50) * 0.005}},
                 "inertia": {"dominant_session": "night"}}
    fake_solar= {"solar_sine": np.sin(2 * np.pi * t / 365),
                 "seasonal_bias": "autumn_volatility_bias",
                 "volatility_flag": True, "days_to_cardinals": {"days_to_cardinals": {}}}

    fake_plan = {
        "symbol": "BTCUSDT",
        "_engines": {"acf": fake_acf, "ar": fake_ar, "solar": fake_solar},
        "_risk": {},
    }
    fig = render(fake_plan)
    fig.savefig("/tmp/phase_chart_test.png", dpi=120, bbox_inches="tight")
    print("✅ Phase chart rendered → /tmp/phase_chart_test.png")
    plt.close(fig)
