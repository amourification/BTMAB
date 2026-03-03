# =============================================================================
#  charts/risk_chart.py — Risk Management Dashboard (Eq 7, 8, 11)
#  Four-panel risk overview: Kelly gauge, stop ladder, Gamma vol surface,
#  portfolio summary. Helpers in risk_chart_utils.py.
#
#  draw() interface: draws onto caller-supplied axes list [ax0..ax3].
#  render() interface: creates and returns a standalone Figure.
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from charts.risk_chart_utils import (
    draw_kelly_gauge, draw_stop_ladder, draw_gamma_vol,
    draw_portfolio_summary, BG_COLOR, PANEL_COLOR, GRID_COLOR,
    TEXT_COLOR, ACCENT_CLR,
)


def _apply_style(fig, axes):
    fig.patch.set_facecolor(BG_COLOR)
    for ax in axes:
        ax.set_facecolor(PANEL_COLOR)
        ax.tick_params(colors=TEXT_COLOR, labelsize=8)
        for spine in ax.spines.values():
            spine.set_color(GRID_COLOR)
        ax.grid(True, color=GRID_COLOR, linewidth=0.4, alpha=0.5)


def draw(axes: list, trade_plan: dict) -> None:
    """
    Draws the four-panel risk dashboard onto caller-supplied axes.

    Parameters
    ----------
    axes       : list of 4 matplotlib Axes  [kelly, stops, gamma, portfolio]
    trade_plan : dict — output of aggregator.run()
    """
    eng     = trade_plan.get("_engines", {})
    risk    = trade_plan.get("_risk",    {})
    kelly_r = eng.get("kelly",      {})
    gamma_r = eng.get("gamma",      {})
    stops_r = risk.get("stops",     {})
    port_r  = risk.get("portfolio", {})

    detrend   = eng.get("detrend", {})
    trend     = np.array(detrend.get("trend",     []))
    detr      = np.array(detrend.get("detrended", []))
    prices    = trend + detr if len(trend) == len(detr) and len(trend) > 0 else trend
    cur_price = float(prices[-1]) if len(prices) > 0 else 0.0

    _apply_style(axes[0].figure, axes)
    draw_kelly_gauge(axes[0],      kelly_r)
    draw_stop_ladder(axes[1],      stops_r, cur_price)
    draw_gamma_vol(axes[2],        gamma_r)
    draw_portfolio_summary(axes[3],port_r,  kelly_r)


def render(trade_plan: dict, figsize: tuple = (14, 10)) -> plt.Figure:
    """
    Creates and returns a standalone risk dashboard Figure.

    Parameters
    ----------
    trade_plan : dict — output of aggregator.run()
    figsize    : (width, height) in inches

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig = plt.figure(figsize=figsize, facecolor=BG_COLOR)
    fig.suptitle(
        f"Risk Management Dashboard — {trade_plan.get('symbol', '?')}",
        color=ACCENT_CLR, fontsize=11, fontweight="bold", y=0.98,
    )
    gs   = GridSpec(2, 2, figure=fig, hspace=0.40, wspace=0.30,
                    top=0.93, bottom=0.07, left=0.06, right=0.97)
    axes = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(2)]
    draw(axes, trade_plan)
    return fig


# ── Smoke test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    np.random.seed(3)
    n = 100; prices = np.cumsum(np.random.randn(n) * 300) + 60000
    vol = np.full(n, np.nan)
    for i in range(20, n):
        vol[i] = np.std(np.diff(np.log(prices[i-20:i]))) * np.sqrt(252)

    fake_plan = {
        "symbol": "BTCUSDT",
        "_engines": {
            "kelly":  {"success": True, "position_pct": 10.5, "confidence": 0.68,
                       "p_win": 0.63, "win_ratio_b": 1.8, "expected_value": 0.041,
                       "position_tier": {"tier": "moderate", "label": "Moderate (10–15%)"}},
            "gamma":  {"success": True,
                       "vol_series": vol, "gamma_proxy": np.random.randn(n) * 0.01,
                       "regime": {"regime": "negative_gamma"},
                       "vol_stats": {"current": 0.55, "percentile": 0.72, "regime": "high"}},
            "detrend":{"trend": prices, "detrended": np.zeros(n)},
        },
        "_risk": {
            "stops": {"success": True, "entry_price": 61000.0, "atr": 450.0,
                      "phase_multiplier": 1.5,
                      "initial_stop":  {"stop_price": 59200.0},
                      "trailing_stop": {"stop_price": 59800.0},
                      "active_stop":   {"stop_type": "trailing", "stop_price": 59800.0}},
            "portfolio": {"success": True,
                          "positions": {"net_pct": 0.08, "gross_pct": 0.38,
                                        "primary": {"size_pct": 0.105},
                                        "hedge":   {"size_pct": 0.025}},
                          "risk_adjusted": {"grade": "B", "risk_reward": 2.3},
                          "overall_risk_score": 0.32,
                          "summary": "Net LONG 8% | Gross 38% | R/R 2.3× (B) | DD 1.2%",
                          "exposure_checks": []},
        },
    }
    fig = render(fake_plan)
    fig.savefig("/tmp/risk_chart_test.png", dpi=120, bbox_inches="tight")
    print(f"✅ Risk chart rendered → /tmp/risk_chart_test.png")
    plt.close(fig)
