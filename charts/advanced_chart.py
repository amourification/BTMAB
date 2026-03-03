from __future__ import annotations

"""
charts/advanced_chart.py — Advanced Regime & Uncertainty Dashboard

Two-panel view for the Advanced mode:
  [0] Multi-scale cycle periods by window (bar chart)
  [1] Uncertainty / stress gauges
"""

from typing import Dict, Any, List

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from gui.theme import get_current_theme

_THEME = get_current_theme()
BG_COLOR = _THEME["BG"]
PANEL_COLOR = _THEME["PANEL"]
TEXT_COLOR = _THEME["FG"]
ACCENT_COLOR = _THEME["ACCENT"]
POS_COLOR = "#4caf50"
NEG_COLOR = "#f44336"


def _apply_style(fig: plt.Figure, axes: List[plt.Axes]) -> None:
    fig.patch.set_facecolor(BG_COLOR)
    for ax in axes:
        ax.set_facecolor(PANEL_COLOR)
        ax.tick_params(colors=TEXT_COLOR, labelsize=8)
        ax.xaxis.label.set_color(TEXT_COLOR)
        ax.yaxis.label.set_color(TEXT_COLOR)
        for spine in ax.spines.values():
            spine.set_color("#444444")
        ax.grid(True, color="#333333", linewidth=0.4, alpha=0.4)


def _draw_cycles_panel(ax: plt.Axes, advanced: Dict[str, Any]) -> None:
    ms = advanced.get("multi_scale_cycles", {}) or {}
    windows = ms.get("windows", []) or []
    if not windows:
        ax.text(0.5, 0.5, "No multi-scale cycle data", transform=ax.transAxes,
                ha="center", va="center", color=TEXT_COLOR)
        ax.set_title("Multi-scale cycles", color=ACCENT_COLOR, fontsize=9, loc="left")
        return

    win_vals = np.array([w.get("window", 0) for w in windows], dtype=float)
    period_vals = np.array([w.get("period", 0.0) for w in windows], dtype=float)
    conf_vals = np.array([w.get("confidence", 0.0) for w in windows], dtype=float)

    idx = np.arange(len(win_vals))
    colors = [ACCENT_COLOR if c >= conf_vals.max() else "#555555" for c in conf_vals]

    ax.bar(idx, period_vals, color=colors, alpha=0.85)
    for i, (w, p, c) in enumerate(zip(win_vals, period_vals, conf_vals)):
        ax.text(i, p, f"{int(p)}", ha="center", va="bottom",
                fontsize=7, color=TEXT_COLOR)
        ax.text(i, 0, f"W={int(w)}\nC={c:.2f}", ha="center", va="bottom",
                fontsize=6, color=TEXT_COLOR)

    ax.set_ylabel("Dominant Period (bars)", color=TEXT_COLOR, fontsize=8)
    ax.set_xticks(idx)
    ax.set_xticklabels([f"{int(w)}" for w in win_vals], fontsize=7, color=TEXT_COLOR)
    ax.set_xlabel("Window length (bars)", color=TEXT_COLOR, fontsize=8)
    ax.set_title("Multi-scale cycles — dominant periods per window",
                 color=ACCENT_COLOR, fontsize=9, loc="left", pad=4)


def _draw_uncertainty_panel(ax: plt.Axes, advanced: Dict[str, Any]) -> None:
    unc = float(advanced.get("uncertainty_score", 0.0) or 0.0)
    stress = float(advanced.get("stress_score", 0.0) or 0.0)
    vol_regime = str(advanced.get("vol_regime", "normal") or "normal")
    trend_regime = str(advanced.get("trend_regime", "sideways") or "sideways")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 3)
    ax.set_yticks([])
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0", "0.25", "0.5", "0.75", "1"], color=TEXT_COLOR, fontsize=7)

    # Uncertainty bar
    ax.barh(2.0, unc, height=0.6, color=NEG_COLOR if unc > 0.7 else ACCENT_COLOR,
            alpha=0.7)
    ax.text(0.01, 2.0, f"Uncertainty = {unc:.2f}", va="center",
            fontsize=8, color=TEXT_COLOR)

    # Stress bar
    ax.barh(1.0, stress, height=0.6, color=NEG_COLOR if stress > 0.7 else POS_COLOR,
            alpha=0.7)
    ax.text(0.01, 1.0, f"Stress = {stress:.2f}", va="center",
            fontsize=8, color=TEXT_COLOR)

    # Regime labels
    ax.text(0.01, 0.25, f"Vol regime: {vol_regime}", va="center",
            fontsize=8, color=TEXT_COLOR)
    ax.text(0.5, 0.25, f"Trend regime: {trend_regime}", va="center",
            fontsize=8, color=TEXT_COLOR)

    ax.set_title("Uncertainty / Stress regime snapshot",
                 color=ACCENT_COLOR, fontsize=9, loc="left", pad=4)


def draw(axes: list, trade_plan: Dict[str, Any]) -> None:
    """
    Draws the advanced dashboard onto caller-supplied axes [ax0, ax1].
    """
    advanced = trade_plan.get("advanced", {}) or {}
    _apply_style(axes[0].figure, axes)
    _draw_cycles_panel(axes[0], advanced)
    _draw_uncertainty_panel(axes[1], advanced)


def render(trade_plan: Dict[str, Any], figsize: tuple = (12, 7)) -> plt.Figure:
    fig = plt.figure(figsize=figsize, facecolor=BG_COLOR)
    fig.suptitle(
        f"Advanced Regime / Uncertainty — {trade_plan.get('symbol','?')}",
        color=ACCENT_COLOR,
        fontsize=11,
        fontweight="bold",
        y=0.97,
    )
    gs = GridSpec(2, 1, figure=fig, hspace=0.35,
                  top=0.9, bottom=0.08, left=0.07, right=0.97)
    axes = [fig.add_subplot(gs[i]) for i in range(2)]
    draw(axes, trade_plan)
    return fig

