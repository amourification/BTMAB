from __future__ import annotations

from datetime import timedelta

import numpy as np
import matplotlib.pyplot as plt
from gui.theme import get_current_theme
from charts.cycle_time_utils import _infer_bar_timedelta, _ts_to_datetime

_THEME         = get_current_theme()
BG_COLOR       = _THEME["BG"]
PANEL_COLOR    = _THEME["PANEL"]
GRID_COLOR     = "#4e1f3d"
TEXT_COLOR     = _THEME["FG"]
ACCENT_COLOR   = _THEME["ACCENT"]
RETR_COLOR     = "#4fc3f7"
EXT_COLOR      = "#ef5350"


def _apply_style(ax: plt.Axes) -> None:
    ax.set_facecolor(PANEL_COLOR)
    ax.tick_params(colors=TEXT_COLOR, labelsize=8)
    ax.xaxis.label.set_color(TEXT_COLOR)
    ax.yaxis.label.set_color(TEXT_COLOR)
    for spine in ax.spines.values():
        spine.set_color(GRID_COLOR)
    ax.grid(True, color=GRID_COLOR, linewidth=0.4, alpha=0.6)


def draw(axes: list, trade_plan: dict) -> None:
    """
    Draws a three-panel Elliott/Fibonacci view:
    - Panel 1: Price + impulse swing + Fibonacci levels + wave labels.
    - Panel 2: RSI with simple divergence markers.
    - Panel 3: MACD line / signal / histogram with divergence markers.
    """
    if not axes:
        return

    eng = trade_plan.get("_engines", {})
    det = eng.get("detrend", {}) or {}
    ef = eng.get("elliott_fib", {}) or {}
    fetch = eng.get("_fetch", {}) or {}

    trend = np.array(det.get("trend", []), dtype=float)
    detr = np.array(det.get("detrended", []), dtype=float)
    prices = trend + detr if len(trend) == len(detr) and len(trend) > 0 else trend
    timestamps = np.array(fetch.get("timestamps", np.arange(len(prices))), dtype=float)

    if prices is None or len(prices) == 0 or not ef.get("success"):
        ax = axes[0]
        _apply_style(ax)
        ax.text(
            0.5,
            0.5,
            "Elliott/Fib: Not enough data",
            transform=ax.transAxes,
            ha="center",
            va="center",
            color=TEXT_COLOR,
        )
        return

    x = np.arange(len(prices))

    # --- Panel 1: price + fib + waves ----------------------------------------
    ax_price = axes[0]
    _apply_style(ax_price)
    ax_price.plot(x, prices, color="#e8e3d5", lw=0.9, alpha=0.9, label="Price")

    swing = ef.get("swing_points", {}) or {}
    s_idx = int(swing.get("start_idx", 0))
    e_idx = int(swing.get("end_idx", len(prices) - 1))

    if 0 <= s_idx < len(prices) and 0 <= e_idx < len(prices):
        ax_price.plot(
            [s_idx, e_idx],
            [prices[s_idx], prices[e_idx]],
            color=ACCENT_COLOR,
            lw=1.4,
            label="Impulse leg",
        )
        ax_price.scatter(
            [s_idx, e_idx],
            [prices[s_idx], prices[e_idx]],
            color=ACCENT_COLOR,
            s=20,
            zorder=4,
        )

    # Retracement bands
    retr = ef.get("fib_retracements", {}) or {}
    for key, level in retr.items():
        ax_price.axhline(
            level,
            color=RETR_COLOR,
            lw=0.8,
            alpha=0.6,
            linestyle="--",
        )
        ax_price.text(
            0.99,
            level,
            f"{key}% {level:,.2f}",
            ha="right",
            va="center",
            color=RETR_COLOR,
            fontsize=7,
        )

    # Extension targets
    exts = ef.get("fib_extensions", {}) or {}
    for key, level in exts.items():
        ax_price.axhline(
            level,
            color=EXT_COLOR,
            lw=0.9,
            alpha=0.7,
            linestyle="-.",
        )
        ax_price.text(
            0.01,
            level,
            f"x{key} {level:,.2f}",
            ha="left",
            va="center",
            color=EXT_COLOR,
            fontsize=7,
        )

    # Wave labels (proposed count)
    wave = ef.get("wave_count") or {}
    impulse_labels = wave.get("impulse_labels") or {}
    for label, idx in impulse_labels.items():
        idx = int(idx)
        if 0 <= idx < len(prices):
            ax_price.scatter(
                [idx],
                [prices[idx]],
                color=ACCENT_COLOR,
                s=30,
                zorder=5,
            )
            ax_price.text(
                idx,
                prices[idx],
                f"{label}",
                color=ACCENT_COLOR,
                fontsize=7,
                ha="center",
                va="bottom",
            )

    direction = ef.get("direction", "neutral")
    sym = trade_plan.get("symbol", "?")
    score = wave.get("score", 0.0) if wave else 0.0
    ax_price.set_title(
        f"Elliott / Fibonacci — {sym} ({direction.upper()} impulse, score={score:.2f})",
        color=ACCENT_COLOR,
        fontsize=9,
        loc="left",
        pad=4,
    )
    ax_price.set_ylabel("Price", color=TEXT_COLOR, fontsize=8)
    ax_price.set_xlim(0, len(prices) - 1)

    # Forward-looking time windows from Elliott / RSI / MACD (bars → timestamps)
    try:
        bar_secs = _infer_bar_timedelta(timestamps)
    except Exception:
        bar_secs = 0.0
    if bar_secs > 0 and len(timestamps) > 0:
        current_dt = _ts_to_datetime(timestamps[-1])
        events = ef.get("time_events", []) or []
        y_text = 0.96
        for ev in events:
            bars_ahead = float(ev.get("bars_ahead", 0.0) or 0.0)
            if bars_ahead <= 0:
                continue
            secs_ahead = bars_ahead * bar_secs
            dt_future = current_dt + timedelta(seconds=secs_ahead)
            label = ev.get("label") or f"{ev.get('source','?')} {ev.get('kind','')}"
            ax_price.text(
                0.99,
                y_text,
                f"{label} ≈ {dt_future.strftime('%Y-%m-%d %H:%M UTC')}",
                transform=ax_price.transAxes,
                ha="right",
                va="top",
                fontsize=7,
                color=TEXT_COLOR,
            )
            y_text -= 0.07

    # --- Panel 2: RSI ---------------------------------------------------------
    if len(axes) >= 2:
        ax_rsi = axes[1]
        _apply_style(ax_rsi)
        rsi = np.array(ef.get("rsi", []), dtype=float)
        if len(rsi) == len(prices):
            ax_rsi.plot(x, rsi, color="#80cbc4", lw=0.9, label="RSI(14)")
            ax_rsi.axhline(70, color="#ffab91", lw=0.6, linestyle="--", alpha=0.7)
            ax_rsi.axhline(30, color="#90caf9", lw=0.6, linestyle="--", alpha=0.7)
            for div in ef.get("rsi_divergences", []):
                idx = int(div.get("price_idx", -1))
                if 0 <= idx < len(prices):
                    color = "#ff7043" if div.get("type") == "bearish" else "#66bb6a"
                    ax_rsi.scatter([idx], [rsi[idx]], color=color, s=20, zorder=5)
        ax_rsi.set_ylabel("RSI", color=TEXT_COLOR, fontsize=8)

    # --- Panel 3: MACD --------------------------------------------------------
    if len(axes) >= 3:
        ax_macd = axes[2]
        _apply_style(ax_macd)
        macd_line = np.array(ef.get("macd_line", []), dtype=float)
        macd_signal = np.array(ef.get("macd_signal", []), dtype=float)
        macd_hist = np.array(ef.get("macd_hist", []), dtype=float)
        if len(macd_line) == len(prices):
            ax_macd.plot(x, macd_line, color="#ffb74d", lw=0.8, label="MACD")
            ax_macd.plot(x, macd_signal, color="#ffffff", lw=0.7, alpha=0.8, label="Signal")
            ax_macd.bar(
                x,
                macd_hist,
                color=np.where(macd_hist >= 0, "#81c784", "#e57373"),
                width=0.8,
                alpha=0.7,
            )
            for div in ef.get("macd_divergences", []):
                idx = int(div.get("price_idx", -1))
                if 0 <= idx < len(prices):
                    color = "#ff7043" if div.get("type") == "bearish" else "#66bb6a"
                    ax_macd.scatter([idx], [macd_line[idx]], color=color, s=20, zorder=5)
        ax_macd.set_ylabel("MACD", color=TEXT_COLOR, fontsize=8)
        ax_macd.set_xlabel("Bars", color=TEXT_COLOR, fontsize=8)

