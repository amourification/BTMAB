# =============================================================================
#  charts/cycle_chart.py — Dominant Cycle Overlay Chart (Eq 1, 3, 4)
#  Renders raw price, SSA reconstruction, FFT cycle envelope, and
#  Hilbert phase-band colouring in a multi-panel matplotlib figure.
#
#  Panel layout:
#    [0] Price + SSA cycle reconstruction + Hilbert phase colouring
#    [1] Detrended oscillator + FFT dominant frequency band
#    [2] Hilbert instantaneous phase (0°–360°) with turn-type zones
#
#  draw() interface: draws onto caller-supplied axes list.
#  render() interface: creates and returns a standalone Figure.
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
from matplotlib.gridspec import GridSpec
from gui.theme import get_current_theme

# ── Colour palette (Ubuntu-like dark) ────────────────────────────────────────
_THEME         = get_current_theme()
BG_COLOR       = _THEME["BG"]
PANEL_COLOR    = _THEME["PANEL"]
PRICE_COLOR    = "#f2f1f0"
SSA_COLOR      = "#19b6ee"
FFT_ENV_COLOR  = "#a56de2"
DETREND_COLOR  = "#8ae234"
PHASE_LINE_CLR = "#e95420"
GRID_COLOR     = "#4e1f3d"
TEXT_COLOR     = _THEME["FG"]
ACCENT_COLOR   = _THEME["ACCENT"]

PHASE_ZONE_COLORS = {
    "early_bullish": "#1b5e20",   # deep green
    "mid_expansion": "#2e7d32",   # medium green
    "distribution":  "#b71c1c",   # deep red
    "accumulation":  "#1a237e",   # deep blue
}

PHASE_ZONE_LABELS = {
    "early_bullish": "Early Bull (0°–90°)",
    "mid_expansion": "Expansion (90°–180°)",
    "distribution":  "Distribution (180°–270°)",
    "accumulation":  "Accumulation (270°–360°)",
}

from charts.cycle_time_utils import compute_time_angle_predictions


# ── Internal helpers ──────────────────────────────────────────────────────────

def _apply_style(fig: plt.Figure, axes: list) -> None:
    """Applies dark terminal styling to fig and all axes."""
    fig.patch.set_facecolor(BG_COLOR)
    for ax in axes:
        ax.set_facecolor(PANEL_COLOR)
        ax.tick_params(colors=TEXT_COLOR, labelsize=8)
        ax.xaxis.label.set_color(TEXT_COLOR)
        ax.yaxis.label.set_color(TEXT_COLOR)
        for spine in ax.spines.values():
            spine.set_color(GRID_COLOR)
        ax.grid(True, color=GRID_COLOR, linewidth=0.4, alpha=0.6)


def _phase_to_zone(phase_deg: float) -> str:
    p = phase_deg % 360
    if   p < 90:  return "early_bullish"
    elif p < 180: return "mid_expansion"
    elif p < 270: return "distribution"
    else:         return "accumulation"


def _coloured_phase_bands(
    ax: plt.Axes,
    phase_arr: np.ndarray,
    prices:    np.ndarray,
) -> None:
    """
    Shades the price panel background according to Hilbert phase zone.
    Each bar gets a translucent colour band based on its phase.
    """
    n = min(len(phase_arr), len(prices))
    x = np.arange(n)
    zone_prev = _phase_to_zone(float(phase_arr[0]))
    start     = 0

    for i in range(1, n + 1):
        zone_curr = _phase_to_zone(float(phase_arr[i - 1])) if i < n else None
        if zone_curr != zone_prev or i == n:
            color = PHASE_ZONE_COLORS[zone_prev]
            ax.axvspan(start, i - 1, alpha=0.12, color=color, lw=0)
            zone_prev = zone_curr
            start     = i


def _fft_envelope(fft_result: dict, n: int) -> tuple:
    """
    Reconstructs a ±amplitude band for the dominant FFT cycle.
    Returns (upper_band, lower_band) arrays of length n.
    """
    primary = fft_result.get("primary_cycle", {})
    period  = float(primary.get("period",    64.0))
    amp     = float(primary.get("amplitude", 1.0))
    # Normalise amplitude to a fraction of price range (for overlay scaling)
    t       = np.arange(n)
    cycle   = amp * np.sin(2 * np.pi * t / period)
    return cycle, period


def _draw_panel_0(
    ax:          plt.Axes,
    prices:      np.ndarray,
    ssa_recon:   np.ndarray,
    phase_arr:   np.ndarray,
    timestamps:  np.ndarray,
    trade_plan:  dict,
) -> None:
    """Price + SSA cycle reconstruction + phase bands."""
    n  = len(prices)
    x  = np.arange(n)

    _coloured_phase_bands(ax, phase_arr[:n], prices)

    ax.plot(x, prices,    color=PRICE_COLOR, lw=0.9, alpha=0.9, label="Price",          zorder=3)

    if len(ssa_recon) == n:
        # Scale SSA recon to price range for visual overlay
        pr   = prices.max() - prices.min()
        sr   = ssa_recon.max() - ssa_recon.min() if ssa_recon.std() > 1e-6 else 1
        recon_scaled = prices.mean() + (ssa_recon - ssa_recon.mean()) / sr * pr * 0.4
        ax.plot(x, recon_scaled, color=SSA_COLOR, lw=1.4, alpha=0.8,
                label="SSA Cycle",  zorder=4, linestyle="--")

    # Annotate turn urgency
    urgency = trade_plan.get("turn_urgency", "low")
    phase   = trade_plan.get("phase_deg",    0.0)
    label   = f"Phase {phase:.0f}°  |  {trade_plan.get('turn_type','').replace('_',' ').title()}"
    ax.set_title(label, color=ACCENT_COLOR, fontsize=9, loc="left", pad=4)
    ax.set_ylabel("Price", color=TEXT_COLOR, fontsize=8)
    ax.legend(loc="upper left", fontsize=7, facecolor=PANEL_COLOR,
              labelcolor=TEXT_COLOR, framealpha=0.6)
    ax.set_xlim(0, n - 1)
    # Ensure y-axis rescales for each symbol/run and stays linear.
    ax.set_yscale("linear")
    ax.relim()
    ax.autoscale_view()


def _draw_panel_1(
    ax:         plt.Axes,
    detrended:  np.ndarray,
    fft_result: dict,
    zero_cross: np.ndarray,
) -> None:
    """Detrended oscillator + FFT envelope band."""
    n  = len(detrended)
    x  = np.arange(n)

    ax.axhline(0, color=GRID_COLOR, lw=0.8, zorder=1)
    ax.fill_between(x, detrended, 0,
                    where=detrended >= 0, color="#2e7d32", alpha=0.35, zorder=2)
    ax.fill_between(x, detrended, 0,
                    where=detrended < 0,  color="#b71c1c", alpha=0.35, zorder=2)
    ax.plot(x, detrended, color=DETREND_COLOR, lw=0.8, alpha=0.9, zorder=3, label="Detrended")

    # FFT cycle overlay (normalised)
    fft_cycle, period = _fft_envelope(fft_result, n)
    if detrended.std() > 1e-6:
        scale = detrended.std() * 1.5 / (fft_cycle.std() + 1e-12)
        fft_scaled = fft_cycle * scale
        ax.plot(x, fft_scaled, color=FFT_ENV_COLOR, lw=1.1, alpha=0.7,
                label=f"FFT {period:.0f}-bar cycle", linestyle="-.")

    # Zero crossings
    if len(zero_cross) > 0:
        valid = zero_cross[zero_cross < n]
        ax.scatter(valid, np.zeros(len(valid)), color=ACCENT_COLOR,
                   s=20, zorder=5, marker="|", linewidths=0.8, label="Zero Cross")

    ax.set_ylabel("Oscillator", color=TEXT_COLOR, fontsize=8)
    ax.legend(loc="upper left", fontsize=7, facecolor=PANEL_COLOR,
              labelcolor=TEXT_COLOR, framealpha=0.6)
    ax.set_xlim(0, n - 1)


def _draw_panel_2(
    ax:        plt.Axes,
    phase_arr: np.ndarray,
    timestamps: np.ndarray,
    trade_plan: dict,
) -> None:
    """Hilbert instantaneous phase with zone shading."""
    n  = len(phase_arr)
    x  = np.arange(n)

    # Draw zone bands
    zone_bounds = [(0, 90, "early_bullish"), (90, 180, "mid_expansion"),
                   (180, 270, "distribution"), (270, 360, "accumulation")]
    for lo, hi, zone in zone_bounds:
        ax.axhspan(lo, hi, alpha=0.15, color=PHASE_ZONE_COLORS[zone], lw=0)
        ax.text(n * 0.99, (lo + hi) / 2, PHASE_ZONE_LABELS[zone],
                ha="right", va="center", color=TEXT_COLOR, fontsize=6.5, alpha=0.7)

    # phase_arr is already in [0, 360] from hilbert.py
    ax.plot(x, phase_arr % 360, color=PHASE_LINE_CLR, lw=1.0, zorder=3)
    ax.set_ylim(0, 360)
    ax.set_yticks([0, 90, 180, 270, 360])
    ax.set_yticklabels(["0°", "90°", "180°", "270°", "360°"],
                       color=TEXT_COLOR, fontsize=7)
    ax.set_ylabel("Phase (°)", color=TEXT_COLOR, fontsize=8)
    ax.set_xlabel("Bars", color=TEXT_COLOR, fontsize=8)
    ax.set_xlim(0, n - 1)

    # Time-based cycle angle projections (change of state, regardless of direction)
    try:
        preds = compute_time_angle_predictions(trade_plan, timestamps)
    except Exception:
        preds = []

    if preds:
        y_text = 0.96
        for p in preds:
            label_time = p.timestamp.strftime("%Y-%m-%d %H:%M UTC")
            ax.text(
                0.02,
                y_text,
                f"{int(p.angle_deg)%360:>3d}° turn ≈ {label_time}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=7,
                color=TEXT_COLOR,
            )
            y_text -= 0.07


# ── Public interfaces ─────────────────────────────────────────────────────────

def draw(axes: list, trade_plan: dict) -> None:
    """
    Draws the cycle chart onto caller-supplied axes [ax0, ax1, ax2].
    Used by main_gui.py which manages its own Figure/GridSpec.

    Parameters
    ----------
    axes       : list of 3 matplotlib Axes
    trade_plan : dict — output of aggregator.run()
    """
    eng       = trade_plan.get("_engines", {})
    hilbert   = eng.get("hilbert",  {})
    ssa       = eng.get("ssa",      {})
    fft       = eng.get("fft",      {})
    detrend   = eng.get("detrend",  {})
    fetch     = eng.get("_fetch",   {})

    prices    = np.array(detrend.get("trend",      []) ) + np.array(detrend.get("detrended", []))
    detrended = np.array(detrend.get("detrended",  []))
    ssa_recon = np.array(ssa.get("reconstruction", []))
    phase_arr = np.degrees(np.array(hilbert.get("phase_rad", []))) % 360
    zero_x    = np.array(detrend.get("zero_crossings", []), dtype=int)
    timestamps = np.array(fetch.get("timestamps", np.arange(len(prices))))

    if len(prices) == 0 or len(detrended) == 0:
        for ax in axes:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                    ha="center", va="center", color=TEXT_COLOR)
        return

    _apply_style(axes[0].figure, axes)
    _draw_panel_0(axes[0], prices, ssa_recon, phase_arr, timestamps, trade_plan)
    _draw_panel_1(axes[1], detrended, fft, zero_x)
    _draw_panel_2(axes[2], phase_arr, timestamps, trade_plan)


def render(trade_plan: dict, figsize: tuple = (14, 9)) -> plt.Figure:
    """
    Creates and returns a standalone Figure for export or display.

    Parameters
    ----------
    trade_plan : dict — output of aggregator.run()
    figsize    : tuple — (width, height) in inches

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig = plt.figure(figsize=figsize, facecolor=BG_COLOR)
    fig.suptitle(
        f"Temporal Cycle Analysis — {trade_plan.get('symbol','?')}",
        color=ACCENT_COLOR, fontsize=11, fontweight="bold", y=0.98,
    )
    gs  = GridSpec(3, 1, figure=fig, hspace=0.35,
                   top=0.93, bottom=0.07, left=0.07, right=0.97)
    axes = [fig.add_subplot(gs[i]) for i in range(3)]
    draw(axes, trade_plan)
    return fig


# ── Smoke test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    np.random.seed(42)
    n = 200; t = np.arange(n)
    prices    = 50000 + 3000 * np.sin(2 * np.pi * t / 64) + np.cumsum(np.random.randn(n) * 100)
    detrended = 3000 * np.sin(2 * np.pi * t / 64) + np.random.randn(n) * 150
    phase_rad = 2 * np.pi * t / 64

    fake_plan = {
        "symbol": "BTCUSDT", "phase_deg": 210.0, "turn_type": "distribution",
        "turn_urgency": "high",
        "_engines": {
            "detrend":  {"trend": prices - detrended, "detrended": detrended,
                         "zero_crossings": np.where(np.diff(np.sign(detrended)))[0]},
            "ssa":      {"reconstruction": detrended * 0.8},
            "fft":      {"primary_cycle": {"period": 64.0, "amplitude": 3000.0}},
            "hilbert":  {"phase_rad": phase_rad},
            "_fetch":   {"timestamps": t},
        },
        "_risk": {},
    }

    fig = render(fake_plan)
    fig.savefig("/tmp/cycle_chart_test.png", dpi=120, bbox_inches="tight")
    print("✅ Cycle chart rendered → /tmp/cycle_chart_test.png")
    plt.close(fig)
