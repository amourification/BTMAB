# =============================================================================
#  gui/gui_charts.py — Matplotlib-in-Tkinter Chart Tab Helpers
#  Each ChartTab embeds a matplotlib Figure inside a tk.Frame using
#  FigureCanvasTkAgg. Provides show(trade_plan) and clear() methods.
#
#  Chart tabs:
#    CycleTab    — cycle_chart (3 panels)
#    MurrayTab   — murray_chart (2 panels)
#    PhaseTab    — phase_chart (3 panels)
#    RiskTab     — risk_chart (2×2 grid)
#    GannTab     — gann_chart (2 panels)
#
#  All tabs inherit from BaseChartTab which manages Figure lifecycle,
#  NavigationToolbar2Tk embedding, and resize debouncing.
# =============================================================================

import tkinter as tk
from tkinter import ttk
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import warnings
from gui.theme import get_current_theme

# Chart render functions
from charts.cycle_chart        import draw as draw_cycle
from charts.murray_chart       import draw as draw_murray
from charts.phase_chart        import draw as draw_phase
from charts.risk_chart         import draw as draw_risk
from charts.gann_chart         import draw as draw_gann
from charts.elliott_fib_chart  import draw as draw_elliott_fib
from charts.advanced_chart     import draw as draw_advanced

_THEME   = get_current_theme()
BG       = _THEME["BG"]
PANEL    = _THEME["PANEL"]
BORDER   = _THEME["BORDER"]
FG       = _THEME["FG"]
ACCENT   = _THEME["ACCENT"]

# Suppress tight_layout compatibility warnings from embedded figures
warnings.filterwarnings(
    "ignore",
    message="This figure includes Axes that are not compatible with tight_layout, so results might be incorrect.",
    category=UserWarning,
)


# ── Base chart tab ────────────────────────────────────────────────────────────

class BaseChartTab(tk.Frame):
    """
    Manages a Figure + FigureCanvasTkAgg inside a tk.Frame.
    Subclasses must implement:
      _make_axes(fig)   → list of Axes matching what their draw_fn expects
      _draw(axes, plan) → calls the appropriate chart draw() function

    refresh(trade_plan) is the only public method callers need.
    """

    def __init__(self, parent, nrows: int = 1, ncols: int = 1,
                 height_ratios: list = None, **kw):
        super().__init__(parent, bg=BG, **kw)
        self._nrows         = nrows
        self._ncols         = ncols
        self._height_ratios = height_ratios
        self._fig:  Figure  = None
        self._canvas        = None
        self._toolbar       = None
        self._axes          = []
        self._build()

    def _build(self):
        self._fig = Figure(facecolor=BG, tight_layout=False)
        self._canvas = FigureCanvasTkAgg(self._fig, master=self)
        widget = self._canvas.get_tk_widget()
        widget.config(bg=BG, highlightthickness=0)
        widget.pack(fill="both", expand=True)

        tb_frame = tk.Frame(self, bg=BG)
        tb_frame.pack(fill="x", side="bottom")
        self._toolbar = NavigationToolbar2Tk(self._canvas, tb_frame)
        self._toolbar.config(background=BG)
        for child in self._toolbar.winfo_children():
            try:
                child.config(background=BG, foreground=FG)
            except Exception:
                pass
        self._toolbar.update()

        self._axes = self._make_axes()

    def _make_axes(self) -> list:
        """Creates and returns axes list. Override in subclasses."""
        gs   = GridSpec(self._nrows, self._ncols, figure=self._fig,
                        height_ratios=self._height_ratios,
                        hspace=0.35, wspace=0.25,
                        top=0.93, bottom=0.08, left=0.07, right=0.97)
        axes = []
        for r in range(self._nrows):
            for c in range(self._ncols):
                ax = self._fig.add_subplot(gs[r, c])
                ax.set_facecolor(PANEL)
                ax.tick_params(colors=FG, labelsize=8)
                for spine in ax.spines.values():
                    spine.set_color(BORDER)
                axes.append(ax)
        return axes

    def _draw(self, axes: list, trade_plan: dict) -> None:
        """Override in subclass — calls the chart draw() function."""
        raise NotImplementedError

    def _clear_axes(self):
        for ax in self._axes:
            ax.cla()

    def refresh(self, trade_plan: dict) -> None:
        """
        Clears and redraws the chart with a new trade plan.
        Safe to call from the Tkinter main thread only.
        """
        try:
            self._clear_axes()
            self._draw(self._axes, trade_plan)
            self._fig.canvas.draw_idle()
        except Exception as exc:
            # Show error inline rather than crashing the GUI
            for ax in self._axes:
                ax.cla()
                ax.set_facecolor(PANEL)
                ax.text(0.5, 0.5, f"Chart error:\n{str(exc)[:120]}",
                        transform=ax.transAxes, ha="center", va="center",
                        color="#ef5350", fontsize=9, wrap=True)
            self._fig.canvas.draw_idle()

    def show_placeholder(self, message: str = "Run analysis to display chart"):
        self._clear_axes()
        for ax in self._axes:
            ax.set_facecolor(PANEL)
            ax.text(0.5, 0.5, message, transform=ax.transAxes,
                    ha="center", va="center", color=FG, fontsize=10)
        self._fig.canvas.draw_idle()


# ── Concrete chart tabs ───────────────────────────────────────────────────────

class CycleTab(BaseChartTab):
    """3-panel: Price+SSA · Detrended oscillator · Hilbert phase."""

    def __init__(self, parent, **kw):
        super().__init__(parent, nrows=3, ncols=1,
                         height_ratios=[3, 2, 1.5], **kw)

    def _draw(self, axes, trade_plan):
        draw_cycle(axes, trade_plan)


class MurrayTab(BaseChartTab):
    """2-panel: OHLC+Murray+Gann · Murray heatbar."""

    def __init__(self, parent, **kw):
        super().__init__(parent, nrows=2, ncols=1,
                         height_ratios=[6, 1], **kw)

    def _draw(self, axes, trade_plan):
        draw_murray(axes, trade_plan)


class PhaseTab(BaseChartTab):
    """3-panel: ACF correlogram · AR forecast · Solar sine."""

    def __init__(self, parent, **kw):
        super().__init__(parent, nrows=3, ncols=1,
                         height_ratios=[2, 2, 1.5], **kw)

    def _draw(self, axes, trade_plan):
        draw_phase(axes, trade_plan)


class RiskTab(BaseChartTab):
    """2×2 grid: Kelly gauge · Stop ladder · Gamma vol · Portfolio summary."""

    def __init__(self, parent, **kw):
        super().__init__(parent, nrows=2, ncols=2, **kw)

    def _make_axes(self) -> list:
        """Override to set equal height ratios for the 2×2 grid."""
        gs   = GridSpec(2, 2, figure=self._fig, hspace=0.40, wspace=0.30,
                        top=0.93, bottom=0.08, left=0.07, right=0.97)
        axes = []
        for r in range(2):
            for c in range(2):
                ax = self._fig.add_subplot(gs[r, c])
                ax.set_facecolor(PANEL)
                ax.tick_params(colors=FG, labelsize=8)
                for spine in ax.spines.values():
                    spine.set_color(BORDER)
                axes.append(ax)
        return axes

    def _draw(self, axes, trade_plan):
        draw_risk(axes, trade_plan)


class GannTab(BaseChartTab):
    """2-panel: Gann fan + breaks · Distance-from-angle bars."""

    def __init__(self, parent, **kw):
        super().__init__(parent, nrows=2, ncols=1,
                         height_ratios=[4, 1], **kw)

    def _draw(self, axes, trade_plan):
        draw_gann(axes, trade_plan)


class ElliottFibTab(BaseChartTab):
    """3-panel Elliott Wave / Fibonacci view (Price · RSI · MACD)."""

    def __init__(self, parent, **kw):
        super().__init__(parent, nrows=3, ncols=1,
                         height_ratios=[3, 1.5, 1.5], **kw)

    def _draw(self, axes, trade_plan):
        draw_elliott_fib(axes, trade_plan)


class AdvancedTab(BaseChartTab):
    """2-panel Advanced dashboard: multi-scale cycles + uncertainty/stress."""

    def __init__(self, parent, **kw):
        super().__init__(parent, nrows=2, ncols=1,
                         height_ratios=[3, 2], **kw)

    def _draw(self, axes, trade_plan):
        draw_advanced(axes, trade_plan)


# ── Tab registry ──────────────────────────────────────────────────────────────

TAB_CLASSES = [
    ("Cycle",       CycleTab),
    ("Murray",      MurrayTab),
    ("Phase",       PhaseTab),
    ("Risk",        RiskTab),
    ("Gann",        GannTab),
    ("Elliott/Fib", ElliottFibTab),
    ("Advanced",    AdvancedTab),
]
