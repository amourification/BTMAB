# =============================================================================
#  gui/gui_widgets.py — Reusable Tkinter Widgets
#  All custom widgets used by the main GUI.
#  Widgets are theme-aware: dark terminal palette matching chart colours.
#
#  Components:
#    ControlPanel   — symbol / interval / bars inputs + Run / Export buttons
#    StatusBar      — bottom bar: state, elapsed time, countdown, error
#    MetricLabel    — single-metric display tile (value + label + colour)
#    TradeplanPanel — scrollable summary of all key trade plan fields
# =============================================================================

import tkinter as tk
from tkinter import ttk
from tktooltip import ToolTip
from gui.gui_widgets_utils import (
    build_tradeplan_canvas, update_tradeplan_rows, clear_tradeplan_rows,
)
from gui.theme import get_current_theme

_THEME   = get_current_theme()
BG       = _THEME["BG"]
PANEL    = _THEME["PANEL"]
BORDER   = _THEME["BORDER"]
FG       = _THEME["FG"]
ACCENT   = _THEME["ACCENT"]
GREEN    = "#66bb6a"
RED      = "#ef5350"
ORANGE   = "#ffa726"
CYAN     = "#4fc3f7"
FONT_MON = ("Courier New", 9)
FONT_LBL = ("Courier New", 8)
FONT_BIG = ("Courier New", 11, "bold")
FONT_HDR = ("Courier New", 10, "bold")

INTERVALS = ["1m","5m","15m","30m","1h","2h","4h","6h",
             "8h","12h","1d","3d","1w","1M"]

# (Label, seconds)
REFRESH_PRESETS = [
    ("30s", 30),
    ("1m", 60),
    ("5m", 300),
    ("15m", 900),
]

class _DarkFrame(tk.Frame):
    def __init__(self, parent, **kw):
        super().__init__(parent, bg=BG, relief="flat", bd=0, **kw)


# ── ControlPanel ──────────────────────────────────────────────────────────────
class ControlPanel(_DarkFrame):
    """
    Top toolbar with symbol, interval, bars inputs and Run / Export / Refresh
    toggle buttons.

    Callbacks injected at construction time:
      on_run(symbol, interval, bars, mode)
      on_export()
      on_backtest()
      on_help()
      on_toggle_refresh(enabled: bool)
      on_change_interval(seconds: int)
      on_change_theme(theme_name: str)
    """

    def __init__(self, parent, on_run, on_export, on_backtest, on_help, on_toggle_refresh,
                 on_change_interval, on_change_theme, **kw):
        super().__init__(parent, **kw)
        self._on_run            = on_run
        self._on_export         = on_export
        self._on_backtest       = on_backtest
        self._on_help           = on_help
        self._on_toggle_refresh = on_toggle_refresh
        self._on_change_interval = on_change_interval
        self._on_change_theme   = on_change_theme
        self._auto_var          = tk.BooleanVar(value=True)
        self._mode_var          = tk.StringVar(value="Classic")
        self._build()

    def _build(self):
        pad = {"padx": 6, "pady": 4}

        # Symbol
        tk.Label(self, text="Symbol", bg=BG, fg=FG, font=FONT_LBL).pack(side="left", **pad)
        self._sym_var = tk.StringVar(value="BTCUSDT")
        sym_e = tk.Entry(self, textvariable=self._sym_var, bg=PANEL, fg=ACCENT,
                         insertbackground=ACCENT, font=FONT_MON, width=10,
                         relief="flat", bd=1, highlightthickness=1,
                         highlightbackground=BORDER, highlightcolor=ACCENT)
        sym_e.pack(side="left", **pad)
        ToolTip(sym_e, msg="Binance symbol, e.g. BTCUSDT or ETHUSDT")

        # Interval
        tk.Label(self, text="Interval", bg=BG, fg=FG, font=FONT_LBL).pack(side="left", **pad)
        self._int_var = tk.StringVar(value="1d")
        int_cb = ttk.Combobox(self, textvariable=self._int_var, values=INTERVALS,
                               width=5, font=FONT_MON, state="readonly")
        int_cb.pack(side="left", **pad)
        ToolTip(int_cb, msg="Timeframe for each candle, e.g. 1h, 4h, 1d")

        # Bars
        tk.Label(self, text="Bars", bg=BG, fg=FG, font=FONT_LBL).pack(side="left", **pad)
        self._bars_var = tk.StringVar(value="512")
        bars_e = tk.Entry(self, textvariable=self._bars_var, bg=PANEL, fg=CYAN,
                          insertbackground=CYAN, font=FONT_MON, width=6,
                          relief="flat", bd=1, highlightthickness=1,
                          highlightbackground=BORDER, highlightcolor=CYAN)
        bars_e.pack(side="left", **pad)
        ToolTip(bars_e, msg="How many past candles to analyse (more = slower, but more context)")

        # Separator
        tk.Label(self, text="│", bg=BG, fg=BORDER, font=FONT_MON).pack(side="left")

        # Run button
        self._run_btn = tk.Button(
            self, text="▶ RUN", command=self._fire_run,
            bg="#2e7d32", fg="white", activebackground="#388e3c",
            activeforeground="white", font=FONT_HDR, relief="flat",
            padx=10, pady=2, cursor="hand2",
        )
        self._run_btn.pack(side="left", **pad)
        ToolTip(self._run_btn, msg="Fetch latest data and run the full analysis once")

        # Export button
        tk.Button(
            self, text="⬇ EXPORT", command=self._on_export,
            bg="#283593", fg="white", activebackground="#303f9f",
            activeforeground="white", font=FONT_MON, relief="flat",
            padx=8, pady=2, cursor="hand2",
        ).pack(side="left", **pad)

        # Backtest button
        tk.Button(
            self, text="🧪 BACKTEST", command=self._on_backtest,
            bg="#4e342e", fg="white", activebackground="#5d4037",
            activeforeground="white", font=FONT_MON, relief="flat",
            padx=8, pady=2, cursor="hand2",
        ).pack(side="left", **pad)

        # Auto-refresh toggle
        tk.Label(self, text="│", bg=BG, fg=BORDER, font=FONT_MON).pack(side="left")
        self._refresh_btn = tk.Checkbutton(
            self, text="Auto-refresh", variable=self._auto_var,
            command=self._toggle_refresh,
            bg=BG, fg=FG, activebackground=BG, activeforeground=ACCENT,
            selectcolor=PANEL, font=FONT_LBL,
        )
        self._refresh_btn.pack(side="left", **pad)
        ToolTip(self._refresh_btn, msg="Automatically re-run analysis every chosen interval")

        # Refresh interval selector
        tk.Label(self, text="Every", bg=BG, fg=FG, font=FONT_LBL).pack(side="left", **pad)
        self._refresh_label_var = tk.StringVar(value="5m")
        refresh_values = [label for (label, _secs) in REFRESH_PRESETS]
        self._refresh_cb = ttk.Combobox(
            self,
            textvariable=self._refresh_label_var,
            values=refresh_values,
            width=4,
            font=FONT_MON,
            state="readonly",
        )
        self._refresh_cb.bind("<<ComboboxSelected>>", self._on_refresh_interval_change)
        self._refresh_cb.pack(side="left", **pad)
        ToolTip(self._refresh_cb, msg="How often to refresh with new data when Auto-refresh is ON")

        # Spacer
        tk.Label(self, text="│", bg=BG, fg=BORDER, font=FONT_MON).pack(side="left")

        # Mode selector
        tk.Label(self, text="Mode", bg=BG, fg=FG, font=FONT_LBL).pack(side="left", **pad)
        mode_cb = ttk.Combobox(
            self,
            textvariable=self._mode_var,
            values=["Classic", "Advanced"],
            width=9,
            font=FONT_MON,
            state="readonly",
        )
        mode_cb.pack(side="left", **pad)
        ToolTip(mode_cb, msg="Select analysis mode: Classic 12-engine or Advanced (regime-aware)")

        # Spacer
        tk.Label(self, text="│", bg=BG, fg=BORDER, font=FONT_MON).pack(side="left")

        # Theme selector
        tk.Label(self, text="Theme", bg=BG, fg=FG, font=FONT_LBL).pack(side="left", **pad)
        self._theme_var = tk.StringVar(
            value="Dark" if _THEME["BG"] == "#2c001e" else "Light"
        )
        theme_cb = ttk.Combobox(
            self,
            textvariable=self._theme_var,
            values=["Dark", "Light"],
            width=6,
            font=FONT_MON,
            state="readonly",
        )
        theme_cb.bind("<<ComboboxSelected>>", self._on_theme_change)
        theme_cb.pack(side="left", **pad)
        ToolTip(theme_cb, msg="Switch between Dark and Light Ubuntu-style themes (requires restart)")

        # Spacer
        tk.Label(self, text="│", bg=BG, fg=BORDER, font=FONT_MON).pack(side="left")

        # Help button
        help_btn = tk.Button(
            self, text="Help", command=self._on_help,
            bg=PANEL, fg=FG, activebackground="#455a64",
            activeforeground="white", font=FONT_MON, relief="flat",
            padx=8, pady=2, cursor="hand2",
        )
        help_btn.pack(side="left", **pad)
        ToolTip(help_btn, msg="Open the README help in a separate window")

    def _fire_run(self):
        try:
            bars = int(self._bars_var.get())
        except ValueError:
            bars = 512
        mode = self._mode_var.get().strip().lower()
        if mode not in ("classic", "advanced"):
            mode = "classic"
        self._on_run(
            self._sym_var.get().strip().upper(),
            self._int_var.get().strip(),
            bars,
            mode,
        )

    def _toggle_refresh(self):
        self._on_toggle_refresh(self._auto_var.get())

    def _on_refresh_interval_change(self, _event=None):
        label = self._refresh_label_var.get()
        for human, secs in REFRESH_PRESETS:
            if human == label:
                self._on_change_interval(secs)
                break

    def _on_theme_change(self, _event=None):
        label = self._theme_var.get().strip().lower()
        if label in ("dark", "light"):
            self._on_change_theme(label)

    def set_running(self, running: bool):
        """Called from GUI thread to disable / re-enable run button."""
        if running:
            self._run_btn.config(text="⏳ RUNNING…", state="disabled",
                                 bg="#263238")
        else:
            self._run_btn.config(text="▶ RUN", state="normal",
                                 bg="#2e7d32")

    def get_inputs(self) -> tuple:
        try:
            bars = int(self._bars_var.get())
        except ValueError:
            bars = 512
        mode = self._mode_var.get().strip().lower()
        if mode not in ("classic", "advanced"):
            mode = "classic"
        return self._sym_var.get().strip().upper(), self._int_var.get().strip(), bars, mode


# ── StatusBar ─────────────────────────────────────────────────────────────────
class StatusBar(_DarkFrame):
    """Bottom status bar showing run state, elapsed, countdown, and errors."""

    def __init__(self, parent, **kw):
        super().__init__(parent, **kw)
        self.config(bd=1, relief="flat",
                    highlightthickness=1, highlightbackground=BORDER)
        self._status_var    = tk.StringVar(value="Ready")
        self._elapsed_var   = tk.StringVar(value="")
        self._countdown_var = tk.StringVar(value="")
        self._error_var     = tk.StringVar(value="")
        self._build()

    def _build(self):
        tk.Label(self, textvariable=self._status_var, bg=BG, fg=GREEN,
                 font=FONT_MON, width=20, anchor="w").pack(side="left", padx=6)
        tk.Label(self, textvariable=self._elapsed_var, bg=BG, fg=FG,
                 font=FONT_LBL).pack(side="left", padx=4)
        tk.Label(self, textvariable=self._countdown_var, bg=BG, fg=CYAN,
                 font=FONT_LBL).pack(side="left", padx=4)
        tk.Label(self, textvariable=self._error_var, bg=BG, fg=RED,
                 font=FONT_LBL, wraplength=600, anchor="w").pack(
                     side="left", padx=8, fill="x", expand=True)

    def set_running(self, symbol: str):
        self._status_var.set(f"⏳  Analysing {symbol}…")
        self._error_var.set("")

    def set_done(self, elapsed: float, symbol: str):
        self._status_var.set(f"✓  {symbol} — complete")
        self._elapsed_var.set(f"{elapsed:.1f}s")

    def set_error(self, msg: str):
        self._status_var.set("✗  Error")
        self._error_var.set(msg[:120])

    def set_countdown(self, secs: int):
        self._countdown_var.set(f"↺ {secs}s" if secs > 0 else "")

    def set_idle(self):
        self._status_var.set("Ready")
        self._elapsed_var.set("")


# ── MetricLabel ───────────────────────────────────────────────────────────────
class MetricLabel(_DarkFrame):
    """
    Single metric tile: large coloured value + small grey label below.
    Used in the quick-metrics strip at the top of the TradeplanPanel.
    """

    def __init__(self, parent, label: str, color: str = ACCENT, **kw):
        super().__init__(parent, **kw)
        self.config(bg=PANEL, bd=1, relief="flat",
                    highlightthickness=1, highlightbackground=BORDER)
        self._val_var = tk.StringVar(value="—")
        tk.Label(self, textvariable=self._val_var, bg=PANEL, fg=color,
                 font=FONT_BIG, width=12, anchor="center").pack(pady=(6, 0))
        tk.Label(self, text=label, bg=PANEL, fg=FG,
                 font=FONT_LBL, anchor="center").pack(pady=(0, 5))

    def set(self, value: str):
        self._val_var.set(str(value))


# ── TradeplanPanel ────────────────────────────────────────────────────────────
class TradeplanPanel(_DarkFrame):
    """
    Scrollable panel showing the full flat trade plan as a formatted table.
    Groups metrics into sections: Cycle · Bias · Trade Plan · Stops · Portfolio.
    """

    _SECTIONS = [
        ("TEMPORAL CYCLE", [
            ("Phase",          "phase_deg",           "°",  ACCENT),
            ("Turn Type",      "turn_type",            "",  FG),
            ("Turn Urgency",   "turn_urgency",         "",  ORANGE),
            ("Bars to Turn",   "bars_to_next_turn",    "b", CYAN),
            ("ETA To Turn",    "next_turn_eta_utc",    "",  CYAN),
            ("Dominant Cycle", "dominant_cycle_bars",  "b", FG),
            ("SSA Period",     "ssa_period",           "b", FG),
            ("SSA Position",   "ssa_position",         "",  FG),
            ("Cycle %",        "cycle_pct_complete",   "%", FG),
            ("Mode",           "mode",                 "",  FG),
        ]),
        ("MARKET CONTEXT", [
            ("Bias",           "market_bias",          "",   GREEN),
            ("Bias Strength",  "bias_strength",        "",   FG),
            ("Murray Level",   "murray_index",         "/8", ACCENT),
            ("Murray Action",  "murray_action",        "",   FG),
            ("Gamma Regime",   "gamma_regime",         "",   ORANGE),
            ("Vol Regime",     "vol_regime",           "",   ORANGE),
            ("Seasonal Bias",  "seasonal_bias",        "",   FG),
            ("Vol Flag",       "volatility_flag",      "",   ORANGE),
            ("Walras Adj",     "walras_adjustment",    "×",  CYAN),
        ]),
        ("TRADE PLAN", [
            ("Kelly Position", "kelly_position_pct",   "%",  GREEN),
            ("Kelly Tier",     "kelly_tier",           "",   FG),
            ("Kelly EV",       "kelly_ev",             "",   CYAN),
            ("Hedge Ratio",    "hedge_pct",            "%",  ORANGE),
            ("Hedge Urgency",  "hedge_urgency",        "",   ORANGE),
            ("Sweep Detected", "sweep_detected",       "",   RED),
            ("Uncertainty",    "advanced.uncertainty_score", "", ORANGE),
            ("Suggested Entry", "suggested_entry_price","",  ACCENT),
            ("Suggested Exit",  "suggested_exit_price", "",  GREEN),
            ("TP1 (Conservative)", "take_profit_1",     "",  GREEN),
            ("TP2 (Standard)",     "take_profit_2",     "",  GREEN),
            ("TP3 (Aggressive)",   "take_profit_3",     "",  GREEN),
        ]),
        ("STOPS & RISK", [
            ("Suggested Stop Loss", "suggested_stop_price", "",   RED),
            ("R/R Grade",      "risk_reward_grade",    "",   ACCENT),
            ("Risk Score",     "overall_risk_score",   "",   ORANGE),
            ("Confidence",     "consensus_confidence", "%",  GREEN),
            ("Analysis Time",  "analysis_time_sec",    "s",  FG),
        ]),
    ]

    def __init__(self, parent, **kw):
        super().__init__(parent, **kw)
        self._rows: dict = {}
        self._build()

    def _build(self):
        self._rows = build_tradeplan_canvas(self, self._SECTIONS)

    def update(self, trade_plan: dict):
        """Populates all rows from a trade_plan dict."""
        update_tradeplan_rows(self._rows, trade_plan)

    def clear(self):
        clear_tradeplan_rows(self._rows)
