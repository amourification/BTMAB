# =============================================================================
#  main_gui.py — Temporal Market Analysis Bot — Main GUI Entry Point
#  Tkinter application window that embeds all 5 chart tabs + trade plan
#  panel + control bar. Background thread runs the full analysis pipeline.
#
#  Layout:
#    ┌─────────────────────────────────────────────────────┐
#    │  ControlPanel (toolbar)                             │
#    ├──────────────────────┬──────────────────────────────┤
#    │  Notebook (5 tabs):  │  TradeplanPanel              │
#    │  Cycle / Murray /    │  (scrollable metrics list)   │
#    │  Phase / Risk / Gann │                              │
#    ├──────────────────────┴──────────────────────────────┤
#    │  StatusBar                                          │
#    └─────────────────────────────────────────────────────┘
#
#  Threading model:
#    - Main thread: Tkinter event loop + .after() poll at POLL_MS
#    - Background daemon thread: runs aggregator.run() — never touches Tk
#    - StateManager (gui_state.py): thread-safe bridge via queue.Queue
# =============================================================================

import sys
import threading
from pathlib import Path

try:
    import tkinter as tk
    from tkinter import ttk, messagebox, filedialog
except ModuleNotFoundError:
    sys.stderr.write(
        "\n[ERROR] Tkinter GUI toolkit is not available in this Python installation.\n\n"
        "On Debian/Ubuntu:\n"
        "  sudo apt update\n"
        "  sudo apt install python3-tk\n\n"
        "On Fedora:\n"
        "  sudo dnf install python3-tkinter\n\n"
        "On Arch Linux:\n"
        "  sudo pacman -S tk\n\n"
        "After installing the Tk packages, re-run:\n"
        "  python main_gui.py\n\n"
    )
    sys.exit(1)

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from gui.gui_state        import get_manager, AnalysisResult, POLL_MS
from gui.gui_widgets      import ControlPanel, StatusBar, TradeplanPanel
from gui.gui_charts       import TAB_CLASSES
from gui.onboarding       import ensure_credentials
from gui.trade_suggestions import compute_trade_suggestions
from gui.theme            import persist_theme_choice
from tktooltip import ToolTip

from charts.exporter import export_all, export_summary_png
from logs_utils import write_analysis_log

import logging
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s")
logger = logging.getLogger("temporal_bot.main_gui")

# ── Palette (keep in sync with gui_widgets/theme) ────────────────────────────
from gui.theme import get_current_theme
_THEME = get_current_theme()
BG      = _THEME["BG"]
PANEL   = _THEME["PANEL"]
BORDER  = _THEME["BORDER"]
FG      = _THEME["FG"]
ACCENT  = _THEME["ACCENT"]
FONT_M  = ("Courier New", 9)
FONT_H  = ("Courier New", 10, "bold")

WIN_W   = 1420
WIN_H   = 860
PLAN_W  = 310    # width of right-side trade plan panel (px)


# ── Background analysis runner ────────────────────────────────────────────────

def _run_analysis(symbol: str, interval: str, bars: int,
                  cfg: dict, manager, mode: str = "classic") -> None:
    """
    Executed in a daemon background thread.
    Imports aggregator lazily to avoid import-time side effects on startup.
    Pushes AnalysisResult into StateManager when done.
    """
    import time
    t0 = time.perf_counter()
    try:
        mode = (mode or "classic").lower()
        if mode == "advanced":
            from consensus.advanced_aggregator import run as agg_run  # type: ignore[no-redef]
        else:
            from consensus.aggregator import run as agg_run  # type: ignore[no-redef]

        plan    = agg_run(symbol, interval, bars, cfg)
        elapsed = time.perf_counter() - t0
        if plan.get("success"):
            manager.push(AnalysisResult(success=True,
                                        trade_plan=plan,
                                        elapsed=elapsed))
        else:
            manager.push(AnalysisResult(success=False,
                                        error=plan.get("error", "Unknown error"),
                                        elapsed=elapsed))
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        logger.exception("Analysis thread failed")
        manager.push(AnalysisResult(success=False,
                                    error=f"{type(exc).__name__}: {exc}",
                                    elapsed=elapsed))


# ── Main application class ────────────────────────────────────────────────────

class App(tk.Tk):
    """
    Main Tkinter application window.
    Manages layout, tab switching, background thread lifecycle,
    and the .after() polling loop.
    """

    def __init__(self, cfg: dict = None):
        super().__init__()
        self._cfg     = cfg or {}
        self._manager = get_manager()
        self._tabs:   list = []   # list of BaseChartTab instances
        self._thread: threading.Thread | None = None
        self._countdown_job  = None
        self._style_ttk()
        self._build_window()
        self._build_layout()

        # First-time API key onboarding (modal dialog)
        project_root = Path(__file__).resolve().parent
        if not ensure_credentials(project_root, self):
            # User cancelled onboarding — close the app cleanly
            self.destroy()
            return

        # Rebuild config dict after onboarding so API keys from .env are included.
        try:
            from config import build_config
            self._cfg = build_config()
        except Exception:
            pass

        self._start_poll()
        self._start_countdown()
        logger.info("GUI started")

    # ── Window setup ──────────────────────────────────────────────────────

    def _style_ttk(self):
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("TNotebook",          background=BG,    borderwidth=0)
        style.configure("TNotebook.Tab",      background=PANEL, foreground=FG,
                         font=FONT_M, padding=[10, 4])
        style.map("TNotebook.Tab",
                  background=[("selected", BG)],
                  foreground=[("selected", ACCENT)])
        style.configure(
            "TCombobox",
            fieldbackground=PANEL,
            background=PANEL,
            foreground=FG,
            bordercolor=BORDER,
            arrowcolor=FG,
        )
        # Ensure selected text remains legible in the entry area
        style.map(
            "TCombobox",
            foreground=[("readonly", FG), ("!disabled", FG)],
            fieldbackground=[("readonly", PANEL), ("!disabled", PANEL)],
            background=[("readonly", PANEL), ("!disabled", PANEL)],
        )
        style.configure(
            "TButton",
            background=PANEL,
            foreground=FG,
            borderwidth=0,
            focusthickness=0,
        )

        # Default Tk widget colors for better legibility
        self.option_add("*Background", BG)
        self.option_add("*Foreground", FG)
        self.option_add("*Entry.Background", PANEL)
        self.option_add("*Entry.Foreground", FG)
        self.option_add("*Label.Background", BG)
        self.option_add("*Label.Foreground", FG)
        self.option_add("*Button.Background", PANEL)
        self.option_add("*Button.Foreground", FG)
        # Combobox dropdown list colors (platform-dependent, best-effort)
        self.option_add("*TCombobox*Listbox*Background", PANEL)
        self.option_add("*TCombobox*Listbox*Foreground", FG)

    def _build_window(self):
        self.title("Temporal Market Analysis Bot")
        self.geometry(f"{WIN_W}x{WIN_H}")
        self.minsize(900, 600)
        self.configure(bg=BG)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_layout(self):
        # ── Top toolbar ──────────────────────────────────────────────────
        self._ctrl = ControlPanel(
            self,
            on_run=self._trigger_run,
            on_export=self._trigger_export,
            on_backtest=self._trigger_backtest,
            on_help=self._show_help,
            on_toggle_refresh=self._toggle_refresh,
            on_change_interval=self._change_refresh_interval,
            on_change_theme=self._change_theme,
        )
        self._ctrl.pack(fill="x", side="top",
                         padx=4, pady=4)

        # ── Thin divider ─────────────────────────────────────────────────
        tk.Frame(self, bg=BORDER, height=1).pack(fill="x")

        # ── Main content area (chart notebook + trade plan panel) ─────────
        content = tk.Frame(self, bg=BG)
        content.pack(fill="both", expand=True, padx=0, pady=0)

        # Right panel: TradeplanPanel
        self._plan_panel = TradeplanPanel(content, width=PLAN_W)
        self._plan_panel.pack(side="right", fill="y", padx=(2, 4), pady=4)

        tk.Frame(content, bg=BORDER, width=1).pack(side="right", fill="y")

        # Left panel: Notebook tabs
        self._notebook = ttk.Notebook(content)
        self._notebook.pack(side="left", fill="both", expand=True,
                             padx=4, pady=4)

        self._tabs = []
        for tab_name, TabClass in TAB_CLASSES:
            tab = TabClass(self._notebook)
            self._tabs.append(tab)
            self._notebook.add(tab, text=f"  {tab_name}  ")
            # Tab-level tooltip for noobs
            help_msg = {
                "Cycle": "Cycle: price + SSA cycle + Hilbert phase. Shows where we are in the dominant cycle.",
                "Murray": "Murray: support/resistance levels and Gann fan. Shows key price geometry.",
                "Phase": "Phase: ACF, AR forecast, and solar/seasonal bias. Helps understand timing.",
                "Risk": "Risk: Kelly sizing, stops, gamma/vol and portfolio view.",
                "Gann": "Gann: Gann fan angles and distance from each angle.",
                "Elliott/Fib": "Elliott/Fib: Elliott wave helper with Fibonacci retracements/extensions.",
                "Advanced": "Advanced: multi-scale cycles plus uncertainty and stress regime snapshot.",
            }.get(tab_name, tab_name)
            ToolTip(tab, msg=help_msg)
            tab.show_placeholder()

        # Track which tabs need refresh after new analysis results
        self._dirty_tabs = set(range(len(self._tabs)))
        self._notebook.bind("<<NotebookTabChanged>>", self._on_tab_changed)

        # ── Bottom status bar ─────────────────────────────────────────────
        tk.Frame(self, bg=BORDER, height=1).pack(fill="x")
        self._status = StatusBar(self)
        self._status.pack(fill="x", side="bottom", padx=4, pady=2)

    # ── Analysis trigger ──────────────────────────────────────────────────

    def _trigger_run(self, symbol: str, interval: str, bars: int, mode: str = "classic"):
        """Called when user presses Run or auto-refresh fires."""
        if self._manager.state.is_running:
            return   # already running — ignore
        self._manager.update_inputs(symbol, interval, bars, mode=mode)
        self._manager.set_running(True)
        self._ctrl.set_running(True)
        self._status.set_running(symbol)

        self._thread = threading.Thread(
            target=_run_analysis,
            args=(symbol, interval, bars, self._cfg, self._manager, mode),
            daemon=True,
        )
        self._thread.start()
        logger.info("Analysis started: %s %s %d bars", symbol, interval, bars)

    # ── Poll loop ──────────────────────────────────────────────────────────

    def _start_poll(self):
        self._poll()

    def _poll(self):
        """
        Called every POLL_MS ms by Tkinter's .after() scheduler.
        Checks for new AnalysisResult from the background thread.
        """
        result = self._manager.poll()
        if result is not None:
            self._manager.apply_result(result)
            self._ctrl.set_running(False)
            if result.success:
                self._refresh_ui(result.trade_plan, result.elapsed)
            else:
                self._status.set_error(result.error)

        self.after(POLL_MS, self._poll)

    def _refresh_ui(self, trade_plan: dict, elapsed: float):
        """Redraws all visible charts and updates the trade plan panel."""
        # Enrich plan with simple, human-friendly trade suggestions
        try:
            suggestions = compute_trade_suggestions(trade_plan) or {}
            trade_plan.update(suggestions)
        except Exception as exc:
            logger.warning("Trade suggestion computation failed: %s", exc)

        sym = trade_plan.get("symbol", "?")
        self._status.set_done(elapsed, sym)
        self._plan_panel.update(trade_plan)

        # Persist full analysis snapshot to per-symbol/per-interval log file
        try:
            log_path = write_analysis_log(trade_plan)
            if log_path is not None:
                logger.info("Analysis log written: %s", log_path)
        except Exception:
            logger.warning("Failed to write analysis log", exc_info=True)

        # Only redraw the active tab immediately; mark others dirty
        active_idx = self._notebook.index(self._notebook.select())
        for i in range(len(self._tabs)):
            if i == active_idx:
                try:
                    self._tabs[i].refresh(trade_plan)
                    if hasattr(self, "_dirty_tabs"):
                        self._dirty_tabs.discard(i)
                except Exception as exc:
                    logger.warning("Tab %d refresh error: %s", i, exc)
            else:
                if hasattr(self, "_dirty_tabs"):
                    self._dirty_tabs.add(i)

        logger.info("UI refreshed (active tab only): %s in %.2fs", sym, elapsed)

    def _on_tab_changed(self, _event=None):
        """
        Lazy-refresh tabs when the user switches to them, to avoid
        unnecessary redraw work while they are hidden.
        """
        if not hasattr(self, "_dirty_tabs"):
            return
        idx = self._notebook.index(self._notebook.select())
        if idx in self._dirty_tabs:
            state = self._manager.snapshot()
            if state.trade_plan is not None:
                try:
                    self._tabs[idx].refresh(state.trade_plan)
                except Exception as exc:
                    logger.warning("Tab %d refresh error (on switch): %s", idx, exc)
            self._dirty_tabs.discard(idx)

    # ── Countdown loop ────────────────────────────────────────────────────

    def _start_countdown(self):
        self._tick_countdown()

    def _tick_countdown(self):
        state = self._manager.snapshot()
        if state.auto_refresh and not state.is_running:
            remaining = self._manager.tick_countdown()
            self._status.set_countdown(remaining)
            if remaining == 0:
                sym, interval, bars, mode = self._ctrl.get_inputs()
                self._trigger_run(sym, interval, bars, mode)
        self._countdown_job = self.after(1000, self._tick_countdown)

    def _toggle_refresh(self, enabled: bool):
        self._manager.state.auto_refresh = enabled
        if enabled:
            self._manager.reset_countdown()

    def _change_refresh_interval(self, seconds: int):
        """
        Called when the user selects a new auto-refresh interval from the toolbar.
        Updates shared state so countdown and next API fetch use the new value.
        """
        self._manager.set_refresh_interval(seconds)

    def _change_theme(self, theme_name: str):
        """
        Persist the requested theme and prompt the user to restart
        so that all palette-dependent modules pick it up cleanly.
        """
        persist_theme_choice(theme_name)
        messagebox.showinfo(
            "Theme changed",
            "Theme preference saved. Please restart the application to apply the new theme.",
            parent=self,
        )

    # ── Export ────────────────────────────────────────────────────────────

    def _trigger_export(self):
        state = self._manager.snapshot()
        if state.trade_plan is None:
            messagebox.showinfo("Export", "Run analysis first before exporting.")
            return

        out_dir = filedialog.askdirectory(title="Select export folder",
                                           initialdir=str(Path.home()))
        if not out_dir:
            return

        try:
            result = export_all(state.trade_plan, Path(out_dir), self._cfg)
            msg    = f"PDF: {Path(result['pdf']).name}\nCSV: {Path(result['csv']).name}"
            messagebox.showinfo("Export complete", msg)
            logger.info("Exported to %s", out_dir)
        except Exception as exc:
            messagebox.showerror("Export failed", str(exc))

    def _trigger_backtest(self):
        """
        Run a full BTC/XRP 2024–2025 backtest in a background thread and
        show a live-updating results window. Uses the offline consensus
        pipeline to compare directional calls vs realised price moves.
        """
        from threading import Thread
        from backtest.full_protocol import run_full_backtest, SYMBOLS as _BT_SYMBOLS

        # Build a results window up front so the user immediately sees
        # that work is in progress and which checks are running.
        win = tk.Toplevel(self)
        win.title("Backtest — BTCUSDT & XRPUSDT (2024–2025)")
        win.geometry("780x460")
        win.configure(bg=BG)

        status_var = tk.StringVar(value="Waiting to start…")
        current_symbol_var = tk.StringVar(value="Current symbol: —")

        # Status line (acts like a mini terminal status)
        lbl = tk.Label(win, textvariable=status_var, bg=BG, fg=FG, anchor="w")
        lbl.pack(fill="x", padx=10, pady=(10, 2))

        # Current "terminal"/task line
        current_lbl = tk.Label(win, textvariable=current_symbol_var, bg=BG, fg=FG, anchor="w")
        current_lbl.pack(fill="x", padx=10, pady=(0, 4))

        # Progress bar (2 symbol-level reports)
        progress = ttk.Progressbar(win, mode="determinate", maximum=len(_BT_SYMBOLS))
        progress.pack(fill="x", padx=10, pady=(0, 4))

        # Within-symbol progress (walk-forward steps)
        symbol_progress = ttk.Progressbar(win, mode="determinate", maximum=100)
        symbol_progress.pack(fill="x", padx=10, pady=(0, 8))

        # Results table
        columns = ("symbol", "bars", "decisions", "trades", "hit_rate", "status", "details")
        tree = ttk.Treeview(
            win, columns=columns, show="headings", height=5, selectmode="none"
        )
        for col, w in (
            ("symbol", 90),
            ("bars", 80),
            ("decisions", 80),
            ("trades", 70),
            ("hit_rate", 80),
            ("status", 80),
            ("details", 260),
        ):
            tree.heading(col, text=col.capitalize())
            tree.column(col, width=w, anchor="center" if col != "details" else "w")

        # Tag colours for pass/fail
        tree.tag_configure("ok", foreground="green")
        tree.tag_configure("fail", foreground="red")

        # Pre-populate rows so user can see what will run.
        for sym in _BT_SYMBOLS:
            tree.insert(
                "",
                "end",
                iid=f"{sym}",
                values=(sym, "-", "-", "-", "-", "Pending", "Waiting to run…"),
            )

        tree.pack(fill="x", padx=10, pady=(0, 10))

        # Log area for more verbose messages
        txt = tk.Text(
            win,
            wrap="word",
            bg=BG,
            fg=FG,
            insertbackground=FG,
            font=("Courier New", 9),
            state="disabled",
            height=8,
        )
        sb = tk.Scrollbar(win, orient="vertical", command=txt.yview)
        txt.configure(yscrollcommand=sb.set)
        txt.pack(side="left", fill="both", expand=True, padx=(10, 0), pady=(0, 10))
        sb.pack(side="right", fill="y", padx=(0, 10), pady=(0, 10))

        def _append_line(line: str) -> None:
            def _inner():
                txt.configure(state="normal")
                txt.insert("end", line + "\n")
                txt.see("end")
                txt.configure(state="disabled")

            self.after(0, _inner)

        def _set_status(message: str) -> None:
            self.after(0, lambda: status_var.set(message))

        def _set_current_symbol(sym: str) -> None:
            self.after(0, lambda: current_symbol_var.set(f"Current symbol: {sym}"))

        def _set_symbol_progress(sym: str, step: int, total: int) -> None:
            if total <= 0:
                return

            frac = max(0.0, min(1.0, float(step) / float(total)))

            def _inner():
                current_symbol_var.set(f"Current symbol: {sym} ({int(frac * 100)}%)")
                symbol_progress["value"] = int(frac * 100)

            self.after(0, _inner)

        completed = 0

        def _update_rows_from_result(rep) -> None:
            nonlocal completed

            def _inner():
                nonlocal completed

                item = f"{rep.symbol}"
                if not tree.exists(item):
                    return

                status = "SUCCESS" if not rep.error and rep.n_trades > 0 else "FAIL"
                detail = (
                    f"{rep.n_trades} trades, hit-rate={rep.hit_rate:.1%}, "
                    f"avg trade return={rep.avg_trade_return:.4f}"
                    if not rep.error and rep.n_trades > 0
                    else (rep.error or "No trades generated; check config / data.")
                )

                tree.set(item, "bars", str(rep.bars))
                tree.set(item, "decisions", str(rep.n_decisions))
                tree.set(item, "trades", str(rep.n_trades))
                tree.set(item, "hit_rate", f"{rep.hit_rate:.1%}")
                tree.set(item, "status", status)
                tree.set(item, "details", detail)
                tree.item(item, tags=("ok" if status == "SUCCESS" else "fail",))

                completed += 1
                progress["value"] = completed

            self.after(0, _inner)

        def _run():
            try:
                _set_status("Running backtest on BTCUSDT and XRPUSDT …")
                _append_line(
                    "Starting full backtest. Success = the engine produced trades and "
                    "their directional calls were profitable more often than not over "
                    "the chosen horizon. Failure = no trades or an error during fetch/"
                    "analysis (see details below)."
                )

                def _progress(rep) -> None:
                    # Update table, current 'terminal' line and log as each symbol completes.
                    _set_current_symbol(rep.symbol)
                    _update_rows_from_result(rep)
                    if rep.error:
                        line = f"{rep.symbol}: ERROR — {rep.error}"
                    else:
                        line = (
                            f"{rep.symbol}: OK — {rep.bars} bars "
                            f"({rep.start.date()} → {rep.end.date()}), "
                            f"trades={rep.n_trades}, hit-rate={rep.hit_rate:.1%}"
                        )
                    _append_line(line)

                reports = run_full_backtest(progress_cb=_progress, step_cb=_set_symbol_progress)
                if not reports:
                    _append_line("No results returned.")

                _set_status("Backtest complete. Review the table above for pass/fail.")
                _append_line(
                    "Backtest complete. Green rows = symbol produced trades with a "
                    "positive hit-rate over the test period. Red rows = no trades or "
                    "an error; read the details column / log for context."
                )
            except Exception as exc:
                _append_line(f"Backtest failed: {exc!r}")
                _set_status("Backtest failed.")

        Thread(target=_run, daemon=True).start()

    def _show_help(self):
        """
        Open README.md in a separate scrollable window.
        """
        import textwrap

        readme_path = Path(__file__).resolve().parent / "README.md"
        if not readme_path.exists():
            messagebox.showinfo("Help", "README.md not found.", parent=self)
            return

        try:
            content = readme_path.read_text(encoding="utf-8")
        except Exception as exc:
            messagebox.showerror("Help", f"Could not read README.md: {exc}", parent=self)
            return

        win = tk.Toplevel(self)
        win.title("Temporal Bot — Help (README)")
        win.geometry("900x700")
        win.configure(bg=BG)

        txt = tk.Text(win, wrap="word", bg=BG, fg=FG, insertbackground=FG,
                      font=("Courier New", 9))
        sb = tk.Scrollbar(win, orient="vertical", command=txt.yview)
        txt.configure(yscrollcommand=sb.set)
        txt.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")

        # Slightly indent headings for readability
        txt.insert("1.0", content)
        txt.config(state="disabled")

    # ── Close ─────────────────────────────────────────────────────────────

    def _on_close(self):
        if self._countdown_job:
            self.after_cancel(self._countdown_job)
        self.destroy()


# ── Entry point ───────────────────────────────────────────────────────────────

def main(cfg: dict = None):
    """Launches the GUI. Pass a cfg dict from config.py if available."""
    if cfg is None:
        try:
            from config import build_config
            cfg = build_config()
        except Exception:
            cfg = {}
    app = App(cfg=cfg)
    app.mainloop()


if __name__ == "__main__":
    main()
