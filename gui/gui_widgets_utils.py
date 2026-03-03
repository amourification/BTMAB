# =============================================================================
#  gui/gui_widgets_utils.py — TradeplanPanel Builder Helpers
#  Split from gui_widgets.py to stay under 300 lines.
# =============================================================================

import tkinter as tk
from tktooltip import ToolTip
from gui.theme import get_current_theme

_THEME = get_current_theme()
BG      = _THEME["BG"]
PANEL   = _THEME["PANEL"]
BORDER  = _THEME["BORDER"]
FG      = _THEME["FG"]
ACCENT  = _THEME["ACCENT"]
FONT_M  = ("Courier New", 9)
FONT_LBL= ("Courier New", 8)
FONT_HDR= ("Courier New", 10, "bold")

FIELD_TOOLTIPS = {
    "phase_deg": "Where we are in the current market cycle (0° bottom, 180° top, 360° bottom again).",
    "bars_to_next_turn": "Rough number of candles until the next expected cycle turning point.",
    "next_turn_eta_utc": "Estimated UTC date/time when the next cycle turn is expected (based on current timeframe).",
    "market_bias": "Overall direction from all engines: Bullish, Bearish or Neutral.",
    "kelly_position_pct": "Suggested position size as % of your account based on edge (not a guarantee).",
    "active_stop_price": "Deprecated in UI (internal only).",
    "active_stop_type": "Deprecated in UI (internal only).",
    "risk_reward_grade": "Letter grade for reward vs. risk on this idea (A is best, D is weakest).",
    "overall_risk_score": "0–1 score for how risky this setup is (higher = more risk).",
    "consensus_confidence": "How many engines agree on the bias. Higher = more agreement.",
    "suggested_entry_price": "Approximate price to ENTER the trade (based on latest close).",
    "suggested_stop_price": "Suggested STOP-LOSS price. If the market hits this, close the trade.",
    "suggested_exit_price": "Default TAKE-PROFIT based on risk/reward (typically TP2).",
    "take_profit_1": "Conservative TAKE-PROFIT (≈1× risk distance from entry).",
    "take_profit_2": "Standard TAKE-PROFIT (≈2× risk distance from entry).",
    "take_profit_3": "Aggressive TAKE-PROFIT (maximal risk/reward within configured cap).",
}


def build_tradeplan_canvas(frame, sections: list) -> dict:
    """
    Creates the scrollable canvas + inner Frame + all metric row widgets.
    Returns a dict mapping field key → (StringVar, suffix, color).
    Called once at TradeplanPanel construction time.
    """
    canvas = tk.Canvas(frame, bg=BG, highlightthickness=0, bd=0)
    sb     = tk.Scrollbar(frame, orient="vertical", command=canvas.yview,
                           bg=PANEL, troughcolor=BG)
    canvas.configure(yscrollcommand=sb.set)
    sb.pack(side="right", fill="y")
    canvas.pack(side="left", fill="both", expand=True)

    inner = tk.Frame(canvas, bg=BG)
    inner_id = canvas.create_window((0, 0), window=inner, anchor="nw")
    inner.bind("<Configure>",
               lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
    canvas.bind("<Configure>",
                lambda e: canvas.itemconfig(inner_id, width=e.width))
    inner.bind("<MouseWheel>",
               lambda e: canvas.yview_scroll(-1*(e.delta//120), "units"))

    rows: dict = {}
    for section_title, fields in sections:
        tk.Label(inner, text=f"── {section_title} ────────────────────",
                 bg=BG, fg=ACCENT, font=FONT_HDR, anchor="w").pack(
                     fill="x", padx=8, pady=(10, 2))
        for label, key, suffix, color in fields:
            row = tk.Frame(inner, bg=PANEL, bd=0)
            row.pack(fill="x", padx=8, pady=1)
            lbl_widget = tk.Label(row, text=f"  {label:<22}", bg=PANEL, fg=FG,
                                  font=FONT_LBL, anchor="w", width=24)
            lbl_widget.pack(side="left")
            if key in FIELD_TOOLTIPS:
                ToolTip(lbl_widget, msg=FIELD_TOOLTIPS[key])
            var = tk.StringVar(value="—")
            rows[key] = (var, suffix, color)
            tk.Label(row, textvariable=var, bg=PANEL, fg=color,
                     font=FONT_M, anchor="w").pack(side="left", padx=4)
    return rows


def update_tradeplan_rows(rows: dict, trade_plan: dict) -> None:
    """Populates all metric row StringVars from a trade_plan dict."""

    def _resolve(tp: dict, dotted_key: str):
        """Supports nested keys like 'advanced.uncertainty_score'."""
        if "." not in dotted_key:
            return tp.get(dotted_key, "—")
        cur = tp
        for part in dotted_key.split("."):
            if not isinstance(cur, dict) or part not in cur:
                return "—"
            cur = cur[part]
        return cur

    for key, (var, suffix, _) in rows.items():
        raw = _resolve(trade_plan, key)
        if isinstance(raw, float):
            val = f"{raw:.2f}{suffix}"
        elif isinstance(raw, bool):
            val = "YES ⚡" if raw else "No"
        elif raw is None:
            val = "—"
        else:
            val = f"{str(raw).replace('_',' ').title()}{suffix}"
        var.set(val)


def clear_tradeplan_rows(rows: dict) -> None:
    for _, (var, _, __) in rows.items():
        var.set("—")
