# =============================================================================
#  bot/message_builder.py — Telegram Message Formatter
#  Converts the aggregator trade plan dict into Telegram MarkdownV2-safe
#  message strings. Three message types:
#
#    build_summary()   — full analysis digest (~40 lines)
#    build_alert()     — urgent signal notification (< 10 lines)
#    build_help()      — command reference card
#
#  Telegram MarkdownV2 rules:
#    • Reserved chars must be escaped: . ! ( ) - _ * [ ] ~ ` > # + = | { } \
#    • Bold: *text*   Italic: _text_   Monospace: `text`   Code block: ```
#    • Pre-formatted block: ```language\ncode\n```
# =============================================================================

import re
from datetime import datetime, timezone


# ── MarkdownV2 escaping ───────────────────────────────────────────────────────

_ESCAPE_RE = re.compile(r'([_*\[\]()~`>#+=|{}.!\\-])')

def _e(text: str) -> str:
    """Escapes a string for Telegram MarkdownV2."""
    return _ESCAPE_RE.sub(r'\\\1', str(text))


def _ef(value: float, decimals: int = 2, suffix: str = "") -> str:
    """Formats a float and escapes it."""
    return _e(f"{value:.{decimals}f}{suffix}")


def _pct(value: float) -> str:
    """Formats a percentage value."""
    return _e(f"{value:.1f}%")


def _price(value: float) -> str:
    """Formats a price with comma separators."""
    return _e(f"{value:,.2f}")


# ── Urgency emoji mapping ─────────────────────────────────────────────────────

URGENCY_EMOJI = {
    "none":      "⚪",
    "low":       "🟡",
    "medium":    "🟠",
    "high":      "🔴",
    "immediate": "🚨",
    "emergency": "🆘",
}

BIAS_EMOJI = {
    "bullish": "📈",
    "bearish": "📉",
    "neutral": "➡️",
}

PHASE_EMOJI = {
    "early_bullish": "🌱",
    "mid_expansion": "🚀",
    "distribution":  "⛰️",
    "accumulation":  "🌊",
    "unknown":       "❓",
}

GRADE_EMOJI = {
    "A": "🟢", "B": "🟡", "C": "🟠", "D": "🔴", "N/A": "⚫",
}


# ── Internal section builders ─────────────────────────────────────────────────

def _header(plan: dict) -> str:
    sym  = _e(plan.get("symbol", "?"))
    ts   = _e(datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"))
    conf = plan.get("consensus_confidence", 0.0)
    conf_bar = "█" * int(conf / 10) + "░" * (10 - int(conf / 10))
    mode = str(plan.get("mode", "classic")).lower()
    mode_label = "ADVANCED" if mode == "advanced" else "CLASSIC"
    return (
        f"*🔭 TEMPORAL ANALYSIS — {sym}*  `[{_e(mode_label)}]`\n"
        f"`{_e(conf_bar)}` {_pct(conf)} confidence\n"
        f"_{ts}_\n"
    )


def _cycle_section(plan: dict) -> str:
    phase     = plan.get("phase_deg",        0.0)
    turn_type = plan.get("turn_type",        "unknown")
    turn_urg  = plan.get("turn_urgency",     "low")
    bars_turn = plan.get("bars_to_next_turn",0)
    dom_cycle = plan.get("dominant_cycle_bars", 0.0)
    ssa_period= plan.get("ssa_period",       0.0)
    ssa_pos   = plan.get("ssa_position",     "unknown")
    cycle_pct = plan.get("cycle_pct_complete",0.0)
    next_turn = plan.get("next_turn_type",   "?")

    phase_em  = PHASE_EMOJI.get(turn_type, "❓")
    urg_em    = URGENCY_EMOJI.get(turn_urg, "⚪")

    return (
        f"\n*📊 TEMPORAL CYCLE*\n"
        f"{phase_em} Phase: *{_ef(phase, 1)}°* — {_e(turn_type.replace('_',' ').title())}\n"
        f"{urg_em} Turn urgency: *{_e(turn_urg.upper())}* \\({_e(str(bars_turn))} bars\\)\n"
        f"Next turn: _{_e(next_turn.replace('_',' ').title())}_\n"
        f"Cycle: {_pct(cycle_pct)} complete \\({_ef(dom_cycle,0)} bars\\)\n"
        f"SSA: {_ef(ssa_period,1)} bars — _{_e(ssa_pos.replace('_',' ').title())}_\n"
    )


def _market_section(plan: dict) -> str:
    bias      = plan.get("market_bias",       "neutral")
    strength  = plan.get("bias_strength",     0.0)
    murray    = plan.get("murray_index",      0.0)
    murray_a  = plan.get("murray_action",     "")
    gamma_r   = plan.get("gamma_regime",      "unknown")
    sol_bias  = plan.get("seasonal_bias",     "unknown")
    vol_flag  = plan.get("volatility_flag",   False)
    walras    = plan.get("walras_adjustment", 1.0)
    sweep     = plan.get("sweep_detected",    False)
    reasons   = plan.get("bias_reasons",      [])

    bias_em   = BIAS_EMOJI.get(bias, "➡️")
    vol_str   = " ⚡ *VOL ZONE*" if vol_flag else ""
    sweep_str = "\n⚡ *H/L SWEEP DETECTED — counter\\-trend setup*" if sweep else ""

    reason_str = ""
    if reasons:
        bullets = "\n".join(f"  • {_e(r)}" for r in reasons[:4])
        reason_str = f"\n{bullets}"

    return (
        f"\n*🌐 MARKET CONTEXT*\n"
        f"{bias_em} Bias: *{_e(bias.upper())}* \\({_pct(strength * 100)}\\){reason_str}\n"
        f"Murray: *{_ef(murray, 2)}/8* — _{_e(murray_a[:40])}_\n"
        f"Gamma: `{_e(gamma_r.replace('_',' '))}`\n"
        f"Seasonal: _{_e(sol_bias.replace('_',' ').title())}_{vol_str}\n"
        f"Walras adj: *{_ef(walras, 2)}×*{sweep_str}\n"
    )


def _trade_section(plan: dict) -> str:
    kelly_pct = plan.get("kelly_position_pct", 0.0)
    kelly_tier= plan.get("kelly_tier",         "")
    kelly_ev  = plan.get("kelly_ev",           0.0)
    hedge_pct = plan.get("hedge_pct",          0.0)
    hedge_urg = plan.get("hedge_urgency",      "none")
    hedge_act = plan.get("hedge_action",       "")
    stop_p    = plan.get("active_stop_price",  0.0)
    stop_type = plan.get("active_stop_type",   "")
    stop_pct  = plan.get("active_stop_pct",    0.0)
    rr_grade  = plan.get("risk_reward_grade",  "N/A")
    risk_score= plan.get("overall_risk_score", 0.5)
    port_sum  = plan.get("portfolio_summary",  "")
    adv       = plan.get("advanced", {}) or {}
    uncert    = float(adv.get("uncertainty_score", 0.0) or 0.0)

    grade_em  = GRADE_EMOJI.get(rr_grade, "⚫")
    hedge_em  = URGENCY_EMOJI.get(hedge_urg, "⚪")
    risk_em   = "🟢" if risk_score < 0.4 else ("🟠" if risk_score < 0.7 else "🔴")
    unc_em    = "🟢" if uncert < 0.3 else ("🟠" if uncert < 0.7 else "🔴")

    return (
        f"\n*💼 ANALYSIS PLAN*\n"
        f"Suggested sizing: *{_pct(kelly_pct)}* — _{_e(kelly_tier)}_ \\(EV: `{_e(f'{kelly_ev:+.4f}')}`\\)\n"
        f"{hedge_em} Suggested hedge: *{_pct(hedge_pct)}* — _{_e(hedge_act[:50])}_\n"
        f"Suggested stop: `{_price(stop_p)}` \\({_e(stop_type)}, {_pct(stop_pct)}\\)\n"
        f"{grade_em} R/R Grade: *{_e(rr_grade)}*  {risk_em} Risk: `{_ef(risk_score, 2)}`  {unc_em} Uncertainty: `{_ef(uncert, 2)}`\n"
        f"_{_e(port_sum[:65])}_\n"
    )


def _footer(plan: dict) -> str:
    elapsed = plan.get("analysis_time_sec", 0.0)
    return (
        f"\n`⏱ {_ef(elapsed,1)}s  •  /run /export /alert /help`"
    )


# ── Public interfaces ─────────────────────────────────────────────────────────

def build_summary(plan: dict) -> str:
    """
    Builds the full analysis summary message for Telegram MarkdownV2.
    Composed of: header + cycle + market + trade sections + footer.

    Parameters
    ----------
    plan : dict — output of aggregator.run()

    Returns
    -------
    str — Telegram-safe MarkdownV2 string (< 4096 chars)
    """
    msg = (
        _header(plan)
        + _cycle_section(plan)
        + _market_section(plan)
        + _trade_section(plan)
        + _footer(plan)
    )
    # Telegram message limit safety trim
    return msg[:4090] + "…" if len(msg) > 4090 else msg


def build_summary_advanced(plan: dict) -> str:
    """
    Alias for build_summary() for now, kept for clarity and future
    advanced-only formatting tweaks.
    """
    return build_summary(plan)


def build_alert(plan: dict, alert_type: str = "turn") -> str:
    """
    Builds a short urgent alert message for high-priority events.

    Alert types:
      "turn"    — Hilbert phase boundary approaching (urgency = high/immediate)
      "sweep"   — Yesterday H/L sweep detected
      "emergency" — Walras liquidity shock

    Parameters
    ----------
    plan       : dict — output of aggregator.run()
    alert_type : str

    Returns
    -------
    str — short MarkdownV2 string (< 500 chars)
    """
    sym    = _e(plan.get("symbol", "?"))
    phase  = plan.get("phase_deg",        0.0)
    turn   = plan.get("turn_type",        "unknown")
    bars   = plan.get("bars_to_next_turn",0)
    kelly  = plan.get("kelly_position_pct",0.0)
    hedge  = plan.get("hedge_pct",         0.0)
    stop_p = plan.get("active_stop_price", 0.0)

    if alert_type == "emergency":
        return (
            f"🆘 *EMERGENCY SIGNAL — {sym}*\n"
            f"Extreme liquidity shock detected\\. Analysis suggests immediate risk review\\.\n"
            f"Suggested stop level: `{_price(stop_p)}`"
        )
    elif alert_type == "sweep":
        bias = plan.get("market_bias", "neutral")
        return (
            f"⚡ *H/L SWEEP SIGNAL — {sym}*\n"
            f"Yesterday level swept → analysis indicates counter\\-trend _{_e(bias)}_ setup\n"
            f"Suggested sizing: *{_pct(kelly)}*  Suggested hedge: *{_pct(hedge)}*\n"
            f"Suggested stop level: `{_price(stop_p)}`"
        )
    else:  # turn alert
        turn_em = PHASE_EMOJI.get(turn, "❓")
        return (
            f"🔴 *CYCLE TURN SIGNAL — {sym}*\n"
            f"{turn_em} *{_e(turn.replace('_',' ').title())}* in *{_e(str(bars))} bars*\n"
            f"Phase: {_ef(phase, 1)}°  Suggested sizing: *{_pct(kelly)}*\n"
            f"Suggested hedge: *{_pct(hedge)}*  Suggested stop: `{_price(stop_p)}`"
        )


def build_help() -> str:
    """Returns the /help command reference card."""
    return (
        "*🤖 Temporal Analysis Bot — Commands*\n\n"
        "_This bot analyses assets and plans\\. It never places orders\\._\n\n"
        "`/run`  — Run full analysis on current pair\n"
        "`/run BTCUSDT 4h 512`  — Run with custom params\n"
        "`/run_adv`  — Run advanced analysis on current pair\n"
        "`/status`  — Show last analysis summary\n"
        "`/status_adv`  — Show last advanced summary\n"
        "`/export`  — Send PDF report \\+ CSV data file\n"
        "`/export_adv`  — Send PDF/CSV for last advanced run\n"
        "`/setpair ETHUSDT 1d`  — Change default pair\n"
        "`/alert on|off`  — Toggle cycle turn alerts\n"
        "`/alert threshold 30`  — Alert when ≤ N bars to turn\n"
        "`/schedule 5m|1h|4h|off`  — Set auto\\-refresh interval\n"
        "`/help`  — Show this message\n\n"
        "_Charts attached after /run: Cycle, Murray, Phase, Risk, Gann_"
    )


def build_schedule_confirm(interval: str, symbol: str) -> str:
    """Confirms schedule change."""
    if interval == "off":
        return f"⏹ Auto\\-analysis *disabled* for `{_e(symbol)}`"
    return (
        f"⏱ Auto\\-analysis set to *{_e(interval)}* for `{_e(symbol)}`\n"
        f"Next run in {_e(interval)}"
    )

# ── Smoke test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    fake = {"symbol": "BTCUSDT", "consensus_confidence": 74.3, "market_bias": "bearish",
            "bias_strength": 0.62, "phase_deg": 215.0, "turn_type": "distribution",
            "turn_urgency": "high", "bars_to_next_turn": 12, "next_turn_type": "accumulation",
            "dominant_cycle_bars": 64.0, "ssa_period": 61.0, "ssa_position": "near_peak",
            "cycle_pct_complete": 60.0, "murray_index": 6.7,
            "murray_action": "Sell — Strong resistance", "gamma_regime": "negative_gamma",
            "seasonal_bias": "autumn_volatility_bias", "volatility_flag": True,
            "walras_adjustment": 0.85, "sweep_detected": True, "bias_reasons": [],
            "kelly_position_pct": 8.0, "kelly_tier": "Small (5–10%)", "kelly_ev": 0.031,
            "hedge_pct": 25.0, "hedge_urgency": "high", "hedge_action": "Open 25% short hedge",
            "active_stop_price": 59400.0, "active_stop_type": "trailing", "active_stop_pct": 2.1,
            "risk_reward_grade": "B", "overall_risk_score": 0.41, "analysis_time_sec": 4.23,
            "portfolio_summary": "Net SHORT 17% | Gross 33% | R/R 2.1× (B)"}
    msg = build_summary(fake)
    assert len(msg) < 4096, f"Summary too long: {len(msg)}"
    assert len(build_alert(fake, "turn"))  < 500
    assert len(build_alert(fake, "sweep")) < 500
    print(f"✅ message_builder OK — summary={len(msg)} chars")
