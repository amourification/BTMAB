# =============================================================================
#  bot/bot_commands.py — Telegram Command Handlers
#  All /command handlers wired to the analysis pipeline.
#  Heavy helpers (chart rendering, export, engine runner) in bot_commands_utils.
#
#  Commands:
#    /start          — Welcome message
#    /help           — Command reference card
#    /run [SYM INT BARS] — Run full analysis, send summary + 5 chart images
#    /status         — Resend last analysis summary (no recompute)
#    /export         — Send PDF report + CSV as Telegram documents
#    /setpair SYM INT — Change default symbol/interval for this chat
#    /alert on|off|threshold N — Toggle alerts / set bar threshold
#    /schedule INT|off — Set auto-refresh schedule cadence
# =============================================================================

import logging
from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import ContextTypes

from bot.message_builder import (
    build_summary, build_summary_advanced, build_help, build_schedule_confirm,
)
from bot.bot_commands_utils import run_engine, run_engine_advanced, send_charts, send_export

logger = logging.getLogger("temporal_bot.bot.commands")

LAST_PLAN_KEY = "last_plan_{chat_id}"
LAST_PLAN_ADV_KEY = "last_plan_adv_{chat_id}"
SYMBOL_KEY    = "symbol"
INTERVAL_KEY  = "interval"
BARS_KEY      = "bars"


def _get_defaults(chat_data: dict) -> tuple:
    return (chat_data.get(SYMBOL_KEY, "BTCUSDT"),
            chat_data.get(INTERVAL_KEY, "1d"),
            int(chat_data.get(BARS_KEY, 512)))

def _store(context, chat_id, plan):
    context.bot_data[LAST_PLAN_KEY.format(chat_id=chat_id)] = plan

def _load(context, chat_id):
    return context.bot_data.get(LAST_PLAN_KEY.format(chat_id=chat_id))


def _store_adv(context, chat_id, plan):
    context.bot_data[LAST_PLAN_ADV_KEY.format(chat_id=chat_id)] = plan


def _load_adv(context, chat_id):
    return context.bot_data.get(LAST_PLAN_ADV_KEY.format(chat_id=chat_id))


# ── Handlers ──────────────────────────────────────────────────────────────────

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "🔭 *Temporal Market Analysis Bot*\n\n"
        "Uses 12 mathematical equations to identify dominant market cycles "
        "and generate deterministic trade plans\\.\n\n"
        "Type /help for commands or /run to start\\.",
        parse_mode=ParseMode.MARKDOWN_V2,
    )


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        build_help(), parse_mode=ParseMode.MARKDOWN_V2,
    )


async def cmd_run(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    /run [SYMBOL] [INTERVAL] [BARS]
    Full pipeline: engine → summary text → 5 chart PNGs → alert check.
    """
    chat_id              = update.effective_chat.id
    sym_def, int_def, bars_def = _get_defaults(context.chat_data)
    args     = context.args or []
    symbol   = args[0].upper() if len(args) > 0 else sym_def
    interval = args[1]          if len(args) > 1 else int_def
    try:    bars = int(args[2]) if len(args) > 2 else bars_def
    except: bars = bars_def

    cfg = context.bot_data.get("cfg", {})

    await update.effective_chat.send_action("typing")
    await update.message.reply_text(
        f"⏳ Analysing `{symbol}` {interval} × {bars} bars…",
        parse_mode=ParseMode.MARKDOWN_V2,
    )

    try:
        plan = await run_engine(symbol, interval, bars, cfg)
    except Exception as exc:
        await update.message.reply_text(f"❌ Engine error: {exc}"); return

    if not plan.get("success"):
        await update.message.reply_text(
            f"❌ Failed: {plan.get('error','unknown')}"
        ); return

    _store(context, chat_id, plan)
    await update.message.reply_text(
        build_summary(plan), parse_mode=ParseMode.MARKDOWN_V2,
    )

    await update.effective_chat.send_action("upload_photo")
    await send_charts(update, plan)

    scheduler = context.bot_data.get("scheduler")
    if scheduler:
        await scheduler.process_alerts(chat_id, plan)

    logger.info("/run OK: %s %s %d chat=%d", symbol, interval, bars, chat_id)


async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/status — Resend last summary without re-running."""
    chat_id = update.effective_chat.id
    plan    = _load(context, chat_id)
    if plan is None:
        await update.message.reply_text(
            "No analysis yet\\. Use /run\\.", parse_mode=ParseMode.MARKDOWN_V2,
        ); return
    await update.message.reply_text(
        build_summary(plan), parse_mode=ParseMode.MARKDOWN_V2,
    )


async def cmd_export(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/export — Send PDF + CSV as Telegram documents."""
    chat_id = update.effective_chat.id
    plan    = _load(context, chat_id)
    if plan is None:
        await update.message.reply_text(
            "Run /run first\\.", parse_mode=ParseMode.MARKDOWN_V2,
        ); return
    await update.effective_chat.send_action("upload_document")
    await send_export(update, plan, context.bot_data.get("cfg", {}))


async def cmd_run_adv(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    /run_adv [SYMBOL] [INTERVAL] [BARS]
    Uses the advanced analysis pipeline and formatting.
    """
    chat_id = update.effective_chat.id
    sym_def, int_def, bars_def = _get_defaults(context.chat_data)
    args = context.args or []
    symbol = args[0].upper() if len(args) > 0 else sym_def
    interval = args[1] if len(args) > 1 else int_def
    try:
        bars = int(args[2]) if len(args) > 2 else bars_def
    except Exception:
        bars = bars_def

    cfg = context.bot_data.get("cfg", {})

    await update.effective_chat.send_action("typing")
    await update.message.reply_text(
        f"⏳ [Advanced] Analysing `{symbol}` {interval} × {bars} bars…",
        parse_mode=ParseMode.MARKDOWN_V2,
    )

    try:
        plan = await run_engine_advanced(symbol, interval, bars, cfg)
    except Exception as exc:
        await update.message.reply_text(f"❌ Engine error: {exc}")
        return

    if not plan.get("success"):
        await update.message.reply_text(
            f"❌ Failed: {plan.get('error','unknown')}"
        )
        return

    _store_adv(context, chat_id, plan)
    await update.message.reply_text(
        build_summary_advanced(plan), parse_mode=ParseMode.MARKDOWN_V2,
    )

    await update.effective_chat.send_action("upload_photo")
    await send_charts(update, plan)

    scheduler = context.bot_data.get("scheduler")
    if scheduler:
        await scheduler.process_alerts(chat_id, plan)

    logger.info("/run_adv OK: %s %s %d chat=%d", symbol, interval, bars, chat_id)


async def cmd_status_adv(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/status_adv — Resend last advanced summary without re-running."""
    chat_id = update.effective_chat.id
    plan = _load_adv(context, chat_id)
    if plan is None:
        await update.message.reply_text(
            "No advanced analysis yet\\. Use /run_adv\\.", parse_mode=ParseMode.MARKDOWN_V2,
        )
        return
    await update.message.reply_text(
        build_summary_advanced(plan), parse_mode=ParseMode.MARKDOWN_V2,
    )


async def cmd_export_adv(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/export_adv — Send PDF + CSV for the last advanced run."""
    chat_id = update.effective_chat.id
    plan = _load_adv(context, chat_id)
    if plan is None:
        await update.message.reply_text(
            "Run /run_adv first\\.", parse_mode=ParseMode.MARKDOWN_V2,
        )
        return
    await update.effective_chat.send_action("upload_document")
    await send_export(update, plan, context.bot_data.get("cfg", {}))


async def cmd_setpair(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/setpair SYMBOL [INTERVAL] [BARS] — Set default pair for this chat."""
    args = context.args or []
    if not args:
        await update.message.reply_text(
            "Usage: `/setpair BTCUSDT 1d 512`",
            parse_mode=ParseMode.MARKDOWN_V2,
        ); return
    context.chat_data[SYMBOL_KEY]   = args[0].upper()
    context.chat_data[INTERVAL_KEY] = args[1] if len(args) > 1 else "1d"
    try:    context.chat_data[BARS_KEY] = int(args[2]) if len(args) > 2 else 512
    except: context.chat_data[BARS_KEY] = 512
    sym = context.chat_data[SYMBOL_KEY]; itv = context.chat_data[INTERVAL_KEY]
    await update.message.reply_text(
        f"✅ Default pair → `{sym}` `{itv}`",
        parse_mode=ParseMode.MARKDOWN_V2,
    )


async def cmd_alert(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/alert on|off|threshold N — Control alert delivery."""
    chat_id   = update.effective_chat.id
    scheduler = context.bot_data.get("scheduler")
    args      = context.args or []
    if not args:
        await update.message.reply_text(
            "Usage: `/alert on` \\| `/alert off` \\| `/alert threshold 20`",
            parse_mode=ParseMode.MARKDOWN_V2,
        ); return

    verb = args[0].lower()
    if verb in ("on", "off"):
        enabled = verb == "on"
        if scheduler: scheduler.toggle_alerts(chat_id, enabled)
        await update.message.reply_text(
            f"Alerts {'enabled ✅' if enabled else 'disabled ⏹'}"
        )
    elif verb == "threshold" and len(args) > 1:
        try:
            bars = int(args[1])
            if scheduler: scheduler.set_alert_threshold(chat_id, bars)
            await update.message.reply_text(
                f"✅ Alert threshold → *{bars}* bars",
                parse_mode=ParseMode.MARKDOWN_V2,
            )
        except ValueError:
            await update.message.reply_text("❌ Invalid number")
    else:
        await update.message.reply_text(
            "Unknown option\\. Try /help", parse_mode=ParseMode.MARKDOWN_V2,
        )


async def cmd_schedule(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/schedule 4h|1h|30m|off — Set or remove auto-analysis schedule."""
    chat_id   = update.effective_chat.id
    scheduler = context.bot_data.get("scheduler")
    args      = context.args or []
    if not args:
        jobs = scheduler.list_jobs() if scheduler else ["Scheduler unavailable"]
        await update.message.reply_text(
            "Schedules:\n`" + "\n".join(jobs) + "`",
            parse_mode=ParseMode.MARKDOWN_V2,
        ); return
    interval_str = args[0].lower()
    sym, _, _    = _get_defaults(context.chat_data)
    if scheduler: scheduler.set_schedule(chat_id, interval_str)
    await update.message.reply_text(
        build_schedule_confirm(interval_str, sym),
        parse_mode=ParseMode.MARKDOWN_V2,
    )
