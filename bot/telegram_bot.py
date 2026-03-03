# =============================================================================
#  bot/telegram_bot.py — Telegram Bot Entry Point
#  Bootstraps the python-telegram-bot Application, registers all command
#  handlers, wires the BotScheduler, and starts the polling loop.
#
#  Usage:
#      python -m bot.telegram_bot                # reads BOT_TOKEN from env
#      python -m bot.telegram_bot --token TOKEN  # explicit token
#
#  Environment variables:
#      BOT_TOKEN        — Telegram bot token (required)
#      ALLOWED_CHAT_IDS — comma-separated chat IDs to whitelist (optional)
#
#  Architecture:
#      Application (python-telegram-bot v20+)
#        ├── CommandHandlers → bot_commands.py
#        ├── BotScheduler   → bot_scheduler.py  (APScheduler)
#        ├── MessageBuilder → message_builder.py (formatting)
#        └── Aggregator     → consensus/aggregator.py (analysis engine)
# =============================================================================

import asyncio
import logging
import os
import sys
from pathlib import Path

from telegram import Update, BotCommand
from telegram.ext import (
    Application, ApplicationBuilder, CommandHandler,
    ContextTypes, filters,
)

from bot.bot_commands import (
    cmd_start, cmd_help, cmd_run, cmd_status,
    cmd_export, cmd_setpair, cmd_alert, cmd_schedule,
)
from bot.bot_scheduler import BotScheduler
from bot.message_builder import build_alert

logger = logging.getLogger("temporal_bot.bot.telegram_bot")

BOT_COMMANDS = [
    ("start",    "Welcome message"),
    ("help",     "Command reference"),
    ("run",      "Run analysis [SYMBOL INTERVAL BARS]"),
    ("status",   "Show last analysis"),
    ("export",   "Download PDF report + CSV"),
    ("setpair",  "Set default pair [SYMBOL INTERVAL]"),
    ("alert",    "Toggle alerts [on|off|threshold N]"),
    ("schedule", "Auto-refresh schedule [Nh|Nm|off]"),
]


# ── Auth middleware ────────────────────────────────────────────────────────────

def _make_auth_filter(allowed_ids: list[int] | None):
    """
    Returns a filter that only passes updates from allowed chat IDs.
    If allowed_ids is empty/None, all chats are permitted.
    """
    if not allowed_ids:
        return filters.ALL
    return filters.Chat(chat_id=allowed_ids)


# ── Scheduler alert callback ──────────────────────────────────────────────────

def _make_alert_callback(app: Application):
    """
    Builds the async alert_fn used by BotScheduler.process_alerts().
    Sends a formatted alert message to the given chat_id.
    """
    async def _alert(chat_id: int, plan: dict, alert_type: str) -> None:
        try:
            from telegram.constants import ParseMode
            msg = build_alert(plan, alert_type)
            await app.bot.send_message(
                chat_id=chat_id,
                text=msg,
                parse_mode=ParseMode.MARKDOWN_V2,
            )
            logger.info("Alert '%s' sent to %d", alert_type, chat_id)
        except Exception as exc:
            logger.error("Alert send failed: %s", exc)
    return _alert


def _make_run_callback(app: Application, cfg: dict):
    """
    Builds the async run_fn used by BotScheduler for scheduled jobs.
    Mimics /run behaviour without an Update object.
    """
    async def _run(chat_id: int, symbol: str, interval: str, bars: int) -> None:
        from bot.bot_commands_utils import run_engine, send_charts
        from bot.message_builder import build_summary
        from telegram.constants import ParseMode

        logger.info("Scheduled run: %s %s %d for chat %d",
                    symbol, interval, bars, chat_id)
        try:
            plan = await run_engine(symbol, interval, bars, cfg)
        except Exception as exc:
            await app.bot.send_message(chat_id, f"❌ Scheduled run failed: {exc}")
            return

        if not plan.get("success"):
            await app.bot.send_message(
                chat_id, f"❌ Analysis failed: {plan.get('error','?')}"
            ); return

        await app.bot.send_message(
            chat_id,
            build_summary(plan),
            parse_mode=ParseMode.MARKDOWN_V2,
        )
        # Attach a lightweight mock Update-like object for send_charts
        class _FakeUpdate:
            class effective_chat:
                @staticmethod
                async def send_photo(**kw): await app.bot.send_photo(chat_id=chat_id, **kw)
                @staticmethod
                async def send_action(_): pass
            class message:
                @staticmethod
                async def reply_text(t, **_): await app.bot.send_message(chat_id, t)
        try:
            await send_charts(_FakeUpdate(), plan)
        except Exception as exc:
            logger.warning("Scheduled chart send failed: %s", exc)

        scheduler = app.bot_data.get("scheduler")
        if scheduler:
            await scheduler.process_alerts(chat_id, plan)
    return _run


# ── Application builder ───────────────────────────────────────────────────────

def build_application(token: str, cfg: dict,
                      allowed_ids: list[int] | None = None) -> Application:
    """
    Constructs and configures the Telegram Application.
    Registers all command handlers and injects the BotScheduler.

    Parameters
    ----------
    token       : str — Telegram bot token
    cfg         : dict — full config from config.py
    allowed_ids : list[int] | None — whitelist of chat IDs (None = all)

    Returns
    -------
    telegram.ext.Application (ready to run)
    """
    app = (
        ApplicationBuilder()
        .token(token)
        .concurrent_updates(True)
        .build()
    )

    # Inject cfg and scheduler into bot_data (shared across all chats)
    app.bot_data["cfg"] = cfg

    # Register commands with auth filter
    auth  = _make_auth_filter(allowed_ids)
    cmds  = [
        ("start",      cmd_start),
        ("help",       cmd_help),
        ("run",        cmd_run),
        ("status",     cmd_status),
        ("export",     cmd_export),
        ("run_adv",    cmd_run_adv),
        ("status_adv", cmd_status_adv),
        ("export_adv", cmd_export_adv),
        ("setpair",    cmd_setpair),
        ("alert",      cmd_alert),
        ("schedule",   cmd_schedule),
    ]
    for name, handler in cmds:
        app.add_handler(CommandHandler(name, handler, filters=auth))

    # Wire scheduler after app is built (needs bot reference for callbacks)
    scheduler = BotScheduler(
        run_fn=_make_run_callback(app, cfg),
        alert_fn=_make_alert_callback(app),
    )
    app.bot_data["scheduler"] = scheduler

    # Lifecycle hooks
    async def _on_startup(application: Application) -> None:
        scheduler.start()
        await application.bot.set_my_commands(
            [BotCommand(c, d) for c, d in BOT_COMMANDS]
        )
        logger.info("Bot started — commands registered")

    async def _on_shutdown(application: Application) -> None:
        scheduler.stop()
        logger.info("Bot stopped")

    app.post_init    = _on_startup
    app.post_shutdown= _on_shutdown

    return app


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    """
    Launches the Telegram bot.
    Reads BOT_TOKEN from environment (or --token CLI arg).
    Reads ALLOWED_CHAT_IDS as comma-separated ints (optional).
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # ── Token resolution ──────────────────────────────────────────────────
    token = os.environ.get("TELEGRAM_BOT_TOKEN") or os.environ.get("BOT_TOKEN", "")
    if "--token" in sys.argv:
        idx = sys.argv.index("--token")
        if idx + 1 < len(sys.argv):
            token = sys.argv[idx + 1]

    if not token:
        print("ERROR: TELEGRAM_BOT_TOKEN not set. Use:\n"
              "  export TELEGRAM_BOT_TOKEN=<your_token>\n"
              "  python -m bot.telegram_bot\n"
              "or:\n"
              "  python -m bot.telegram_bot --token <your_token>",
              file=sys.stderr)
        sys.exit(1)

    # ── Allowed chat IDs ──────────────────────────────────────────────────
    raw_ids = os.environ.get("ALLOWED_CHAT_IDS", "")
    allowed_ids: list[int] | None = None
    if raw_ids.strip():
        try:
            allowed_ids = [int(x.strip()) for x in raw_ids.split(",") if x.strip()]
            logger.info("Whitelist: %s", allowed_ids)
        except ValueError:
            logger.warning("ALLOWED_CHAT_IDS parse failed — allowing all chats")

    # ── Config ────────────────────────────────────────────────────────────
    cfg: dict = {}
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from config import build_config
        cfg = build_config()
    except Exception as exc:
        logger.warning("Config load failed (%s) — using defaults", exc)

    # ── Launch ────────────────────────────────────────────────────────────
    app = build_application(token, cfg, allowed_ids)
    logger.info("Starting polling…")
    app.run_polling(allowed_updates=Update.ALL_TYPES, drop_pending_updates=True)


if __name__ == "__main__":
    main()
