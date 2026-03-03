# =============================================================================
#  bot/bot_commands_utils.py — Command Handler Helpers
#  Split from bot_commands.py to keep both modules under 300 lines.
#  Contains: chart rendering pipeline, file export, engine runner.
# =============================================================================

import io
import logging
import tempfile
from pathlib import Path

from telegram import Update
from telegram.ext import ContextTypes

logger = logging.getLogger("temporal_bot.bot.commands_utils")


async def run_engine(symbol: str, interval: str, bars: int, cfg: dict) -> dict:
    """
    Runs the full aggregator pipeline off the event loop thread.
    Returns the trade plan dict (success flag inside).
    """
    import asyncio
    from concurrent.futures import ThreadPoolExecutor
    from consensus.aggregator import run as agg_run

    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor(max_workers=1) as ex:
        plan = await loop.run_in_executor(
            ex, lambda: agg_run(symbol, interval, bars, cfg)
        )
    return plan


async def run_engine_advanced(symbol: str, interval: str, bars: int, cfg: dict) -> dict:
    """
    Advanced pipeline variant that calls consensus.advanced_aggregator.
    Keeps the same coroutine interface as run_engine().
    """
    import asyncio
    from concurrent.futures import ThreadPoolExecutor
    from consensus.advanced_aggregator import run as agg_run_adv

    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor(max_workers=1) as ex:
        plan = await loop.run_in_executor(
            ex, lambda: agg_run_adv(symbol, interval, bars, cfg)
        )
    return plan


async def send_charts(update: Update, plan: dict) -> None:
    """
    Renders all 5 charts as PNGs and sends each as a Telegram photo.
    Individual sends used (Telegram limits 10 media/group; we have 5,
    but individual sends give cleaner captions per chart).
    """
    from charts.cycle_chart  import render as r_cycle
    from charts.murray_chart import render as r_murray
    from charts.phase_chart  import render as r_phase
    from charts.risk_chart   import render as r_risk
    from charts.gann_chart   import render as r_gann
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    renderers = [
        ("Cycle",  r_cycle),
        ("Murray", r_murray),
        ("Phase",  r_phase),
        ("Risk",   r_risk),
        ("Gann",   r_gann),
    ]
    for name, fn in renderers:
        try:
            fig = fn(plan, figsize=(12, 7))
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=100,
                        bbox_inches="tight", facecolor="#0d0f14")
            plt.close(fig)
            buf.seek(0)
            await update.effective_chat.send_photo(
                photo=buf, caption=f"📊 {name} Chart"
            )
        except Exception as exc:
            logger.warning("Chart '%s' send failed: %s", name, exc)


async def send_export(update: Update, plan: dict, cfg: dict) -> None:
    """
    Generates PDF report + CSV and sends both as Telegram document files.
    Uses a temporary directory that is auto-cleaned after sending.
    """
    from charts.exporter import export_all

    try:
        with tempfile.TemporaryDirectory() as tmp:
            result = export_all(plan, Path(tmp), cfg)
            if result.get("pdf"):
                with open(result["pdf"], "rb") as f:
                    await update.effective_chat.send_document(
                        document=f,
                        filename=Path(result["pdf"]).name,
                        caption="📄 Full PDF Report",
                    )
            if result.get("csv"):
                with open(result["csv"], "rb") as f:
                    await update.effective_chat.send_document(
                        document=f,
                        filename=Path(result["csv"]).name,
                        caption="📊 Trade Plan CSV",
                    )
    except Exception as exc:
        await update.message.reply_text(f"❌ Export error: {exc}")
        logger.exception("Export failed")
