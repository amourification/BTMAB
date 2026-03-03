# =============================================================================
#  bot/bot_scheduler.py — Background Scheduler & Alert Throttle
#  Uses APScheduler to run periodic analysis jobs and fire Telegram alerts
#  when high-urgency conditions are detected.
#
#  Responsibilities:
#    - Schedule periodic /run jobs per chat_id (each chat can have its own
#      symbol, interval, and schedule cadence)
#    - Throttle alerts so the same signal type doesn't spam every cycle
#    - Detect alert-worthy conditions from a trade plan dict
#    - Push results back to the bot via an async callback
#
#  Design: one AsyncIOScheduler shared across all chats. Each chat gets
#  its own APScheduler job keyed by job_id = f"analysis_{chat_id}".
# =============================================================================

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Callable, Awaitable

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval  import IntervalTrigger

logger = logging.getLogger("temporal_bot.bot.scheduler")

# Alert throttle: minimum seconds between same alert type per chat
ALERT_THROTTLE_SECS: int = 3600    # 1 hour between same-type alerts
MAX_JOBS_PER_CHAT:   int = 1       # only one schedule per chat at a time

# Alert condition thresholds
TURN_ALERT_BARS:     int   = 20    # alert when bars_to_next_turn ≤ this
URGENCY_ALERT_SET   = {"high", "immediate"}
EMERGENCY_SHOCK_SIG = 5.0          # walras shock σ threshold


# ── Alert state tracker ───────────────────────────────────────────────────────

class AlertThrottle:
    """
    Tracks the last time each alert type was sent per chat_id.
    Prevents spam: same alert type suppressed for ALERT_THROTTLE_SECS.
    """

    def __init__(self) -> None:
        # {chat_id: {alert_type: last_sent_ts}}
        self._last: dict[int, dict[str, float]] = {}

    def can_send(self, chat_id: int, alert_type: str) -> bool:
        now    = time.monotonic()
        chat   = self._last.setdefault(chat_id, {})
        last_t = chat.get(alert_type, 0.0)
        return (now - last_t) >= ALERT_THROTTLE_SECS

    def mark_sent(self, chat_id: int, alert_type: str) -> None:
        self._last.setdefault(chat_id, {})[alert_type] = time.monotonic()

    def reset(self, chat_id: int) -> None:
        self._last.pop(chat_id, None)

    def set_throttle(self, secs: int) -> None:
        """Override throttle window (used by /alert threshold command)."""
        global ALERT_THROTTLE_SECS
        ALERT_THROTTLE_SECS = max(60, int(secs))


# ── Alert condition detector ──────────────────────────────────────────────────

def detect_alerts(plan: dict) -> list[str]:
    """
    Inspects a trade plan dict and returns a list of alert type strings
    that should be sent for this result.

    Alert types:
      "turn"      — Hilbert phase boundary within TURN_ALERT_BARS bars
      "sweep"     — Yesterday H/L sweep was detected
      "emergency" — Walras emergency stop triggered

    Parameters
    ----------
    plan : dict — output of aggregator.run()

    Returns
    -------
    list[str] — list of alert type strings (may be empty)
    """
    alerts: list[str] = []

    # Emergency stop (highest priority)
    risk  = plan.get("_risk", {})
    stops = risk.get("stops", {})
    emrg  = stops.get("emergency_stop", {})
    if emrg.get("triggered", False):
        alerts.append("emergency")
        return alerts   # emergency overrides everything

    # Cycle turn approaching
    bars_to_turn = int(plan.get("bars_to_next_turn", 999))
    turn_urgency = plan.get("turn_urgency", "low")
    if bars_to_turn <= TURN_ALERT_BARS or turn_urgency in URGENCY_ALERT_SET:
        alerts.append("turn")

    # Sweep detected
    if plan.get("sweep_detected", False):
        alerts.append("sweep")

    return alerts


# ── Chat job config ───────────────────────────────────────────────────────────

class ChatConfig:
    """Per-chat configuration for scheduled analysis."""

    __slots__ = ("symbol", "interval", "bars", "schedule_interval",
                 "alerts_enabled", "alert_bars_threshold")

    def __init__(self, symbol: str = "BTCUSDT", interval: str = "1d",
                 bars: int = 512) -> None:
        self.symbol             = symbol
        self.interval           = interval
        self.bars               = bars
        self.schedule_interval  = None    # None = no schedule; str like "4h"
        self.alerts_enabled     = True
        self.alert_bars_threshold = TURN_ALERT_BARS

    def parse_schedule(self) -> dict | None:
        """
        Converts a human interval string to APScheduler IntervalTrigger kwargs.
        Supports: Nm (minutes), Nh (hours), Nd (days).
        Returns None if schedule is "off" or unparseable.
        """
        s = self.schedule_interval
        if not s or s == "off":
            return None
        s = s.strip().lower()
        try:
            if s.endswith("m"):
                return {"minutes": int(s[:-1])}
            elif s.endswith("h"):
                return {"hours": int(s[:-1])}
            elif s.endswith("d"):
                return {"days": int(s[:-1])}
        except ValueError:
            pass
        return None


# ── Scheduler manager ─────────────────────────────────────────────────────────

class BotScheduler:
    """
    Wraps an APScheduler AsyncIOScheduler.
    Manages one analysis job per chat_id and handles alert throttling.

    The analysis callback `run_fn` is async and receives:
        run_fn(chat_id, symbol, interval, bars) -> None
    It is responsible for actually running the engine and sending results.

    The alert callback `alert_fn` is async and receives:
        alert_fn(chat_id, plan, alert_type) -> None
    """

    def __init__(
        self,
        run_fn:   Callable[[int, str, str, int], Awaitable[None]],
        alert_fn: Callable[[int, dict, str], Awaitable[None]],
    ) -> None:
        self._run_fn    = run_fn
        self._alert_fn  = alert_fn
        self._throttle  = AlertThrottle()
        self._configs:  dict[int, ChatConfig] = {}
        self._scheduler = AsyncIOScheduler(timezone="UTC")

    def start(self) -> None:
        """Starts the APScheduler event loop. Call once at bot startup."""
        if not self._scheduler.running:
            self._scheduler.start()
            logger.info("Scheduler started")

    def stop(self) -> None:
        if self._scheduler.running:
            self._scheduler.shutdown(wait=False)
            logger.info("Scheduler stopped")

    def get_config(self, chat_id: int) -> ChatConfig:
        """Returns (creating if needed) the ChatConfig for a chat."""
        if chat_id not in self._configs:
            self._configs[chat_id] = ChatConfig()
        return self._configs[chat_id]

    def set_schedule(self, chat_id: int, interval_str: str) -> bool:
        """
        Sets or removes the scheduled analysis job for a chat.
        Returns True if a new job was scheduled, False if disabled.
        """
        cfg = self.get_config(chat_id)
        cfg.schedule_interval = interval_str
        job_id = f"analysis_{chat_id}"

        # Remove existing job if any
        if self._scheduler.get_job(job_id):
            self._scheduler.remove_job(job_id)

        trigger_kw = cfg.parse_schedule()
        if trigger_kw is None:
            logger.info("Schedule disabled for chat %d", chat_id)
            return False

        async def _job():
            await self._run_fn(chat_id, cfg.symbol, cfg.interval, cfg.bars)

        self._scheduler.add_job(
            _job,
            trigger=IntervalTrigger(**trigger_kw),
            id=job_id,
            replace_existing=True,
            misfire_grace_time=60,
        )
        logger.info("Scheduled %s for chat %d (%s)", cfg.symbol, chat_id, interval_str)
        return True

    async def process_alerts(self, chat_id: int, plan: dict) -> None:
        """
        Called after every analysis run. Checks for alert conditions and
        fires the alert callback for any un-throttled alert types.
        """
        cfg = self.get_config(chat_id)
        if not cfg.alerts_enabled:
            return

        alert_types = detect_alerts(plan)
        for atype in alert_types:
            if self._throttle.can_send(chat_id, atype):
                try:
                    await self._alert_fn(chat_id, plan, atype)
                    self._throttle.mark_sent(chat_id, atype)
                    logger.info("Alert '%s' sent to chat %d", atype, chat_id)
                except Exception as exc:
                    logger.error("Alert send failed (%s): %s", atype, exc)

    def toggle_alerts(self, chat_id: int, enabled: bool) -> None:
        self.get_config(chat_id).alerts_enabled = enabled

    def set_alert_threshold(self, chat_id: int, bars: int) -> None:
        global TURN_ALERT_BARS
        cfg = self.get_config(chat_id)
        cfg.alert_bars_threshold = bars
        TURN_ALERT_BARS = bars   # global override for this session

    def list_jobs(self) -> list[str]:
        """Returns a human-readable list of all active scheduled jobs."""
        jobs = self._scheduler.get_jobs()
        return [f"{j.id}: next={j.next_run_time}" for j in jobs] if jobs else ["No jobs scheduled"]


# ── Smoke test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Test alert detection without starting the scheduler
    fake_plan = {
        "bars_to_next_turn": 10, "turn_urgency": "high",
        "sweep_detected": True,
        "_risk": {"stops": {"emergency_stop": {"triggered": False}}},
    }
    alerts = detect_alerts(fake_plan)
    print(f"✅ Detected alerts: {alerts}")
    assert "turn"  in alerts, "Expected turn alert"
    assert "sweep" in alerts, "Expected sweep alert"

    # Test throttle
    t = AlertThrottle()
    assert t.can_send(1, "turn") is True
    t.mark_sent(1, "turn")
    assert t.can_send(1, "turn") is False
    assert t.can_send(1, "sweep") is True
    print("✅ AlertThrottle OK")

    # Test ChatConfig parse_schedule
    cfg = ChatConfig()
    cfg.schedule_interval = "4h"
    assert cfg.parse_schedule() == {"hours": 4}
    cfg.schedule_interval = "30m"
    assert cfg.parse_schedule() == {"minutes": 30}
    cfg.schedule_interval = "off"
    assert cfg.parse_schedule() is None
    print("✅ ChatConfig.parse_schedule OK")
    print("✅ bot_scheduler smoke test complete")
