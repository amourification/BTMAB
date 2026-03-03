# =============================================================================
#  gui/gui_state.py — Shared GUI State & Thread-Safe Result Queue
#  Holds the single mutable state object that the background analysis thread
#  writes to and the Tkinter GUI thread reads from.
#
#  Pattern: producer/consumer with a queue.Queue of at most 1 item.
#  The GUI polls the queue every POLL_MS milliseconds via .after().
# =============================================================================

import queue
import threading
from dataclasses import dataclass, field
from typing import Any


POLL_MS:      int = 400    # GUI polls for new results every 400 ms
REFRESH_SECS: int = 300    # default auto-refresh interval (5 minutes)


@dataclass
class AppState:
    """
    Single source of truth for the running application.
    Written by background thread, read by GUI thread.
    All writes should go through the result_queue — direct field mutation
    is allowed only from the Tkinter main thread.
    """
    # ── User inputs (set by control panel widgets) ────────────────────────
    symbol:   str = "BTCUSDT"
    interval: str = "1d"
    bars:     int = 512
    mode:     str = "classic"  # "classic" or "advanced"

    # ── Analysis state ────────────────────────────────────────────────────
    is_running:    bool  = False
    last_error:    str   = ""
    last_run_secs: float = 0.0

    # ── Latest trade plan (None until first successful run) ───────────────
    trade_plan: Any = None      # dict from aggregator.run()

    # ── Auto-refresh ─────────────────────────────────────────────────────
    auto_refresh:       bool = True
    refresh_interval:   int  = REFRESH_SECS
    countdown_secs:     int  = REFRESH_SECS

    # ── Export paths from last export ────────────────────────────────────
    last_pdf: str = ""
    last_csv: str = ""

    # ── Active tab index ─────────────────────────────────────────────────
    active_tab: int = 0


@dataclass
class AnalysisResult:
    """
    Sent from background thread → GUI thread via result_queue.
    success=False means the run failed; error contains the reason.
    """
    success:    bool
    trade_plan: Any   = None
    error:      str   = ""
    elapsed:    float = 0.0


class StateManager:
    """
    Thread-safe wrapper around AppState.

    Background thread pushes AnalysisResult via push().
    GUI thread polls via poll() and applies results to AppState.
    """

    def __init__(self) -> None:
        self.state  = AppState()
        self._queue: queue.Queue[AnalysisResult] = queue.Queue(maxsize=1)
        self._lock  = threading.Lock()

    def push(self, result: AnalysisResult) -> None:
        """
        Called from the background analysis thread.
        Drops any pending result that hasn't been consumed yet
        (analysis thread is faster than GUI refresh — newest wins).
        """
        try:
            self._queue.get_nowait()   # discard stale result
        except queue.Empty:
            pass
        try:
            self._queue.put_nowait(result)
        except queue.Full:
            pass   # still full somehow — drop silently

    def poll(self) -> AnalysisResult | None:
        """
        Called from the Tkinter main thread (via .after() poll loop).
        Returns the latest AnalysisResult or None if queue is empty.
        """
        try:
            return self._queue.get_nowait()
        except queue.Empty:
            return None

    def update_inputs(self, symbol: str, interval: str, bars: int, mode: str | None = None) -> None:
        """Called from GUI thread when user changes inputs."""
        with self._lock:
            self.state.symbol   = symbol.strip().upper()
            self.state.interval = interval.strip()
            self.state.bars     = max(64, min(4096, int(bars)))
            if mode is not None:
                self.state.mode = (mode or "classic").strip().lower()

    def set_running(self, running: bool) -> None:
        with self._lock:
            self.state.is_running = running

    def apply_result(self, result: AnalysisResult) -> None:
        """Merges a successful AnalysisResult into AppState (GUI thread only)."""
        with self._lock:
            self.state.is_running    = False
            self.state.last_run_secs = result.elapsed
            if result.success:
                self.state.trade_plan = result.trade_plan
                self.state.last_error = ""
                self.state.countdown_secs = self.state.refresh_interval
            else:
                self.state.last_error = result.error

    def tick_countdown(self) -> int:
        """
        Decrements auto-refresh countdown by 1.
        Returns the new countdown value (0 means it's time to refresh).
        """
        with self._lock:
            self.state.countdown_secs = max(0, self.state.countdown_secs - 1)
            return self.state.countdown_secs

    def reset_countdown(self) -> None:
        with self._lock:
            self.state.countdown_secs = self.state.refresh_interval

    def set_refresh_interval(self, seconds: int) -> None:
        """
        Updates the auto-refresh interval (in seconds) and resets countdown.
        """
        seconds = max(10, int(seconds))
        with self._lock:
            self.state.refresh_interval = seconds
            self.state.countdown_secs = seconds

    def snapshot(self) -> AppState:
        """Returns a shallow copy of AppState for GUI rendering (no lock held during render)."""
        with self._lock:
            import copy
            return copy.copy(self.state)


# ── Singleton ─────────────────────────────────────────────────────────────────
_manager: StateManager | None = None


def get_manager() -> StateManager:
    """Returns the global StateManager singleton, creating it if needed."""
    global _manager
    if _manager is None:
        _manager = StateManager()
    return _manager


def reset_manager() -> None:
    """Resets the singleton (used in testing)."""
    global _manager
    _manager = None
