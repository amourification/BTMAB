# =============================================================================
#  config.py — Central Configuration & Constants
#  Loads environment variables and defines all system-wide parameters.
#  All other modules import from here — never hardcode values elsewhere.
# =============================================================================

import os
import logging
from pathlib import Path
from dotenv import load_dotenv

# ── Locate and load .env ──────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"

if ENV_PATH.exists():
    load_dotenv(dotenv_path=ENV_PATH)
else:
    # Fall back to .env.example so the app doesn't crash before setup
    load_dotenv(dotenv_path=BASE_DIR / ".env.example")
    print(
        "[config] WARNING: .env not found. "
        "Copy .env.example to .env and fill in your API keys."
    )

# ── API Credentials ───────────────────────────────────────────────────────────
BINANCE_API_KEY: str    = os.getenv("BINANCE_API_KEY", "")
BINANCE_API_SECRET: str = os.getenv("BINANCE_API_SECRET", "")
TELEGRAM_BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
POLYGON_API_KEY: str    = os.getenv("POLYGON_API_KEY", "")

# ── Binance Data Settings ─────────────────────────────────────────────────────
DEFAULT_SYMBOL: str   = os.getenv("DEFAULT_SYMBOL", "BTCUSDT")
DEFAULT_INTERVAL: str = os.getenv("DEFAULT_INTERVAL", "1d")

# ── GUI Settings ──────────────────────────────────────────────────────────────
# Preferred GUI theme: "dark" or "light"
GUI_THEME: str = os.getenv("GUI_THEME", "dark").lower()

# Number of bars to fetch and keep in memory.
# 2048 ensures enough history for 512-day cycle detection with headroom.
DEFAULT_BARS: int = int(os.getenv("DEFAULT_BARS", 2048))

# Valid Binance kline intervals (used for input validation in the GUI)
VALID_INTERVALS: list[str] = [
    "1m", "3m", "5m", "15m", "30m",
    "1h", "2h", "4h", "6h", "8h", "12h",
    "1d", "3d", "1w", "1M",
]

# ── Temporal Engine Constants ─────────────────────────────────────────────────

# Dominant macro cycle in bars (days when using 1d interval).
# Corresponds to Bitcoin halving / major macro rhythms.
DOMINANT_CYCLE_BARS: int = 512

# SSA: Embedding window length (L). Rule of thumb: N/2 or next power of 2.
# Larger L → finer frequency resolution. Must be < DEFAULT_BARS / 2.
SSA_WINDOW_LENGTH: int = 256

# SSA: Number of leading eigentriples to reconstruct the signal.
# Higher = captures more components; risk of including noise.
SSA_NUM_COMPONENTS: int = 6

# ACF: Maximum lag to scan (in bars).
ACF_MAX_LAG: int = 100

# FFT: Minimum cycle period to consider (bars). Filters out noise spikes.
FFT_MIN_PERIOD: int = 10

# Hilbert Transform: Phase boundaries for turn-type classification (degrees).
PHASE_EARLY_BULLISH  = (0,   90)    # Early bullish phase
PHASE_MID_EXPANSION  = (90,  180)   # Mid-cycle expansion
PHASE_DISTRIBUTION   = (180, 270)   # Distribution / cyclical top zone
PHASE_ACCUMULATION   = (270, 360)   # Late bearish / accumulation

# ── Murray Math Constants ─────────────────────────────────────────────────────
# Price range lookback for Murray Math level calculation (bars).
MURRAY_LOOKBACK: int = 64

# Murray level labels — index 0 = 0/8, index 8 = 8/8
MURRAY_LABELS: list[str] = [
    "0/8 — Major Support",
    "1/8 — Weak Support",
    "2/8 — Strong Support",
    "3/8 — Lower Pivot",
    "4/8 — Major Pivot (Mid)",
    "5/8 — Upper Pivot",
    "6/8 — Strong Resistance",
    "7/8 — Weak Resistance",
    "8/8 — Major Resistance",
]

# ── Gann Constants ────────────────────────────────────────────────────────────
# Lookback window to find the most recent significant high/low (bars).
GANN_LOOKBACK: int = 128

# Gann angle scales to draw (price units per time unit).
# 1x1 = 45°, 2x1 = steeper, 1x2 = shallower.
GANN_SCALES: list[tuple] = [
    (1, 8),   # 1×8 shallowest
    (1, 4),
    (1, 2),
    (1, 1),   # 1×1 master angle
    (2, 1),
    (4, 1),
    (8, 1),   # 8×1 steepest
]

# ── Kelly Criterion ───────────────────────────────────────────────────────────
# Half-Kelly multiplier — reduces position size to manage drawdown risk.
KELLY_FRACTION: float = 0.5

# Minimum historical cycles required before Kelly is trusted.
KELLY_MIN_CYCLES: int = 5

# ── Risk Management ───────────────────────────────────────────────────────────
# Maximum position size as % of portfolio regardless of Kelly output.
MAX_POSITION_PCT: float = 0.25   # 25%

# Gamma threshold below which hedging is triggered (negative gamma zone).
GAMMA_HEDGE_THRESHOLD: float = -0.02

# Default hedge ratio when gamma trigger fires.
DEFAULT_HEDGE_RATIO: float = 0.30   # 30% of position

# ── AR Model (Day/Night Inertia) ──────────────────────────────────────────────
# Order of the autoregressive model.
AR_ORDER: int = 5

# Weight multiplier applied to overnight sentiment vs intraday.
NIGHT_SENTIMENT_WEIGHT: float = 1.35

# ── Solar Cycle ───────────────────────────────────────────────────────────────
# Seasonal bias threshold — if solar phase within this many days of
# solstice/equinox, a heightened-volatility flag is raised.
SOLAR_PROXIMITY_DAYS: int = 7

# ── Charting & Export ─────────────────────────────────────────────────────────
# Default export directory (created automatically if it doesn't exist).
EXPORT_DIR: Path = BASE_DIR / "exports"
EXPORT_DIR.mkdir(exist_ok=True)

# Chart DPI for PNG/PDF export.
CHART_DPI: int = 150

# Colour palette used across all charts.
CHART_COLORS: dict = {
    "price":        "#00C9FF",   # Cyan — price line
    "detrend":      "#FFD700",   # Gold — detrended oscillator
    "murray_sup":   "#00FF88",   # Green — support levels
    "murray_res":   "#FF4444",   # Red — resistance levels
    "murray_mid":   "#AAAAAA",   # Grey — mid/neutral levels
    "ssa":          "#FF8C00",   # Orange — SSA reconstruction
    "fft":          "#BF5FFF",   # Purple — FFT power spectrum
    "phase_bull":   "#00FF88",   # Green — bullish phase
    "phase_bear":   "#FF4444",   # Red — bearish phase
    "gann":         "#FFAA00",   # Amber — Gann angle lines
    "background":   "#0D0D0D",   # Near-black background
    "grid":         "#1E1E1E",   # Subtle grid
    "text":         "#E0E0E0",   # Light text
}

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger("temporal_bot")

# ── Validation on import ──────────────────────────────────────────────────────
def validate_config() -> list[str]:
    """
    Returns a list of warnings for any missing or suspicious config values.
    Called by main_gui.py on startup to alert the user before analysis runs.
    """
    warnings: list[str] = []

    if not BINANCE_API_KEY or BINANCE_API_KEY == "your_binance_api_key_here":
        warnings.append("BINANCE_API_KEY is not set. Data fetch will fail.")

    if not BINANCE_API_SECRET or BINANCE_API_SECRET == "your_binance_api_secret_here":
        warnings.append("BINANCE_API_SECRET is not set. Data fetch will fail.")

    if DEFAULT_BARS < DOMINANT_CYCLE_BARS * 2:
        warnings.append(
            f"DEFAULT_BARS ({DEFAULT_BARS}) should be at least "
            f"2× DOMINANT_CYCLE_BARS ({DOMINANT_CYCLE_BARS}) for reliable SSA."
        )

    if SSA_WINDOW_LENGTH >= DEFAULT_BARS // 2:
        warnings.append(
            f"SSA_WINDOW_LENGTH ({SSA_WINDOW_LENGTH}) must be < "
            f"DEFAULT_BARS/2 ({DEFAULT_BARS // 2})."
        )

    return warnings


# Run validation and log any warnings immediately on import
_config_warnings = validate_config()
for _w in _config_warnings:
    logger.warning("[config] %s", _w)


# ── Config dict builder (for GUI / bot entry points) ───────────────────────────
def build_config() -> dict:
    """
    Returns a dict of all config values for use by aggregator, fetcher, etc.
    Used by main_gui.py and telegram_bot.py as the cfg argument.
    """
    return {
        "BINANCE_API_KEY":       BINANCE_API_KEY,
        "BINANCE_API_SECRET":    BINANCE_API_SECRET,
        "TELEGRAM_BOT_TOKEN":   TELEGRAM_BOT_TOKEN,
        "POLYGON_API_KEY":      POLYGON_API_KEY,
        "DEFAULT_SYMBOL":        DEFAULT_SYMBOL,
        "DEFAULT_INTERVAL":     DEFAULT_INTERVAL,
        "GUI_THEME":            GUI_THEME,
        "DEFAULT_BARS":         DEFAULT_BARS,
        "VALID_INTERVALS":      VALID_INTERVALS,
        "DOMINANT_CYCLE_BARS":  DOMINANT_CYCLE_BARS,
        "SSA_WINDOW_LENGTH":    SSA_WINDOW_LENGTH,
        "SSA_NUM_COMPONENTS":   SSA_NUM_COMPONENTS,
        "ACF_MAX_LAG":          ACF_MAX_LAG,
        "FFT_MIN_PERIOD":       FFT_MIN_PERIOD,
        "MURRAY_LOOKBACK":      MURRAY_LOOKBACK,
        "MURRAY_LABELS":        MURRAY_LABELS,
        "GANN_LOOKBACK":        GANN_LOOKBACK,
        "GANN_SCALES":          GANN_SCALES,
        "KELLY_FRACTION":       KELLY_FRACTION,
        "KELLY_MIN_CYCLES":     KELLY_MIN_CYCLES,
        "MAX_POSITION_PCT":     MAX_POSITION_PCT,
        "GAMMA_HEDGE_THRESHOLD": GAMMA_HEDGE_THRESHOLD,
        "DEFAULT_HEDGE_RATIO":   DEFAULT_HEDGE_RATIO,
        "AR_ORDER":             AR_ORDER,
        "NIGHT_SENTIMENT_WEIGHT": NIGHT_SENTIMENT_WEIGHT,
        "SOLAR_PROXIMITY_DAYS":  SOLAR_PROXIMITY_DAYS,
        "EXPORT_DIR":           EXPORT_DIR,
        "CHART_DPI":            CHART_DPI,
        "CHART_COLORS":         CHART_COLORS,
        "TRADE_DIRECTION":      "long",
        "PORTFOLIO_VALUE":      10000.0,
    }
