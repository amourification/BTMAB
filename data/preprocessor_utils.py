# =============================================================================
#  data/preprocessor_utils.py — Preprocessing Helper Functions
#  Internal utilities used by preprocessor.py. Split to keep both ≤ 300 lines.
# =============================================================================

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timezone, time as dt_time

logger = logging.getLogger("temporal_bot.preprocessor_utils")

# Julian Date constants
J2000_JD: float       = 2_451_545.0
DAY_SESSION_START     = dt_time(8, 0, 0)   # 08:00 UTC
DAY_SESSION_END       = dt_time(22, 0, 0)  # 22:00 UTC


def unix_ms_to_julian(unix_ms: np.ndarray) -> np.ndarray:
    """
    Converts Unix timestamps (ms) to Julian Dates.
    Formula: JD = (unix_sec / 86400) + 2440587.5
    """
    return (unix_ms / 1000.0 / 86400.0) + 2_440_587.5


def julian_to_centurial_T(jd: np.ndarray) -> np.ndarray:
    """
    Converts Julian Dates to Julian Centuries from J2000.0.
    Formula: T = (JD - 2451545.0) / 36525.0
    """
    return (jd - J2000_JD) / 36_525.0


def label_sessions(timestamps_ms: np.ndarray) -> np.ndarray:
    """
    Labels each bar as 'day' or 'night' based on UTC open time.
    Day session: 08:00–22:00 UTC. Everything else is 'night'.
    """
    sessions = np.empty(len(timestamps_ms), dtype=object)
    for i, ts_ms in enumerate(timestamps_ms):
        dt_utc   = datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc)
        bar_time = dt_utc.time()
        sessions[i] = "day" if DAY_SESSION_START <= bar_time < DAY_SESSION_END else "night"
    return sessions


def normalize_minmax(arr: np.ndarray) -> np.ndarray:
    """Min-max normalization to [0, 1]. Returns zeros if range is zero."""
    mn, mx = arr.min(), arr.max()
    if mx - mn == 0:
        logger.warning("normalize_minmax: range is zero — returning zeros.")
        return np.zeros_like(arr)
    return (arr - mn) / (mx - mn)


def normalize_zscore(arr: np.ndarray) -> np.ndarray:
    """Z-score normalization. Returns zeros if std is zero."""
    std = arr.std()
    if std == 0:
        logger.warning("normalize_zscore: std is zero — returning zeros.")
        return np.zeros_like(arr)
    return (arr - arr.mean()) / std


def quality_check(df: pd.DataFrame) -> list[str]:
    """Returns list of data quality warnings (empty = clean)."""
    warnings: list[str] = []
    for col, count in df.isnull().sum().items():
        if count > 0:
            warnings.append(f"Column '{col}' has {count} NaN values.")
    zero_closes = (df["close"] == 0).sum()
    if zero_closes > 0:
        warnings.append(f"{zero_closes} bars have zero close price.")
    if df.index.duplicated().sum() > 0:
        warnings.append(f"{df.index.duplicated().sum()} duplicate timestamps — will drop.")
    if not df.index.is_monotonic_increasing:
        warnings.append("Timestamps not sorted ascending — will sort.")
    return warnings


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Drops duplicates, sorts, forward-fills NaNs, drops zero closes."""
    df = df[~df.index.duplicated(keep="last")]
    df = df.sort_index()
    df = df.replace(0, np.nan)
    df = df.ffill(limit=3)
    df = df.dropna(subset=["close"])
    return df
