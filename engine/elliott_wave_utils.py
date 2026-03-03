from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

import numpy as np


@dataclass
class WaveCount:
    """Simplified Elliott wave count representation."""
    impulse_labels: Dict[str, int]  # "1".."5" -> pivot index
    correction_labels: Dict[str, int]  # "A","B","C" -> pivot index (optional)
    direction: str  # "up" or "down"
    score: float  # guideline score 0..1
    valid: bool


def _find_pivots(prices: np.ndarray, window: int = 3) -> List[int]:
    """
    Detect simple zigzag pivots using a rolling window.
    Returns list of indices that are local minima or maxima.
    """
    pivots: List[int] = []
    if len(prices) < window * 2 + 1:
        return pivots

    for i in range(window, len(prices) - window):
        left = prices[i - window : i]
        right = prices[i + 1 : i + 1 + window]
        p = prices[i]
        if p == max(np.concatenate([left, right, [p]])) or p == min(
            np.concatenate([left, right, [p]])
        ):
            pivots.append(i)
    return pivots


def _enforce_impulse_rules(prices: np.ndarray, pivots: List[int]) -> bool:
    """
    Basic Elliott impulse rules:
    - Wave 2 does not retrace beyond start of Wave 1.
    - Wave 3 is not the shortest among 1,3,5.
    - Wave 4 does not overlap Wave 1 price territory.
    """
    if len(pivots) < 6:
        return False

    i1, i2, i3, i4, i5, i6 = pivots[-6:]
    p1, p2, p3, p4, p5, p6 = prices[[i1, i2, i3, i4, i5, i6]]

    # Determine direction from 1→3 move
    direction = "up" if p6 > p1 else "down"

    if direction == "up":
        # Wave 2 (> p1) and does not drop below p1
        if p3 <= p1:
            return False
        # Wave 4 does not enter wave 1 price territory
        w1_lo, w1_hi = min(p1, p2), max(p1, p2)
        if w1_lo <= p5 <= w1_hi:
            return False
        # Wave lengths
        w1 = abs(p2 - p1)
        w3 = abs(p4 - p3)
        w5 = abs(p6 - p5)
    else:
        # Down impulse
        if p3 >= p1:
            return False
        w1_lo, w1_hi = min(p1, p2), max(p1, p2)
        if w1_lo <= p5 <= w1_hi:
            return False
        w1 = abs(p2 - p1)
        w3 = abs(p4 - p3)
        w5 = abs(p6 - p5)

    shortest = min(w1, w3, w5)
    # Wave 3 cannot be the shortest
    if w3 == shortest:
        return False
    return True


def _guideline_score(prices: np.ndarray, pivots: List[int]) -> float:
    """
    Crude guideline score based on Fibonacci relationships.
    Returns 0..1 (higher = more like textbook Elliott).
    """
    if len(pivots) < 6:
        return 0.0
    i1, i2, i3, i4, i5, i6 = pivots[-6:]
    p1, p2, p3, p4, p5, p6 = prices[[i1, i2, i3, i4, i5, i6]]
    direction = "up" if p6 > p1 else "down"

    def _ratio(val, base):
        if base == 0:
            return 0.0
        return abs(val / base)

    if direction == "up":
        w1 = p2 - p1
        w2 = p3 - p2
        w3 = p4 - p3
        w4 = p5 - p4
        w5 = p6 - p5
    else:
        w1 = p1 - p2
        w2 = p2 - p3
        w3 = p3 - p4
        w4 = p4 - p5
        w5 = p5 - p6

    score = 0.0
    checks = 0

    # Wave 2 retrace of wave 1 ~ 0.5–0.618
    retr2 = _ratio(w2, w1)
    if 0.4 <= retr2 <= 0.7:
        score += 1.0
    checks += 1

    # Wave 3 extension of wave 1 ~ >=1.2
    ext3 = _ratio(w3, w1)
    if ext3 >= 1.2:
        score += 1.0
    checks += 1

    # Wave 4 retrace of wave 3 ~ 0.2–0.5
    retr4 = _ratio(w4, w3)
    if 0.2 <= retr4 <= 0.6:
        score += 1.0
    checks += 1

    # Wave 5 similar to wave 1
    rel5 = _ratio(w5, w1)
    if 0.6 <= rel5 <= 1.6:
        score += 1.0
    checks += 1

    return score / max(checks, 1)


def find_primary_wave_count(prices: np.ndarray) -> WaveCount | None:
    """
    Attempt to find a basic 5-wave impulse at the right edge of the series.
    This is intentionally conservative and only returns a count when
    rules are satisfied.
    """
    pivots = _find_pivots(prices, window=3)
    if len(pivots) < 6:
        return None

    cand = pivots[-6:]
    if not _enforce_impulse_rules(prices, cand):
        return None

    score = _guideline_score(prices, cand)
    i1, i2, i3, i4, i5, i6 = cand
    direction = "up" if prices[i6] > prices[i1] else "down"

    return WaveCount(
        impulse_labels={"1": i1, "2": i2, "3": i3, "4": i4, "5": i6},
        correction_labels={},  # basic version: no ABC detection yet
        direction=direction,
        score=score,
        valid=True,
    )


# --- RSI / MACD / Divergences -------------------------------------------------


def compute_rsi(prices: np.ndarray, period: int = 14) -> np.ndarray:
    if len(prices) < period + 1:
        return np.full_like(prices, np.nan, dtype=float)
    deltas = np.diff(prices)
    gains = np.clip(deltas, 0, None)
    losses = -np.clip(deltas, None, 0)
    avg_gain = np.convolve(gains, np.ones(period), "full")[: len(gains)] / period
    avg_loss = np.convolve(losses, np.ones(period), "full")[: len(losses)] / period
    avg_loss[avg_loss == 0] = 1e-12
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = np.concatenate([[np.nan], rsi])  # align with prices length
    return rsi


def compute_macd(
    prices: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(prices) < slow + signal:
        nan_arr = np.full_like(prices, np.nan, dtype=float)
        return nan_arr, nan_arr, nan_arr

    def ema(x: np.ndarray, span: int) -> np.ndarray:
        alpha = 2 / (span + 1)
        out = np.zeros_like(x, dtype=float)
        out[0] = x[0]
        for i in range(1, len(x)):
            out[i] = alpha * x[i] + (1 - alpha) * out[i - 1]
        return out

    ema_fast = ema(prices, fast)
    ema_slow = ema(prices, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def detect_divergences(
    prices: np.ndarray, indicator: np.ndarray, lookback: int = 40
) -> List[Dict[str, Any]]:
    """
    Very simple divergence detection on the last couple of highs/lows.
    Returns a list of dicts with fields:
      type: "bullish" or "bearish"
      price_idx: index of the second pivot
    """
    res: List[Dict[str, Any]] = []
    if len(prices) < lookback + 5:
        return res

    start = len(prices) - lookback
    p = prices[start:]
    ind = indicator[start:]

    pivots = _find_pivots(p, window=2)
    if len(pivots) < 4:
        return res

    # Compare last two highs and last two lows
    highs = [i for i in pivots if p[i] == max(p[max(0, i - 2) : i + 3])]
    lows = [i for i in pivots if p[i] == min(p[max(0, i - 2) : i + 3])]

    if len(highs) >= 2:
        h1, h2 = highs[-2], highs[-1]
        if p[h2] > p[h1] and ind[h2] < ind[h1]:
            res.append({"type": "bearish", "price_idx": start + h2})

    if len(lows) >= 2:
        l1, l2 = lows[-2], lows[-1]
        if p[l2] < p[l1] and ind[l2] > ind[l1]:
            res.append({"type": "bullish", "price_idx": start + l2})

    return res

