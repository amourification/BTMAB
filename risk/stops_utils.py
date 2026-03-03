# =============================================================================
#  risk/stops_utils.py — Stop Management Internal Helpers
#  Split from stops.py to keep both modules under 300 lines.
# =============================================================================
import numpy as np

ATR_PERIOD:          int   = 14
ATR_MULTIPLIER_BASE: float = 2.0
MIN_STOP_PCT:        float = 0.005
MAX_STOP_PCT:        float = 0.15


def compute_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> float:
    if len(close) < period + 1:
        return float((high - low)[-period:].mean()) if len(high) >= period else float(close[-1] * 0.02)
    tr = np.array([max(float(high[i]-low[i]), abs(float(high[i]-close[i-1])),
                       abs(float(low[i]-close[i-1]))) for i in range(1, len(close))])
    return float(tr[-period:].mean())


def phase_stop_multiplier(phase_deg: float, turn_type: str) -> float:
    phase = phase_deg % 360
    if turn_type == "distribution"  or (180 <= phase < 270): return 1.5
    if turn_type == "accumulation"  or (270 <= phase < 360): return 1.0
    if turn_type == "mid_expansion" or (90  <= phase < 180): return 2.5
    return 3.0


def nearest_murray_stop(entry: float, atr_stop: float, levels: np.ndarray, direction: str) -> float:
    if levels is None or len(levels) == 0:
        return atr_stop
    if direction == "long":
        cands = [l for l in levels if l < atr_stop]
        return float(max(cands)) if cands else atr_stop
    else:
        cands = [l for l in levels if l > atr_stop]
        return float(min(cands)) if cands else atr_stop


def initial_stop(entry: float, atr: float, mult: float, levels: np.ndarray, direction: str) -> dict:
    dist = float(np.clip(atr * mult, entry * MIN_STOP_PCT, entry * MAX_STOP_PCT))
    raw  = entry - dist if direction == "long" else entry + dist
    snap = nearest_murray_stop(entry, raw, levels, direction)
    return {"stop_price": round(snap, 4), "atr_stop": round(raw, 4),
            "stop_pct": round(abs(entry - snap) / entry * 100, 3),
            "atr_distance": round(dist, 4), "stop_type": "initial"}


def trailing_stop(prices: np.ndarray, entry: float, atr: float, phase: float,
                  direction: str, levels: np.ndarray) -> dict:
    trail_dist = atr * max(1.0, (360 - phase) / 360 * 2.5)
    if direction == "long":
        raw = float(prices.max()) - trail_dist
    else:
        raw = float(prices.min()) + trail_dist
    snap = nearest_murray_stop(entry, raw, levels, direction)
    snap = max(snap, entry * (1 - MAX_STOP_PCT)) if direction == "long" \
           else min(snap, entry * (1 + MAX_STOP_PCT))
    locked = (float(prices[-1]) - snap) / float(prices[-1]) * 100
    return {"stop_price": round(snap, 4), "stop_type": "trailing",
            "trail_dist": round(trail_dist, 4), "locked_pct": round(locked, 3),
            "peak_price": round(float(prices.max() if direction == "long" else prices.min()), 4)}


def emergency_stop(prices: np.ndarray, walras: dict) -> dict:
    shock = walras.get("liquidity_shock", {})
    trig  = shock.get("shock_detected", False) and shock.get("shock_magnitude", 0) > 5.0
    return {"triggered": trig, "stop_price": round(float(prices[-1]), 4) if trig else None,
            "reason": f"Liquidity shock: {shock.get('shock_magnitude',0):.1f}σ" if trig else None,
            "stop_type": "emergency"}


def stop_schedule(phase: float, entry: float, atr: float, direction: str) -> list:
    deg_per_bar = 360 / 50
    sched = []
    for pt in [180, 270, 315, 360]:
        if pt <= phase % 360:
            continue
        bars = round((pt - phase % 360) / deg_per_bar)
        mult = phase_stop_multiplier(float(pt), "")
        proj = entry - atr * mult if direction == "long" else entry + atr * mult
        sched.append({"phase_target": pt, "bars_ahead": bars,
                      "projected_stop": round(proj, 4), "multiplier": mult})
    return sched
