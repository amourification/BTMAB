# =============================================================================
#  engine/solar.py — Equation 5: Solar Cycle & Julian Date Synchronization
#  Synchronizes market time with astronomical cycles. Implements Gann's
#  belief that market movements reflect natural law — specifically the
#  solar cycle and its relationship to seasonal volatility patterns.
#
#  Computes sun's mean longitude (L₀), mean anomaly (M), and proximity
#  to solstices/equinoxes which historically coincide with volatility spikes.
#
#  Standard interface:
#      result = run(preprocessed_result, config) -> dict
# =============================================================================

import logging
import numpy as np

logger = logging.getLogger("temporal_bot.engine.solar")

# Solar cycle astronomical constants
SOLAR_PROXIMITY_DAYS: int = 7    # days within solstice/equinox = heightened flag

# Solstice / equinox mean longitudes (degrees)
EQUINOX_SPRING_DEG:  float = 0.0
SOLSTICE_SUMMER_DEG: float = 90.0
EQUINOX_AUTUMN_DEG:  float = 180.0
SOLSTICE_WINTER_DEG: float = 270.0

CARDINAL_POINTS = {
    "spring_equinox":  EQUINOX_SPRING_DEG,
    "summer_solstice": SOLSTICE_SUMMER_DEG,
    "autumn_equinox":  EQUINOX_AUTUMN_DEG,
    "winter_solstice": SOLSTICE_WINTER_DEG,
}


# ── Internal helpers ──────────────────────────────────────────────────────────

def _solar_mean_longitude(T: float) -> float:
    """
    Sun's mean longitude L₀ in degrees (Jean Meeus, Astronomical Algorithms).
    T = Julian centuries from J2000.0

    L₀ = 280.46646 + 36000.76983·T + 0.0003032·T²  (mod 360)
    """
    L0 = 280.46646 + 36000.76983 * T + 0.0003032 * T ** 2
    return float(L0 % 360)


def _solar_mean_anomaly(T: float) -> float:
    """
    Sun's mean anomaly M in degrees.
    M = 357.52911 + 35999.05029·T − 0.0001537·T²  (mod 360)
    """
    M = 357.52911 + 35999.05029 * T - 0.0001537 * T ** 2
    return float(M % 360)


def _equation_of_center(M_deg: float) -> float:
    """
    Equation of center C — correction from mean to true anomaly.
    C = (1.9146 − 0.004817·T − 0.000014·T²)·sin(M)
      + (0.019993 − 0.000101·T)·sin(2M)
      + 0.00029·sin(3M)

    Simplified here with T≈0 (modern era, error < 0.01°):
    """
    M_rad = np.radians(M_deg)
    C = (1.9146 * np.sin(M_rad)
         + 0.019993 * np.sin(2 * M_rad)
         + 0.00029 * np.sin(3 * M_rad))
    return float(C)


def _true_longitude(L0: float, C: float) -> float:
    """Sun's true longitude = mean longitude + equation of center."""
    return float((L0 + C) % 360)


def _solar_sine_value(true_lon: float) -> float:
    """
    Converts true longitude to a sine-wave value in [−1, 1].
    Used as the "solar transformed" signal that highlights seasonal bias.
    Peak (+1) at summer solstice (90°), trough (−1) at winter solstice (270°).
    """
    return float(np.sin(np.radians(true_lon)))


def _proximity_to_cardinal(true_lon: float, proximity_deg: float) -> dict:
    """
    Checks if the sun's true longitude is within `proximity_deg` of any
    cardinal point (solstice/equinox). Returns the nearest point and distance.

    proximity_deg corresponds to SOLAR_PROXIMITY_DAYS × ~1°/day solar motion.
    """
    nearest_name  = None
    nearest_dist  = 360.0
    triggered     = False

    for name, cardinal_lon in CARDINAL_POINTS.items():
        dist = abs(true_lon - cardinal_lon)
        # Handle wrap-around at 0°/360°
        dist = min(dist, 360 - dist)
        if dist < nearest_dist:
            nearest_dist  = dist
            nearest_name  = name
        if dist <= proximity_deg:
            triggered = True

    return {
        "nearest_cardinal": nearest_name,
        "degrees_from_cardinal": round(nearest_dist, 2),
        "volatility_flag": triggered,
    }


def _seasonal_bias(true_lon: float) -> str:
    """
    Returns a seasonal bias label based on solar longitude.
    Bias reflects historical tendencies observed in market data:
    - Q1 (0°–90°):   Spring rally bias (post-tax, new fiscal year flows)
    - Q2 (90°–180°): Summer drift — often lower volume
    - Q3 (180°–270°): Autumn volatility — historically high crash risk
    - Q4 (270°–360°): Year-end rally bias (window dressing, Santa rally)
    """
    if 0 <= true_lon < 90:
        return "spring_rally_bias"
    elif 90 <= true_lon < 180:
        return "summer_drift_bias"
    elif 180 <= true_lon < 270:
        return "autumn_volatility_bias"
    else:
        return "yearend_rally_bias"


def _compute_series(T_array: np.ndarray) -> dict:
    """
    Vectorized computation of solar values for the entire bar series.
    Returns arrays of L0, M, true_longitude, and sine values.
    """
    L0_arr     = np.array([_solar_mean_longitude(T) for T in T_array])
    M_arr      = np.array([_solar_mean_anomaly(T) for T in T_array])
    C_arr      = np.array([_equation_of_center(M) for M in M_arr])
    trulon_arr = (L0_arr + C_arr) % 360
    sine_arr   = np.sin(np.radians(trulon_arr))
    return {
        "L0":            L0_arr,
        "M":             M_arr,
        "true_longitude": trulon_arr,
        "solar_sine":    sine_arr,
    }


def _days_to_next_cardinal(true_lon: float) -> dict:
    """
    Estimates bars (days) until the next cardinal solar point.
    Sun moves ~1° per day, so degrees ≈ days.
    """
    results = {}
    for name, cardinal in CARDINAL_POINTS.items():
        delta = (cardinal - true_lon) % 360
        results[name] = round(delta, 1)   # degrees ≈ days
    nearest = min(results, key=results.get)
    return {"days_to_cardinals": results, "nearest_upcoming": nearest,
            "days_to_nearest": results[nearest]}


# ── Public interface ──────────────────────────────────────────────────────────

def run(preprocessed: dict, cfg: dict) -> dict:
    """
    Computes solar cycle positions for all bars and flags volatility zones.

    Parameters
    ----------
    preprocessed : dict — output of data/preprocessor.py run()
                   Must contain: "centurial_T" (np.ndarray)
    cfg          : dict — "SOLAR_PROXIMITY_DAYS" (int, default 7)

    Returns
    -------
    dict with keys:
        "success"              : bool
        "solar_sine"           : np.ndarray  — solar sine series [−1, 1]
        "true_longitude_series": np.ndarray  — sun's true longitude per bar
        "current_L0"           : float       — mean longitude (last bar)
        "current_M"            : float       — mean anomaly (last bar)
        "current_true_lon"     : float       — true longitude (last bar)
        "current_sine"         : float       — solar sine value (last bar)
        "seasonal_bias"        : str         — seasonal tendency label
        "cardinal_proximity"   : dict        — nearest cardinal + flag
        "days_to_cardinals"    : dict        — bars to each solstice/equinox
        "volatility_flag"      : bool        — True if near solstice/equinox
        "confidence"           : float       — [0, 1]
        "metadata"             : dict
        "error"                : str | None
    """
    _empty = {
        "success":               False,
        "solar_sine":            np.array([]),
        "true_longitude_series": np.array([]),
        "current_L0":            0.0,
        "current_M":             0.0,
        "current_true_lon":      0.0,
        "current_sine":          0.0,
        "seasonal_bias":         "unknown",
        "cardinal_proximity":    {},
        "days_to_cardinals":     {},
        "volatility_flag":       False,
        "confidence":            0.0,
        "metadata":              {},
        "error":                 None,
    }

    T_array = preprocessed.get("centurial_T")
    if T_array is None or len(T_array) == 0:
        _empty["error"] = "Solar: centurial_T array missing from preprocessed data."
        logger.error(_empty["error"])
        return _empty

    proximity_days = int(cfg.get("SOLAR_PROXIMITY_DAYS", SOLAR_PROXIMITY_DAYS))
    # Sun moves ~1°/day → convert days to degrees
    proximity_deg  = float(proximity_days)

    try:
        series      = _compute_series(T_array)
        T_last      = float(T_array[-1])
        L0_last     = _solar_mean_longitude(T_last)
        M_last      = _solar_mean_anomaly(T_last)
        C_last      = _equation_of_center(M_last)
        trulon_last = float((L0_last + C_last) % 360)
        sine_last   = _solar_sine_value(trulon_last)

        proximity   = _proximity_to_cardinal(trulon_last, proximity_deg)
        bias        = _seasonal_bias(trulon_last)
        days_cards  = _days_to_next_cardinal(trulon_last)

        # Confidence: higher near cardinal points (more historically significant)
        dist_norm   = min(proximity["degrees_from_cardinal"] / 90.0, 1.0)
        confidence  = round(float(1.0 - dist_norm * 0.5), 4)   # [0.5, 1.0]

        logger.info(
            "Solar OK: true_lon=%.1f° bias=%s volatility_flag=%s "
            "nearest=%s in %.1f days confidence=%.3f",
            trulon_last, bias, proximity["volatility_flag"],
            days_cards["nearest_upcoming"], days_cards["days_to_nearest"],
            confidence,
        )

        return {
            "success":               True,
            "solar_sine":            series["solar_sine"],
            "true_longitude_series": series["true_longitude"],
            "current_L0":            L0_last,
            "current_M":             M_last,
            "current_true_lon":      trulon_last,
            "current_sine":          sine_last,
            "seasonal_bias":         bias,
            "cardinal_proximity":    proximity,
            "days_to_cardinals":     days_cards,
            "volatility_flag":       proximity["volatility_flag"],
            "confidence":            confidence,
            "metadata":              {"n_bars": len(T_array), "proximity_deg": proximity_deg},
            "error":                 None,
        }

    except Exception as exc:
        msg = f"{type(exc).__name__}: {exc}"
        logger.exception("Solar failed: %s", msg)
        _empty["error"] = msg
        return _empty


# ── Smoke test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Simulate T values around 2024 (T ≈ 0.24)
    T_arr = np.linspace(0.23, 0.25, 100)
    fake_preprocessed = {"centurial_T": T_arr}

    result = run(fake_preprocessed, {"SOLAR_PROXIMITY_DAYS": 7})
    if result["success"]:
        print(f"✅ Solar OK")
        print(f"   True longitude : {result['current_true_lon']:.2f}°")
        print(f"   Solar sine     : {result['current_sine']:.4f}")
        print(f"   Seasonal bias  : {result['seasonal_bias']}")
        print(f"   Volatility flag: {result['volatility_flag']}")
        print(f"   Nearest cardinal: {result['cardinal_proximity']}")
        print(f"   Days to cards  : {result['days_to_cardinals']}")
        print(f"   Confidence     : {result['confidence']:.3f}")
    else:
        print(f"❌ {result['error']}")
