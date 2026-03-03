# =============================================================================
#  engine/fft.py — Equation 3: Fast Fourier Transform (FFT)
#  Transforms the detrended signal into the frequency domain to identify
#  dominant cycles, amplitudes, and current phase position.
#
#  Key outputs: primary cycle period, 512-day macro cycle proximity,
#  and "Day N of cycle" position for the Telegram/GUI output.
#
#  Heavy spectral helpers live in fft_utils.py.
#
#  Standard interface:
#      result = run(detrended, config) -> dict
# =============================================================================

import logging
import numpy as np
from engine.fft_utils import (
    compute_power_spectrum,
    find_dominant_cycles,
    estimate_cycle_phase,
    spectral_reliability,
)

logger = logging.getLogger("temporal_bot.engine.fft")

DEFAULT_MIN_PERIOD: int = 10


def run(detrended: np.ndarray, cfg: dict) -> dict:
    """
    Computes the FFT power spectrum and identifies dominant market cycles.

    Parameters
    ----------
    detrended : np.ndarray — detrended signal from detrend.py
    cfg       : dict — keys: "FFT_MIN_PERIOD" (int), "DOMINANT_CYCLE_BARS" (int)

    Returns
    -------
    dict with keys:
        success, freqs, periods, power, amplitude,
        dominant_cycles, primary_cycle, macro_cycle,
        cycle_phase, spectral_reliability, confidence,
        metadata, error
    """
    _empty = {
        "success":              False,
        "freqs":                np.array([]),
        "periods":              np.array([]),
        "power":                np.array([]),
        "amplitude":            np.array([]),
        "dominant_cycles":      [],
        "primary_cycle":        {},
        "macro_cycle":          {},
        "cycle_phase":          {},
        "spectral_reliability": 0.0,
        "confidence":           0.0,
        "metadata":             {},
        "error":                None,
    }

    if detrended is None or len(detrended) < 20:
        _empty["error"] = (
            f"FFT needs >= 20 samples, "
            f"got {len(detrended) if detrended is not None else 0}."
        )
        logger.error(_empty["error"])
        return _empty

    N          = len(detrended)
    min_period = int(cfg.get("FFT_MIN_PERIOD", DEFAULT_MIN_PERIOD))
    macro_bars = int(cfg.get("DOMINANT_CYCLE_BARS", 512))

    try:
        # ── Power spectrum ────────────────────────────────────────────────────
        freqs, periods, power, amplitude = compute_power_spectrum(detrended)

        # ── Dominant cycles ───────────────────────────────────────────────────
        dom_cycles = find_dominant_cycles(
            freqs, periods, power, amplitude, min_period, N
        )

        if not dom_cycles:
            _empty["error"] = "No dominant cycles found above noise floor."
            logger.warning(_empty["error"])
            return _empty

        primary_cycle = dom_cycles[0]

        # ── Macro cycle: closest to DOMINANT_CYCLE_BARS ───────────────────────
        macro_cycle = min(
            dom_cycles, key=lambda c: abs(c["period"] - macro_bars)
        )

        # ── Current phase of primary cycle ────────────────────────────────────
        cycle_phase = estimate_cycle_phase(detrended, primary_cycle["period"])

        # ── Spectral reliability ──────────────────────────────────────────────
        total_pwr  = float(power.sum())
        reliability = spectral_reliability(primary_cycle["power"], total_pwr)

        # ── Confidence: 60% reliability + 40% primary strength ───────────────
        confidence = round(
            max(0.0, min(1.0, 0.6 * reliability + 0.4 * primary_cycle["strength"])),
            4,
        )

        logger.info(
            "FFT OK: primary=%.1f bars macro=%.1f bars "
            "reliability=%.3f confidence=%.3f",
            primary_cycle["period"], macro_cycle["period"],
            reliability, confidence,
        )

        return {
            "success":              True,
            "freqs":                freqs,
            "periods":              periods,
            "power":                power,
            "amplitude":            amplitude,
            "dominant_cycles":      dom_cycles,
            "primary_cycle":        primary_cycle,
            "macro_cycle":          macro_cycle,
            "cycle_phase":          cycle_phase,
            "spectral_reliability": reliability,
            "confidence":           confidence,
            "metadata":             {"N": N, "min_period": min_period},
            "error":                None,
        }

    except Exception as exc:
        msg = f"{type(exc).__name__}: {exc}"
        logger.exception("FFT failed: %s", msg)
        _empty["error"] = msg
        return _empty


# ── Smoke test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    np.random.seed(3)
    n = 512
    t = np.arange(n)
    signal = (20 * np.sin(2 * np.pi * t / 128)
              + 5 * np.sin(2 * np.pi * t / 32)
              + np.random.randn(n) * 3)

    result = run(signal, {"FFT_MIN_PERIOD": 10, "DOMINANT_CYCLE_BARS": 128})
    if result["success"]:
        print(f"✅ FFT OK")
        print(f"   Primary : {result['primary_cycle']['period']:.1f} bars "
              f"(strength {result['primary_cycle']['strength']:.2f})")
        print(f"   Macro   : {result['macro_cycle']['period']:.1f} bars")
        print(f"   Phase   : {result['cycle_phase']}")
        print(f"   Reliability: {result['spectral_reliability']:.3f}")
        print(f"   Confidence : {result['confidence']:.3f}")
    else:
        print(f"❌ {result['error']}")
