# =============================================================================
#  engine/fft_utils.py — FFT Internal Helpers
#  Split from fft.py to keep both modules under the 300-line limit.
#  Not intended to be called directly by other modules.
# =============================================================================

import numpy as np
from scipy.signal import find_peaks

DEFAULT_TOP_N: int = 5


def compute_power_spectrum(signal: np.ndarray) -> tuple:
    """
    Computes the one-sided power spectrum via real FFT with Hanning window.

    Returns: freqs, periods, power, amplitude (all np.ndarray)
    """
    N        = len(signal)
    window   = np.hanning(N)
    fft_vals = np.fft.rfft(signal * window)
    freqs    = np.fft.rfftfreq(N)
    amplitude = np.abs(fft_vals)
    power     = amplitude ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        periods = np.where(freqs > 0, 1.0 / freqs, np.inf)
    return freqs, periods, power, amplitude


def find_dominant_cycles(
    freqs: np.ndarray,
    periods: np.ndarray,
    power: np.ndarray,
    amplitude: np.ndarray,
    min_period: int,
    n_bars: int,
    top_n: int = DEFAULT_TOP_N,
) -> list[dict]:
    """
    Returns the top_n dominant cycles by power within valid period range.
    """
    valid = (periods >= min_period) & (periods <= n_bars // 2)
    if not valid.any():
        return []

    vi   = np.where(valid)[0]
    vp   = power[vi]
    vper = periods[vi]
    va   = amplitude[vi]
    vf   = freqs[vi]

    peaks, _ = find_peaks(vp, height=vp.max() * 0.05)
    if len(peaks) == 0:
        peaks = np.argsort(vp)[::-1][:top_n]

    peaks = sorted(peaks, key=lambda i: vp[i], reverse=True)
    total = vp.sum() + 1e-12

    return [
        {
            "period":    float(vper[i]),
            "frequency": float(vf[i]),
            "power":     float(vp[i]),
            "amplitude": float(va[i]),
            "power_pct": float(vp[i] / total * 100),
            "strength":  float(min(va[i] / (va.max() + 1e-12), 1.0)),
        }
        for i in peaks[:top_n]
    ]


def estimate_cycle_phase(detrended: np.ndarray, period: float) -> dict:
    """
    Estimates current phase of a specific cycle via least-squares sine fit.
    Returns: day_in_cycle, pct_complete, phase_rad
    """
    if period < 2:
        return {"day_in_cycle": 0, "pct_complete": 0.0, "phase_rad": 0.0}

    N     = len(detrended)
    t     = np.arange(N)
    omega = 2 * np.pi / period
    M     = np.column_stack([np.cos(omega * t), np.sin(omega * t)])

    try:
        coeffs, _, _, _ = np.linalg.lstsq(M, detrended, rcond=None)
        A, B = coeffs
    except np.linalg.LinAlgError:
        return {"day_in_cycle": 0, "pct_complete": 0.0, "phase_rad": 0.0}

    phase_rad = float(np.arctan2(B, A))
    if phase_rad < 0:
        phase_rad += 2 * np.pi

    return {
        "day_in_cycle":  round(float(phase_rad / (2 * np.pi) * period), 1),
        "pct_complete":  round(float(phase_rad / (2 * np.pi)), 4),
        "phase_rad":     round(phase_rad, 4),
    }


def spectral_reliability(dominant_power: float, total_power: float) -> float:
    """Fraction of total spectral power in the dominant cycle. [0, 1]"""
    if total_power < 1e-12:
        return 0.0
    return float(min(dominant_power / total_power, 1.0))
