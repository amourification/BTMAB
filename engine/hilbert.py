# =============================================================================
#  engine/hilbert.py — Equation 4: Hilbert Transform & Instantaneous Phase
#  Pinpoints the exact moment of a trend reversal by computing the
#  instantaneous phase of the detrended cycle in radians and degrees.
#
#  Phase Classifications (from research doc):
#    0°–90°   → Early Bullish Phase    (enter longs)
#    90°–180° → Mid-Cycle Expansion    (hold / trail stop)
#    180°–270°→ Distribution/Top       (hedge / take profit)
#    270°–360°→ Accumulation/Bottom    (prepare for new cycle)
#
#  Standard interface:
#      result = run(detrended, config) -> dict
# =============================================================================

import logging
import numpy as np
from scipy.signal import hilbert

logger = logging.getLogger("temporal_bot.engine.hilbert")

# Phase boundary constants (degrees) — from research doc table
PHASE_BOUNDARIES = {
    "early_bullish":  (0,   90),
    "mid_expansion":  (90,  180),
    "distribution":   (180, 270),
    "accumulation":   (270, 360),
}

TURN_TYPE_LABELS = {
    "early_bullish": "Early Bullish — Build Long Position",
    "mid_expansion": "Mid-Cycle Expansion — Ride Trend",
    "distribution":  "Distribution / Cyclical Top — Hedge / Take Profit",
    "accumulation":  "Accumulation / Cyclical Bottom — Prepare for New Cycle",
}

TURN_URGENCY = {
    "early_bullish": "low",
    "mid_expansion": "low",
    "distribution":  "high",    # action required
    "accumulation":  "medium",
}


# ── Internal helpers ──────────────────────────────────────────────────────────

def _apply_hilbert(signal: np.ndarray) -> tuple:
    """
    Applies the Hilbert Transform to extract the analytic signal.

    analytic_signal = signal + j * H(signal)
    where H is the Hilbert Transform (90° phase-shifted version).

    Returns:
        analytic   : complex np.ndarray
        amplitude  : instantaneous amplitude (envelope)
        phase_rad  : instantaneous phase in radians [−π, π]
        phase_deg  : unwrapped phase in degrees [0, 360) for last sample
    """
    analytic   = hilbert(signal)
    amplitude  = np.abs(analytic)
    phase_rad  = np.angle(analytic)              # wrapped: [−π, π]
    phase_unwrapped = np.unwrap(phase_rad)       # continuous phase

    # Convert last phase to [0, 360) for turn-type classification
    last_phase_deg = float((phase_rad[-1] % (2 * np.pi)) * 180 / np.pi)

    return analytic, amplitude, phase_rad, phase_unwrapped, last_phase_deg


def _classify_phase(phase_deg: float) -> str:
    """
    Maps a phase angle in degrees [0, 360) to a turn-type category.
    """
    phase_deg = phase_deg % 360
    for label, (lo, hi) in PHASE_BOUNDARIES.items():
        if lo <= phase_deg < hi:
            return label
    return "accumulation"   # 360° wraps to 0° → accumulation zone


def _bars_to_next_phase_boundary(
    phase_unwrapped: np.ndarray,
    current_phase_deg: float,
    period: float,
) -> dict:
    """
    Estimates how many bars until the next major phase boundary (90°, 180°, 270°, 360°).

    Uses the instantaneous frequency (derivative of unwrapped phase) to estimate
    the current rate of phase progression per bar.

    Returns dict: {next_boundary_deg, bars_to_boundary, next_turn_type}
    """
    if period < 2 or len(phase_unwrapped) < 5:
        return {"next_boundary_deg": 0, "bars_to_boundary": 0, "next_turn_type": "unknown"}

    # Instantaneous frequency: d(phase)/dt in radians per bar
    d_phase     = np.diff(phase_unwrapped)
    inst_freq   = float(np.median(d_phase[-20:])) if len(d_phase) >= 20 else float(d_phase.mean())
    deg_per_bar = float(inst_freq * 180 / np.pi)

    if abs(deg_per_bar) < 0.01:
        deg_per_bar = 360.0 / max(period, 1)   # fallback: uniform phase progression

    # Find the next boundary ahead
    boundaries      = [90.0, 180.0, 270.0, 360.0, 450.0]   # 450 = 90° of next cycle
    current_wrapped = current_phase_deg % 360

    for b in boundaries:
        delta = b - current_wrapped
        if delta < 0:
            delta += 360
        if delta > 0:
            bars = delta / abs(deg_per_bar)
            next_type_deg = b % 360
            next_type = _classify_phase(next_type_deg + 0.01)   # just past boundary
            return {
                "next_boundary_deg": float(b % 360),
                "bars_to_boundary":  round(bars, 1),
                "next_turn_type":    next_type,
            }

    return {"next_boundary_deg": 0, "bars_to_boundary": 0, "next_turn_type": "unknown"}


def _phase_history(phase_rad: np.ndarray, window: int = 20) -> dict:
    """
    Summarises recent phase dynamics to detect acceleration or deceleration
    near turning points.
    """
    if len(phase_rad) < window:
        window = len(phase_rad)

    recent    = phase_rad[-window:]
    d_phase   = np.diff(recent)
    mean_freq = float(d_phase.mean())
    std_freq  = float(d_phase.std())

    # Acceleration: second derivative of phase
    if len(d_phase) > 1:
        accel = float(np.diff(d_phase).mean())
    else:
        accel = 0.0

    return {
        "mean_freq_rad_per_bar": mean_freq,
        "std_freq":              std_freq,
        "acceleration":          accel,
        "is_decelerating":       accel < 0,
    }


def _compute_confidence(
    amplitude: np.ndarray,
    phase_unwrapped: np.ndarray,
    turn_type: str,
) -> float:
    """
    Confidence of the Hilbert phase reading:
    - High amplitude relative to its own mean = cleaner analytic signal
    - Stable instantaneous frequency (low std) = reliable phase estimate
    - Phase in 'distribution' or 'accumulation' = highest actionability
    """
    # Amplitude stability: current amplitude vs recent mean
    recent_amp = amplitude[-20:] if len(amplitude) >= 20 else amplitude
    amp_ratio  = float(amplitude[-1] / (recent_amp.mean() + 1e-12))
    amp_score  = float(min(amp_ratio, 2.0) / 2.0)   # normalise to [0, 1]

    # Phase stability: low std of instantaneous frequency = stable
    d_phase      = np.diff(phase_unwrapped[-20:]) if len(phase_unwrapped) >= 21 else np.diff(phase_unwrapped)
    freq_std     = float(d_phase.std())
    freq_score   = float(max(0.0, 1.0 - freq_std / (np.pi + 1e-12)))

    # Actionability bonus for reversal zones
    action_bonus = 0.15 if turn_type in ("distribution", "accumulation") else 0.0

    confidence = float(0.5 * amp_score + 0.35 * freq_score + action_bonus)
    return round(max(0.0, min(1.0, confidence)), 4)


# ── Public interface ──────────────────────────────────────────────────────────

def run(detrended: np.ndarray, cfg: dict) -> dict:
    """
    Applies the Hilbert Transform to compute instantaneous phase and
    classify the current market turn type.

    Parameters
    ----------
    detrended : np.ndarray — detrended signal from detrend.py
    cfg       : dict — supports key "DOMINANT_CYCLE_BARS" (float, default 512)
                       and optionally "FFT_PRIMARY_PERIOD" from fft.run()

    Returns
    -------
    dict with keys:
        "success"           : bool
        "analytic_signal"   : np.ndarray complex
        "amplitude"         : np.ndarray  — instantaneous amplitude (envelope)
        "phase_rad"         : np.ndarray  — wrapped phase [−π, π]
        "phase_unwrapped"   : np.ndarray  — continuous unwrapped phase
        "phase_deg"         : float       — current phase [0, 360)
        "turn_type"         : str         — category key
        "turn_label"        : str         — human-readable label
        "turn_urgency"      : str         — "low" | "medium" | "high"
        "bars_to_next_turn" : dict        — next_boundary_deg, bars_to_boundary
        "phase_history"     : dict        — frequency dynamics
        "confidence"        : float       — [0, 1] for aggregator
        "metadata"          : dict        — N, period used
        "error"             : str | None
    """
    _empty = {
        "success":           False,
        "analytic_signal":   np.array([]),
        "amplitude":         np.array([]),
        "phase_rad":         np.array([]),
        "phase_unwrapped":   np.array([]),
        "phase_deg":         0.0,
        "turn_type":         "unknown",
        "turn_label":        "Unknown",
        "turn_urgency":      "low",
        "bars_to_next_turn": {},
        "phase_history":     {},
        "confidence":        0.0,
        "metadata":          {},
        "error":             None,
    }

    if detrended is None or len(detrended) < 10:
        _empty["error"] = f"Hilbert needs >= 10 samples, got {len(detrended) if detrended is not None else 0}."
        logger.error(_empty["error"])
        return _empty

    period = float(cfg.get("FFT_PRIMARY_PERIOD", cfg.get("DOMINANT_CYCLE_BARS", 512)))

    try:
        analytic, amplitude, phase_rad, phase_unwrapped, phase_deg = \
            _apply_hilbert(detrended)

        turn_type  = _classify_phase(phase_deg)
        turn_label = TURN_TYPE_LABELS[turn_type]
        urgency    = TURN_URGENCY[turn_type]

        bars_to_next = _bars_to_next_phase_boundary(phase_unwrapped, phase_deg, period)
        hist         = _phase_history(phase_rad)
        confidence   = _compute_confidence(amplitude, phase_unwrapped, turn_type)

        logger.info(
            "Hilbert OK: phase=%.1f° turn=%s urgency=%s bars_to_turn=%.1f confidence=%.3f",
            phase_deg, turn_type, urgency,
            bars_to_next.get("bars_to_boundary", 0), confidence,
        )

        return {
            "success":           True,
            "analytic_signal":   analytic,
            "amplitude":         amplitude,
            "phase_rad":         phase_rad,
            "phase_unwrapped":   phase_unwrapped,
            "phase_deg":         phase_deg,
            "turn_type":         turn_type,
            "turn_label":        turn_label,
            "turn_urgency":      urgency,
            "bars_to_next_turn": bars_to_next,
            "phase_history":     hist,
            "confidence":        confidence,
            "metadata":          {"N": len(detrended), "period": period},
            "error":             None,
        }

    except Exception as exc:
        msg = f"{type(exc).__name__}: {exc}"
        logger.exception("Hilbert failed: %s", msg)
        _empty["error"] = msg
        return _empty


# ── Smoke test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    np.random.seed(42)
    n      = 300
    t      = np.linspace(0, 6 * np.pi, n)
    signal = 10 * np.sin(t) + np.random.randn(n) * 0.5

    result = run(signal, {"DOMINANT_CYCLE_BARS": 100})
    if result["success"]:
        print(f"✅ Hilbert OK")
        print(f"   Phase          : {result['phase_deg']:.1f}°")
        print(f"   Turn type      : {result['turn_type']}")
        print(f"   Turn label     : {result['turn_label']}")
        print(f"   Urgency        : {result['turn_urgency']}")
        print(f"   Next turn      : {result['bars_to_next_turn']}")
        print(f"   Confidence     : {result['confidence']:.3f}")
    else:
        print(f"❌ {result['error']}")
