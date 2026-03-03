# =============================================================================
#  engine/ssa_utils.py — Equation 1b: SSA Confidence & Cycle Strength
#  Wraps ssa_core.py and adds the Frobenius norm separability check,
#  cycle strength scoring, and the final confidence score consumed by
#  aggregator.py and the Kelly Criterion (kelly.py).
#
#  Standard interface:
#      result = run(detrended, config) -> dict   ← calls ssa_core internally
# =============================================================================

import logging
import numpy as np
from engine.ssa_core import run as ssa_core_run

logger = logging.getLogger("temporal_bot.engine.ssa_utils")


# ── Frobenius norm helpers ────────────────────────────────────────────────────

def _frobenius_norm(matrix: np.ndarray) -> float:
    """
    Computes the Frobenius norm: sqrt(sum of squared elements).
    ||A||_F = sqrt(Σ a_ij²)
    """
    return float(np.sqrt(np.sum(matrix ** 2)))


def _low_rank_approximation(
    U: np.ndarray,
    eigenvalues: np.ndarray,
    V: np.ndarray,
    indices: list[int],
    L: int,
    K: int,
) -> np.ndarray:
    """
    Reconstructs the trajectory matrix using only the selected component indices.
    Used to measure how well oscillation components explain the original matrix.

    X_approx = Σ_{i ∈ indices} sqrt(λ_i) * U_i * V_i^T
    """
    X_approx = np.zeros((L, K))
    sigma = np.sqrt(np.maximum(eigenvalues, 0))
    for i in indices:
        if i < U.shape[1] and i < len(sigma):
            X_approx += sigma[i] * np.outer(U[:, i], V[:, i])
    return X_approx


def _separability_score(
    X: np.ndarray,
    U: np.ndarray,
    eigenvalues: np.ndarray,
    V: np.ndarray,
    osc_indices: list[int],
) -> float:
    """
    Frobenius norm separability check (from research doc):
    Measures what fraction of the original matrix's energy is captured
    by the oscillation components.

    separability = 1 - (||X - X_osc||_F / ||X||_F)

    A high score (close to 1) means the cycle is "dominant" and
    statistically significant rather than random fluctuation.
    """
    if not osc_indices:
        return 0.0

    L, K     = X.shape
    X_osc    = _low_rank_approximation(U, eigenvalues, V, osc_indices, L, K)
    norm_X   = _frobenius_norm(X)
    norm_err = _frobenius_norm(X - X_osc)

    if norm_X < 1e-12:
        return 0.0

    score = 1.0 - (norm_err / norm_X)
    return float(max(0.0, min(1.0, score)))


def _cycle_strength(variance_pct: np.ndarray, osc_indices: list[int]) -> float:
    """
    Sum of variance percentages of the oscillation components.
    Normalised to [0, 1]. High = cycle is statistically significant.
    """
    if not osc_indices or len(variance_pct) == 0:
        return 0.0
    total_osc_var = sum(
        variance_pct[i] for i in osc_indices if i < len(variance_pct)
    )
    return float(min(total_osc_var / 100.0, 1.0))


def _estimate_cycle_position(reconstruction: np.ndarray) -> dict:
    """
    Estimates where we currently are within the identified cycle.

    Finds the last local maximum and minimum in the reconstruction
    to determine if we're in the ascending or descending half,
    and estimates bars-until-next-turn.
    """
    if len(reconstruction) < 10:
        return {"position": "unknown", "bars_to_next_turn": 0, "phase_pct": 0.0}

    # Simple peak/trough detection: compare last value to recent mean
    last    = reconstruction[-1]
    recent  = reconstruction[-20:] if len(reconstruction) >= 20 else reconstruction
    mean_r  = recent.mean()
    std_r   = recent.std()

    if last > mean_r + 0.3 * std_r:
        position = "near_peak"
    elif last < mean_r - 0.3 * std_r:
        position = "near_trough"
    elif last > mean_r:
        position = "ascending"
    else:
        position = "descending"

    # Rough phase % based on recent range
    rng = reconstruction.max() - reconstruction.min()
    if rng > 1e-12:
        phase_pct = float((last - reconstruction.min()) / rng)
    else:
        phase_pct = 0.5

    return {
        "position":          position,
        "bars_to_next_turn": 0,    # refined by hilbert.py
        "phase_pct":         phase_pct,
    }


# ── Public interface ──────────────────────────────────────────────────────────

def run(detrended: np.ndarray, cfg: dict) -> dict:
    """
    Full SSA pipeline: core decomposition + confidence scoring.
    This is the module that aggregator.py and the GUI call directly.

    Parameters
    ----------
    detrended : np.ndarray — detrended price signal from detrend.py
    cfg       : dict — "SSA_WINDOW_LENGTH", "SSA_NUM_COMPONENTS"

    Returns
    -------
    dict — all keys from ssa_core.run() PLUS:
        "separability"    : float  — Frobenius norm score [0, 1]
        "cycle_strength"  : float  — variance % of oscillation components [0, 1]
        "cycle_position"  : dict   — position / phase_pct / bars_to_next_turn
        "confidence"      : float  — combined confidence score [0, 1]
        "summary"         : str    — human-readable status line
    """
    # ── Run core ──────────────────────────────────────────────────────────────
    core = ssa_core_run(detrended, cfg)

    if not core["success"]:
        core["separability"]   = 0.0
        core["cycle_strength"] = 0.0
        core["cycle_position"] = {}
        core["summary"]        = f"SSA failed: {core['error']}"
        return core

    # ── Extract needed objects ─────────────────────────────────────────────────
    X            = core["trajectory_matrix"]
    U            = core["left_svecs"]
    eigenvalues  = core["eigenvalues"]
    V            = core["right_svecs"]
    variance_pct = core["variance_pct"]
    osc_indices  = core["groups"].get("oscillation", [])
    reconstruction = core["reconstruction"]

    # ── Frobenius separability score ──────────────────────────────────────────
    separability = _separability_score(X, U, eigenvalues, V, osc_indices)

    # ── Cycle strength (variance-based) ──────────────────────────────────────
    strength = _cycle_strength(variance_pct, osc_indices)

    # ── Cycle position ─────────────────────────────────────────────────────────
    cycle_position = _estimate_cycle_position(reconstruction)

    # ── Combined confidence ────────────────────────────────────────────────────
    # Weighted: 60% Frobenius separability + 40% variance strength
    confidence = float(0.60 * separability + 0.40 * strength)
    confidence = round(max(0.0, min(1.0, confidence)), 4)

    # ── Human-readable summary ────────────────────────────────────────────────
    pct_str = f"{confidence * 100:.0f}%"
    period  = core["dominant_period"]
    pos     = cycle_position["position"]
    summary = (
        f"Dominant cycle: {period:.0f} bars | "
        f"Position: {pos} | "
        f"Confidence: {pct_str}"
    )

    logger.info(
        "SSA Utils: separability=%.3f strength=%.3f confidence=%.3f period=%.1f",
        separability, strength, confidence, period,
    )

    # Merge into core result
    core["separability"]   = separability
    core["cycle_strength"] = strength
    core["cycle_position"] = cycle_position
    core["confidence"]     = confidence
    core["summary"]        = summary

    return core


# ── Smoke test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    np.random.seed(0)
    n      = 300
    t      = np.linspace(0, 8 * np.pi, n)
    signal = 10 * np.sin(t) + 3 * np.sin(3 * t) + np.random.randn(n)

    result = run(signal, {"SSA_WINDOW_LENGTH": 50, "SSA_NUM_COMPONENTS": 6})
    if result["success"]:
        print(f"✅ SSA Utils OK")
        print(f"   Separability  : {result['separability']:.3f}")
        print(f"   Cycle Strength: {result['cycle_strength']:.3f}")
        print(f"   Confidence    : {result['confidence']:.3f}")
        print(f"   Position      : {result['cycle_position']}")
        print(f"   Summary       : {result['summary']}")
    else:
        print(f"❌ {result['error']}")
