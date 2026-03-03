# =============================================================================
#  engine/ssa_core.py — Equation 1a: Singular Spectrum Analysis (Core)
#  Implements the SSA trajectory matrix, SVD decomposition, and component
#  reconstruction. The Frobenius norm confidence score lives in ssa_utils.py.
#
#  Theory: SSA decomposes a time series into trend, oscillatory components,
#  and noise WITHOUT assuming stationarity or sinusoidal shapes. This makes
#  it superior to FFT for non-stationary crypto price cycles.
#
#  Pipeline: detrend.py → ssa_core.py → ssa_utils.py
#
#  Standard interface:
#      result = run(detrended_signal, config) -> dict
# =============================================================================

import logging
import numpy as np
from numpy.linalg import svd

logger = logging.getLogger("temporal_bot.engine.ssa_core")


# ── Internal helpers ──────────────────────────────────────────────────────────

def _build_trajectory_matrix(signal: np.ndarray, L: int) -> np.ndarray:
    """
    Embeds the time series into an L × K trajectory (Hankel) matrix.

    Given signal Y of length N and window length L:
        K = N - L + 1
        X[i, j] = Y[i + j]   for i in [0, L), j in [0, K)

    Each column is a lagged copy of the signal — this "unfolding" in time
    allows SVD to extract structured oscillations from the embedded space.
    """
    N = len(signal)
    K = N - L + 1
    if K < 1:
        raise ValueError(f"Window length L={L} too large for signal length N={N}. Need L < N.")

    # Efficient Hankel construction via stride tricks
    X = np.zeros((L, K))
    for i in range(K):
        X[:, i] = signal[i: i + L]
    return X


def _svd_decompose(X: np.ndarray, n_components: int) -> tuple:
    """
    Performs full SVD on the trajectory matrix X and returns the
    leading `n_components` eigentriples (eigenvalue, left SV, right SV).

    X = Σ sqrt(λ_i) * U_i * V_i^T    (master equation from research doc)

    Returns:
        eigenvalues  : np.ndarray shape (n_components,)
        left_svecs   : np.ndarray shape (L, n_components)  — U_i columns
        right_svecs  : np.ndarray shape (K, n_components)  — V_i columns
        variance_pct : np.ndarray — % variance captured per component
    """
    # economy SVD: only compute leading singular values
    U, sigma, Vt = svd(X, full_matrices=False)

    eigenvalues  = sigma ** 2
    total_var    = eigenvalues.sum()
    variance_pct = eigenvalues / (total_var + 1e-12) * 100.0

    n = min(n_components, len(sigma))
    return (
        eigenvalues[:n],
        U[:, :n],
        Vt[:n, :].T,       # shape (K, n) — V_i columns
        variance_pct[:n],
    )


def _reconstruct_component(
    U_i: np.ndarray,
    sigma_i: float,
    V_i: np.ndarray,
    N: int,
    L: int,
) -> np.ndarray:
    """
    Reconstructs a single elementary matrix from one eigentriple and
    diagonally averages (hankelizes) it back to a 1-D time series of length N.

    Diagonal averaging (also called "anti-diagonal averaging") converts the
    rank-1 matrix X_i = sqrt(λ_i) * U_i * V_i^T back to the time domain.
    """
    K   = N - L + 1
    Xi  = sigma_i * np.outer(U_i, V_i)   # L × K elementary matrix

    # Diagonal averaging — average along each anti-diagonal
    reconstructed = np.zeros(N)
    counts        = np.zeros(N)

    for i in range(L):
        for j in range(K):
            reconstructed[i + j] += Xi[i, j]
            counts[i + j]        += 1

    reconstructed /= (counts + 1e-12)
    return reconstructed


def _group_components(
    eigenvalues: np.ndarray,
    variance_pct: np.ndarray,
    n_components: int,
) -> dict:
    """
    Groups eigentriples into categories based on variance contribution:
        - "trend"      : top 1-2 components (highest variance)
        - "oscillation": middle components (significant but not dominant)
        - "noise"      : low variance tail

    Returns a dict mapping group name → list of component indices.
    """
    groups: dict[str, list[int]] = {"trend": [], "oscillation": [], "noise": []}
    cumvar = 0.0
    for i in range(n_components):
        pct     = variance_pct[i]
        cumvar += pct
        if i < 2 and pct > 10.0:
            groups["trend"].append(i)
        elif pct > 2.0:
            groups["oscillation"].append(i)
        else:
            groups["noise"].append(i)
    return groups


# ── Public interface ──────────────────────────────────────────────────────────

def run(detrended: np.ndarray, cfg: dict) -> dict:
    """
    Runs SSA on a detrended signal and reconstructs the dominant cycle components.

    Parameters
    ----------
    detrended : np.ndarray — output of detrend.py ("detrended" key)
    cfg       : dict — supports keys:
                    "SSA_WINDOW_LENGTH"  (int, default 256)
                    "SSA_NUM_COMPONENTS" (int, default 6)

    Returns
    -------
    dict with keys:
        "success"          : bool
        "trajectory_matrix": np.ndarray  — L × K Hankel matrix
        "eigenvalues"      : np.ndarray  — leading eigenvalues
        "left_svecs"       : np.ndarray  — U matrix (L × n_components)
        "right_svecs"      : np.ndarray  — V matrix (K × n_components)
        "variance_pct"     : np.ndarray  — % variance per component
        "components"       : list[np.ndarray] — reconstructed 1-D components
        "reconstruction"   : np.ndarray  — sum of oscillation components
        "groups"           : dict        — trend/oscillation/noise grouping
        "dominant_period"  : float       — estimated period of leading oscillation (bars)
        "confidence"       : float       — set by ssa_utils.py (placeholder here)
        "metadata"         : dict        — L, K, N, n_components
        "error"            : str | None
    """
    _empty = {
        "success":           False,
        "trajectory_matrix": np.array([[]]),
        "eigenvalues":       np.array([]),
        "left_svecs":        np.array([[]]),
        "right_svecs":       np.array([[]]),
        "variance_pct":      np.array([]),
        "components":        [],
        "reconstruction":    np.array([]),
        "groups":            {},
        "dominant_period":   0.0,
        "confidence":        0.0,
        "metadata":          {},
        "error":             None,
    }

    # ── Validate ──────────────────────────────────────────────────────────────
    if detrended is None or len(detrended) < 20:
        _empty["error"] = f"SSA needs at least 20 samples, got {len(detrended) if detrended is not None else 0}."
        logger.error(_empty["error"])
        return _empty

    N            = len(detrended)
    L            = int(cfg.get("SSA_WINDOW_LENGTH", 256))
    n_components = int(cfg.get("SSA_NUM_COMPONENTS", 6))

    # Clamp L: must satisfy 2 ≤ L ≤ N//2
    L = max(2, min(L, N // 2))
    n_components = max(1, min(n_components, L))

    try:
        # ── Step 1: Embed ─────────────────────────────────────────────────────
        logger.info("SSA embed: N=%d L=%d n_components=%d", N, L, n_components)
        X = _build_trajectory_matrix(detrended, L)
        K = X.shape[1]

        # ── Step 2: Decompose ─────────────────────────────────────────────────
        eigenvalues, U, V, variance_pct = _svd_decompose(X, n_components)
        sigma = np.sqrt(np.maximum(eigenvalues, 0))

        # ── Step 3: Reconstruct individual components ─────────────────────────
        components: list[np.ndarray] = []
        for i in range(len(eigenvalues)):
            comp = _reconstruct_component(U[:, i], sigma[i], V[:, i], N, L)
            components.append(comp)

        # ── Step 4: Group and sum oscillation components ──────────────────────
        groups = _group_components(eigenvalues, variance_pct, len(eigenvalues))
        osc_indices = groups.get("oscillation", [])

        if osc_indices:
            reconstruction = sum(components[i] for i in osc_indices)
        elif components:
            reconstruction = components[0]   # fallback
        else:
            reconstruction = np.zeros(N)

        # ── Step 5: Estimate dominant period from leading oscillation ─────────
        # FFT on the first oscillation component to find its period
        dominant_period = 0.0
        if osc_indices and len(components) > osc_indices[0]:
            comp0 = components[osc_indices[0]]
            fft_mag  = np.abs(np.fft.rfft(comp0))
            freqs    = np.fft.rfftfreq(N)
            # Ignore DC (index 0) and find peak frequency
            fft_mag[0] = 0
            peak_freq  = freqs[np.argmax(fft_mag)]
            dominant_period = float(1.0 / peak_freq) if peak_freq > 0 else 0.0

        logger.info(
            "SSA OK: eigenvalues=%s variance=%s dominant_period=%.1f bars",
            eigenvalues.round(2), variance_pct.round(1), dominant_period,
        )

        return {
            "success":           True,
            "trajectory_matrix": X,
            "eigenvalues":       eigenvalues,
            "left_svecs":        U,
            "right_svecs":       V,
            "variance_pct":      variance_pct,
            "components":        components,
            "reconstruction":    reconstruction,
            "groups":            groups,
            "dominant_period":   dominant_period,
            "confidence":        0.0,   # filled in by ssa_utils.run()
            "metadata":          {"N": N, "L": L, "K": K, "n_components": len(eigenvalues)},
            "error":             None,
        }

    except Exception as exc:
        msg = f"{type(exc).__name__}: {exc}"
        logger.exception("SSA core failed: %s", msg)
        _empty["error"] = msg
        return _empty


# ── Smoke test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    np.random.seed(42)
    n      = 300
    t      = np.linspace(0, 8 * np.pi, n)
    signal = 10 * np.sin(t) + 4 * np.sin(3 * t) + np.random.randn(n)

    result = run(signal, {"SSA_WINDOW_LENGTH": 50, "SSA_NUM_COMPONENTS": 6})
    if result["success"]:
        print(f"✅ SSA Core OK")
        print(f"   Eigenvalues  : {result['eigenvalues'].round(2)}")
        print(f"   Variance %   : {result['variance_pct'].round(1)}")
        print(f"   Groups       : {result['groups']}")
        print(f"   Dom. Period  : {result['dominant_period']:.1f} bars")
        print(f"   Recon shape  : {result['reconstruction'].shape}")
    else:
        print(f"❌ {result['error']}")
