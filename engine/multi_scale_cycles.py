from __future__ import annotations

from typing import Dict, Any, List

import logging
import numpy as np

from engine.ssa_utils import run as run_ssa
from engine.fft import run as run_fft
from engine.hilbert import run as run_hilbert

logger = logging.getLogger("temporal_bot.engine.multi_scale_cycles")


def _safe_float(val: Any, default: float = 0.0) -> float:
    try:
        return float(val)
    except Exception:
        return default


def run(detrended: np.ndarray, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute multi-window cycle diagnostics by running the existing SSA/FFT/
    Hilbert engines over several embedding lengths. This is a thin wrapper
    that leaves the original engines unchanged.

    Returns a dict with:
      - success: bool
      - windows: list of per-window diagnostics
      - primary: best-effort primary cycle summary
    """
    result: Dict[str, Any] = {"success": False, "windows": [], "primary": {}, "error": None}

    try:
        if detrended is None or len(detrended) < 64:
            msg = "Not enough data for multi-scale cycles."
            logger.debug("multi_scale_cycles: %s (len=%s)", msg, 0 if detrended is None else len(detrended))
            result["error"] = msg
            return result

        # Default windows derived from DOMINANT_CYCLE_BARS if not provided
        dom = int(cfg.get("DOMINANT_CYCLE_BARS", 512))
        base_windows: List[int] = [max(64, dom // 4), max(96, dom // 2), max(128, dom)]
        custom = cfg.get("ADV_CYCLE_WINDOWS")
        if isinstance(custom, (list, tuple)) and custom:
            windows = [int(max(32, w)) for w in custom]
        else:
            windows = sorted({w for w in base_windows if w < len(detrended)})

        entries = []
        for w in windows:
            # Override SSA window length for this call only
            local_cfg = {**cfg, "SSA_WINDOW_LENGTH": min(w, len(detrended) // 2)}

            ssa_r = run_ssa(detrended[-w:], local_cfg)
            fft_r = run_fft(detrended[-w:], local_cfg)
            hil_r = run_hilbert(detrended[-w:], local_cfg)

            if not (ssa_r.get("success") and fft_r.get("success") and hil_r.get("success")):
                continue

            period_fft = _safe_float(
                fft_r.get("primary_cycle", {}).get("period", local_cfg.get("DOMINANT_CYCLE_BARS", 0.0))
            )
            period_ssa = _safe_float(ssa_r.get("dominant_period", period_fft))
            phase_deg = _safe_float(hil_r.get("phase_deg", 0.0))

            conf_ssa = _safe_float(ssa_r.get("confidence", 0.0))
            conf_fft = _safe_float(fft_r.get("confidence", 0.0))
            conf_hil = _safe_float(hil_r.get("confidence", 0.0))

            combined_conf = max(0.0, min(1.0, (conf_ssa + conf_fft + conf_hil) / 3.0))

            entries.append(
                {
                    "success": True,
                    "window": w,
                    "period": period_fft,
                    "ssa_period": period_ssa,
                    "phase_deg": phase_deg,
                    "confidence": combined_conf,
                    "ssa": ssa_r,
                    "fft": fft_r,
                    "hilbert": hil_r,
                }
            )

        if not entries:
            msg = "No successful multi-scale cycle windows."
            logger.debug("multi_scale_cycles: %s", msg)
            result["error"] = msg
            return result

        # Choose a primary cycle as the highest-confidence entry
        primary = max(entries, key=lambda e: e.get("confidence", 0.0))

        logger.debug(
            "multi_scale_cycles: %d windows, primary_period=%.2f window=%d",
            len(entries),
            primary.get("period", 0.0),
            primary.get("window", 0),
        )
        result.update({"success": True, "windows": entries, "primary": primary})
        return result
    except Exception as exc:
        msg = f"{type(exc).__name__}: {exc}"
        logger.warning("multi_scale_cycles error: %s", msg)
        result["error"] = msg
        return result

