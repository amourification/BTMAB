# =============================================================================
#  engine/ar_model.py — Equation 10: AR Model & Affective Inertia
#  Models price momentum and sentiment persistence across day/night sessions.
#  Helpers live in ar_utils.py.
#
#  Standard interface:
#      result = run(preprocessed, sentiment_result, config) -> dict
# =============================================================================

import logging
import numpy as np
from engine.ar_utils import (
    fit_ar_model, numpy_ar_fit, forecast_ar,
    split_by_session, inertia_score, blend_sentiment,
    NIGHT_SENTIMENT_WEIGHT,
)

logger = logging.getLogger("temporal_bot.engine.ar_model")
AR_ORDER:           int = 5
MAX_FORECAST_BARS:  int = 10


def run(preprocessed: dict, sentiment_result: dict, cfg: dict) -> dict:
    """
    Fits an AR model to log-returns and forecasts with sentiment adjustment.

    Parameters
    ----------
    preprocessed    : dict — from preprocessor.run() — needs "close", "sessions"
    sentiment_result: dict — from sentiment.run()
    cfg             : dict — "AR_ORDER", "NIGHT_SENTIMENT_WEIGHT"

    Returns
    -------
    dict — success, ar_coeffs, ar_intercept, forecast_raw, forecast_adjusted,
           forecast_direction, session_split, inertia, sentiment_score,
           last_session, residual_std, confidence, metadata, error
    """
    _empty = {
        "success": False, "ar_coeffs": np.array([]), "ar_intercept": 0.0,
        "forecast_raw": np.array([]), "forecast_adjusted": np.array([]),
        "forecast_direction": "flat", "session_split": {}, "inertia": {},
        "sentiment_score": 0.0, "last_session": "day",
        "residual_std": 0.0, "confidence": 0.0, "metadata": {}, "error": None,
    }

    close    = preprocessed.get("close")
    sessions = preprocessed.get("sessions")

    if close is None or len(close) < AR_ORDER + 5:
        _empty["error"] = f"AR: need >= {AR_ORDER + 5} prices."
        logger.error(_empty["error"])
        return _empty

    if sessions is None or len(sessions) != len(close):
        sessions = np.array(["day"] * len(close), dtype=object)

    order    = int(cfg.get("AR_ORDER", AR_ORDER))
    night_wt = float(cfg.get("NIGHT_SENTIMENT_WEIGHT", NIGHT_SENTIMENT_WEIGHT))
    steps    = int(cfg.get("AR_FORECAST_BARS", MAX_FORECAST_BARS))

    try:
        log_returns  = np.diff(np.log(close + 1e-12))
        ret_sessions = sessions[1:]

        ar_result = fit_ar_model(log_returns, order)
        if ar_result is not None:
            ar_params = {"coeffs": np.array(ar_result.params[1:]),
                         "intercept": float(ar_result.params[0])}
            residuals = np.array(ar_result.resid)
        else:
            ar_params = numpy_ar_fit(log_returns, order)
            residuals = ar_params["residuals"]

        forecast_raw  = forecast_ar(log_returns, ar_params, order, steps)
        session_split = split_by_session(log_returns, ret_sessions)
        inertia       = inertia_score(session_split, night_wt)
        last_session  = str(sessions[-1]) if len(sessions) > 0 else "day"
        sent_score    = float(sentiment_result.get("score", 0.0))
        forecast_adj  = blend_sentiment(forecast_raw, sent_score, last_session, night_wt)

        avg = float(forecast_adj.mean())
        direction = "up" if avg > 0.0005 else ("down" if avg < -0.0005 else "flat")
        res_std   = float(residuals.std()) if len(residuals) > 0 else 0.0

        inertia_str = abs(inertia["night_inertia_weighted"])
        dir_clarity = min(abs(avg) / (res_std + 1e-12), 1.0)
        confidence  = round(max(0.0, min(1.0, 0.5 * min(inertia_str, 1.0) + 0.5 * dir_clarity)), 4)

        logger.info("AR OK: order=%d dir=%s inertia=%s sentiment=%.3f confidence=%.3f",
                    order, direction, inertia["dominant_session"], sent_score, confidence)

        return {
            "success": True, "ar_coeffs": ar_params["coeffs"],
            "ar_intercept": ar_params["intercept"], "forecast_raw": forecast_raw,
            "forecast_adjusted": forecast_adj, "forecast_direction": direction,
            "session_split": session_split, "inertia": inertia,
            "sentiment_score": sent_score, "last_session": last_session,
            "residual_std": res_std, "confidence": confidence,
            "metadata": {"order": order, "n_bars": len(close), "steps": steps},
            "error": None,
        }

    except Exception as exc:
        msg = f"{type(exc).__name__}: {exc}"
        logger.exception("AR model failed: %s", msg)
        _empty["error"] = msg
        return _empty


if __name__ == "__main__":
    np.random.seed(11)
    prices   = np.cumsum(np.random.randn(300) * 500) + 50000
    sessions = np.array(["day" if i % 2 == 0 else "night" for i in range(300)])
    r = run({"close": prices, "sessions": sessions, "bars": 300}, {"score": 0.15}, {})
    if r["success"]:
        print(f"✅ AR OK | dir={r['forecast_direction']} | inertia={r['inertia']['dominant_session']} | conf={r['confidence']:.3f}")
    else:
        print(f"❌ {r['error']}")
