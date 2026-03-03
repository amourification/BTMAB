# =============================================================================
#  data/sentiment.py — Sentiment Analysis Module (Stub)
#  Provides a real-time sentiment score for the asset being analyzed.
#  Currently stubbed with a neutral score — will be wired to a Crypto-BERT
#  model and live feeds (X/Twitter, Reddit, news) in a future stage.
#
#  The module is already wired into the pipeline so ar_model.py (Eq 10)
#  can consume sentiment scores immediately once the live feed is enabled.
#
#  Standard interface:
#      result = run(symbol, session, config) -> dict
# =============================================================================

import logging
import numpy as np
from datetime import datetime, timezone

logger = logging.getLogger("temporal_bot.sentiment")

# ── Sentiment score scale ──────────────────────────────────────────────────────
# Score range: -1.0 (extreme fear) to +1.0 (extreme greed)
# 0.0 = neutral
SCORE_MIN:  float = -1.0
SCORE_MAX:  float =  1.0
SCORE_NEUTRAL: float = 0.0

# ── Source weights (used once live feeds are enabled) ─────────────────────────
# Overnight / night-session sentiment is weighted higher per the AR model
# research: "overnight inertia exceeds within-day inertia."
SOURCE_WEIGHTS = {
    "twitter":  0.30,
    "reddit":   0.25,
    "news":     0.30,
    "fear_greed_index": 0.15,
}

NIGHT_SESSION_MULTIPLIER: float = 1.35   # from config.py NIGHT_SENTIMENT_WEIGHT


# ── Internal helpers ──────────────────────────────────────────────────────────

def _fetch_fear_greed() -> float:
    """
    Stub: Will call the Alternative.me Fear & Greed API.
    Returns a normalized score in [-1, 1].
    API endpoint: https://api.alternative.me/fng/
    """
    # TODO: implement live call
    # response = requests.get("https://api.alternative.me/fng/?limit=1")
    # raw = response.json()["data"][0]["value"]   # 0–100
    # return (float(raw) - 50) / 50               # normalize to [-1, 1]
    logger.debug("fear_greed_index: stub returning 0.0")
    return SCORE_NEUTRAL


def _fetch_twitter_sentiment(symbol: str) -> float:
    """
    Stub: Will use Crypto-BERT to score recent tweets containing $symbol.
    Model: ElKulako/cryptobert (HuggingFace)
    """
    # TODO: implement
    # from transformers import pipeline
    # classifier = pipeline("text-classification", model="ElKulako/cryptobert")
    # tweets = scrape_twitter(f"${symbol}", limit=100)
    # scores = [classifier(t)[0] for t in tweets]
    # return np.mean([s["score"] if s["label"]=="Bullish" else -s["score"] for s in scores])
    logger.debug("twitter_sentiment: stub returning 0.0 for %s", symbol)
    return SCORE_NEUTRAL


def _fetch_reddit_sentiment(symbol: str) -> float:
    """
    Stub: Will use PRAW + Crypto-BERT on r/CryptoCurrency, r/Bitcoin, etc.
    """
    # TODO: implement
    logger.debug("reddit_sentiment: stub returning 0.0 for %s", symbol)
    return SCORE_NEUTRAL


def _fetch_news_sentiment(symbol: str) -> float:
    """
    Stub: Will scrape CoinDesk, CoinTelegraph, and Decrypt headlines
    and score them with Crypto-BERT.
    """
    # TODO: implement
    logger.debug("news_sentiment: stub returning 0.0 for %s", symbol)
    return SCORE_NEUTRAL


def _aggregate_scores(scores: dict, session: str) -> float:
    """
    Combines individual source scores using SOURCE_WEIGHTS.
    Applies NIGHT_SESSION_MULTIPLIER if session == 'night', then clips to [-1, 1].

    This weighting logic implements the AR model research finding that
    overnight sentiment has stronger persistent effects on the next day's open.
    """
    weighted = sum(
        scores.get(source, 0.0) * weight
        for source, weight in SOURCE_WEIGHTS.items()
    )

    if session == "night":
        weighted *= NIGHT_SESSION_MULTIPLIER
        logger.debug(
            "Night session multiplier applied (×%.2f): %.4f → %.4f",
            NIGHT_SESSION_MULTIPLIER, weighted / NIGHT_SESSION_MULTIPLIER, weighted,
        )

    return float(np.clip(weighted, SCORE_MIN, SCORE_MAX))


# ── Public interface ──────────────────────────────────────────────────────────

def run(symbol: str, session: str, cfg: dict) -> dict:
    """
    Main entry point for the sentiment module.

    Parameters
    ----------
    symbol  : Trading pair base asset, e.g. "BTCUSDT" or "BTC"
    session : 'day' or 'night' — affects sentiment weighting
    cfg     : config dict (reserved for future API key injection)

    Returns
    -------
    dict with keys:
        "success"       : bool
        "symbol"        : str
        "session"       : str — 'day' or 'night'
        "score"         : float — aggregate score in [-1.0, 1.0]
        "interpretation": str  — human-readable label
        "sources"       : dict — individual source scores
        "is_stub"       : bool — True until live feeds are wired in
        "confidence"    : float — 0.0 when stub, 0.0–1.0 when live
        "fetched_at"    : str  — ISO 8601 UTC
        "error"         : str | None
    """
    logger.info("Sentiment: symbol=%s session=%s", symbol, session)

    # Normalize symbol (strip quote asset if present)
    base = symbol.upper().replace("USDT", "").replace("BTC", "BTC")

    # ── Fetch individual source scores ────────────────────────────────────────
    sources = {
        "twitter":          _fetch_twitter_sentiment(base),
        "reddit":           _fetch_reddit_sentiment(base),
        "news":             _fetch_news_sentiment(base),
        "fear_greed_index": _fetch_fear_greed(),
    }

    # ── Aggregate ─────────────────────────────────────────────────────────────
    score = _aggregate_scores(sources, session)

    # ── Interpret ─────────────────────────────────────────────────────────────
    if score >= 0.6:
        interpretation = "Extreme Greed 🚀"
    elif score >= 0.2:
        interpretation = "Greed 📈"
    elif score >= -0.2:
        interpretation = "Neutral ↔"
    elif score >= -0.6:
        interpretation = "Fear 📉"
    else:
        interpretation = "Extreme Fear 😱"

    result = {
        "success":        True,
        "symbol":         symbol.upper(),
        "session":        session,
        "score":          score,
        "interpretation": interpretation,
        "sources":        sources,
        "is_stub":        True,     # flip to False when live feeds are wired in
        "confidence":     0.0,      # 0.0 while stub; will be 0.5–1.0 with live data
        "fetched_at":     datetime.now(tz=timezone.utc).isoformat(),
        "error":          None,
    }

    logger.info(
        "Sentiment result: score=%.3f (%s) [stub=%s]",
        score, interpretation, result["is_stub"],
    )
    return result


def get_last_session_sentiment(
    symbol: str,
    sessions: np.ndarray,
    cfg: dict,
) -> dict:
    """
    Convenience wrapper called by ar_model.py.
    Determines the most recent session type from the preprocessor's
    sessions array and fetches the appropriate weighted sentiment.

    Parameters
    ----------
    symbol   : Trading pair
    sessions : np.ndarray of 'day'/'night' strings from preprocessor
    cfg      : config dict

    Returns
    -------
    Same dict as run()
    """
    last_session = sessions[-1] if len(sessions) > 0 else "day"
    return run(symbol, last_session, cfg)


# ── Quick smoke test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    result = run("BTCUSDT", "night", {})
    print("\n📊 Sentiment Result:")
    print(f"   Symbol       : {result['symbol']}")
    print(f"   Session      : {result['session']}")
    print(f"   Score        : {result['score']:.3f}")
    print(f"   Interpretation: {result['interpretation']}")
    print(f"   Is stub      : {result['is_stub']}")
    print(f"   Sources      : {result['sources']}")
