# =============================================================================
#  test_integration.py — Integration Tests for Main File Interactions
#  Tests how the data layer, engines, risk modules, consensus, exporter,
#  and GUI state interact end-to-end. Uses synthetic data and mocks to avoid
#  live API calls.
# =============================================================================

import sys
from pathlib import Path
from datetime import datetime, timezone, timedelta
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent))


# ── Fixtures: Synthetic data & config ────────────────────────────────────────

@pytest.fixture
def minimal_config():
    """Minimal config dict for pipeline (no API keys required when fetcher is mocked)."""
    return {
        "BINANCE_API_KEY":    "test_key",
        "BINANCE_API_SECRET": "test_secret",
        "DEFAULT_SYMBOL":     "BTCUSDT",
        "DOMINANT_CYCLE_BARS": 128,
        "SSA_WINDOW_LENGTH":  50,
        "SSA_NUM_COMPONENTS": 6,
        "ACF_MAX_LAG":        80,
        "FFT_MIN_PERIOD":     10,
        "MURRAY_LOOKBACK":    64,
        "GANN_LOOKBACK":      64,
        "KELLY_FRACTION":     0.5,
        "KELLY_MIN_CYCLES":   3,
        "AR_ORDER":           5,
        "NIGHT_SENTIMENT_WEIGHT": 1.35,
        "TRADE_DIRECTION":    "long",
        "PORTFOLIO_VALUE":    10000.0,
        "EXPORT_DIR":         Path(__file__).parent / "test_exports",
    }


@pytest.fixture
def synthetic_ohlcv():
    """Synthetic OHLCV data matching fetcher output format."""
    np.random.seed(42)
    n = 400
    t = np.arange(n)
    base = 50000
    close = base + 5000 * np.sin(2 * np.pi * t / 128) + np.cumsum(np.random.randn(n) * 200)
    high = close + np.abs(np.random.randn(n) * 300)
    low = close - np.abs(np.random.randn(n) * 300)
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    volume = np.abs(np.random.randn(n) * 1e6) + 1e5

    # Timestamps: 1d interval, starting 400 days ago
    start = datetime.now(tz=timezone.utc) - timedelta(days=n)
    timestamps_ms = np.array([
        int((start + timedelta(days=i)).timestamp() * 1000)
        for i in range(n)
    ])

    df = pd.DataFrame({
        "timestamp": timestamps_ms,
        "open":      open_,
        "high":      high,
        "low":       low,
        "close":     close,
        "volume":    volume,
    })
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df.set_index("datetime", inplace=True)
    df.sort_index(inplace=True)

    return {
        "success":    True,
        "symbol":     "BTCUSDT",
        "interval":   "1d",
        "bars":       n,
        "df":         df,
        "close":      df["close"].to_numpy(dtype=np.float64),
        "high":       df["high"].to_numpy(dtype=np.float64),
        "low":        df["low"].to_numpy(dtype=np.float64),
        "volume":     df["volume"].to_numpy(dtype=np.float64),
        "timestamps": df["timestamp"].to_numpy(dtype=np.int64),
        "error":      None,
    }


# ── Integration: Full pipeline (aggregator) ────────────────────────────────────

def test_aggregator_full_pipeline_with_mocked_fetcher(synthetic_ohlcv, minimal_config):
    """
    Full aggregator pipeline: mocked fetcher → preprocessor → engines → risk → consensus.
    Verifies all stages integrate correctly and produce a valid trade plan.
    """
    with patch("consensus.aggregator.fetch_data", return_value=synthetic_ohlcv):
        from consensus.aggregator import run

        plan = run("BTCUSDT", "1d", 400, minimal_config)

    assert plan["success"] is True
    assert plan["symbol"] == "BTCUSDT"
    assert plan["error"] is None

    # Trade plan keys consumed by GUI and exporter
    assert "consensus_confidence" in plan
    assert "market_bias" in plan
    assert "dominant_cycle_bars" in plan
    assert "phase_deg" in plan
    assert "turn_type" in plan
    assert "kelly_position_pct" in plan
    assert "hedge_pct" in plan
    assert "active_stop_price" in plan
    assert "_engines" in plan
    assert "_risk" in plan

    # Engine results present
    eng = plan["_engines"]
    assert "detrend" in eng
    assert "ssa" in eng
    assert "fft" in eng
    assert "hilbert" in eng
    assert "murray" in eng
    assert "kelly" in eng

    # Risk results present
    risk = plan["_risk"]
    assert "hedging" in risk
    assert "stops" in risk
    assert "portfolio" in risk


def test_advanced_aggregator_full_pipeline_with_mocked_fetcher(
    synthetic_ohlcv, minimal_config
):
    """
    Advanced aggregator pipeline: mocked fetcher → full stack.
    Ensures the advanced mode produces a valid trade plan with
    additional advanced fields while remaining compatible with
    the classic plan shape.
    """
    with patch("consensus.advanced_aggregator.fetch_data", return_value=synthetic_ohlcv):
        from consensus.advanced_aggregator import run as run_adv

        plan = run_adv("BTCUSDT", "1d", 400, minimal_config)

    assert plan["success"] is True
    assert plan["symbol"] == "BTCUSDT"
    assert plan["error"] is None

    # Mode and advanced block
    assert plan.get("mode") == "advanced"
    adv = plan.get("advanced", {})
    assert isinstance(adv, dict)
    assert "uncertainty_score" in adv

    # Classic keys still present
    assert "consensus_confidence" in plan
    assert "market_bias" in plan
    assert "dominant_cycle_bars" in plan
    assert "phase_deg" in plan
    assert "kelly_position_pct" in plan
    assert "_engines" in plan
    assert "_risk" in plan


def test_aggregator_fails_gracefully_on_fetch_error(minimal_config):
    """Aggregator returns structured error when fetcher fails."""
    failed_fetch = {"success": False, "error": "API key invalid"}

    with patch("consensus.aggregator.fetch_data", return_value=failed_fetch):
        from consensus.aggregator import run

        plan = run("BTCUSDT", "1d", 100, minimal_config)

    assert plan["success"] is False
    assert "Fetch:" in plan["error"]
    assert plan["symbol"] == "BTCUSDT"


# ── Integration: Data layer (fetcher → preprocessor → sentiment) ──────────────

def test_preprocessor_consumes_fetch_result(synthetic_ohlcv, minimal_config):
    """Preprocessor correctly transforms fetcher output for engines."""
    from data.preprocessor import run as preprocess

    prep = preprocess(synthetic_ohlcv, minimal_config)

    assert prep["success"] is True
    assert prep["bars"] == synthetic_ohlcv["bars"]
    assert "close" in prep
    assert "close_norm" in prep
    assert "close_z" in prep
    assert "sessions" in prep
    assert "julian_dates" in prep
    assert len(prep["close"]) == len(prep["sessions"])


def test_sentiment_integrates_with_ar_model(minimal_config):
    """Sentiment module returns structure consumable by AR model."""
    from data.sentiment import run as get_sentiment

    result = get_sentiment("BTCUSDT", "2024-01-15", minimal_config)

    assert "success" in result
    assert "score" in result
    assert -1.0 <= result["score"] <= 1.0


# ── Integration: Engine chain (detrend → core → support → kelly) ───────────────

def test_engine_chain_detrend_to_kelly(synthetic_ohlcv, minimal_config):
    """Engines pass data correctly: detrend → SSA/FFT/Hilbert → Kelly."""
    from data.preprocessor import run as preprocess
    from engine.detrend import run as run_detrend
    from engine.ssa_utils import run as run_ssa
    from engine.fft import run as run_fft
    from engine.hilbert import run as run_hilbert
    from engine.kelly import run as run_kelly

    prep = preprocess(synthetic_ohlcv, minimal_config)
    det = run_detrend(prep["close"], minimal_config)
    assert det["success"]

    ssa = run_ssa(det["detrended"], minimal_config)
    fft = run_fft(det["detrended"], minimal_config)
    hil = run_hilbert(det["detrended"], {
        **minimal_config,
        "FFT_PRIMARY_PERIOD": fft.get("primary_cycle", {}).get("period", 128),
    })

    eng = {
        "detrend": det,
        "ssa": ssa,
        "fft": fft,
        "hilbert": hil,
        "prices": prep["close"],
        "_fetch": {"high": prep["high"], "low": prep["low"]},
    }
    # Kelly needs murray, gamma, etc. — use minimal stubs for chain test
    eng["murray"] = {"success": True, "confluence": {"signal": "neutral"}}
    eng["gamma"] = {"success": True, "regime": {}, "vol_stats": {}}
    eng["gann"] = {"success": True, "price_vs_master": ""}
    eng["walras"] = {"success": True, "kelly_multiplier": 1.0}
    eng["solar"] = {"success": True, "seasonal_bias": "unknown"}
    eng["ar"] = {"success": True, "forecast_direction": "flat"}

    kel = run_kelly(eng, minimal_config)

    assert kel["success"]
    assert "position_pct" in kel
    assert "position_tier" in kel


# ── Integration: Consensus (aggregator_utils) ─────────────────────────────────

def test_consensus_weighted_confidence_and_bias():
    """weighted_confidence and determine_market_bias integrate with engine results."""
    from consensus.aggregator_utils import (
        weighted_confidence,
        determine_market_bias,
        build_trade_plan,
    )

    engine_results = {
        "hilbert": {"success": True, "turn_type": "early_bullish", "phase_deg": 45},
        "fft":     {"success": True, "cycle_phase": {"pct_complete": 0.2}},
        "murray":  {"success": True, "confluence": {"signal": "buy"}},
        "kelly":   {"success": True, "position_pct": 10.0},
    }

    conf = weighted_confidence(engine_results)
    assert 0 <= conf <= 1

    bias = determine_market_bias(engine_results)
    assert bias["bias"] in ("bullish", "bearish", "neutral")
    assert "votes" in bias
    assert "reasons" in bias


def test_build_trade_plan_assembles_all_sections():
    """build_trade_plan produces dict consumable by GUI and exporter."""
    from consensus.aggregator_utils import build_trade_plan

    engine_results = {
        "hilbert": {"phase_deg": 90, "turn_type": "mid_expansion", "turn_label": "Expansion",
                    "turn_urgency": "low", "bars_to_next_turn": {"bars_to_boundary": 20, "next_turn_type": "distribution"}},
        "fft":     {"primary_cycle": {"period": 64}, "macro_cycle": {"period": 256},
                    "cycle_phase": {"pct_complete": 0.5}},
        "ssa":     {"dominant_period": 62, "confidence": 0.75, "cycle_position": {"position": "mid"}},
        "murray":  {"murray_index": 4.5, "nearest_levels": {"resistance_price": 52000, "support_price": 48000},
                   "recommended_action": "Hold"},
        "kelly":   {"position_pct": 8.0, "position_tier": {"label": "Small"}, "expected_value": 0.02},
        "gamma":   {"regime": {"regime": "neutral"}, "vol_stats": {"regime": "normal"}},
        "walras":  {"kelly_multiplier": 1.0},
        "solar":   {"seasonal_bias": "spring_rally_bias", "volatility_flag": False},
    }
    risk_results = {
        "hedging": {"unified_hedge": {"ratio": 0.1, "pct": 10.0, "action": "hold", "urgency": "low"},
                    "sweep_detection": {"detected": False}},
        "stops":   {"active_stop": {"stop_price": 47000, "stop_type": "trailing", "stop_pct": 5.0},
                    "stop_schedule": []},
        "portfolio": {"summary": "Net +8%", "risk_adjusted": {"grade": "B"}, "overall_risk_score": 0.4,
                      "drawdown": {}},
    }
    bias = {"bias": "bullish", "strength": 0.7, "votes": {}, "reasons": []}

    plan = build_trade_plan("BTCUSDT", engine_results, risk_results, bias, 0.72, 12.5)

    assert plan["symbol"] == "BTCUSDT"
    assert plan["consensus_confidence"] == 72.0
    assert plan["market_bias"] == "bullish"
    assert plan["phase_deg"] == 90
    assert plan["dominant_cycle_bars"] == 64
    assert plan["kelly_position_pct"] == 8.0
    assert plan["hedge_pct"] == 10.0
    assert plan["active_stop_price"] == 47000


# ── Integration: Config → Pipeline ───────────────────────────────────────────

def test_config_build_config_integrates_with_aggregator():
    """build_config() produces dict usable by aggregator (when fetcher mocked)."""
    from config import build_config

    cfg = build_config()

    required_keys = [
        "BINANCE_API_KEY", "DOMINANT_CYCLE_BARS", "SSA_WINDOW_LENGTH",
        "ACF_MAX_LAG", "MURRAY_LOOKBACK", "TRADE_DIRECTION", "PORTFOLIO_VALUE",
    ]
    for k in required_keys:
        assert k in cfg, f"build_config missing key: {k}"


# ── Integration: Exporter ← Trade plan ────────────────────────────────────────

def test_exporter_consumes_trade_plan(minimal_config, tmp_path):
    """export_all and export_summary_png accept aggregator output."""
    from charts.exporter import export_all, export_summary_png
    from charts.exporter_utils import make_smoke_test_plan

    plan = make_smoke_test_plan()
    plan["_engines"] = {}
    plan["_risk"] = {}

    out_dir = tmp_path / "exports"
    result = export_all(plan, out_dir, minimal_config)

    assert result["success"] is True
    assert Path(result["pdf"]).exists()
    assert Path(result["csv"]).exists()

    summary_path = export_summary_png(plan, out_dir)
    assert Path(summary_path).exists()


def test_flatten_trade_plan_for_csv():
    """flatten_trade_plan produces CSV-friendly dict."""
    from charts.exporter_utils import flatten_trade_plan

    plan = {"symbol": "BTCUSDT", "consensus_confidence": 75.0, "market_bias": "bullish"}
    flat = flatten_trade_plan(plan)

    assert "symbol" in flat
    assert "consensus_confidence" in flat
    assert "market_bias" in flat
    assert "_engines" not in flat


# ── Integration: GUI state manager ───────────────────────────────────────────

def test_gui_state_manager_integrates_with_analysis_result():
    """StateManager correctly applies AnalysisResult from background thread."""
    from gui.gui_state import get_manager, AnalysisResult, reset_manager

    reset_manager()
    manager = get_manager()

    # Simulate background thread pushing result
    plan = {"symbol": "BTCUSDT", "consensus_confidence": 80.0}
    result = AnalysisResult(success=True, trade_plan=plan, elapsed=5.2)
    manager.push(result)

    # Simulate GUI thread polling
    polled = manager.poll()
    assert polled is not None
    assert polled.success is True
    assert polled.trade_plan["symbol"] == "BTCUSDT"

    manager.apply_result(polled)
    state = manager.snapshot()
    assert state.trade_plan["symbol"] == "BTCUSDT"
    assert state.last_run_secs == 5.2
    assert state.is_running is False


def test_gui_state_handles_failed_analysis():
    """StateManager stores error when analysis fails."""
    from gui.gui_state import get_manager, AnalysisResult, reset_manager

    reset_manager()
    manager = get_manager()

    result = AnalysisResult(success=False, error="Fetch failed", elapsed=0.5)
    manager.push(result)
    polled = manager.poll()
    manager.apply_result(polled)

    state = manager.snapshot()
    assert state.last_error == "Fetch failed"
    assert state.trade_plan is None
