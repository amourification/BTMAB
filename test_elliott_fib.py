import numpy as np

from engine.elliott_fib import run as run_elliott_fib


def test_elliott_fib_up_impulse_basic_levels():
    # Simple upward impulse from 100 → 200 over 10 bars
    prices = np.linspace(100.0, 200.0, 20)
    cfg = {"GANN_LOOKBACK": 20}

    result = run_elliott_fib(prices, cfg)
    assert result["success"] is True
    assert result["direction"] in ("up", "down")
    swing = result["swing_points"]
    assert "start_idx" in swing and "end_idx" in swing
    assert swing["end_price"] >= swing["start_price"]

    retr = result["fib_retracements"]
    exts = result["fib_extensions"]
    # Retracements and extensions should be non-empty and within a plausible price range
    assert retr and exts
    for lvl in retr.values():
        assert 0 < lvl < 1e6
    for lvl in exts.values():
        assert 0 < lvl < 1e6

    # Wave count payload and oscillators
    wave = result["wave_count"]
    assert "rsi" in result and "macd_line" in result and "macd_signal" in result
    assert isinstance(result["rsi"], list)
    assert isinstance(result["macd_line"], list)
    assert isinstance(result["macd_signal"], list)

    if wave is not None:
        assert "impulse_labels" in wave
        assert "direction" in wave

    # Time events should be present and strictly forward in bars
    events = result.get("time_events", [])
    assert isinstance(events, list)
    for ev in events:
        assert ev["source"] in {"elliott", "rsi", "macd"}
        assert ev["bars_ahead"] > 0

