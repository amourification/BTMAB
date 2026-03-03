import numpy as np

from engine.elliott_wave_utils import (
    find_primary_wave_count,
    compute_rsi,
    compute_macd,
    detect_divergences,
)


def test_find_primary_wave_count_basic_impulse_up():
    # Construct a synthetic 5-wave up impulse pattern
    prices = np.array(
        [
            100.0,
            102.0,
            105.0,  # wave 1
            103.0,
            108.0,  # wave 3
            106.0,
            112.0,  # wave 5
        ],
        dtype=float,
    )
    # add some padding so pivot detection has enough context
    prices = np.concatenate([np.linspace(98.0, 100.0, 5), prices, np.linspace(112.0, 113.0, 5)])

    wave = find_primary_wave_count(prices)
    assert wave is not None
    assert wave.valid is True
    assert wave.direction in ("up", "down")
    assert set(wave.impulse_labels.keys()) == {"1", "2", "3", "4", "5"}
    assert 0.0 <= wave.score <= 1.0


def test_rsi_and_macd_shapes():
    prices = np.linspace(100.0, 120.0, 200)
    rsi = compute_rsi(prices, period=14)
    macd_line, macd_signal, macd_hist = compute_macd(prices)

    assert rsi.shape == prices.shape
    assert macd_line.shape == prices.shape
    assert macd_signal.shape == prices.shape
    assert macd_hist.shape == prices.shape


def test_detect_divergences_returns_list():
    prices = np.linspace(100.0, 120.0, 200)
    rsi = compute_rsi(prices, period=14)
    divs = detect_divergences(prices, rsi)
    assert isinstance(divs, list)
    for d in divs:
        assert "type" in d and "price_idx" in d


def test_find_primary_wave_count_rejects_noise():
    # High-noise series should not accidentally produce a valid textbook impulse
    rng = np.random.default_rng(42)
    prices = 100 + rng.standard_normal(300).cumsum()
    wave = find_primary_wave_count(prices)
    # Either no count or an extremely low guideline score
    if wave is None:
        assert True
    else:
        assert wave.score < 0.2

