import numpy as np

from risk.stops_utils import (
    compute_atr,
    initial_stop,
    trailing_stop,
    stop_schedule,
)


def test_compute_atr_reasonable_value():
    close = np.linspace(100.0, 120.0, 30)
    high = close + 2.0
    low = close - 2.0
    atr = compute_atr(high, low, close, period=14)
    assert atr > 0
    assert atr < 10  # with such mild moves, ATR should be modest


def test_initial_stop_with_murray_levels_long():
    entry = 100.0
    atr = 2.0
    mult = 2.0
    levels = np.array([80.0, 90.0, 95.0, 99.0])
    res = initial_stop(entry, atr, mult, levels, direction="long")
    assert res["stop_price"] < entry
    assert res["stop_type"] == "initial"


def test_trailing_stop_moves_with_price():
    prices = np.array([100.0, 105.0, 110.0, 115.0, 120.0])
    entry = 100.0
    atr = 2.0
    phase = 200.0
    levels = np.array([])
    res = trailing_stop(prices, entry, atr, phase, direction="long", levels=levels)
    assert res["stop_price"] < prices.max()
    assert res["stop_type"] == "trailing"


def test_stop_schedule_generates_future_points():
    sched = stop_schedule(phase=100.0, entry=100.0, atr=2.0, direction="long")
    assert sched
    for item in sched:
        assert item["bars_ahead"] > 0
        assert isinstance(item["projected_stop"], float)

