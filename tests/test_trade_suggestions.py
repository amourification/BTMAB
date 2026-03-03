import numpy as np

from gui.trade_suggestions import compute_trade_suggestions


def test_compute_trade_suggestions_long_with_portfolio():
    prices = np.linspace(90.0, 110.0, 50)
    trade_plan = {
        "_engines": {
            "detrend": {
                "trend": prices,
                "detrended": np.zeros_like(prices),
            }
        },
        "_risk": {
            "stops": {
                "entry_price": 100.0,
                "direction": "long",
                "active_stop": {"stop_price": 95.0},
            },
            "portfolio": {
                "risk_adjusted": {"risk_reward": 2.0},
            },
        },
    }

    res = compute_trade_suggestions(trade_plan)
    assert res["suggested_entry_price"] == 100.0
    assert res["suggested_stop_price"] == 95.0
    # With RR=2, target distance should be about 2 * (entry - stop) = 10
    assert 109.5 <= res["suggested_exit_price"] <= 110.5


def test_compute_trade_suggestions_handles_missing_stop():
    prices = np.linspace(90.0, 110.0, 50)
    trade_plan = {
        "_engines": {
            "detrend": {
                "trend": prices,
                "detrended": np.zeros_like(prices),
            }
        },
        "_risk": {
            "stops": {
                "entry_price": 100.0,
                "direction": "long",
                # no active or initial stop
            },
            "portfolio": {},
        },
    }

    res = compute_trade_suggestions(trade_plan)
    # Should at least return a suggested entry and not crash
    assert res["suggested_entry_price"] == 100.0
