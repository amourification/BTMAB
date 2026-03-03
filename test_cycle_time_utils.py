from datetime import datetime, timedelta, timezone

import numpy as np

from charts.cycle_time_utils import compute_time_angle_predictions


def test_compute_time_angle_predictions_basic_daily():
    # 10 daily bars ending "now"
    now = datetime(2026, 3, 1, 12, 0, tzinfo=timezone.utc)
    timestamps = np.array(
        [now.timestamp() - (9 - i) * 86400 for i in range(10)], dtype=float
    )

    trade_plan = {
        "phase_deg": 45.0,
        "dominant_cycle_bars": 90.0,  # 4° per bar
    }

    preds = compute_time_angle_predictions(trade_plan, timestamps, max_events=3)
    # We expect some forward projections
    assert len(preds) > 0

    # Ensure predictions are in the future relative to last timestamp
    last_ts = timestamps[-1]
    for p in preds:
        assert p.timestamp.timestamp() > last_ts
        assert 0.0 <= p.angle_deg <= 360.0


def test_cycle_time_predictions_exact_zero_angle_timestamp():
    # Exactly daily bars, phase at 0°, dominant cycle 90 bars.
    # The implementation returns the SOONEST angle crossing, so for phase=0°
    # that is the 90° crossing, 90/4 = 22.5 bars ahead (not the 0° wrap).
    now = datetime(2026, 3, 1, 12, 0, tzinfo=timezone.utc)
    timestamps = np.array(
        [now.timestamp() - (9 - i) * 86400 for i in range(10)], dtype=float
    )

    trade_plan = {
        "phase_deg": 0.0,
        "dominant_cycle_bars": 90.0,
    }

    preds = compute_time_angle_predictions(trade_plan, timestamps, max_events=1)
    assert len(preds) == 1
    p = preds[0]

    # With phase 0 and 90-bar dominant cycle, earliest crossing is 90°
    # at 90/4 = 22.5 bars ahead.
    assert p.angle_deg == 90.0
    assert abs(p.bars_ahead - 22.5) < 1e-9

    expected_dt = now + timedelta(days=22.5)
    assert abs((p.timestamp - expected_dt).total_seconds()) < 1e-3
