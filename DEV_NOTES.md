## Temporal Bot — Developer Notes

### Modes: Classic vs Advanced

- **Classic mode**:
  - Entry points:
    - GUI: `main_gui.py` with mode selector set to `Classic`.
    - Telegram: `/run`, `/status`, `/export`.
  - Pipeline:
    - `consensus/aggregator.py` orchestrates data → engines → risk → consensus.
  - Trade plan:
    - Flat dict produced by `consensus/aggregator_utils.build_trade_plan()`.

- **Advanced mode**:
  - Entry points:
    - GUI: `main_gui.py` mode selector set to `Advanced`.
    - Telegram: `/run_adv`, `/status_adv`, `/export_adv`.
  - Pipeline:
    - `consensus/advanced_aggregator.py` wraps the classic stack and adds:
      - `engine/multi_scale_cycles.py`
      - `engine/regime.py`
      - `engine/gamma_adv.py`
      - `engine/walras_adv.py`
  - Trade plan:
    - Compatible with the classic plan but with:
      - Top-level `mode: "advanced"`.
      - `advanced` block containing:
        - `vol_regime`, `trend_regime`, `stress_flag`, `stress_score`.
        - `uncertainty_score`.
        - `multi_scale_cycles` diagnostics.
        - `meta` (optional meta-model outputs).

### Meta-model hook

- Location: `advanced/meta_model.py`.
- Function: `predict_meta_signal(features: dict) -> dict`.
  - `features` contains:
    - Symbol, elapsed time.
    - Regime and cycle diagnostics.
    - Raw `engine_results` and `risk_results`.
    - Baseline bias and confidence.
  - Return a dict such as:
    - `{"meta_bias": "bullish", "meta_bias_confidence": 0.8, "meta_position_multiplier": 0.6}`.
- The advanced consensus (`consensus/advanced_consensus.py`) injects this result into:
  - `plan["advanced"]["meta"]`.
- Default implementation is a stub returning `{}` so it is safe to import even without a model.

### Testing and backtesting

- **Standard tests**:
  - Run `python run_all_tests.py`.
  - By default, tests marked `@pytest.mark.slow` (e.g. live Binance backtests) are skipped.
  - Set `INCLUDE_SLOW_TESTS=1` to include them:
    - `INCLUDE_SLOW_TESTS=1 python run_all_tests.py`.

- **Backtests (BTCUSDT/XRPUSDT, 2024–2025)**:
  - File: `test_backtest_binance_range.py`.
  - Requires:
    - `BINANCE_API_KEY` and `BINANCE_API_SECRET` in the environment.
    - `python-binance` installed.
  - Behaviour:
    - Fetches daily OHLCV for `BTCUSDT` and `XRPUSDT` from 2024-01-01 to 2025-12-31.
    - Runs **both** `consensus.aggregator.run` and `consensus.advanced_aggregator.run`.
    - Asserts success and validates advanced-specific fields.

