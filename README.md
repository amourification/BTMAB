# Temporal Market Analysis Bot

A deterministic, cycle-based cryptocurrency market analysis system built on 12 mathematical equations. The bot identifies dominant price cycles, calculates optimal position sizing, and generates complete trade plans — delivered through a dark-themed desktop GUI and a Telegram bot with full chart exports.

---

## Table of Contents

1. [What This Bot Does](#1-what-this-bot-does)
2. [The 12 Equations](#2-the-12-equations)
3. [System Architecture](#3-system-architecture)
4. [Installation](#4-installation)
5. [Configuration](#5-configuration)
6. [Running the Desktop GUI](#6-running-the-desktop-gui)
7. [Running the Telegram Bot](#7-running-the-telegram-bot)
8. [Telegram Commands Reference](#8-telegram-commands-reference)
9. [Understanding the Output](#9-understanding-the-output)
10. [The Five Charts Explained](#10-the-five-charts-explained)
11. [Risk Management Logic](#11-risk-management-logic)
12. [Export System](#12-export-system)
13. [Module Reference](#13-module-reference)
14. [Configuration Reference](#14-configuration-reference)
15. [Extending the Bot](#15-extending-the-bot)
16. [Troubleshooting](#16-troubleshooting)

---

## 1. What This Bot Does

The Temporal Market Analysis Bot approaches markets from a purely mathematical perspective: it assumes that price movement contains measurable, recurring cycles that can be detected, quantified, and acted upon.

For any given trading pair and interval, the bot:

- Fetches live OHLCV data from Binance and processes it through a preprocessing pipeline
- Runs 12 independent mathematical engines in parallel, each examining a different aspect of price cycle structure
- Aggregates all 12 outputs through a consensus layer that resolves conflicts and assigns confidence scores
- Generates a complete trade plan including position sizing, stop loss placement, hedging recommendations, and a risk grade
- Renders five detailed charts visualising the cycle state from multiple analytical perspectives
- Delivers everything through a Telegram message, a PDF report, or a live-refreshing desktop GUI

The system is **deterministic**: given identical input data, it always produces identical output. There is no randomness, no black-box ML, and no discretionary overrides. Every number in the output can be traced back to a specific equation.

---

## 2. The 12 Equations

The analytical engine is built around twelve equations, each contributing a specific signal to the final consensus.

### Equation 1 — Singular Spectrum Analysis (SSA)
**Module:** `engine/ssa_core.py`

SSA decomposes the price series into additive components by constructing a trajectory matrix, performing SVD, and reconstructing using the leading eigentriples. It extracts the dominant oscillatory mode from price data — the underlying "heartbeat" of the instrument without the noise.

**Output:** SSA-reconstructed cycle, SSA period estimate, current position within the cycle (early/mid/late ascending/descending), percentage of cycle completed.

**Key parameters:**
- `SSA_WINDOW_LENGTH = 256` — embedding dimension (frequency resolution)
- `SSA_NUM_COMPONENTS = 6` — number of leading eigentriples to reconstruct

### Equation 2 — Autocorrelation Function (ACF)
**Module:** `engine/acf.py`

Computes the ACF of log-returns up to `ACF_MAX_LAG` bars. The lag at which the first significant positive peak occurs identifies the dominant repetition cycle. Confidence bands (±1.96/√N) separate meaningful correlation from noise. The best lag is used to predict the next likely directional move.

**Output:** Best cycle lag, ACF strength at that lag, predicted next move direction, top three correlated lags.

### Equation 3 — Fast Fourier Transform (FFT)
**Module:** `engine/fft.py`

Transforms the detrended price series into the frequency domain, identifies the dominant frequency bin (peak power), converts back to a period in bars, and generates a time-domain envelope. Multiple dominant frequencies are ranked by amplitude. The FFT confirms or contradicts the SSA period estimate — agreement between the two increases consensus confidence.

**Output:** Dominant cycle period, top three periods by power, FFT envelope, spectral centroid.

### Equation 4 — Hilbert Transform (Instantaneous Phase)
**Module:** `engine/hilbert.py`

Applies the Hilbert transform to the detrended series to produce the analytic signal, from which instantaneous phase (0°–360°) and instantaneous amplitude (envelope) are extracted. Phase determines the precise position within the current cycle, classified into four zones: early bullish (0°–90°), mid-expansion (90°–180°), distribution (180°–270°), and accumulation (270°–360°).

**Output:** Current phase in degrees, phase zone, cycle turn type, bars to next phase boundary, turn urgency classification.

### Equation 5 — Solar Cycle / Seasonal Model
**Module:** `engine/solar.py`

Uses the `ephem` library to compute the sun's ecliptic longitude, producing a sine wave proxy for the annual solar cycle. Overlaid with four empirically-observed seasonal biases in crypto markets: spring rally (March–May), summer drift (June–August), autumn volatility (September–November), and year-end rally (December–January). Proximity to solstices and equinoxes raises a volatility flag.

**Output:** Solar phase (degrees), seasonal bias label, volatility flag, seasonal strength score, proximity to next solstice/equinox.

### Equation 6 — Murray Math Levels
**Module:** `engine/murray.py`

Divides the price range of the last `MURRAY_LOOKBACK` bars into eighths, producing nine price levels (0/8 through 8/8). These act as significant support and resistance zones. The 0/8 and 8/8 levels are major pivots; 4/8 is the mid-pivot around which price tends to oscillate. Murray Math classifies price action relative to these levels and generates a directional bias.

**Output:** Murray index (current level as a float 0.0–8.0), nearest level labels, action recommendation, distance to next level, confluence zones.

### Equation 7 — Kelly Criterion
**Module:** `engine/kelly.py`

Implements the Kelly Criterion with historical win rate and average win/loss ratio estimated from the cycle history. A half-Kelly multiplier (`KELLY_FRACTION = 0.5`) is applied to reduce drawdown risk. Position sizes are classified into six tiers from "No Trade" to "Maximum". Expected value per trade is calculated to confirm the bet is worth taking.

**Output:** Optimal position size (%), Kelly tier label, expected value per trade, minimum required cycles before Kelly is trusted.

### Equation 8 — Gamma / Realized Volatility
**Module:** `engine/gamma.py`

Computes a rolling realized volatility proxy and constructs a "gamma" regime indicator — analogous to dealer gamma in options markets — to identify whether the current volatility environment will amplify or dampen price moves. Negative gamma regimes correlate with explosive, trend-accelerating moves. Positive gamma regimes produce mean-reversion.

**Output:** Gamma proxy value, gamma regime (positive/negative/neutral), realized vol, regime confidence, hedge trigger flag.

### Equation 9 — Gann Fan Angles
**Module:** `engine/gann.py`

Identifies the most recent significant swing high or low within `GANN_LOOKBACK` bars, then projects seven geometric fan lines from that anchor at angles: 1×8, 1×4, 1×2, 1×1 (master), 2×1, 4×1, and 8×1. When price crosses a Gann angle, it signals a potential change in trend velocity. The 1×1 angle is considered the natural rate of price appreciation.

**Output:** Anchor price and bar, all seven angle prices at the current bar, most recently broken angle, distance from current price to each angle, directional bias.

### Equation 10 — Autoregressive Model (AR)
**Module:** `engine/ar_model.py`

Fits an AR(5) model to log-returns to capture momentum inertia. The model produces a short-term (10-bar) forecast adjusted by an overnight sentiment weight (`NIGHT_SENTIMENT_WEIGHT = 1.35`) reflecting asymmetric information content during low-liquidity hours. The dominant trading session (Asian/European/US) is detected from the instrument's most active volume period.

**Output:** AR(5) coefficients, 10-bar return forecast, sentiment-adjusted forecast, dominant session, model confidence (AIC-based).

### Equation 11 — Walras Market Clearing
**Module:** `engine/walras.py`

Applies Walrasian price-adjustment theory to the observed bid/ask imbalance and volume profile. The adjustment factor models how far price must move to clear the current order imbalance, generating a directional pressure estimate and a price level multiplier. Extreme Walras values signal potential liquidity shocks and trigger emergency stop logic.

**Output:** Walras adjustment factor, directional pressure, excess demand/supply estimate, liquidity shock flag, emergency stop trigger.

### Equation 12 — Detrending / Cycle Isolation
**Module:** `engine/detrend.py`

The foundation of the system. Removes the long-term trend from the price series using a combination of Hodrick-Prescott filtering and polynomial detrending, isolating the pure oscillatory component. Zero-crossings in the detrended series mark potential cycle turns. The detrended series is the input to SSA, FFT, Hilbert, and ACF.

**Output:** Detrended price series, trend component, zero-crossing bars (previous and predicted next), oscillator extremes.

---

## 3. System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        DATA LAYER                               │
│  data/fetcher.py ──► data/preprocessor.py ──► data/sentiment.py│
│  (Binance OHLCV)     (detrend, normalise)    (session detect)  │
└────────────────────────────┬────────────────────────────────────┘
                             │  clean OHLCV + preprocessed series
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    ENGINE LAYER  (12 equations)                 │
│  ssa_core ── fft ── hilbert ── acf ── detrend  (Cycle group)   │
│  solar ── murray ── kelly ── gamma ── gann ── ar ── walras      │
└────────────────────────────┬────────────────────────────────────┘
                             │  12 result dicts
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                       RISK LAYER                                │
│  risk/hedging.py ──► risk/stops.py ──► risk/portfolio.py       │
│  (hedge sizing)     (stop placement)  (exposure summary)       │
└────────────────────────────┬────────────────────────────────────┘
                             │  risk plan dict
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                     CONSENSUS LAYER                             │
│  consensus/aggregator.py — resolves conflicts, weights signals, │
│  generates final trade_plan dict with confidence score          │
└──────────────┬────────────────────────┬────────────────────────┘
               │                        │
               ▼                        ▼
┌──────────────────────┐    ┌───────────────────────────────────┐
│   DESKTOP GUI        │    │         TELEGRAM BOT              │
│  main_gui.py         │    │  bot/telegram_bot.py              │
│  ├── 5 chart tabs    │    │  ├── /run → summary + 5 charts    │
│  ├── trade plan panel│    │  ├── /export → PDF + CSV          │
│  └── auto-refresh    │    │  ├── /alert → turn notifications  │
│                      │    │  └── /schedule → auto-analysis    │
└──────────────────────┘    └───────────────────────────────────┘
               │                        │
               └─────────┬──────────────┘
                         ▼
           ┌──────────────────────────┐
           │     EXPORT SYSTEM        │
           │  charts/exporter.py      │
           │  ├── PDF (6-page report) │
           │  ├── PNG (per chart)     │
           │  └── CSV (flat metrics)  │
           └──────────────────────────┘
```

### Threading Model (Desktop GUI)

The GUI uses a producer/consumer pattern to keep the interface responsive during analysis (which takes 3–10 seconds):

- **Tkinter main thread** — runs the event loop, renders charts, handles button clicks
- **Background daemon thread** — runs the entire analysis pipeline; never touches Tkinter directly
- **StateManager** (`gui/gui_state.py`) — thread-safe bridge using `queue.Queue(maxsize=1)`. The background thread pushes `AnalysisResult` objects; the main thread polls every 400ms via `.after()`

### Threading Model (Telegram Bot)

The bot uses `python-telegram-bot`'s built-in `asyncio` event loop with `concurrent_updates=True`. Engine analysis runs in a `ThreadPoolExecutor` via `asyncio.run_in_executor()` so it does not block the event loop while computations are in progress.

---

## 4. Installation

### Prerequisites

- Python 3.11 or higher
- A Binance account with API access (read-only keys are sufficient — the bot never places orders)
- For the Telegram bot: a bot token from [@BotFather](https://t.me/botfather)

### Step 1 — Clone the project

```bash
git clone https://your-repo-url/temporal_bot.git
cd temporal_bot
```

### Step 2 — Quick setup (recommended for non-technical users)

In the project root you will find two helper scripts:

- `setup.ps1` — Windows (PowerShell)
- `setup.sh` — macOS / Linux / WSL

They create a `.venv` virtual environment and install all dependencies.

**Windows (PowerShell):**

```powershell
cd path\to\temporal_bot
.\setup.ps1
```

**macOS / Linux / WSL:**

```bash
cd /path/to/temporal_bot
chmod +x setup.sh        # first time only
./setup.sh
```

### Step 3 — Manual virtual environment setup (advanced users)

```bash
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
.venv\Scripts\activate           # Windows
```

### Step 4 — Install dependencies

```bash
pip install -r requirements.txt -r requirements-dev.txt
```

Core runtime dependencies:

| Package | Purpose |
|---------|---------|
| `numpy`, `scipy` | Core mathematics for all 12 engines |
| `pandas` | OHLCV data handling and CSV export |
| `statsmodels` | ACF computation and AR model fitting |
| `matplotlib` | All chart rendering and PDF/PNG export |
| `mplfinance` | Candlestick chart rendering |
| `ephem` | Astronomical calculations for the solar engine |
| `python-binance` | Live data from Binance |
| `python-telegram-bot` | Telegram bot framework (v20+, async) |
| `apscheduler` | Scheduled auto-analysis jobs |
| `python-dotenv` | `.env` file loading |

Developer / testing dependencies (from `requirements-dev.txt`):

| Package | Purpose |
|---------|---------|
| `pytest` | Unit and integration tests (`run_all_tests.py`, `test_gui.py`) |
| `tqdm` | Progress bars in `run_all_tests.py` |
| `ruff`, `black` (optional) | Linting and formatting during development |

### Step 5 — Create your `.env` file

```bash
cp .env.example .env
# Open .env and fill in your API credentials
```

### Step 6 — Verify installation

```bash
python -c "from consensus.aggregator import run; print('Import OK')"
```

---

## 5. Configuration

All configuration lives in a single `.env` file in the project root. The `config.py` module loads these at startup and exposes them as typed constants throughout the codebase.

### Required settings

```env
# Binance API — read-only keys are sufficient
BINANCE_API_KEY=your_binance_api_key_here
BINANCE_API_SECRET=your_binance_api_secret_here
```

### Telegram settings

```env
# Required only for the Telegram bot
TELEGRAM_BOT_TOKEN=your_telegram_bot_token_here

# Optional: comma-separated chat ID whitelist (leave blank to allow all)
ALLOWED_CHAT_IDS=123456789,987654321
```

### Analysis defaults

```env
DEFAULT_SYMBOL=BTCUSDT
DEFAULT_INTERVAL=1d

# Number of bars to fetch — must be at least 2× the dominant cycle
DEFAULT_BARS=2048
```

### Logging

```env
# DEBUG, INFO, WARNING, or ERROR
LOG_LEVEL=INFO
```

### GUI theme

```env
# GUI theme: "dark" (Ubuntu-style aubergine) or "light"
GUI_THEME=dark
```

### Advanced engine constants

Edit directly in `config.py` to tune behaviour:

| Constant | Default | Effect |
|----------|---------|--------|
| `DOMINANT_CYCLE_BARS` | 512 | Expected dominant cycle length |
| `SSA_WINDOW_LENGTH` | 256 | SSA frequency resolution (higher = finer) |
| `SSA_NUM_COMPONENTS` | 6 | Number of SSA components to reconstruct |
| `ACF_MAX_LAG` | 100 | Maximum autocorrelation lag to scan |
| `FFT_MIN_PERIOD` | 10 | Minimum detectable cycle (bars) |
| `MURRAY_LOOKBACK` | 64 | Price range window for Murray Math levels |
| `GANN_LOOKBACK` | 128 | Lookback window for swing high/low anchor |
| `KELLY_FRACTION` | 0.5 | Half-Kelly multiplier for position sizing |
| `MAX_POSITION_PCT` | 0.25 | Hard cap on any single position (25%) |
| `GAMMA_HEDGE_THRESHOLD` | -0.02 | Gamma level that triggers hedging |
| `AR_ORDER` | 5 | Order of the autoregressive model |
| `CHART_DPI` | 150 | Resolution of exported PNG/PDF charts |

---

## 6. Running the Desktop GUI

### Easiest way — helper scripts (`run_gui`)

After running the setup script once (`setup.ps1` on Windows or `setup.sh` on macOS/Linux/WSL), you can start the GUI with a single command.

**Windows (PowerShell):**

```powershell
cd path\to\temporal_bot
.\run_gui.ps1
```

**macOS / Linux / WSL:**

```bash
cd /path/to/temporal_bot
chmod +x run_gui.sh    # first time only
./run_gui.sh
```

The scripts automatically use the `.venv` virtual environment and run `main_gui.py`. If `.venv` is missing, they tell you to run the appropriate setup script.

### Running tests and the test GUI

After running the setup script once:

- **CLI test runner:**

  ```bash
  python run_all_tests.py
  ```

  By default this skips tests marked as `@pytest.mark.slow` (for example, the live Binance backtest). To include them:

  ```bash
  INCLUDE_SLOW_TESTS=1 python run_all_tests.py
  ```

- **Graphical test runner:**

  ```bash
  python test_gui.py
  ```

  Lets you pick which test modules to run, toggle slow tests, see a progress bar, and inspect pytest output in a terminal-style panel.

### Theme selector

The desktop GUI uses a theme system inspired by modern Ubuntu:

- Default: `GUI_THEME=dark` (aubergine background + orange accents)
- Optional: `GUI_THEME=light` (light panels, aubergine text)

You can also change this **inside the GUI** via the `Theme` dropdown in the top toolbar (right side). When you switch theme, the preference is written to `.env` and you’ll be prompted to restart the app so that all charts and panels pick up the new palette.

### Interface layout

```
┌──────────────────────────────────────────────────────────────────┐
│  Symbol [BTCUSDT] Interval [1d▼] Bars [512] [▶ RUN] [⬇ EXPORT] │
│  Auto-refresh ☑                                                  │
├────────────────────────────────┬─────────────────────────────────┤
│  [Cycle][Murray][Phase][Risk]  │  ── TEMPORAL CYCLE ──────────   │
│  [Gann ]                       │  Phase         215.00°          │
│                                │  Turn Type     Distribution     │
│  ┌─────────────────────────┐   │  Turn Urgency  HIGH             │
│  │                         │   │  Bars to Turn  12               │
│  │  Matplotlib chart       │   │                                 │
│  │  (active tab)           │   │  ── MARKET CONTEXT ──────────   │
│  │                         │   │  Bias          BEARISH          │
│  │                         │   │  Murray Level  6.70/8           │
│  └─────────────────────────┘   │  Gamma Regime  Negative         │
│  [Navigation toolbar]          │  Seasonal      Autumn Volatility│
│                                │                                 │
│                                │  ── TRADE PLAN ──────────────   │
│                                │  Kelly %       8.00%            │
│                                │  Hedge Ratio   25.00%           │
│                                │  Active Stop   59,400.00        │
│                                │  R/R Grade     B                │
├────────────────────────────────┴─────────────────────────────────┤
│  ✓ BTCUSDT — complete    4.3s    ↺ 287s                          │
└──────────────────────────────────────────────────────────────────┘
```

### Controls

**Symbol field** — Any Binance spot pair (e.g., `ETHUSDT`, `SOLUSDT`, `BNBUSDT`). Automatically uppercased.

**Interval dropdown** — Standard Binance kline intervals from `1m` through `1M`. For cycle detection, `4h` or higher is recommended.

**Bars field** — Number of historical bars to fetch. Range: 64–4096. For daily cycle analysis, 512–2048 is recommended. More bars improve SSA accuracy but increase analysis time.

**▶ RUN button** — Starts the analysis. Disabled while analysis is running. The status bar shows progress.

**⬇ EXPORT button** — Opens a folder picker and generates a full PDF report + CSV in the selected location. Requires a completed analysis run.

**Auto-refresh checkbox** — When ticked, re-runs analysis every 5 minutes. The countdown (`↺ 287s`) resets after each successful run. Untick to run only on demand.

### Chart tabs

Each of the five tabs shows a different analytical perspective. All charts are refreshed in the background after each run — switching tabs is instant.

The **Matplotlib navigation toolbar** below each chart provides: pan, zoom, zoom-to-rectangle, and save-as-PNG.

---

## 7. Running the Telegram Bot

### Step 1 — Get a bot token

Message [@BotFather](https://t.me/botfather) on Telegram, send `/newbot`, follow the prompts, and copy the token.

### Step 2 — Configure credentials

In your `.env`:
```env
TELEGRAM_BOT_TOKEN=1234567890:ABCdefGhIJKlmNoPQRsTUVwxyZ
```

Or export directly:
```bash
export BOT_TOKEN=1234567890:ABCdefGhIJKlmNoPQRsTUVwxyZ
```

### Step 3 — Restrict access (recommended)

Find your personal Telegram chat ID by messaging [@userinfobot](https://t.me/userinfobot), then set:
```env
ALLOWED_CHAT_IDS=123456789
```

### Step 4 — Start the bot

```bash
python -m bot.telegram_bot
```

Expected output:
```
2026-03-01 12:00:00 [INFO] temporal_bot.bot.telegram_bot — Bot started — commands registered
2026-03-01 12:00:00 [INFO] temporal_bot.bot.telegram_bot — Starting polling…
```

Send `/start` in Telegram to verify the bot is running.

### Running as a background service

```bash
# Start in background
nohup python -m bot.telegram_bot > logs/bot.log 2>&1 &
echo $! > bot.pid

# Stop
kill $(cat bot.pid)
```

With `systemd` (`/etc/systemd/system/temporal-bot.service`):

```ini
[Unit]
Description=Temporal Market Analysis Bot
After=network.target

[Service]
Type=simple
WorkingDirectory=/path/to/temporal_bot
EnvironmentFile=/path/to/temporal_bot/.env
ExecStart=/path/to/.venv/bin/python -m bot.telegram_bot
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Then:
```bash
sudo systemctl daemon-reload
sudo systemctl enable temporal-bot
sudo systemctl start temporal-bot
sudo systemctl status temporal-bot
```

---

## 8. Telegram Commands Reference

### `/start`
Welcome message with bot description and pointer to `/help`.

### `/help`
Full command reference card listing all commands, syntax, and descriptions.

### `/run [SYMBOL] [INTERVAL] [BARS]`
Runs the full analysis pipeline and delivers summary + five charts.

```
/run                       Uses your chat's default pair
/run BTCUSDT               BTCUSDT with default interval and bars
/run ETHUSDT 4h            ETHUSDT on 4h, default bar count
/run SOLUSDT 1d 1024       SOLUSDT daily with 1024 bars
```

**Sequence of events after `/run`:**
1. Bot sends "⏳ Analysing…" acknowledgement immediately
2. Full pipeline runs in background (3–10 seconds)
3. Formatted summary message is sent with all key metrics
4. Five chart images are sent individually (Cycle, Murray, Phase, Risk, Gann)
5. Alert conditions are evaluated — if a turn is imminent, a separate alert fires automatically

### `/run_adv [SYMBOL] [INTERVAL] [BARS]`
Runs the **advanced** analysis pipeline (regime-aware, with uncertainty metrics) and delivers summary + five charts, analogous to `/run`.

The syntax mirrors `/run`:

```
/run_adv                    Uses your chat's default pair
/run_adv BTCUSDT            BTCUSDT with default interval and bars
/run_adv ETHUSDT 4h         ETHUSDT on 4h, default bar count
/run_adv SOLUSDT 1d 1024    SOLUSDT daily with 1024 bars
```

### `/status`
Resends the summary from the most recent `/run` without rerunning the analysis. Useful for quickly rechecking the trade plan without waiting.

### `/status_adv`
Same as `/status`, but for the most recent `/run_adv` result (advanced pipeline).

### `/export`
Generates and delivers two files as Telegram document attachments:

- **PDF (6 pages)** — Summary page + all five charts at full resolution
- **CSV** — Every metric from the trade plan in a flat, spreadsheet-ready file

### `/export_adv`
Advanced-mode export: sends PDF + CSV for the most recent `/run_adv` analysis.

### `/setpair SYMBOL [INTERVAL] [BARS]`
Sets the default pair for your chat. Persists for the current bot session.

```
/setpair ETHUSDT
/setpair SOLUSDT 4h
/setpair BNBUSDT 1d 2048
```

### `/alert on|off`
Enables or disables automatic alert messages after each analysis run.

### `/alert threshold N`
Sets the number of bars-to-turn at which a turn alert fires. Default: 20 bars.

```
/alert threshold 20    Alert when ≤ 20 bars to turn (default)
/alert threshold 5     Only very imminent turns
/alert threshold 50    Early warning at 50 bars out
```

### `/schedule INTERVAL|off`
Configures automatic recurring analysis for your chat.

```
/schedule 5m     Every 5 minutes
/schedule 1h     Every hour
/schedule 4h     Every 4 hours
/schedule 1d     Once per day
/schedule off    Disable scheduled analysis
/schedule        Show current schedules
```

### Alert types

Three automatic alert types are sent when conditions are detected (subject to a 1-hour throttle per type per chat):

| Alert | Trigger | Emoji |
|-------|---------|-------|
| Turn Alert | Bars-to-turn ≤ threshold, or urgency = high/immediate | 🔴 |
| Sweep Alert | Walras engine detects previous session H/L was swept | ⚡ |
| Emergency Alert | Walras liquidity shock triggers emergency stop — overrides throttle | 🆘 |

---

## 9. Understanding the Output

### Confidence Score

The confidence percentage measures how many of the 12 engines agree on the current directional bias. Above 70% means strong consensus (8+ engines aligned). Below 40% indicates disagreement — position sizing is automatically reduced by the Kelly criterion.

### Phase and Turn Type

Phase (0°–360°) represents the precise position in the dominant cycle as measured by the Hilbert transform:

| Zone | Phase | Market Condition |
|------|-------|-----------------|
| Early Bullish | 0°–90° | Fresh cycle low behind us, uptrend beginning |
| Mid Expansion | 90°–180° | Mid-cycle momentum, trend accelerating |
| Distribution | 180°–270° | Cycle peak zone, trend exhausting |
| Accumulation | 270°–360° | Cycle bottom zone, selling exhaustion |

**Turn Urgency levels:**

| Level | Bars to Turn | Meaning |
|-------|-------------|---------|
| `none` | > 60 | Far from any transition |
| `low` | 40–60 | Approaching but not imminent |
| `medium` | 20–40 | Getting close |
| `high` | 10–20 | Turn is nearby |
| `immediate` | < 10 | Turn is extremely close |
| `emergency` | — | Active liquidity shock |

### Murray Level (0/8 – 8/8)

| Level | Significance |
|-------|-------------|
| 0/8 | Major support — historically strong buying zone |
| 1/8 | Weak support |
| 2/8 | Strong support |
| 3/8 | Lower pivot |
| 4/8 | Major pivot — equal support and resistance |
| 5/8 | Upper pivot |
| 6/8 | Strong resistance |
| 7/8 | Weak resistance |
| 8/8 | Major resistance — historically strong selling zone |

Price at or above 7/8 is considered technically overbought. At or below 1/8, technically oversold.

### Kelly Position Size

| Tier | Size | Meaning |
|------|------|---------|
| No Trade | 0% | Expected value ≤ 0 — skip |
| Minimal | < 2% | Very low edge |
| Small | 2–5% | Below-average confidence |
| Moderate | 5–10% | Normal operating range |
| Large | 10–20% | High-confidence setup |
| Maximum | > 20% | Exceptional setup |

The 25% hard cap (`MAX_POSITION_PCT`) applies regardless of Kelly output.

### Risk/Reward Grade

| Grade | R/R Ratio | Meaning |
|-------|-----------|---------|
| A | > 3:1 | Premium setup |
| B | 2:1 to 3:1 | Good — deploy normal size |
| C | 1:1 to 2:1 | Acceptable — consider reduced size |
| D | < 1:1 | Unfavourable — consider skipping |

---

## 10. The Five Charts Explained

### Cycle Chart (Equations 1, 3, 4)
Three vertically stacked panels.

**Top panel — Price with SSA overlay:** The raw price line in cyan with the SSA-reconstructed cycle in orange. Background shading by Hilbert phase zone: green during bullish phases, red during bearish phases. Cycle zero-crossings marked with vertical dashed lines.

**Middle panel — Detrended oscillator:** Price with trend removed (gold), with the FFT-derived amplitude envelope (purple) overlaid. Moving above or below zero suggests potential transitions.

**Bottom panel — Hilbert phase:** Continuous 0°–360° phase curve. Phase zone boundaries are marked. As the curve approaches 180° (cycle peak) or 360° (cycle trough), a turn is near.

### Murray Chart (Equations 6, 9)
Two panels.

**Top panel — OHLC with Murray levels and Gann fan:** Candlestick chart with all nine Murray levels as horizontal lines (green for support, grey for pivot, red for resistance). 0/8, 4/8, and 8/8 are drawn thicker. Seven Gann fan angles radiate from the most recent swing point, with the 1×1 master angle in gold. Recent angle breaks annotated with directional arrows.

**Bottom panel — Murray heatbar:** Horizontal colour gradient (green left to red right) with a white needle showing the current Murray position.

### Phase Chart (Equations 2, 5, 10)
Three panels.

**Top panel — ACF correlogram:** Bar chart of the autocorrelation function. Significant lags (outside ±1.96/√N confidence bands shown as dotted lines) are highlighted in gold. The best cycle lag is annotated with its period length.

**Middle panel — AR forecast:** Historical log-returns for the last 40 bars followed by the 10-bar AR model forecast (amber dashed) and the sentiment-adjusted forecast (cyan dashed). Dominant trading session labelled.

**Bottom panel — Solar cycle:** Solar sine wave with seasonal bias shading. Spring rally in dark green, summer drift in amber, autumn volatility in deep red, year-end rally in deep blue. Volatility flag displayed when within 7 days of a solstice or equinox.

### Risk Chart (Equations 7, 8, 11)
2×2 grid of four panels.

**Kelly Gauge (top-left):** Semicircular arc from 0% to 25% with a needle pointing to the Kelly-optimal position size. Arc colour-coded by tier: grey (no trade) → yellow (minimal) → green (moderate) → orange (large) → red (maximum).

**Stop Ladder (top-right):** Three horizontal bars showing initial stop, trailing stop, and the active stop (currently enforced). The active stop bar is highlighted. All shown relative to current price.

**Gamma/Volatility (bottom-left):** Realized volatility sparkline with gamma proxy overlaid. Background shaded by regime: red for negative gamma (explosive moves expected), blue for positive gamma (mean-reversion expected).

**Portfolio Summary (bottom-right):** Horizontal exposure bars for net, gross, primary, and hedge positions as percentages. R/R letter grade displayed as a colour-coded badge.

### Gann Chart (Equation 9)
Two panels.

**Top panel — Gann fan:** Full-width price chart with all seven fan lines from the anchor point (marked with a triangle — green upward for swing low, red downward for swing high). Lines colour-graduated from cyan (shallow) through gold (1×1 master) to red (steep). Recent angle breaks annotated.

**Bottom panel — Distance bars:** Horizontal bars showing signed distance between current price and each angle. Bars pointing right = price above that angle (supporting); bars pointing left = price below (resistance). Visual scan of overall support/resistance posture.

---

## 11. Risk Management Logic

### Stop Loss Placement

The stops engine calculates three levels simultaneously:

**Initial stop** — Placed just beyond the most recent significant swing in the opposite direction of the trade, using the Gann anchor as reference.

**Trailing stop** — A volatility-scaled dynamic stop that tightens as the trade moves in your favour. The trailing percentage is widened during high-volatility regimes to avoid premature stop-outs.

**Emergency stop** — Triggered by the Walras engine upon detecting a liquidity shock. This is a hard exit signal that overrides all other levels.

The **active stop** is whichever of the three is most conservative (closest to current price while still being executable).

### Hedging

Hedging is triggered when two or more of these conditions are met:

- Gamma engine reports negative gamma (value below `GAMMA_HEDGE_THRESHOLD = -0.02`)
- Hilbert phase is in the distribution or accumulation zone (180°–360°)
- Murray level is above 6/8 for a long position, or below 2/8 for a short position

Starting hedge ratio is `DEFAULT_HEDGE_RATIO = 0.30` (30%), scaled by urgency. High urgency pushes the ratio toward 50%.

### Portfolio Sizing

The portfolio engine applies this chain: Kelly output → half-Kelly fraction → `MAX_POSITION_PCT` cap → hedge deduction → final net exposure.

The overall risk score (0.0 to 1.0) is a weighted combination of: phase proximity to distribution zone, Murray level extremity, gamma regime severity, and Walras directional pressure.

---

## 12. Export System

### PDF Report (6 pages)

1. **Summary** — Text metrics dashboard with confidence bar, all trade plan fields grouped into four sections
2. **Cycle chart** — Full-page three-panel cycle visualisation
3. **Murray chart** — Full-page Murray Math + Gann fan
4. **Phase chart** — Full-page ACF / AR / Solar dashboard
5. **Risk chart** — Full-page 2×2 risk grid
6. **Gann chart** — Full-page dedicated Gann analysis

All pages use the dark terminal aesthetic matching the live GUI.

### PNG Export

Individual charts exported as PNG. Resolution controlled by `CHART_DPI = 150` in `config.py`.

### CSV Export

A flat file with every metric from the trade plan dict — all 12 engine outputs plus risk and consensus values in one row. Column headers are human-readable. Suitable for logging setups over time or importing into Excel/Google Sheets for backtesting analysis.

### File naming convention

```
BTCUSDT_1d_20260301_120000.pdf
BTCUSDT_1d_20260301_120000.csv
cycle_chart_BTCUSDT_1d_20260301_120000.png
```

---

## 13. Module Reference

### Data layer

| Module | Role |
|--------|------|
| `data/fetcher.py` | Fetches OHLCV from Binance. Handles rate limiting, pagination, retries |
| `data/fetcher_utils.py` | Interval conversion, data validation, gap detection |
| `data/preprocessor.py` | HP filter detrending, log-returns, normalisation, rolling statistics |
| `data/preprocessor_utils.py` | Rolling window utilities, outlier detection, interpolation |
| `data/sentiment.py` | Session detection (Asian/EU/US), overnight gap scoring |

### Engine layer

| Module | Equation | Key Outputs |
|--------|----------|------------|
| `engine/ssa_core.py` | Eq 1 | Cycle reconstruction, period, position |
| `engine/ssa_utils.py` | Eq 1 | SVD utilities, component grouping |
| `engine/acf.py` | Eq 2 | Best lag, cycle strength, directional prediction |
| `engine/fft.py` | Eq 3 | Dominant period, spectral envelope, top periods |
| `engine/fft_utils.py` | Eq 3 | Window functions, frequency bin utilities |
| `engine/hilbert.py` | Eq 4 | Phase (°), zone, turn urgency, bars to turn |
| `engine/solar.py` | Eq 5 | Solar phase, seasonal bias, volatility flag |
| `engine/murray.py` | Eq 6 | Price level (0–8), action, confluence |
| `engine/murray_utils.py` | Eq 6 | Level calculation, heatbar data |
| `engine/kelly.py` | Eq 7 | Position %, tier, expected value |
| `engine/gamma.py` | Eq 8 | Gamma proxy, regime, realized vol |
| `engine/gamma_utils.py` | Eq 8 | Vol computation, regime classification |
| `engine/gann.py` | Eq 9 | Anchor, 7 angle prices, breaks, distances |
| `engine/gann_utils.py` | Eq 9 | Angle projection, break detection |
| `engine/ar_model.py` | Eq 10 | 10-bar forecast, sentiment adjustment |
| `engine/ar_utils.py` | Eq 10 | Model fitting, AIC scoring |
| `engine/walras.py` | Eq 11 | Adjustment factor, pressure, shock flag |
| `engine/detrend.py` | Eq 12 | Detrended series, trend, zero-crossings |

### Risk layer

| Module | Role |
|--------|------|
| `risk/hedging.py` | Hedge ratio, urgency, action recommendation |
| `risk/hedging_utils.py` | Trigger detection, ratio scaling |
| `risk/stops.py` | Initial / trailing / emergency stop calculation |
| `risk/stops_utils.py` | Swing detection, ATR, stop type selection |
| `risk/portfolio.py` | Net/gross exposure, R/R grade, risk score |
| `risk/portfolio_utils.py` | Exposure arithmetic, grade computation |

### Consensus layer

| Module | Role |
|--------|------|
| `consensus/aggregator.py` | Runs all 12 engines, resolves conflicts, builds trade plan |
| `consensus/aggregator_utils.py` | Conflict detection, confidence scoring, bias aggregation |

### Charts layer

| Module | Panels |
|--------|--------|
| `charts/cycle_chart.py` | Price+SSA / Detrended+FFT / Hilbert phase |
| `charts/murray_chart.py` | OHLC+Murray+Gann / Murray heatbar |
| `charts/phase_chart.py` | ACF / AR forecast / Solar |
| `charts/risk_chart.py` | Kelly gauge / Stop ladder / Gamma vol / Portfolio |
| `charts/risk_chart_utils.py` | Individual panel draw functions |
| `charts/gann_chart.py` | Gann fan / Angle distance bars |
| `charts/exporter.py` | PDF / PNG / CSV export |
| `charts/exporter_utils.py` | Summary page renderer, trade plan flattener |

### GUI layer

| Module | Role |
|--------|------|
| `main_gui.py` | `App(tk.Tk)` — entry point, layout, thread management |
| `gui/gui_state.py` | Thread-safe `StateManager`, `AppState`, poll queue |
| `gui/gui_widgets.py` | `ControlPanel`, `StatusBar`, `MetricLabel`, `TradeplanPanel` |
| `gui/gui_widgets_utils.py` | TradeplanPanel scrollable canvas builder |
| `gui/gui_charts.py` | `BaseChartTab` + five concrete tab classes |

### Bot layer

| Module | Role |
|--------|------|
| `bot/telegram_bot.py` | Application bootstrap, command registration, `main()` |
| `bot/bot_commands.py` | Eight `/command` handlers |
| `bot/bot_commands_utils.py` | `run_engine()`, `send_charts()`, `send_export()` |
| `bot/bot_scheduler.py` | `BotScheduler`, `AlertThrottle`, `detect_alerts()` |
| `bot/message_builder.py` | All Telegram MarkdownV2 message formatting |

---

## 14. Configuration Reference

### `.env` keys

| Key | Type | Required | Default | Description |
|-----|------|----------|---------|-------------|
| `BINANCE_API_KEY` | str | Yes | — | Binance API key (read-only) |
| `BINANCE_API_SECRET` | str | Yes | — | Binance API secret |
| `TELEGRAM_BOT_TOKEN` | str | Bot only | — | Telegram token from BotFather |
| `ALLOWED_CHAT_IDS` | str | No | `""` | Comma-separated whitelist |
| `DEFAULT_SYMBOL` | str | No | `BTCUSDT` | Default pair |
| `DEFAULT_INTERVAL` | str | No | `1d` | Default kline interval |
| `DEFAULT_BARS` | int | No | `2048` | Default bar count |
| `LOG_LEVEL` | str | No | `INFO` | Logging verbosity |

### `config.py` constants

| Constant | Default | Description |
|----------|---------|-------------|
| `DOMINANT_CYCLE_BARS` | `512` | Expected dominant cycle length |
| `SSA_WINDOW_LENGTH` | `256` | SSA trajectory matrix width |
| `SSA_NUM_COMPONENTS` | `6` | SSA eigentriples to reconstruct |
| `ACF_MAX_LAG` | `100` | Max ACF lag |
| `FFT_MIN_PERIOD` | `10` | Min detectable FFT cycle |
| `MURRAY_LOOKBACK` | `64` | Murray Math price range window |
| `GANN_LOOKBACK` | `128` | Swing high/low search window |
| `KELLY_FRACTION` | `0.5` | Half-Kelly multiplier |
| `KELLY_MIN_CYCLES` | `5` | Min cycles before Kelly is trusted |
| `MAX_POSITION_PCT` | `0.25` | Position size hard cap |
| `GAMMA_HEDGE_THRESHOLD` | `-0.02` | Gamma level triggering hedge |
| `DEFAULT_HEDGE_RATIO` | `0.30` | Starting hedge ratio |
| `AR_ORDER` | `5` | Autoregressive model order |
| `NIGHT_SENTIMENT_WEIGHT` | `1.35` | Overnight gap sentiment multiplier |
| `SOLAR_PROXIMITY_DAYS` | `7` | Days from solstice to raise vol flag |
| `CHART_DPI` | `150` | Chart export resolution |

---

## 15. Extending the Bot

### Adding a new engine

1. Create `engine/your_engine.py` with `run(ohlcv_df, cfg) -> dict`
2. Create `engine/your_engine_utils.py` for helpers (keep both under 300 lines)
3. Import and call it in `consensus/aggregator.py` inside the engine execution block
4. Add its key to the aggregator's consensus-scoring logic in `consensus/aggregator_utils.py`
5. Add its metric to `gui/gui_widgets.py`'s `_SECTIONS` list for GUI display
6. Add its metric to `bot/message_builder.py` if warranted in the Telegram summary

### Adding a new chart

1. Create `charts/your_chart.py` with `draw(axes, trade_plan)` and `render(trade_plan)` functions
2. Follow the dual-interface pattern: `draw()` accepts caller-provided axes; `render()` creates its own Figure and is used for exports
3. Add `YourTab(BaseChartTab)` to `gui/gui_charts.py` and append it to `TAB_CLASSES`
4. Add the render call to `charts/exporter.py`'s `export_pdf()` for a new PDF page
5. Add the render call to `bot/bot_commands_utils.py`'s `send_charts()` for Telegram delivery

### Adding a new Telegram command

1. Write `async def cmd_yourcommand(update, context)` in `bot/bot_commands.py`
2. Register it in `bot/telegram_bot.py`'s `build_application()` with `CommandHandler`
3. Add the entry to `BOT_COMMANDS` in `bot/telegram_bot.py`
4. Add the `/yourcommand` description to `build_help()` in `bot/message_builder.py`

### Changing the data source

`data/fetcher.py` exposes `fetch_ohlcv(symbol, interval, bars, cfg) -> pd.DataFrame`. The DataFrame must have columns `open`, `high`, `low`, `close`, `volume` with a `datetime` index. Replace the Binance client with any other exchange (Bybit, Coinbase, Yahoo Finance, CSV files) by matching this interface.

---

## 16. Troubleshooting

**"BINANCE_API_KEY is not set. Data fetch will fail."**
The `.env` file is missing or still contains placeholder values. Run `cp .env.example .env` and fill in real credentials. Read-only API keys are sufficient.

**Analysis takes very long**
Reduce the Bars value. With 2048 bars, SSA performs SVD on a large matrix — try 512 bars for faster runs. For the Telegram bot, reduce scheduling frequency.

**Charts are blank or show "Chart error"**
One or more engines returned insufficient data. The most common cause is a bar count too low for cycle detection — SSA requires at least `2 × SSA_WINDOW_LENGTH` (512 bars minimum). Verify the symbol and interval are valid Binance pairs.

**Telegram bot doesn't respond**
Verify the token is correct and the bot is running. If `ALLOWED_CHAT_IDS` is set, confirm your chat ID is included — use [@userinfobot](https://t.me/userinfobot) to find it.

**"No module named 'ephem'"**
```bash
pip install ephem
```

**"No module named 'apscheduler'"**
```bash
pip install apscheduler
```

**Alerts are not firing**
Alerts are throttled — the same alert type is suppressed for 1 hour per chat. Restart the bot to reset the throttle (it is in-memory only). Check that alerts are enabled with `/alert on` and the threshold is appropriate for your interval.

**Murray levels appear wrong**
Murray Math requires a sufficient price range within the lookback window. If the instrument has been in a narrow range, levels will cluster. Increase `MURRAY_LOOKBACK` in `config.py` for a wider historical range.

**The GUI freezes during analysis**
Analysis runs entirely in the background thread. If freezes occur, check the console for `[WARNING]` lines — chart rendering errors are caught and displayed inline rather than crashing the GUI.

---

## Project Statistics

| Metric | Value |
|--------|-------|
| Total Python modules | 50 |
| Total lines of code | 9,141 |
| Modules over 300 lines | 0 |
| Mathematical equations | 12 |
| Chart types | 5 |
| Telegram commands | 8 |
| Export formats | 3 (PDF, PNG, CSV) |
| Supported kline intervals | 14 |
| Alert types | 3 (Turn, Sweep, Emergency) |

---

*Built with Python 3.11 · matplotlib · python-telegram-bot v20 · APScheduler · NumPy · SciPy · statsmodels · ephem · python-binance*
