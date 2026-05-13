# StockProb - Quick Start Guide

## System Overview

StockProb is a quantitative multi-agent stock analysis engine that computes the probability a stock moves UP over a user-defined time window. The system uses **6 specialized sub-agents** working in coordination:

1. **Price Historian** - Historical OHLCV data, log returns, moving averages, regime detection
2. **Volatility Engine** - Multi-window realized vol, Parkinson estimator, GARCH(1,1), VIX correlation
3. **Earnings Analyst** - Historical earnings reactions, upcoming event risk, EPS surprise trends
4. **Sentiment Scanner** - News sentiment scoring, SEC filings, insider trading direction
5. **Macro Puller** - FRED macro indicators (rates, yields, CPI, unemployment), sector performance
6. **Monte Carlo** - 10,000-path GBM simulation with drift adjustments and earnings shock injection

## Installation

```bash
cd /workspace
export PYTHONPATH=/workspace:$PYTHONPATH
```

Dependencies already installed:
- yfinance (market data)
- numpy, pandas, scipy (numerical computing)
- pydantic (data validation)
- arch (GARCH modeling)
- streamlit (web UI)
- plotly (visualization)

## Usage Options

### Option 1: Web Interface (Recommended)

Launch the Streamlit web app:

```bash
cd /workspace
streamlit run ui/app.py --server.address=0.0.0.0 --server.port=8501
```

Then open your browser to `http://localhost:8501`

**Features:**
- Interactive input form with sliders and date picker
- Visual probability card with color coding
- Signal dashboard showing all 6 agent outputs
- Return distribution table (P10-P90)
- Risk metrics and flags
- Agent health status indicators
- JSON export option

### Option 2: Command Line Interface

Quick terminal-based analysis:

```bash
# Basic usage
python ui/run_cli.py --ticker AAPL --window 30

# With all options
python ui/run_cli.py --ticker TSLA --window 7 --risk high --date 2026-05-13

# Use real market data (not mock)
python ui/run_cli.py --ticker MSFT --window 30 --real-data

# Output raw JSON only
python ui/run_cli.py --ticker AAPL --window 30 --json
```

**CLI Arguments:**
- `--ticker, -t`: Stock symbol (required)
- `--window, -w`: Days horizon (default: 30)
- `--date, -d`: Analysis date YYYY-MM-DD (default: today)
- `--risk, -r`: low|moderate|high (default: moderate)
- `--real-data`: Use real yfinance data instead of mock
- `--json`: Output raw JSON only

### Option 3: Python API

Direct integration in your code:

```python
from datetime import date
from stockprob.services.orchestrator import run_analysis
from stockprob.core.models import RiskTolerance

result = run_analysis(
    ticker='AAPL',
    window_days=30,
    as_of_date=date(2026, 5, 13),
    risk_tolerance=RiskTolerance.MODERATE,
    use_mock_data=True  # Set False for real data
)

print(f"Probability UP: {result.probability_up:.1%}")
print(f"Confidence: {result.confidence_score}/100")
print(f"P50 Return: {result.monte_carlo.p50_return:.1%}")
```

## Example Output

```
📈 Probability UP: 60.8%
   over 30 days from 2026-05-13

🎯 Confidence Score: 55/100

📡 Signal Dashboard:
   Risk:    elevated     (Beta: 0.03)
   Trend:   transitional (vs MA200: below)
   Earnings:⚠️ In Window (mixed)
   Sentiment:+0.05      (Insiders: selling)
   Macro:   recovery     (Yield Curve: -0.15%)

📈 Return Distribution (10,000 paths):
   P10:  -10.3%  |  P25:   -4.1%  |  P50:    3.1%  |  P75:   10.8%  |  P90:   18.0%

⚠️ Risk Metrics:
   Prob Loss > 10%:  10.5%
   Prob Gain > 20%:  7.5%
   Median Max DD:    -8.4%

🚩 Risk Flags:
   ⚠️ Earnings report falls within your window. Expect fat tails.
   ⚠️ Vol regime elevated. Widen your expected range.

🤖 Agent Status:
   ✅ Price Historian   : ok
   ✅ Volatility Engine : ok
   ✅ Earnings Analyst  : ok
   ✅ Sentiment Scanner : ok
   ✅ Macro Puller      : ok
   ✅ Monte Carlo       : ok
```

## Key Design Principles

1. **Math, Not Opinion**: Every output derived from counted frequencies, historical distributions, and statistical inference
2. **No Fabrication**: Missing data = null fields + reduced confidence score
3. **Isolated Signals**: Agents operate independently; only Monte Carlo sees all signals as numeric adjustments
4. **Fat Tails Real**: Earnings shocks injected as discrete events, not smoothed
5. **Dated Data**: All macro data timestamped; stale data triggers warnings
6. **No Buy/Sell Calls**: System provides probabilities only; user decides

## Configuration

Edit `/workspace/stockprob/config/settings.py` to customize:
- Sector ETF mappings
- FRED series IDs
- Volatility regime thresholds
- Confidence score adjustments
- Monte Carlo parameters (default: 10,000 paths)

## Mock vs Real Data

**Mock Data (Default)**: Generates realistic synthetic data for testing without API calls. Useful for:
- Development and testing
- Demonstrations
- Environments without internet access

**Real Data**: Fetches actual market data from yfinance and FRED. Enable with:
- CLI: `--real-data` flag
- Python API: `use_mock_data=False`
- Web UI: Uncheck "Use Mock Data" checkbox

## Troubleshooting

**Module not found errors:**
```bash
export PYTHONPATH=/workspace:$PYTHONPATH
```

**Streamlit port already in use:**
```bash
streamlit run ui/app.py --server.port=8502
```

**Real data fetch failures:**
- Check internet connectivity
- Verify ticker symbol is valid
- Some tickers may have limited history

## Disclaimer

**This is math. Not advice.** Past distributions do not guarantee future outcomes. The system provides probabilistic estimates based on historical data and statistical models. Always conduct your own research before making investment decisions.
