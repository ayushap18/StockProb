# StockProb — Multi-Agent Stock Probability Engine

StockProb is a Python library that estimates the probability of a stock moving up over a user-defined forward window using a pipeline of six specialized agents. Each agent runs independently, and their outputs are aggregated by an orchestrator that produces a single structured JSON payload.

> **Disclaimer:** This is math, not financial advice. Past distributions do not guarantee future outcomes.

---

## How It Works

```
Ticker + Window → [6 Agents] → Probability(up) + Confidence Score + Risk Flags
```

Six agents run in sequence on every analysis request:

| # | Agent | Responsibility |
|---|-------|----------------|
| 1 | **Price Historian** | Fetches full OHLCV history, computes log returns, detects price regime via 50/200-day MA |
| 2 | **Volatility Engine** | Realized vol (10/21/63/252-day), Parkinson estimator, GARCH(1,1) forecast, VIX percentile, beta |
| 3 | **Earnings Analyst** | Earnings reaction history, next earnings date, EPS surprise trend, earnings-in-window flag |
| 4 | **Sentiment Scanner** | News headline sentiment (FinBERT/VADER), SEC filings, insider net direction, short interest |
| 5 | **Macro Puller** | FRED data (fed funds, treasury yields, CPI, unemployment), yield curve, sector ETF performance |
| 6 | **Monte Carlo** | 10,000-path GBM simulation using blended volatility; returns full return distribution |

The **Orchestrator** then combines all six outputs into a `confidence_score` (0–100) and a set of plain-English `risk_flags`.

---

## Installation

```bash
pip install -r stockprob/requirements.txt
```

### API Keys (optional but recommended)

| Variable | Source | Used By |
|----------|--------|---------|
| `POLYGON_API_KEY` | [polygon.io](https://polygon.io) | Higher-quality price data |
| `NEWSAPI_KEY` | [newsapi.org](https://newsapi.org) | News sentiment |
| `FRED_API_KEY` | [fred.stlouisfed.org](https://fred.stlouisfed.org) | Macro indicators |

Set them in a `.env` file or as environment variables. Without API keys the engine falls back to `yfinance` (free, no key required).

---

## Quick Start

```python
from stockprob.services.orchestrator import run_analysis

result = run_analysis(
    ticker="AAPL",
    window_days=30,       # Forward-looking horizon
    risk_tolerance="moderate",
    n_mc_paths=10000,
)

print(f"P(up): {result.probability_up:.1%}")
print(f"Confidence: {result.confidence_score}/100")
print(f"Risk flags: {result.risk_flags}")
```

---

## Input

```python
from stockprob import AnalysisRequest
from datetime import date

request = AnalysisRequest(
    ticker="AAPL",
    window_days=30,          # 1–252 trading days
    as_of_date=date.today(),
    risk_tolerance="moderate"  # low | moderate | high
)
```

---

## Output

```json
{
  "ticker": "AAPL",
  "as_of_date": "2026-05-13",
  "window_days": 30,
  "probability_up": 0.623,
  "confidence_score": 74,
  "signal_dashboard": {
    "risk":     { "vol_regime": "elevated", "beta": 1.18 },
    "trend":    { "regime": "bull", "vs_ma200": "above" },
    "earnings": { "in_window": false, "trend": "beat_streak" },
    "sentiment":{ "score": 0.21, "insiders": "neutral" },
    "macro":    { "regime": "expansion", "yield_curve": 0.34 }
  },
  "monte_carlo": {
    "paths_run": 10000,
    "probability_up": 0.623,
    "p10_return": -0.087,
    "p25_return": -0.031,
    "p50_return": 0.019,
    "p75_return": 0.068,
    "p90_return": 0.142,
    "expected_return": 0.022,
    "median_max_drawdown": -0.063,
    "probability_loss_gt_10pct": 0.091,
    "probability_gain_gt_20pct": 0.044
  },
  "risk_flags": [
    "Vol regime elevated. Widen your expected range.",
    "Beta 1.18 — moves harder than the market in both directions."
  ],
  "agent_status": {
    "PRICE_HISTORIAN": "ok",
    "VOLATILITY_ENGINE": "ok",
    "EARNINGS_ANALYST": "ok",
    "SENTIMENT_SCANNER": "ok",
    "MACRO_PULLER": "ok",
    "MONTE_CARLO": "ok"
  },
  "disclaimer": "This is math. Not advice. Past distributions do not guarantee future outcomes."
}
```

---

## Confidence Score

The confidence score (0–100) starts at **50** and is adjusted by signal quality:

**Penalties**
- `< 252` sessions of price history → **-20**
- Earnings report falls inside the analysis window → **-10**
- Volatility regime is `crisis` → **-15**
- Macro regime is `recession` → **-10**
- Net sentiment score `< -0.5` → **-5**
- Window `> 252` days → capped at **60**

**Bonuses**
- `> 1000` sessions of price history → **+10**
- EPS surprise trend is `beat_streak` → **+10**
- Sector ETF outperforming SPY → **+5**
- Net insider direction is `buying` → **+5**

---

## Project Structure

```
stockprob/
├── __init__.py              # Public API exports
├── requirements.txt
├── agents/
│   ├── price_historian.py   # Agent 1
│   ├── volatility_engine.py # Agent 2
│   ├── earnings_analyst.py  # Agent 3
│   ├── sentiment_scanner.py # Agent 4
│   ├── macro_puller.py      # Agent 5
│   └── monte_carlo.py       # Agent 6
├── config/
│   └── settings.py          # Config dataclass, sector ETF map, FRED series
├── core/
│   └── models.py            # Pydantic input/output schemas
├── data/
│   └── sources.py           # Data source abstraction (yfinance, Polygon)
├── services/
│   └── orchestrator.py      # Main Orchestrator + run_analysis convenience fn
└── utils/
    └── helpers.py           # Logging, caching, financial math helpers
```

---

## Configuration

Override defaults by instantiating `Config`:

```python
from stockprob import Config
from stockprob.services.orchestrator import Orchestrator

config = Config(
    MONTE_CARLO_PATHS=50000,
    POLYGON_API_KEY="your-key",
    NEWSAPI_KEY="your-key",
    FRED_API_KEY="your-key",
    CACHE_TTL_SECONDS=1800,
)
```

---

## Dependencies

| Category | Libraries |
|----------|-----------|
| Data fetching | `yfinance`, `polygon-api-client`, `newsapi-python`, `fredapi` |
| Financial math | `numpy`, `pandas`, `scipy`, `arch` (GARCH) |
| Sentiment | `transformers`, `torch`, `nltk`, `vaderSentiment` |
| Simulation | `numba` |
| API / validation | `fastapi`, `uvicorn`, `pydantic` |
| Utilities | `loguru`, `python-dotenv`, `redis`, `joblib` |
| Testing | `pytest`, `pytest-asyncio`, `pytest-cov` |

---

## License

MIT
