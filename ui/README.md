# StockProb UI - Web Interface

## Quick Start

### Run the Streamlit App

```bash
cd /workspace
streamlit run ui/app.py --server.address=0.0.0.0 --server.port=8501
```

Then open your browser to `http://localhost:8501`

### Features

- **📊 Probability Card**: Large, clear display of the UP probability
- **📡 Signal Dashboard**: Real-time view of all 6 agent signals
- **🎯 Confidence Score**: Data quality metric (0-100)
- **📈 Return Distribution**: P10-P90 percentile breakdown
- **⚠️ Risk Metrics**: Loss/gain probabilities and drawdown estimates
- **🚩 Risk Flags**: Plain-English warnings
- **🤖 Agent Status**: Health check for all sub-agents
- **📄 JSON Export**: View/copy raw output data

### Input Parameters

- **Ticker Symbol**: Any valid stock ticker (AAPL, MSFT, TSLA, etc.)
- **Time Horizon**: 1-365 days forward-looking window
- **Analysis Date**: Anchor date for the analysis
- **Risk Tolerance**: Low | Moderate | High
- **Mock Data Toggle**: Test with simulated data or use real market data

## Alternative: Simple Terminal Runner

For quick terminal-based analysis without the web UI:

```bash
python ui/run_cli.py --ticker AAPL --window 30
```

## Screenshots

The UI provides:
- Clean, modern interface with gradient cards
- Color-coded probability display (green > 60%, yellow 40-60%, red < 40%)
- Organized signal dashboard with all agent outputs
- Interactive tables and metrics
- Expandable JSON output for developers

## Requirements

- Python 3.10+
- streamlit
- plotly (optional, for future charts)
- stockprob package (installed in /workspace)
