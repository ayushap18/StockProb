"""
StockProb Web UI - Streamlit Interface
======================================
Simple, clean interface to run the StockProb multi-agent analysis.
"""

import streamlit as st
from datetime import date, timedelta
from typing import Optional
import json
import sys
from pathlib import Path

# Ensure repository root is importable when running from ui/ entrypoint
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import the orchestrator
try:
    from stockprob.services.orchestrator import run_analysis
    from stockprob.core.models import RiskTolerance
except ImportError as e:
    st.error(
        "StockProb dependencies are missing. Install from requirements.txt at the repository root."
    )
    st.code(str(e))
    st.stop()

# Page configuration
st.set_page_config(
    page_title="StockProb | Quantitative Stock Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f2937;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #6b7280;
        margin-bottom: 2rem;
    }
    .probability-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 1rem;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    .probability-value {
        font-size: 4rem;
        font-weight: 800;
    }
    .probability-label {
        font-size: 1.2rem;
        opacity: 0.9;
    }
    .metric-card {
        background: #f9fafb;
        padding: 1.5rem;
        border-radius: 0.75rem;
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
    }
    .metric-label {
        font-size: 0.875rem;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-value {
        font-size: 1.75rem;
        font-weight: 700;
        color: #1f2937;
    }
    .risk-flag {
        background: #fef3c7;
        border-left: 4px solid #f59e0b;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0.25rem;
    }
    .agent-status-ok {
        color: #10b981;
        font-weight: 600;
    }
    .agent-status-degraded {
        color: #f59e0b;
        font-weight: 600;
    }
    .agent-status-failed {
        color: #ef4444;
        font-weight: 600;
    }
    .disclaimer {
        background: #fee2e2;
        border-left: 4px solid #ef4444;
        padding: 1rem;
        margin-top: 2rem;
        border-radius: 0.25rem;
        font-size: 0.875rem;
        color: #991b1b;
    }
</style>
""", unsafe_allow_html=True)


def format_probability(prob: float) -> str:
    """Format probability as percentage with color."""
    if prob >= 0.6:
        color = "#10b981"  # Green
    elif prob >= 0.4:
        color = "#f59e0b"  # Yellow
    else:
        color = "#ef4444"  # Red
    return f'<span style="color: {color}; font-weight: bold;">{prob:.1%}</span>'


def format_confidence(score: int) -> str:
    """Format confidence score with emoji."""
    if score >= 70:
        return f"{score}/100 ✅"
    elif score >= 50:
        return f"{score}/100 ⚠️"
    else:
        return f"{score}/100 ❌"


def main():
    # Header
    st.markdown('<h1 class="main-header">📊 StockProb</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Quantitative Multi-Agent Stock Analysis Engine</p>', unsafe_allow_html=True)
    
    # Sidebar - Input Parameters
    with st.sidebar:
        st.header("⚙️ Analysis Parameters")
        
        ticker = st.text_input(
            "Ticker Symbol",
            value="AAPL",
            help="Stock ticker symbol (e.g., AAPL, MSFT, TSLA)"
        ).upper()
        
        window_days = st.slider(
            "Time Horizon (Days)",
            min_value=1,
            max_value=365,
            value=30,
            step=1,
            help="Forward-looking prediction window in days"
        )
        
        as_of_date = st.date_input(
            "Analysis Date",
            value=date.today(),
            help="Anchor date for the analysis"
        )
        
        risk_tolerance = st.selectbox(
            "Risk Tolerance",
            options=["low", "moderate", "high"],
            index=1,
            help="Your risk tolerance level"
        )
        
        use_mock_data = st.checkbox(
            "Use Mock Data",
            value=True,
            help="Check to use simulated data (for testing). Uncheck for real market data."
        )
        
        st.divider()
        
        run_button = st.button(
            "🚀 Run Analysis",
            type="primary",
            use_container_width=True
        )
    
    # Main content area
    if run_button:
        if not ticker:
            st.error("Please enter a ticker symbol.")
            st.stop()
        
        with st.spinner(f"Running StockProb analysis for {ticker}..."):
            try:
                # Run the analysis
                result = run_analysis(
                    ticker=ticker,
                    window_days=window_days,
                    as_of_date=as_of_date,
                    risk_tolerance=RiskTolerance(risk_tolerance),
                    use_mock_data=use_mock_data
                )
                
                # Display results
                display_results(result, ticker, window_days, as_of_date)
                
            except Exception as e:
                st.error(f"Analysis failed: {str(e)}")
                st.exception(e)
    
    else:
        # Welcome message when no analysis has been run
        st.info("👈 Configure your analysis parameters in the sidebar and click 'Run Analysis' to get started.")
        
        # Example output preview
        st.markdown("### What You'll Get")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Probability UP", "—")
            st.caption("Monte Carlo simulation result")
        with col2:
            st.metric("Confidence Score", "—")
            st.caption("Data quality & signal strength")
        with col3:
            st.metric("Risk Flags", "—")
            st.caption("Important warnings")


def display_results(result, ticker: str, window_days: int, as_of_date: date):
    """Display the analysis results in a structured format."""
    
    # Top row: Probability card
    col1, col2 = st.columns([2, 3])
    
    with col1:
        prob_up = result.probability_up
        st.markdown(f"""
        <div class="probability-card">
            <div class="probability-value">{prob_up:.1%}</div>
            <div class="probability-label">Probability Stock Moves UP</div>
            <div style="margin-top: 1rem; font-size: 0.9rem;">
                over {window_days} days from {as_of_date}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Signal Dashboard
        st.subheader("📡 Signal Dashboard")
        
        signal = result.signal_dashboard
        
        sig_col1, sig_col2 = st.columns(2)
        
        with sig_col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Risk Regime</div>
                <div class="metric-value">{signal.risk.get('vol_regime', 'N/A')}</div>
                <div style="font-size: 0.875rem; color: #6b7280;">Beta: {signal.risk.get('beta', 0):.2f}</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Trend</div>
                <div class="metric-value">{signal.trend.get('regime', 'N/A')}</div>
                <div style="font-size: 0.875rem; color: #6b7280;">vs MA200: {signal.trend.get('vs_ma200', 'N/A')}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with sig_col2:
            earnings_in_window = signal.earnings.get('in_window', False)
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Earnings</div>
                <div class="metric-value">{"⚠️ In Window" if earnings_in_window else "✅ Clear"}</div>
                <div style="font-size: 0.875rem; color: #6b7280;">Trend: {signal.earnings.get('trend', 'N/A')}</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Sentiment</div>
                <div class="metric-value">{signal.sentiment.get('score', 0):.2f}</div>
                <div style="font-size: 0.875rem; color: #6b7280;">Insiders: {signal.sentiment.get('insiders', 'N/A')}</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Macro regime full width
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Macro Environment</div>
            <div class="metric-value">{signal.macro.get('regime', 'N/A')}</div>
            <div style="font-size: 0.875rem; color: #6b7280;">
                Yield Curve Spread: {signal.macro.get('yield_curve', 0):.2f}% | 
                Fed Funds: {signal.macro.get('fed_funds', 'N/A')}%
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.divider()
    
    # Second row: Confidence and Monte Carlo details
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Confidence Score")
        confidence = result.confidence_score
        st.metric("Score", format_confidence(confidence))
        
        # Breakdown of adjustments
        st.caption("Based on data quality, earnings proximity, volatility regime, and signal strength.")
    
    with col2:
        st.subheader("📈 Return Distribution")
        mc = result.monte_carlo
        
        dist_data = {
            "Percentile": ["P10", "P25", "P50 (Median)", "P75", "P90"],
            "Return": [
                f"{mc.p10_return:.1%}",
                f"{mc.p25_return:.1%}",
                f"{mc.p50_return:.1%}",
                f"{mc.p75_return:.1%}",
                f"{mc.p90_return:.1%}"
            ]
        }
        st.table(dist_data)
    
    st.divider()
    
    # Third row: Risk metrics and flags
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("⚠️ Risk Metrics")
        mc = result.monte_carlo
        
        risk_col1, risk_col2 = st.columns(2)
        with risk_col1:
            st.metric(
                "Prob Loss > 10%",
                f"{mc.probability_loss_gt_10pct:.1%}",
                delta=None
            )
        with risk_col2:
            st.metric(
                "Prob Gain > 20%",
                f"{mc.probability_gain_gt_20pct:.1%}",
                delta=None
            )
        
        st.metric(
            "Median Max Drawdown",
            f"{mc.median_max_drawdown:.1%}"
        )
    
    with col2:
        st.subheader("🚩 Risk Flags")
        if result.risk_flags:
            for flag in result.risk_flags:
                st.markdown(f'<div class="risk-flag">⚠️ {flag}</div>', unsafe_allow_html=True)
        else:
            st.success("No significant risk flags detected.")
    
    st.divider()
    
    # Agent status
    st.subheader("🤖 Agent Status")
    agent_cols = st.columns(6)
    agent_names = [
        "Price Historian",
        "Volatility Engine",
        "Earnings Analyst",
        "Sentiment Scanner",
        "Macro Puller",
        "Monte Carlo"
    ]
    
    agent_attrs = [
        "PRICE_HISTORIAN",
        "VOLATILITY_ENGINE", 
        "EARNINGS_ANALYST",
        "SENTIMENT_SCANNER",
        "MACRO_PULLER",
        "MONTE_CARLO"
    ]
    
    for col, name, attr in zip(agent_cols, agent_names, agent_attrs):
        status = getattr(result.agent_status, attr, "unknown")
        
        if status == "ok":
            status_class = "agent-status-ok"
            icon = "✅"
        elif status == "degraded":
            status_class = "agent-status-degraded"
            icon = "⚠️"
        else:
            status_class = "agent-status-failed"
            icon = "❌"
        
        with col:
            st.markdown(f"<div style='text-align: center;'><span class='{status_class}'>{icon} {name}</span></div>", unsafe_allow_html=True)
    
    # Disclaimer
    st.markdown(f"""
    <div class="disclaimer">
        <strong>⚠️ Disclaimer:</strong> {result.disclaimer}
    </div>
    """, unsafe_allow_html=True)
    
    # JSON export option
    with st.expander("📄 View Raw JSON Output"):
        st.json(json.dumps(result.model_dump(), indent=2, default=str))


if __name__ == "__main__":
    main()
