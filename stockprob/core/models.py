"""
StockProb Multi-Agent Orchestration Engine
Core data models and schemas using Pydantic
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import date, datetime
from enum import Enum


# ============================================================================
# ENUMS FOR TYPE SAFETY
# ============================================================================

class PriceRegime(str, Enum):
    BULL = "bull"
    BEAR = "bear"
    TRANSITIONAL = "transitional"


class VolRegime(str, Enum):
    LOW = "low"
    ELEVATED = "elevated"
    CRISIS = "crisis"


class MacroRegime(str, Enum):
    EXPANSION = "expansion"
    STAGFLATION = "stagflation"
    RECESSION = "recession"
    RECOVERY = "recovery"


class SectorRelativeStrength(str, Enum):
    OUTPERFORMING = "outperforming"
    INLINE = "inline"
    UNDERPERFORMING = "underperforming"


class InsiderDirection(str, Enum):
    BUYING = "buying"
    SELLING = "selling"
    NEUTRAL = "neutral"


class SentimentTrend(str, Enum):
    IMPROVING = "improving"
    DETERIORATING = "deteriorating"
    FLAT = "flat"


class EPSSurpriseTrend(str, Enum):
    BEAT_STREAK = "beat_streak"
    MISS_STREAK = "miss_streak"
    MIXED = "mixed"


class RiskTolerance(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"


# ============================================================================
# INPUT SCHEMAS
# ============================================================================

class AnalysisRequest(BaseModel):
    """User input for stock analysis"""
    ticker: str = Field(..., description="Stock ticker symbol", example="AAPL")
    window_days: int = Field(30, ge=1, le=252, description="Forward-looking horizon in days")
    as_of_date: date = Field(..., description="Analysis anchor date")
    risk_tolerance: RiskTolerance = Field(RiskTolerance.MODERATE, description="User risk tolerance")

    class Config:
        json_schema_extra = {
            "example": {
                "ticker": "AAPL",
                "window_days": 30,
                "as_of_date": "2026-05-13",
                "risk_tolerance": "moderate"
            }
        }


# ============================================================================
# AGENT OUTPUT SCHEMAS
# ============================================================================

class PriceHistorianOutput(BaseModel):
    """Output from Agent 1: Price Historian"""
    total_sessions: int = Field(..., description="Total trading sessions available")
    current_price: float = Field(..., description="Current stock price")
    log_return_mean_252d: float = Field(..., description="Mean of 252-day log returns")
    log_return_std_252d: float = Field(..., description="Std dev of 252-day log returns")
    ma_50: float = Field(..., description="50-day moving average")
    ma_200: float = Field(..., description="200-day moving average")
    price_vs_ma200: str = Field(..., description="Price position vs 200-day MA")
    regime: PriceRegime = Field(..., description="Current price regime")
    raw_returns: List[float] = Field(..., description="Last 756+ sessions of log returns")


class VolatilityEngineOutput(BaseModel):
    """Output from Agent 2: Volatility Engine"""
    realized_vol_10d: float = Field(..., description="10-day annualized realized volatility")
    realized_vol_21d: float = Field(..., description="21-day annualized realized volatility")
    realized_vol_63d: float = Field(..., description="63-day annualized realized volatility")
    realized_vol_252d: float = Field(..., description="252-day annualized realized volatility")
    parkinson_vol: float = Field(..., description="Parkinson high-low volatility estimator")
    garch_vol_1d_ahead: Optional[float] = Field(None, description="GARCH(1,1) 1-day ahead forecast")
    vix_current: float = Field(..., description="Current VIX level")
    vix_percentile_52w: float = Field(..., description="VIX 52-week percentile")
    beta_252d: float = Field(..., description="Beta vs SPY over 252 days")
    vol_regime: VolRegime = Field(..., description="Classified volatility regime")


class EarningsAnalystOutput(BaseModel):
    """Output from Agent 3: Earnings Analyst"""
    earnings_dates: List[str] = Field(..., description="Last 12 earnings dates")
    post_earnings_returns_1d: List[float] = Field(..., description="Post-earnings 1-day returns")
    mean_reaction: float = Field(..., description="Mean earnings reaction")
    median_reaction: float = Field(..., description="Median earnings reaction")
    pct_positive_reactions: float = Field(..., description="Percentage of positive reactions")
    next_earnings_date: Optional[str] = Field(None, description="Next earnings date")
    earnings_in_window: bool = Field(..., description="Whether earnings falls in analysis window")
    estimated_earnings_move: float = Field(..., description="Estimated earnings move magnitude")
    eps_surprise_trend: EPSSurpriseTrend = Field(..., description="EPS surprise trend")


class SentimentScannerOutput(BaseModel):
    """Output from Agent 4: Sentiment Scanner"""
    headline_count: int = Field(..., description="Number of news headlines analyzed")
    net_sentiment_score: float = Field(..., ge=-1.0, le=1.0, description="Net sentiment score")
    sentiment_trend: SentimentTrend = Field(..., description="Sentiment trend direction")
    last_10k_date: Optional[str] = Field(None, description="Last 10-K filing date")
    last_10q_date: Optional[str] = Field(None, description="Last 10-Q filing date")
    insider_net_direction: InsiderDirection = Field(..., description="Insider trading direction")
    short_interest_ratio: Optional[float] = Field(None, description="Days to cover ratio")


class MacroPullerOutput(BaseModel):
    """Output from Agent 5: Macro Puller"""
    fed_funds_rate: float = Field(..., description="Federal Funds Rate")
    treasury_10y: float = Field(..., description="10-Year Treasury Yield")
    treasury_2y: float = Field(..., description="2-Year Treasury Yield")
    yield_curve_spread: float = Field(..., description="10Y - 2Y spread")
    cpi_yoy: float = Field(..., description="CPI Year-over-Year")
    unemployment_rate: float = Field(..., description="Unemployment Rate")
    macro_regime: MacroRegime = Field(..., description="Classified macro regime")
    sector_etf: str = Field(..., description="Sector ETF ticker")
    sector_vs_spy_3m: float = Field(..., description="Sector vs SPY 3-month return")
    sector_relative_strength: SectorRelativeStrength = Field(..., description="Sector relative strength")
    fred_pulled_at: date = Field(..., description="Date FRED data was pulled")


class MonteCarloOutput(BaseModel):
    """Output from Agent 6: Monte Carlo Simulator"""
    paths_run: int = Field(10000, description="Number of simulation paths")
    probability_up: float = Field(..., ge=0.0, le=1.0, description="Probability stock moves up")
    p10_return: float = Field(..., description="10th percentile return")
    p25_return: float = Field(..., description="25th percentile return")
    p50_return: float = Field(..., description="50th percentile return (median)")
    p75_return: float = Field(..., description="75th percentile return")
    p90_return: float = Field(..., description="90th percentile return")
    expected_return: float = Field(..., description="Expected (mean) return")
    median_max_drawdown: float = Field(..., description="Median maximum drawdown across paths")
    probability_loss_gt_10pct: float = Field(..., ge=0.0, le=1.0, description="Prob of loss > 10%")
    probability_gain_gt_20pct: float = Field(..., ge=0.0, le=1.0, description="Prob of gain > 20%")


# ============================================================================
# SIGNAL DASHBOARD SCHEMA
# ============================================================================

class SignalDashboard(BaseModel):
    """Aggregated signal dashboard from all agents"""
    risk: Dict[str, Any] = Field(..., description="Risk signals from Volatility Engine")
    trend: Dict[str, Any] = Field(..., description="Trend signals from Price Historian")
    earnings: Dict[str, Any] = Field(..., description="Earnings signals from Earnings Analyst")
    sentiment: Dict[str, Any] = Field(..., description="Sentiment signals from Sentiment Scanner")
    macro: Dict[str, Any] = Field(..., description="Macro signals from Macro Puller")


# ============================================================================
# FINAL RESPONSE SCHEMA
# ============================================================================

class AgentStatus(BaseModel):
    """Status of each agent execution"""
    PRICE_HISTORIAN: str = "ok"
    VOLATILITY_ENGINE: str = "ok"
    EARNINGS_ANALYST: str = "ok"
    SENTIMENT_SCANNER: str = "ok"
    MACRO_PULLER: str = "ok"
    MONTE_CARLO: str = "ok"


class AnalysisResponse(BaseModel):
    """Final structured response from the orchestrator"""
    ticker: str = Field(..., description="Stock ticker symbol")
    as_of_date: date = Field(..., description="Analysis anchor date")
    window_days: int = Field(..., description="Forward-looking horizon")
    probability_up: float = Field(..., ge=0.0, le=1.0, description="PRIMARY: Probability stock moves up")
    confidence_score: int = Field(..., ge=0, le=100, description="Confidence score (0-100)")
    signal_dashboard: SignalDashboard = Field(..., description="Signal dashboard from all agents")
    monte_carlo: MonteCarloOutput = Field(..., description="Monte Carlo simulation results")
    risk_flags: List[str] = Field(..., description="Plain-English risk warnings")
    agent_status: AgentStatus = Field(..., description="Status of each agent")
    disclaimer: str = "This is math. Not advice. Past distributions do not guarantee future outcomes."

    class Config:
        json_schema_extra = {
            "example": {
                "ticker": "AAPL",
                "as_of_date": "2026-05-13",
                "window_days": 30,
                "probability_up": 0.623,
                "confidence_score": 74,
                "signal_dashboard": {
                    "risk": {"vol_regime": "elevated", "beta": 1.18},
                    "trend": {"regime": "bull", "vs_ma200": "above"},
                    "earnings": {"in_window": False, "trend": "beat_streak"},
                    "sentiment": {"score": 0.21, "insiders": "neutral"},
                    "macro": {"regime": "expansion", "yield_curve": 0.34}
                },
                "monte_carlo": {
                    "paths": 10000,
                    "p10": -0.087,
                    "p25": -0.031,
                    "p50": 0.019,
                    "p75": 0.068,
                    "p90": 0.142,
                    "prob_loss_gt_10pct": 0.091,
                    "prob_gain_gt_20pct": 0.044,
                    "median_max_drawdown": -0.063
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
        }


# ============================================================================
# INTERNAL DATA MODELS
# ============================================================================

class OHLCVData(BaseModel):
    """Raw OHLCV data structure"""
    date: date
    open: float
    high: float
    low: float
    close: float
    volume: int
    adjusted_close: Optional[float] = None


class LogReturnStats(BaseModel):
    """Log return statistics"""
    mean: float
    std: float
    skewness: float
    kurtosis: float
    min: float
    max: float
    count: int


class GARCHResults(BaseModel):
    """GARCH model results"""
    omega: float  # Constant term
    alpha: float  # ARCH term coefficient
    beta: float   # GARCH term coefficient
    conditional_volatility: List[float]
    forecast_1d: float
    forecast_5d: float
    forecast_21d: float


class NewsHeadline(BaseModel):
    """News headline with sentiment"""
    title: str
    source: str
    published_at: datetime
    sentiment_score: float  # -1 to +1
    url: Optional[str] = None


class InsiderTrade(BaseModel):
    """SEC Form 4 insider trade"""
    filing_date: date
    transaction_date: date
    insider_name: str
    title: str
    transaction_type: str  # P=purchase, S=sale
    shares: int
    price_per_share: float
    total_value: float
    shares_owned_after: int


class EarningsEvent(BaseModel):
    """Earnings event data"""
    report_date: date
    eps_estimate: float
    eps_actual: Optional[float] = None
    surprise_pct: Optional[float] = None
    revenue_estimate: Optional[float] = None
    revenue_actual: Optional[float] = None


class MacroIndicators(BaseModel):
    """Macro economic indicators"""
    fed_funds_rate: float
    treasury_10y: float
    treasury_2y: float
    yield_curve_spread: float
    cpi_yoy: float
    unemployment_rate: float
    pmi_manufacturing: Optional[float] = None
    pull_date: date
