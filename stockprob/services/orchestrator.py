"""
StockProb Multi-Agent Orchestration Engine
Main Orchestrator - Coordinates all 6 agents and aggregates outputs
"""

import numpy as np
from datetime import date, timedelta
from typing import Dict, List, Optional, Any

from ..core.models import (
    AnalysisRequest,
    AnalysisResponse,
    PriceHistorianOutput,
    VolatilityEngineOutput,
    EarningsAnalystOutput,
    SentimentScannerOutput,
    MacroPullerOutput,
    MonteCarloOutput,
    SignalDashboard,
    AgentStatus,
    RiskTolerance,
)
from ..agents import (
    PriceHistorian,
    VolatilityEngine,
    EarningsAnalyst,
    SentimentScanner,
    MacroPuller,
    MonteCarloSimulator,
)
from ..data.sources import get_data_source, DataSource
from ..utils import log


class Orchestrator:
    """
    Main Orchestrator for StockProb Multi-Agent System
    
    Coordinates all 6 specialized sub-agents:
    1. Price Historian - Historical price data analysis
    2. Volatility Engine - Multi-window realized volatility
    3. Earnings Analyst - Earnings reactions and event risk
    4. Sentiment Scanner - News and SEC filing sentiment
    5. Macro Puller - Macroeconomic indicators
    6. Monte Carlo - 10,000-path simulation
    
    Aggregates outputs into a single structured JSON payload.
    """
    
    def __init__(
        self, 
        data_source: Optional[DataSource] = None,
        n_mc_paths: int = 10000,
        mc_seed: Optional[int] = None
    ):
        """
        Initialize the Orchestrator
        
        Args:
            data_source: Optional custom data source
            n_mc_paths: Number of Monte Carlo paths
            mc_seed: Random seed for Monte Carlo (for reproducibility)
        """
        self.data_source = data_source or get_data_source()
        self.n_mc_paths = n_mc_paths
        self.mc_seed = mc_seed
        
        # Initialize all agents
        self.price_historian = PriceHistorian(self.data_source)
        self.volatility_engine = VolatilityEngine(self.data_source)
        self.earnings_analyst = EarningsAnalyst(self.data_source)
        self.sentiment_scanner = SentimentScanner(self.data_source)
        self.macro_puller = MacroPuller(self.data_source)
        self.monte_carlo = MonteCarloSimulator(n_paths=n_mc_paths, seed=mc_seed)
        
        log.info("Orchestrator initialized with all 6 agents")
    
    def _compute_confidence_score(
        self,
        price_output: PriceHistorianOutput,
        volatility_output: VolatilityEngineOutput,
        earnings_output: EarningsAnalystOutput,
        sentiment_output: SentimentScannerOutput,
        macro_output: MacroPullerOutput,
        window_days: int
    ) -> int:
        """
        Compute confidence score (0-100) based on signal quality
        
        Penalties:
        - < 252 sessions of price history: -20
        - earnings_in_window = true: -10 (fat tail risk)
        - vol_regime = "crisis": -15
        - macro_regime = "recession": -10
        - net_sentiment_score < -0.5: -5
        
        Bonuses:
        - > 1000 sessions: +10
        - eps_surprise_trend = "beat_streak": +10
        - sector_relative_strength = "outperforming": +5
        - insider_net_direction = "buying": +5
        
        Args:
            price_output: Output from Price Historian
            volatility_output: Output from Volatility Engine
            earnings_output: Output from Earnings Analyst
            sentiment_output: Output from Sentiment Scanner
            macro_output: Output from Macro Puller
            window_days: Analysis window
            
        Returns:
            Confidence score (0-100)
        """
        score = 50  # Base score
        
        # === PENALTIES ===
        
        # Insufficient history
        if price_output.total_sessions < 252:
            score -= 20
            log.debug("Penalty: insufficient history (-20)")
        
        # Earnings in window
        if earnings_output.earnings_in_window:
            score -= 10
            log.debug("Penalty: earnings in window (-10)")
        
        # Crisis volatility regime
        if volatility_output.vol_regime.value == "crisis":
            score -= 15
            log.debug("Penalty: crisis vol regime (-15)")
        
        # Recession macro regime
        if macro_output.macro_regime.value == "recession":
            score -= 10
            log.debug("Penalty: recession macro (-10)")
        
        # Negative sentiment
        if sentiment_output.net_sentiment_score < -0.5:
            score -= 5
            log.debug("Penalty: negative sentiment (-5)")
        
        # Long horizon warning
        if window_days > 252:
            score = min(score, 60)  # Cap at 60 for long horizons
            log.debug("Confidence capped at 60 for long horizon")
        
        # === BONUSES ===
        
        # Extensive history
        if price_output.total_sessions > 1000:
            score += 10
            log.debug("Bonus: extensive history (+10)")
        
        # Earnings beat streak
        if earnings_output.eps_surprise_trend.value == "beat_streak":
            score += 10
            log.debug("Bonus: earnings beat streak (+10)")
        
        # Sector outperforming
        if macro_output.sector_relative_strength.value == "outperforming":
            score += 5
            log.debug("Bonus: sector outperforming (+5)")
        
        # Insider buying
        if sentiment_output.insider_net_direction.value == "buying":
            score += 5
            log.debug("Bonus: insider buying (+5)")
        
        # Clamp to 0-100
        score = max(0, min(100, score))
        
        return int(score)
    
    def _generate_risk_flags(
        self,
        volatility_output: VolatilityEngineOutput,
        earnings_output: EarningsAnalystOutput,
        sentiment_output: SentimentScannerOutput,
        macro_output: MacroPullerOutput,
        window_days: int
    ) -> List[str]:
        """
        Generate plain-English risk warnings
        
        Args:
            volatility_output: Output from Volatility Engine
            earnings_output: Output from Earnings Analyst
            sentiment_output: Output from Sentiment Scanner
            macro_output: Output from Macro Puller
            window_days: Analysis window
            
        Returns:
            List of risk flag strings
        """
        flags = []
        
        # Earnings risk
        if earnings_output.earnings_in_window:
            flags.append("Earnings report falls within your window. Expect fat tails.")
        
        # Yield curve inversion
        if macro_output.yield_curve_spread < -0.25:
            flags.append("Yield curve inverted. Historically precedes contraction.")
        
        # Short interest
        if sentiment_output.short_interest_ratio and sentiment_output.short_interest_ratio > 10:
            flags.append("Short interest elevated. Squeeze risk or informed selling.")
        
        # Volatility regime
        if volatility_output.vol_regime.value == "crisis":
            flags.append("Vol regime: crisis. Simulation width is wide.")
        elif volatility_output.vol_regime.value == "elevated":
            flags.append("Vol regime elevated. Widen your expected range.")
        
        # Beta warning
        if volatility_output.beta_252d > 1.3:
            flags.append(f"Beta {volatility_output.beta_252d:.2f} — moves harder than the market in both directions.")
        elif volatility_output.beta_252d < 0.7:
            flags.append(f"Beta {volatility_output.beta_252d:.2f} — lower volatility than the market.")
        
        # Insider selling
        if sentiment_output.insider_net_direction.value == "selling":
            flags.append("Insider net selling in last 90 days.")
        
        # Stale macro data warning
        macro_age = (date.today() - macro_output.fred_pulled_at).days
        if macro_age > 7:
            flags.append("Macro data stale. Refresh before trading.")
        
        # Long horizon warning
        if window_days > 252:
            flags.append("Long horizons have compounding uncertainty. Confidence score capped at 60.")
        
        return flags
    
    def _build_signal_dashboard(
        self,
        price_output: PriceHistorianOutput,
        volatility_output: VolatilityEngineOutput,
        earnings_output: EarningsAnalystOutput,
        sentiment_output: SentimentScannerOutput,
        macro_output: MacroPullerOutput
    ) -> SignalDashboard:
        """
        Build signal dashboard from all agent outputs
        
        Args:
            price_output: Output from Price Historian
            volatility_output: Output from Volatility Engine
            earnings_output: Output from Earnings Analyst
            sentiment_output: Output from Sentiment Scanner
            macro_output: Output from Macro Puller
            
        Returns:
            SignalDashboard instance
        """
        return SignalDashboard(
            risk={
                "vol_regime": volatility_output.vol_regime.value,
                "beta": round(volatility_output.beta_252d, 2),
            },
            trend={
                "regime": price_output.regime.value,
                "vs_ma200": price_output.price_vs_ma200,
            },
            earnings={
                "in_window": earnings_output.earnings_in_window,
                "trend": earnings_output.eps_surprise_trend.value,
            },
            sentiment={
                "score": round(sentiment_output.net_sentiment_score, 2),
                "insiders": sentiment_output.insider_net_direction.value,
            },
            macro={
                "regime": macro_output.macro_regime.value,
                "yield_curve": round(macro_output.yield_curve_spread, 2),
            },
        )
    
    def analyze(self, request: AnalysisRequest) -> AnalysisResponse:
        """
        Main analysis method - runs all 6 agents and aggregates results
        
        Args:
            request: AnalysisRequest with ticker, window_days, as_of_date, risk_tolerance
            
        Returns:
            AnalysisResponse with probability_up and all supporting data
        """
        ticker = request.ticker
        window_days = request.window_days
        as_of_date = request.as_of_date
        risk_tolerance = request.risk_tolerance
        
        log.info(f"Starting full analysis for {ticker}: {window_days} days from {as_of_date}")
        
        # Track agent status
        agent_status = AgentStatus()
        
        # === AGENT 1: PRICE HISTORIAN ===
        try:
            price_output = self.price_historian.analyze(ticker, as_of_date)
            agent_status.PRICE_HISTORIAN = "ok"
        except Exception as e:
            log.error(f"Price Historian failed: {e}")
            price_output = self.price_historian._create_empty_output(ticker)
            agent_status.PRICE_HISTORIAN = "failed"
        
        # === AGENT 2: VOLATILITY ENGINE ===
        try:
            volatility_output = self.volatility_engine.analyze(ticker, as_of_date)
            agent_status.VOLATILITY_ENGINE = "ok"
        except Exception as e:
            log.error(f"Volatility Engine failed: {e}")
            volatility_output = self.volatility_engine._create_empty_output()
            agent_status.VOLATILITY_ENGINE = "failed"
        
        # === AGENT 3: EARNINGS ANALYST ===
        try:
            earnings_output = self.earnings_analyst.analyze(ticker, as_of_date, window_days)
            agent_status.EARNINGS_ANALYST = "ok"
        except Exception as e:
            log.error(f"Earnings Analyst failed: {e}")
            earnings_output = self.earnings_analyst._create_empty_output()
            agent_status.EARNINGS_ANALYST = "failed"
        
        # === AGENT 4: SENTIMENT SCANNER ===
        try:
            sentiment_output = self.sentiment_scanner.analyze(ticker, as_of_date)
            agent_status.SENTIMENT_SCANNER = "ok"
        except Exception as e:
            log.error(f"Sentiment Scanner failed: {e}")
            sentiment_output = self.sentiment_scanner._create_empty_output()
            agent_status.SENTIMENT_SCANNER = "failed"
        
        # === AGENT 5: MACRO PULLER ===
        try:
            macro_output = self.macro_puller.analyze(ticker, as_of_date)
            agent_status.MACRO_PULLER = "ok"
        except Exception as e:
            log.error(f"Macro Puller failed: {e}")
            macro_output = self.macro_puller._create_empty_output(as_of_date)
            agent_status.MACRO_PULLER = "failed"
        
        # === AGENT 6: MONTE CARLO ===
        try:
            monte_carlo_output = self.monte_carlo.run_simulation(
                ticker=ticker,
                as_of_date=as_of_date,
                window_days=window_days,
                price_output=price_output,
                volatility_output=volatility_output,
                earnings_output=earnings_output,
                sentiment_output=sentiment_output,
                macro_output=macro_output,
            )
            agent_status.MONTE_CARLO = "ok"
        except Exception as e:
            log.error(f"Monte Carlo failed: {e}")
            monte_carlo_output = MonteCarloOutput(
                paths_run=self.n_mc_paths,
                probability_up=0.5,
                p10_return=-0.10,
                p25_return=-0.05,
                p50_return=0.0,
                p75_return=0.05,
                p90_return=0.10,
                expected_return=0.0,
                median_max_drawdown=-0.08,
                probability_loss_gt_10pct=0.10,
                probability_gain_gt_20pct=0.10,
            )
            agent_status.MONTE_CARLO = "failed"
        
        # === AGGREGATION ===
        
        # Primary output: probability_up from Monte Carlo
        probability_up = monte_carlo_output.probability_up
        
        # Compute confidence score
        confidence_score = self._compute_confidence_score(
            price_output,
            volatility_output,
            earnings_output,
            sentiment_output,
            macro_output,
            window_days
        )
        
        # Build signal dashboard
        signal_dashboard = self._build_signal_dashboard(
            price_output,
            volatility_output,
            earnings_output,
            sentiment_output,
            macro_output
        )
        
        # Generate risk flags
        risk_flags = self._generate_risk_flags(
            volatility_output,
            earnings_output,
            sentiment_output,
            macro_output,
            window_days
        )
        
        log.info(
            f"Analysis complete for {ticker}: P(up)={probability_up:.2%}, "
            f"confidence={confidence_score}"
        )
        
        # Build final response
        return AnalysisResponse(
            ticker=ticker,
            as_of_date=as_of_date,
            window_days=window_days,
            probability_up=probability_up,
            confidence_score=confidence_score,
            signal_dashboard=signal_dashboard,
            monte_carlo=monte_carlo_output,
            risk_flags=risk_flags,
            agent_status=agent_status,
            disclaimer="This is math. Not advice. Past distributions do not guarantee future outcomes.",
        )
    
    def analyze_simple(
        self,
        ticker: str,
        window_days: int = 30,
        as_of_date: Optional[date] = None,
        risk_tolerance: str = "moderate"
    ) -> AnalysisResponse:
        """
        Simplified analysis method with basic parameters
        
        Args:
            ticker: Stock ticker symbol
            window_days: Analysis window in days (default: 30)
            as_of_date: Analysis anchor date (default: today)
            risk_tolerance: User risk tolerance (low|moderate|high)
            
        Returns:
            AnalysisResponse
        """
        if as_of_date is None:
            as_of_date = date.today()
        
        request = AnalysisRequest(
            ticker=ticker,
            window_days=window_days,
            as_of_date=as_of_date,
            risk_tolerance=RiskTolerance(risk_tolerance),
        )
        
        return self.analyze(request)


# Convenience function
def run_analysis(
    ticker: str,
    window_days: int = 30,
    as_of_date: Optional[date] = None,
    risk_tolerance: str = "moderate",
    n_mc_paths: int = 10000,
    use_mock_data: bool = False
) -> AnalysisResponse:
    """
    Convenience function to run full StockProb analysis
    
    Args:
        ticker: Stock ticker symbol
        window_days: Analysis window in days
        as_of_date: Analysis anchor date
        risk_tolerance: User risk tolerance
        n_mc_paths: Number of Monte Carlo paths
        use_mock_data: If True, use mock data source
        
    Returns:
        AnalysisResponse
    """
    data_source = get_data_source(use_mock=use_mock_data)
    orchestrator = Orchestrator(data_source=data_source, n_mc_paths=n_mc_paths)
    
    return orchestrator.analyze_simple(
        ticker=ticker,
        window_days=window_days,
        as_of_date=as_of_date,
        risk_tolerance=risk_tolerance,
    )
