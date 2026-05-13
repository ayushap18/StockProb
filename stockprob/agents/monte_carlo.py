"""
StockProb Multi-Agent Orchestration Engine
Agent 6: Monte Carlo - 10,000-path simulation with all signal adjustments
"""

import numpy as np
from datetime import date
from typing import Dict, List, Optional, Tuple

from ..core.models import (
    MonteCarloOutput,
    PriceHistorianOutput,
    VolatilityEngineOutput,
    EarningsAnalystOutput,
    SentimentScannerOutput,
    MacroPullerOutput,
    MacroRegime,
    SectorRelativeStrength,
)
from ..utils import log, percentile, max_drawdown


class MonteCarloSimulator:
    """
    Agent 6: Monte Carlo Simulator
    
    Role: 10,000-path simulation. Counted, not predicted.
    
    Responsibilities:
    - Use price historian log_return_mean and log_return_std as GBM parameters
    - Adjust σ using GARCH vol if available, else realized_vol_21d
    - Inject earnings shock if earnings_in_window = true
    - Adjust drift by macro regime, sentiment, sector strength
    - Run 10,000 paths of window_days length
    - Count paths ending above today's price
    - Compute percentile distribution and risk metrics
    """
    
    def __init__(self, n_paths: int = 10000, seed: Optional[int] = None):
        """
        Initialize Monte Carlo Simulator
        
        Args:
            n_paths: Number of simulation paths
            seed: Random seed for reproducibility
        """
        self.n_paths = n_paths
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)
        log.info(f"Monte Carlo Simulator initialized with {n_paths} paths")
    
    def _adjust_drift(
        self,
        base_drift: float,
        macro_output: MacroPullerOutput,
        sentiment_output: SentimentScannerOutput,
        volatility_output: VolatilityEngineOutput
    ) -> float:
        """
        Adjust drift based on signals
        
        Adjustments:
        - +0.002/day if macro_regime = "expansion"
        - -0.002/day if macro_regime = "recession"
        - +0.001/day if sentiment net_score > 0.3
        - -0.001/day if sentiment net_score < -0.3
        - +0.001/day if sector_relative_strength = "outperforming"
        
        Args:
            base_drift: Base daily drift from historical returns
            macro_output: Output from Macro Puller
            sentiment_output: Output from Sentiment Scanner
            volatility_output: Output from Volatility Engine
            
        Returns:
            Adjusted daily drift
        """
        adjusted_drift = base_drift
        
        # Macro regime adjustment
        if macro_output.macro_regime == MacroRegime.EXPANSION:
            adjusted_drift += 0.002
            log.debug("Macro expansion: +0.002/day drift")
        elif macro_output.macro_regime == MacroRegime.RECESSION:
            adjusted_drift -= 0.002
            log.debug("Macro recession: -0.002/day drift")
        
        # Sentiment adjustment
        sentiment_score = sentiment_output.net_sentiment_score
        if sentiment_score > 0.3:
            adjusted_drift += 0.001
            log.debug("Positive sentiment: +0.001/day drift")
        elif sentiment_score < -0.3:
            adjusted_drift -= 0.001
            log.debug("Negative sentiment: -0.001/day drift")
        
        # Sector strength adjustment
        if macro_output.sector_relative_strength == SectorRelativeStrength.OUTPERFORMING:
            adjusted_drift += 0.001
            log.debug("Sector outperforming: +0.001/day drift")
        
        return adjusted_drift
    
    def _get_adjusted_volatility(
        self,
        volatility_output: VolatilityEngineOutput,
        window_days: int
    ) -> float:
        """
        Get adjusted volatility for simulation
        
        Uses GARCH vol if available, else realized_vol_21d
        
        Args:
            volatility_output: Output from Volatility Engine
            window_days: Simulation window
            
        Returns:
            Daily volatility for simulation
        """
        # Prefer GARCH forecast if available
        if volatility_output.garch_vol_1d_ahead is not None:
            daily_vol = volatility_output.garch_vol_1d_ahead
            log.debug(f"Using GARCH vol: {daily_vol:.4f}")
        else:
            # Use realized 21-day vol, convert to daily
            annualized_vol = volatility_output.realized_vol_21d
            daily_vol = annualized_vol / np.sqrt(252)
            log.debug(f"Using realized vol: {daily_vol:.4f}")
        
        # Ensure minimum volatility
        daily_vol = max(daily_vol, 0.005)  # At least 0.5% daily vol
        
        return daily_vol
    
    def _simulate_paths_gbm(
        self,
        initial_price: float,
        drift: float,
        volatility: float,
        window_days: int,
        earnings_shock: Optional[float] = None,
        earnings_day: Optional[int] = None
    ) -> np.ndarray:
        """
        Simulate price paths using Geometric Brownian Motion
        
        dS/S = μ dt + σ dW
        
        Args:
            initial_price: Starting price
            drift: Daily drift (μ)
            volatility: Daily volatility (σ)
            window_days: Number of days to simulate
            earnings_shock: Optional earnings shock magnitude
            earnings_day: Day of earnings shock (if applicable)
            
        Returns:
            Array of shape (n_paths, window_days+1) with simulated prices
        """
        # Generate random shocks for all paths and days
        dW = np.random.normal(0, 1, (self.n_paths, window_days))
        
        # Initialize price array
        prices = np.zeros((self.n_paths, window_days + 1))
        prices[:, 0] = initial_price
        
        # Simulate each day
        for t in range(window_days):
            # GBM step
            returns = drift + volatility * dW[:, t]
            prices[:, t + 1] = prices[:, t] * np.exp(returns)
            
            # Apply earnings shock if applicable
            if earnings_shock is not None and earnings_day is not None:
                if t == earnings_day:
                    # Apply random shock from earnings distribution
                    shock_direction = np.random.choice([-1, 1], self.n_paths)
                    shock_magnitude = np.abs(np.random.normal(0, earnings_shock, self.n_paths))
                    shock = shock_direction * shock_magnitude
                    prices[:, t + 1] = prices[:, t + 1] * (1 + shock)
                    log.debug(f"Applied earnings shock on day {t+1}")
        
        return prices
    
    def _compute_path_statistics(
        self,
        prices: np.ndarray,
        initial_price: float
    ) -> Dict[str, float]:
        """
        Compute statistics from simulated paths
        
        Args:
            prices: Array of simulated prices
            initial_price: Starting price
            
        Returns:
            Dict of computed statistics
        """
        # Calculate final returns
        final_prices = prices[:, -1]
        final_returns = (final_prices - initial_price) / initial_price
        
        # Probability of ending up
        probability_up = float(np.mean(final_returns > 0))
        
        # Percentiles of returns
        p10 = float(percentile(final_returns.tolist(), 10))
        p25 = float(percentile(final_returns.tolist(), 25))
        p50 = float(percentile(final_returns.tolist(), 50))
        p75 = float(percentile(final_returns.tolist(), 75))
        p90 = float(percentile(final_returns.tolist(), 90))
        
        # Expected return (mean)
        expected_return = float(np.mean(final_returns))
        
        # Probability of loss > 10%
        prob_loss_gt_10 = float(np.mean(final_returns < -0.10))
        
        # Probability of gain > 20%
        prob_gain_gt_20 = float(np.mean(final_returns > 0.20))
        
        # Maximum drawdown for each path
        max_drawdowns = []
        for path_idx in range(self.n_paths):
            path_returns = (prices[path_idx, 1:] / prices[path_idx, :-1]) - 1
            md = max_drawdown(path_returns.tolist())
            max_drawdowns.append(md)
        
        median_max_drawdown = float(np.median(max_drawdowns))
        
        return {
            "probability_up": probability_up,
            "p10_return": p10,
            "p25_return": p25,
            "p50_return": p50,
            "p75_return": p75,
            "p90_return": p90,
            "expected_return": expected_return,
            "probability_loss_gt_10pct": prob_loss_gt_10,
            "probability_gain_gt_20pct": prob_gain_gt_20,
            "median_max_drawdown": median_max_drawdown,
        }
    
    def run_simulation(
        self,
        ticker: str,
        as_of_date: date,
        window_days: int,
        price_output: PriceHistorianOutput,
        volatility_output: VolatilityEngineOutput,
        earnings_output: EarningsAnalystOutput,
        sentiment_output: SentimentScannerOutput,
        macro_output: MacroPullerOutput
    ) -> MonteCarloOutput:
        """
        Main simulation method - runs full Monte Carlo with all adjustments
        
        Args:
            ticker: Stock ticker symbol
            as_of_date: Analysis anchor date
            window_days: Simulation horizon in days
            price_output: Output from Price Historian
            volatility_output: Output from Volatility Engine
            earnings_output: Output from Earnings Analyst
            sentiment_output: Output from Sentiment Scanner
            macro_output: Output from Macro Puller
            
        Returns:
            MonteCarloOutput with simulation results
        """
        log.info(f"Starting Monte Carlo simulation for {ticker}: {window_days} days, {self.n_paths} paths")
        
        # Extract base parameters
        initial_price = price_output.current_price
        base_drift = price_output.log_return_mean_252d / 252  # Convert annual to daily
        base_volatility = self._get_adjusted_volatility(volatility_output, window_days)
        
        # Adjust drift based on signals
        adjusted_drift = self._adjust_drift(
            base_drift,
            macro_output,
            sentiment_output,
            volatility_output
        )
        
        log.info(
            f"Simulation params: drift={adjusted_drift:.6f}/day, "
            f"vol={base_volatility:.4f}/day, initial_price={initial_price:.2f}"
        )
        
        # Determine earnings shock parameters
        earnings_shock = None
        earnings_day = None
        
        if earnings_output.earnings_in_window and earnings_output.next_earnings_date:
            # Parse earnings date and calculate day number
            from datetime import datetime
            next_earnings = datetime.fromisoformat(earnings_output.next_earnings_date).date()
            days_until_earnings = (next_earnings - as_of_date).days
            
            if 0 < days_until_earnings <= window_days:
                earnings_shock = earnings_output.estimated_earnings_move
                earnings_day = days_until_earnings - 1  # 0-indexed
                log.info(f"Earnings shock enabled: {earnings_shock:.2%} on day {days_until_earnings}")
        
        # Run simulation
        prices = self._simulate_paths_gbm(
            initial_price=initial_price,
            drift=adjusted_drift,
            volatility=base_volatility,
            window_days=window_days,
            earnings_shock=earnings_shock,
            earnings_day=earnings_day
        )
        
        # Compute statistics
        stats = self._compute_path_statistics(prices, initial_price)
        
        log.info(
            f"Simulation complete: P(up)={stats['probability_up']:.2%}, "
            f"P(loss>10%)={stats['probability_loss_gt_10pct']:.2%}, "
            f"median MDD={stats['median_max_drawdown']:.2%}"
        )
        
        return MonteCarloOutput(
            paths_run=self.n_paths,
            probability_up=stats['probability_up'],
            p10_return=stats['p10_return'],
            p25_return=stats['p25_return'],
            p50_return=stats['p50_return'],
            p75_return=stats['p75_return'],
            p90_return=stats['p90_return'],
            expected_return=stats['expected_return'],
            median_max_drawdown=stats['median_max_drawdown'],
            probability_loss_gt_10pct=stats['probability_loss_gt_10pct'],
            probability_gain_gt_20pct=stats['probability_gain_gt_20pct'],
        )


# Convenience function
def run_monte_carlo(
    ticker: str,
    as_of_date: date,
    window_days: int,
    price_output: PriceHistorianOutput,
    volatility_output: VolatilityEngineOutput,
    earnings_output: EarningsAnalystOutput,
    sentiment_output: SentimentScannerOutput,
    macro_output: MacroPullerOutput,
    n_paths: int = 10000,
    seed: Optional[int] = None
) -> MonteCarloOutput:
    """
    Convenience function to run Monte Carlo simulation
    
    Args:
        ticker: Stock ticker symbol
        as_of_date: Analysis anchor date
        window_days: Simulation horizon in days
        price_output: Output from Price Historian
        volatility_output: Output from Volatility Engine
        earnings_output: Output from Earnings Analyst
        sentiment_output: Output from Sentiment Scanner
        macro_output: Output from Macro Puller
        n_paths: Number of simulation paths
        seed: Random seed
        
    Returns:
        MonteCarloOutput
    """
    simulator = MonteCarloSimulator(n_paths=n_paths, seed=seed)
    return simulator.run_simulation(
        ticker=ticker,
        as_of_date=as_of_date,
        window_days=window_days,
        price_output=price_output,
        volatility_output=volatility_output,
        earnings_output=earnings_output,
        sentiment_output=sentiment_output,
        macro_output=macro_output,
    )
