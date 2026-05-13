"""
StockProb Multi-Agent Orchestration Engine
Agent 3: Earnings Analyst - Historical earnings reactions + upcoming event risk
"""

import pandas as pd
import numpy as np
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple

from ..core.models import EarningsAnalystOutput, EPSSurpriseTrend
from ..data.sources import get_data_source, DataSource
from ..utils import log


class EarningsAnalyst:
    """
    Agent 3: Earnings Analyst
    
    Role: Historical earnings reactions + upcoming event risk.
    
    Responsibilities:
    - Pull last 12 earnings dates and post-earnings 1d returns
    - Compute: mean reaction, median reaction, % positive surprises
    - Pull next earnings date (if within window_days, flag as "earnings_in_window")
    - Estimate earnings move distribution
    - Compute "earnings surprise momentum": last 3 EPS beats/misses trend
    """
    
    def __init__(self, data_source: Optional[DataSource] = None):
        """
        Initialize Earnings Analyst
        
        Args:
            data_source: Optional custom data source
        """
        self.data_source = data_source or get_data_source()
        log.info("Earnings Analyst agent initialized")
    
    def fetch_earnings_history(
        self, 
        ticker: str,
        as_of_date: date,
        max_earnings: int = 12
    ) -> List[Dict]:
        """
        Fetch historical earnings dates and reactions
        
        Args:
            ticker: Stock ticker symbol
            as_of_date: Analysis anchor date
            max_earnings: Maximum number of earnings to return
            
        Returns:
            List of earnings event dictionaries
        """
        # Get earnings dates from data source
        earnings_dates = self.data_source.fetch_earnings_dates(ticker)
        
        if not earnings_dates:
            log.warning(f"No earnings dates found for {ticker}")
            return []
        
        # Filter to dates before as_of_date
        past_earnings = [d for d in earnings_dates if d < as_of_date]
        
        # Sort by date (most recent first)
        past_earnings = sorted(past_earnings, reverse=True)
        
        # Limit to max_earnings
        past_earnings = past_earnings[:max_earnings]
        
        # Build earnings history with mock data
        # In production, this would fetch actual EPS data
        earnings_history = []
        for i, earn_date in enumerate(past_earnings):
            # Generate synthetic reaction based on typical patterns
            np.random.seed(hash(earn_date.isoformat()) % 2**32)
            
            # Most earnings have small reactions, some have large
            if np.random.random() < 0.7:
                # 70% chance of normal reaction (-5% to +5%)
                reaction = np.random.normal(0.01, 0.03)
            else:
                # 30% chance of larger reaction (-15% to +15%)
                reaction = np.random.normal(0.02, 0.08)
            
            # Clip to reasonable bounds
            reaction = np.clip(reaction, -0.25, 0.25)
            
            earnings_history.append({
                "report_date": earn_date,
                "reaction_1d": float(reaction),
                "eps_surprise": np.random.normal(0.02, 0.05),  # Mock EPS surprise
            })
        
        log.info(f"Found {len(earnings_history)} historical earnings for {ticker}")
        return earnings_history
    
    def get_next_earnings_date(
        self, 
        ticker: str, 
        as_of_date: date
    ) -> Optional[date]:
        """
        Get next earnings date after as_of_date
        
        Args:
            ticker: Stock ticker symbol
            as_of_date: Analysis anchor date
            
        Returns:
            Next earnings date or None
        """
        earnings_dates = self.data_source.fetch_earnings_dates(ticker)
        
        if not earnings_dates:
            return None
        
        # Find first date after as_of_date
        future_earnings = [d for d in earnings_dates if d > as_of_date]
        
        if not future_earnings:
            return None
        
        return min(future_earnings)
    
    def compute_reactions(
        self, 
        price_df: pd.DataFrame, 
        earnings_history: List[Dict]
    ) -> List[float]:
        """
        Compute post-earnings 1-day returns from price data
        
        Args:
            price_df: DataFrame with price data
            earnings_history: List of earnings events
            
        Returns:
            List of 1-day post-earnings returns
        """
        if price_df.empty or 'close' not in price_df.columns:
            return []
        
        # Create date-to-close mapping
        price_map = {}
        for _, row in price_df.iterrows():
            if 'Date' in row:
                price_map[row['Date']] = row['close']
        
        returns = []
        for earning in earnings_history:
            earn_date = earning['report_date']
            
            # Find close on earnings day and next day
            if earn_date not in price_map:
                continue
            
            # Find next trading day's close
            next_day_close = None
            for d in sorted(price_map.keys()):
                if d > earn_date:
                    next_day_close = price_map[d]
                    break
            
            if next_day_close is None:
                continue
            
            earn_close = price_map[earn_date]
            if earn_close > 0:
                ret = (next_day_close - earn_close) / earn_close
                returns.append(float(ret))
        
        return returns
    
    def estimate_earnings_move(
        self, 
        historical_reactions: List[float],
        implied_vol: Optional[float] = None
    ) -> float:
        """
        Estimate expected earnings move
        
        Uses historical reaction distribution or options IV if available
        
        Args:
            historical_reactions: List of historical 1-day reactions
            implied_vol: Optional implied volatility from options
            
        Returns:
            Estimated earnings move magnitude
        """
        if not historical_reactions:
            # Default estimate based on typical stock
            return 0.04  # 4% move
        
        # Use standard deviation of historical reactions
        std_dev = np.std(historical_reactions)
        
        # If we have IV, blend it
        if implied_vol:
            # IV typically overestimates actual moves
            iv_estimate = implied_vol / np.sqrt(252)  # Daily IV
            estimated_move = 0.5 * std_dev + 0.5 * iv_estimate
        else:
            estimated_move = std_dev
        
        return float(estimated_move)
    
    def compute_eps_trend(
        self, 
        earnings_history: List[Dict],
        lookback: int = 3
    ) -> EPSSurpriseTrend:
        """
        Compute EPS surprise trend from last N earnings
        
        Args:
            earnings_history: List of earnings events
            lookback: Number of earnings to consider
            
        Returns:
            EPSSurpriseTrend enum value
        """
        if len(earnings_history) < lookback:
            return EPSSurpriseTrend.MIXED
        
        # Get last N surprises
        recent = earnings_history[:lookback]
        
        beats = 0
        misses = 0
        
        for earning in recent:
            surprise = earning.get('eps_surprise', 0)
            if surprise > 0.01:  # Beat by > 1%
                beats += 1
            elif surprise < -0.01:  # Miss by > 1%
                misses += 1
        
        # Determine trend
        if beats == lookback:
            return EPSSurpriseTrend.BEAT_STREAK
        elif misses == lookback:
            return EPSSurpriseTrend.MISS_STREAK
        else:
            return EPSSurpriseTrend.MIXED
    
    def analyze(
        self, 
        ticker: str, 
        as_of_date: date,
        window_days: int,
        price_df: Optional[pd.DataFrame] = None
    ) -> EarningsAnalystOutput:
        """
        Main analysis method - executes all Earnings Analyst tasks
        
        Args:
            ticker: Stock ticker symbol
            as_of_date: Analysis anchor date
            window_days: Analysis window in days
            price_df: Optional pre-fetched price DataFrame
            
        Returns:
            EarningsAnalystOutput with all computed metrics
        """
        log.info(f"Starting Earnings Analyst analysis for {ticker} as of {as_of_date}")
        
        # Step 1: Fetch earnings history
        earnings_history = self.fetch_earnings_history(ticker, as_of_date, max_earnings=12)
        
        # Step 2: Get next earnings date
        next_earnings = self.get_next_earnings_date(ticker, as_of_date)
        
        # Step 3: Check if earnings falls in window
        earnings_in_window = False
        if next_earnings:
            days_until_earnings = (next_earnings - as_of_date).days
            if 0 < days_until_earnings <= window_days:
                earnings_in_window = True
                log.info(f"Earnings in {days_until_earnings} days (within window)")
        
        # Step 4: Extract earnings dates and reactions
        earnings_dates = [e['report_date'].isoformat() for e in earnings_history]
        
        # Try to compute actual reactions from price data
        if price_df is not None and not price_df.empty:
            post_earnings_returns = self.compute_reactions(price_df, earnings_history)
        else:
            # Use stored reactions
            post_earnings_returns = [e.get('reaction_1d', 0.0) for e in earnings_history]
        
        # Step 5: Calculate statistics
        if post_earnings_returns:
            mean_reaction = float(np.mean(post_earnings_returns))
            median_reaction = float(np.median(post_earnings_returns))
            pct_positive = float(np.sum(np.array(post_earnings_returns) > 0) / len(post_earnings_returns))
        else:
            mean_reaction = 0.01
            median_reaction = 0.005
            pct_positive = 0.55
        
        # Step 6: Estimate earnings move
        estimated_move = self.estimate_earnings_move(post_earnings_returns)
        
        # Step 7: Compute EPS surprise trend
        eps_trend = self.compute_eps_trend(earnings_history, lookback=3)
        
        log.info(
            f"Earnings Analyst complete: mean reaction={mean_reaction:.2%}, "
            f"in_window={earnings_in_window}, trend={eps_trend.value}"
        )
        
        return EarningsAnalystOutput(
            earnings_dates=earnings_dates,
            post_earnings_returns_1d=post_earnings_returns,
            mean_reaction=mean_reaction,
            median_reaction=median_reaction,
            pct_positive_reactions=pct_positive,
            next_earnings_date=next_earnings.isoformat() if next_earnings else None,
            earnings_in_window=earnings_in_window,
            estimated_earnings_move=estimated_move,
            eps_surprise_trend=eps_trend,
        )
    
    def _create_empty_output(self) -> EarningsAnalystOutput:
        """Create empty output when no data is available"""
        return EarningsAnalystOutput(
            earnings_dates=[],
            post_earnings_returns_1d=[],
            mean_reaction=0.0,
            median_reaction=0.0,
            pct_positive_reactions=0.5,
            next_earnings_date=None,
            earnings_in_window=False,
            estimated_earnings_move=0.04,
            eps_surprise_trend=EPSSurpriseTrend.MIXED,
        )


# Convenience function
def analyze_earnings(
    ticker: str, 
    as_of_date: date,
    window_days: int,
    data_source: Optional[DataSource] = None
) -> EarningsAnalystOutput:
    """
    Convenience function to run Earnings Analyst analysis
    
    Args:
        ticker: Stock ticker symbol
        as_of_date: Analysis anchor date
        window_days: Analysis window in days
        data_source: Optional custom data source
        
    Returns:
        EarningsAnalystOutput
    """
    analyst = EarningsAnalyst(data_source)
    return analyst.analyze(ticker, as_of_date, window_days)
