"""
StockProb Multi-Agent Orchestration Engine
Agent 1: Price Historian - Historical price data analysis
"""

import pandas as pd
import numpy as np
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple

from ..core.models import PriceHistorianOutput, PriceRegime
from ..data.sources import get_data_source, DataSource
from ..utils import log, calculate_log_returns


class PriceHistorian:
    """
    Agent 1: Price Historian
    
    Role: Pull and clean all historical OHLCV data since IPO.
    
    Responsibilities:
    - Fetch daily close prices
    - Compute log returns for every session
    - Calculate rolling windows: 10d, 21d, 63d, 252d
    - Detect regime changes using 200d MA crossovers
    - Flag if current price is above/below all key MAs
    """
    
    def __init__(self, data_source: Optional[DataSource] = None):
        """
        Initialize Price Historian
        
        Args:
            data_source: Optional custom data source (uses default if None)
        """
        self.data_source = data_source or get_data_source()
        log.info("Price Historian agent initialized")
    
    def fetch_historical_data(
        self, 
        ticker: str, 
        as_of_date: date,
        lookback_years: int = 10
    ) -> pd.DataFrame:
        """
        Fetch historical price data
        
        Args:
            ticker: Stock ticker symbol
            as_of_date: Analysis anchor date
            lookback_years: How many years of history to fetch
            
        Returns:
            DataFrame with historical OHLCV data
        """
        start_date = as_of_date - timedelta(days=lookback_years * 365)
        end_date = as_of_date
        
        log.info(f"Fetching price history for {ticker} from {start_date} to {end_date}")
        
        df = self.data_source.fetch_price_data(ticker, start_date, end_date)
        
        if df.empty:
            log.warning(f"No price data found for {ticker}")
            return pd.DataFrame()
        
        # Ensure we have required columns
        required_cols = ['Date', 'close', 'open', 'high', 'low', 'volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            log.error(f"Missing required columns: {missing_cols}")
            return pd.DataFrame()
        
        # Sort by date
        df = df.sort_values('Date').reset_index(drop=True)
        
        log.info(f"Fetched {len(df)} trading sessions for {ticker}")
        return df
    
    def compute_log_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute log returns from price data
        
        Args:
            df: DataFrame with 'close' column
            
        Returns:
            DataFrame with added 'log_return' column
        """
        if df.empty or 'close' not in df.columns:
            return df
        
        df = df.copy()
        
        # Calculate log returns
        closes = df['close'].values
        
        log_returns = []
        for i in range(1, len(closes)):
            if closes[i-1] > 0 and closes[i] > 0:
                lr = np.log(closes[i] / closes[i-1])
            else:
                lr = 0.0
            log_returns.append(lr)
        
        # First row has no return
        df['log_return'] = [np.nan] + log_returns
        
        return df
    
    def calculate_moving_averages(
        self, 
        df: pd.DataFrame, 
        windows: List[int] = None
    ) -> pd.DataFrame:
        """
        Calculate moving averages for specified windows
        
        Args:
            df: DataFrame with 'close' column
            windows: List of window sizes (default: [10, 21, 63, 252])
            
        Returns:
            DataFrame with added MA columns
        """
        if windows is None:
            windows = [10, 21, 63, 252]
        
        if df.empty or 'close' not in df.columns:
            return df
        
        df = df.copy()
        
        for window in windows:
            col_name = f'ma_{window}'
            df[col_name] = df['close'].rolling(window=window, min_periods=1).mean()
        
        return df
    
    def calculate_rolling_stats(
        self, 
        df: pd.DataFrame, 
        windows: List[int] = None
    ) -> pd.DataFrame:
        """
        Calculate rolling mean and std of log returns
        
        Args:
            df: DataFrame with 'log_return' column
            windows: List of window sizes
            
        Returns:
            DataFrame with added rolling stat columns
        """
        if windows is None:
            windows = [10, 21, 63, 252]
        
        if df.empty or 'log_return' not in df.columns:
            return df
        
        df = df.copy()
        
        for window in windows:
            # Rolling mean
            df[f'log_return_mean_{window}d'] = (
                df['log_return'].rolling(window=window, min_periods=1).mean()
            )
            
            # Rolling std (annualized)
            df[f'log_return_std_{window}d'] = (
                df['log_return'].rolling(window=window, min_periods=1).std() * np.sqrt(252)
            )
        
        return df
    
    def detect_regime(self, df: pd.DataFrame) -> PriceRegime:
        """
        Detect price regime using 200-day MA crossover logic
        
        Rules:
        - Bull: Price > MA200 AND MA50 > MA200
        - Bear: Price < MA200 AND MA50 < MA200
        - Transitional: Mixed signals
        
        Args:
            df: DataFrame with MA columns
            
        Returns:
            PriceRegime enum value
        """
        if df.empty:
            return PriceRegime.TRANSITIONAL
        
        # Get latest row
        latest = df.iloc[-1]
        
        # Check if we have MA data
        if 'ma_50' not in latest or 'ma_200' not in latest:
            return PriceRegime.TRANSITIONAL
        
        current_price = latest['close']
        ma_50 = latest['ma_50']
        ma_200 = latest['ma_200']
        
        # Bullish: price above both MAs and short MA above long MA
        if current_price > ma_200 and ma_50 > ma_200:
            return PriceRegime.BULL
        
        # Bearish: price below both MAs and short MA below long MA
        if current_price < ma_200 and ma_50 < ma_200:
            return PriceRegime.BEAR
        
        # Mixed signals
        return PriceRegime.TRANSITIONAL
    
    def analyze(self, ticker: str, as_of_date: date) -> PriceHistorianOutput:
        """
        Main analysis method - executes all Price Historian tasks
        
        Args:
            ticker: Stock ticker symbol
            as_of_date: Analysis anchor date
            
        Returns:
            PriceHistorianOutput with all computed metrics
        """
        log.info(f"Starting Price Historian analysis for {ticker} as of {as_of_date}")
        
        # Step 1: Fetch historical data
        df = self.fetch_historical_data(ticker, as_of_date, lookback_years=10)
        
        if df.empty:
            log.error("No data available for analysis")
            return self._create_empty_output(ticker)
        
        # Step 2: Compute log returns
        df = self.compute_log_returns(df)
        
        # Step 3: Calculate moving averages
        df = self.calculate_moving_averages(df)
        
        # Step 4: Calculate rolling statistics
        df = self.calculate_rolling_stats(df)
        
        # Step 5: Extract metrics from latest row
        latest = df.iloc[-1]
        
        total_sessions = len(df)
        current_price = float(latest['close'])
        
        # Get 252-day log return stats
        if 'log_return_mean_252d' in latest:
            log_return_mean_252d = float(latest['log_return_mean_252d'])
            log_return_std_252d = float(latest['log_return_std_252d'])
        else:
            # Fallback: calculate from raw returns
            returns_252 = df['log_return'].dropna().tail(252).values
            log_return_mean_252d = float(np.mean(returns_252)) if len(returns_252) > 0 else 0.0
            log_return_std_252d = float(np.std(returns_252) * np.sqrt(252)) if len(returns_252) > 0 else 0.0
        
        # Moving averages
        ma_50 = float(latest.get('ma_50', current_price))
        ma_200 = float(latest.get('ma_200', current_price))
        
        # Price vs MA200
        price_vs_ma200 = "above" if current_price > ma_200 else "below"
        
        # Detect regime
        regime = self.detect_regime(df)
        
        # Get raw returns (last 756 sessions minimum)
        raw_returns = df['log_return'].dropna().tail(max(756, len(df))).tolist()
        
        # Ensure we have at least some returns
        if not raw_returns:
            raw_returns = [0.0]
        
        log.info(
            f"Price Historian complete: {total_sessions} sessions, "
            f"price={current_price:.2f}, regime={regime.value}"
        )
        
        return PriceHistorianOutput(
            total_sessions=total_sessions,
            current_price=current_price,
            log_return_mean_252d=log_return_mean_252d,
            log_return_std_252d=log_return_std_252d,
            ma_50=ma_50,
            ma_200=ma_200,
            price_vs_ma200=price_vs_ma200,
            regime=regime,
            raw_returns=raw_returns,
        )
    
    def _create_empty_output(self, ticker: str) -> PriceHistorianOutput:
        """Create empty output when no data is available"""
        return PriceHistorianOutput(
            total_sessions=0,
            current_price=0.0,
            log_return_mean_252d=0.0,
            log_return_std_252d=0.0,
            ma_50=0.0,
            ma_200=0.0,
            price_vs_ma200="below",
            regime=PriceRegime.TRANSITIONAL,
            raw_returns=[0.0],
        )


# Convenience function for direct usage
def analyze_price_history(
    ticker: str, 
    as_of_date: date,
    data_source: Optional[DataSource] = None
) -> PriceHistorianOutput:
    """
    Convenience function to run Price Historian analysis
    
    Args:
        ticker: Stock ticker symbol
        as_of_date: Analysis anchor date
        data_source: Optional custom data source
        
    Returns:
        PriceHistorianOutput
    """
    historian = PriceHistorian(data_source)
    return historian.analyze(ticker, as_of_date)
