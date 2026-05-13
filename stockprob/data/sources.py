"""
StockProb Multi-Agent Orchestration Engine
Data services for fetching and managing market data
"""

import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple
from abc import ABC, abstractmethod

from ..utils import log, cache, cached
from ..core.models import OHLCVData


class DataSource(ABC):
    """Abstract base class for data sources"""
    
    @abstractmethod
    def fetch_price_data(self, ticker: str, start_date: date, end_date: date) -> pd.DataFrame:
        pass
    
    @abstractmethod
    def fetch_earnings_dates(self, ticker: str) -> List[date]:
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        pass


class YFinanceDataSource(DataSource):
    """
    Data source using yfinance library
    Free, no API key required
    """
    
    def __init__(self):
        self._yfinance = None
        self._initialize()
    
    def _initialize(self):
        """Lazy import of yfinance"""
        try:
            import yfinance as yf
            self._yfinance = yf
            log.info("YFinance data source initialized")
        except ImportError:
            log.error("yfinance not installed. Run: pip install yfinance")
            self._yfinance = None
    
    def is_available(self) -> bool:
        return self._yfinance is not None
    
    @cached(ttl=3600, key_prefix="yf_prices:")
    def fetch_price_data(
        self, 
        ticker: str, 
        start_date: date, 
        end_date: date
    ) -> pd.DataFrame:
        """
        Fetch historical OHLCV data from Yahoo Finance
        
        Args:
            ticker: Stock ticker symbol
            start_date: Start date for data
            end_date: End date for data
            
        Returns:
            DataFrame with columns: Date, Open, High, Low, Close, Adj Close, Volume
        """
        if not self._yfinance:
            raise RuntimeError("YFinance not available")
        
        try:
            ticker_obj = self._yfinance.Ticker(ticker)
            
            # Fetch data (yfinance expects dates as strings)
            start_str = start_date.strftime("%Y-%m-%d")
            end_str = (end_date + timedelta(days=1)).strftime("%Y-%m-%d")  # Inclusive end
            
            df = ticker_obj.history(start=start_str, end=end_str)
            
            if df.empty:
                log.warning(f"No data returned for {ticker}")
                return pd.DataFrame()
            
            # Reset index to have Date as column
            df = df.reset_index()
            
            # Ensure Date column is datetime
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date']).dt.date
            
            # Standardize column names
            df = df.rename(columns={
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Adj Close': 'adjusted_close',
                'Volume': 'volume',
            })
            
            log.info(f"Fetched {len(df)} rows for {ticker} from {start_date} to {end_date}")
            return df
            
        except Exception as e:
            log.error(f"Error fetching data for {ticker}: {str(e)}")
            return pd.DataFrame()
    
    @cached(ttl=86400, key_prefix="yf_earnings:")
    def fetch_earnings_dates(self, ticker: str) -> List[date]:
        """
        Fetch earnings dates from Yahoo Finance
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            List of earnings dates
        """
        if not self._yfinance:
            return []
        
        try:
            ticker_obj = self._yfinance.Ticker(ticker)
            
            # Get earnings calendar
            earnings_calendar = ticker_obj.calendar
            
            if earnings_calendar is None or earnings_calendar.empty:
                return []
            
            # Extract dates
            dates = []
            if 'Earnings Date' in earnings_calendar.index:
                earnings_date = earnings_calendar.loc['Earnings Date']
                if isinstance(earnings_date, pd.Timestamp):
                    dates.append(earnings_date.date())
                elif hasattr(earnings_date, 'date'):
                    dates.append(earnings_date.date())
            
            # Also get historical earnings from earnings history
            try:
                earnings_history = ticker_obj.earnings_history
                if earnings_history is not None and not earnings_history.empty:
                    for idx in earnings_history.index:
                        if isinstance(idx, pd.Timestamp):
                            dates.append(idx.date())
                        elif hasattr(idx, 'date'):
                            dates.append(idx.date())
            except:
                pass
            
            # Remove duplicates and sort
            dates = sorted(list(set(dates)))
            
            log.info(f"Found {len(dates)} earnings dates for {ticker}")
            return dates
            
        except Exception as e:
            log.error(f"Error fetching earnings dates for {ticker}: {str(e)}")
            return []
    
    def fetch_sector_info(self, ticker: str) -> Dict[str, str]:
        """
        Fetch sector and industry information
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Dict with sector and industry info
        """
        if not self._yfinance:
            return {"sector": "Unknown", "industry": "Unknown"}
        
        try:
            ticker_obj = self._yfinance.Ticker(ticker)
            info = ticker_obj.info
            
            return {
                "sector": info.get("sector", "Unknown"),
                "industry": info.get("industry", "Unknown"),
                "market_cap": info.get("marketCap", None),
                "beta": info.get("beta", None),
            }
            
        except Exception as e:
            log.error(f"Error fetching sector info for {ticker}: {str(e)}")
            return {"sector": "Unknown", "industry": "Unknown"}
    
    def fetch_vix_data(self) -> Dict[str, float]:
        """
        Fetch VIX (volatility index) data
        
        Returns:
            Dict with current VIX level and historical percentile
        """
        if not self._yfinance:
            return {"vix_current": 20.0, "vix_percentile_52w": 50.0}
        
        try:
            vix_ticker = self._yfinance.Ticker("^VIX")
            
            # Get current VIX
            hist = vix_ticker.history(period="1d")
            if hist.empty:
                vix_current = 20.0
            else:
                vix_current = hist['Close'].iloc[-1]
            
            # Get 52-week range for percentile calculation
            hist_52w = vix_ticker.history(period="1y")
            if len(hist_52w) > 0:
                vix_min = hist_52w['Close'].min()
                vix_max = hist_52w['Close'].max()
                
                if vix_max > vix_min:
                    vix_percentile = ((vix_current - vix_min) / (vix_max - vix_min)) * 100
                else:
                    vix_percentile = 50.0
            else:
                vix_percentile = 50.0
            
            log.info(f"VIX: {vix_current:.2f}, 52w percentile: {vix_percentile:.1f}")
            
            return {
                "vix_current": float(vix_current),
                "vix_percentile_52w": float(vix_percentile),
            }
            
        except Exception as e:
            log.error(f"Error fetching VIX data: {str(e)}")
            return {"vix_current": 20.0, "vix_percentile_52w": 50.0}
    
    def fetch_spy_data(self, start_date: date, end_date: date) -> pd.DataFrame:
        """
        Fetch SPY (S&P 500 ETF) data for beta calculation
        
        Args:
            start_date: Start date
            end_date: End date
            
        Returns:
            DataFrame with SPY prices
        """
        return self.fetch_price_data("SPY", start_date, end_date)
    
    def fetch_insider_trades(self, ticker: str) -> List[Dict]:
        """
        Fetch insider trading data (Form 4)
        Note: Limited availability through yfinance
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            List of insider trade dictionaries
        """
        if not self._yfinance:
            return []
        
        try:
            ticker_obj = self._yfinance.Ticker(ticker)
            
            # Get insider transactions
            try:
                insider_df = ticker_obj.insider_transactions
                
                if insider_df is None or insider_df.empty:
                    return []
                
                trades = []
                for _, row in insider_df.iterrows():
                    trades.append({
                        "insider_name": row.get('Insider', 'Unknown'),
                        "title": row.get('Position', 'Unknown'),
                        "transaction_type": row.get('Transaction', 'Unknown'),
                        "shares": int(row.get('Shares', 0)),
                        "price_per_share": float(row.get('Value', 0)) / max(int(row.get('Shares', 1)), 1),
                        "transaction_date": row.get('Date', None),
                    })
                
                log.info(f"Found {len(trades)} insider trades for {ticker}")
                return trades
                
            except Exception as e:
                log.debug(f"No insider data for {ticker}: {str(e)}")
                return []
                
        except Exception as e:
            log.error(f"Error fetching insider trades for {ticker}: {str(e)}")
            return []


class MockDataSource(DataSource):
    """
    Mock data source for testing
    Generates realistic synthetic data
    """
    
    def __init__(self, seed: int = 42):
        np.random.seed(seed)
        log.info("Mock data source initialized")
    
    def is_available(self) -> bool:
        return True
    
    def fetch_price_data(
        self, 
        ticker: str, 
        start_date: date, 
        end_date: date
    ) -> pd.DataFrame:
        """Generate synthetic price data using geometric Brownian motion"""
        
        # Calculate number of trading days
        total_days = (end_date - start_date).days
        trading_days = int(total_days * 5 / 7)  # Approximate
        
        if trading_days < 1:
            return pd.DataFrame()
        
        # Generate GBM path
        mu = 0.0002  # Daily drift
        sigma = 0.02  # Daily volatility
        
        initial_price = 150.0  # Starting price
        
        returns = np.random.normal(mu, sigma, trading_days)
        prices = initial_price * np.exp(np.cumsum(returns))
        
        # Create DataFrame
        dates = pd.date_range(start=start_date, periods=trading_days, freq='B')
        
        df = pd.DataFrame({
            'Date': dates.date,
            'open': prices * (1 + np.random.uniform(-0.005, 0.005, trading_days)),
            'high': prices * (1 + np.random.uniform(0, 0.02, trading_days)),
            'low': prices * (1 + np.random.uniform(-0.02, 0, trading_days)),
            'close': prices,
            'adjusted_close': prices,
            'volume': np.random.randint(1000000, 100000000, trading_days),
        })
        
        log.info(f"Generated {len(df)} mock rows for {ticker}")
        return df
    
    def fetch_earnings_dates(self, ticker: str) -> List[date]:
        """Generate synthetic earnings dates"""
        today = date.today()
        
        # Generate quarterly earnings dates
        dates = []
        for months_ago in range(0, 48, 3):  # Last 4 years
            d = today - timedelta(days=months_ago * 30)
            # Set to approximate earnings day (15th of month)
            d = d.replace(day=15)
            dates.append(d)
        
        # Add next earnings date
        next_earnings = today + timedelta(days=np.random.randint(10, 50))
        dates.append(next_earnings)
        
        return sorted(dates)
    
    def fetch_sector_info(self, ticker: str) -> Dict[str, str]:
        """Return mock sector info"""
        sectors = ["Technology", "Health Care", "Financials", "Consumer Discretionary"]
        return {
            "sector": np.random.choice(sectors),
            "industry": "Mock Industry",
            "market_cap": 1000000000000,
            "beta": 1.15,
        }
    
    def fetch_vix_data(self) -> Dict[str, float]:
        """Return mock VIX data"""
        return {
            "vix_current": float(15 + np.random.exponential(5)),
            "vix_percentile_52w": float(np.random.uniform(30, 70)),
        }
    
    def fetch_spy_data(self, start_date: date, end_date: date) -> pd.DataFrame:
        """Generate mock SPY data"""
        return self.fetch_price_data("SPY", start_date, end_date)
    
    def fetch_insider_trades(self, ticker: str) -> List[Dict]:
        """Return mock insider trades"""
        n_trades = np.random.randint(0, 10)
        trades = []
        
        for _ in range(n_trades):
            trade_type = np.random.choice(['P', 'S'], p=[0.3, 0.7])
            trades.append({
                "insider_name": f"Executive {_}",
                "title": "Officer",
                "transaction_type": trade_type,
                "shares": int(np.random.exponential(10000)),
                "price_per_share": float(100 + np.random.normal(0, 20)),
                "transaction_date": date.today() - timedelta(days=np.random.randint(1, 90)),
            })
        
        return trades


# ============================================================================
# DATA SOURCE FACTORY
# ============================================================================

def get_data_source(use_mock: bool = False) -> DataSource:
    """
    Factory function to get appropriate data source
    
    Args:
        use_mock: If True, use mock data source regardless
        
    Returns:
        DataSource instance
    """
    if use_mock:
        return MockDataSource()
    
    # Try real data source first
    yf_source = YFinanceDataSource()
    if yf_source.is_available():
        return yf_source
    
    # Fall back to mock
    log.warning("Real data source not available, falling back to mock")
    return MockDataSource()
