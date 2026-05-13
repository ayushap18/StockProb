"""
StockProb Multi-Agent Orchestration Engine
Agent 2: Volatility Engine - Multi-window realized volatility + VIX correlation
"""

import pandas as pd
import numpy as np
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple

from ..core.models import VolatilityEngineOutput, VolRegime
from ..data.sources import get_data_source, DataSource
from ..utils import log


class VolatilityEngine:
    """
    Agent 2: Volatility Engine
    
    Role: Multi-window realized volatility + VIX correlation.
    
    Responsibilities:
    - Compute annualized σ over 10d, 21d, 63d, 252d windows
    - Compute Parkinson volatility (high-low estimator)
    - Compute GARCH(1,1) conditional volatility if >500 sessions available
    - Pull VIX current level and 52w percentile
    - Compute beta vs SPY over 252d
    - Classify vol regime: low | elevated | crisis
    """
    
    def __init__(self, data_source: Optional[DataSource] = None):
        """
        Initialize Volatility Engine
        
        Args:
            data_source: Optional custom data source
        """
        self.data_source = data_source or get_data_source()
        self.vol_windows = [10, 21, 63, 252]
        log.info("Volatility Engine agent initialized")
    
    def compute_realized_volatility(
        self, 
        returns: np.ndarray, 
        window: int
    ) -> float:
        """
        Compute annualized realized volatility over a window
        
        Args:
            returns: Array of log returns
            window: Window size in days
            
        Returns:
            Annualized volatility
        """
        if len(returns) < window:
            returns_window = returns
        else:
            returns_window = returns[-window:]
        
        if len(returns_window) < 2:
            return 0.0
        
        # Daily volatility
        daily_vol = np.std(returns_window, ddof=1)
        
        # Annualize (252 trading days)
        annualized_vol = daily_vol * np.sqrt(252)
        
        return float(annualized_vol)
    
    def compute_parkinson_volatility(
        self, 
        df: pd.DataFrame, 
        window: int = 21
    ) -> float:
        """
        Compute Parkinson volatility using high-low estimator
        
        Parkinson estimator: σ = sqrt(1/(4n*ln2) * Σ(ln(H/L))^2)
        
        Args:
            df: DataFrame with 'high' and 'low' columns
            window: Window size for calculation
            
        Returns:
            Parkinson volatility (annualized)
        """
        if df.empty or 'high' not in df.columns or 'low' not in df.columns:
            return 0.0
        
        # Get recent data
        df_recent = df.tail(window).copy()
        
        if len(df_recent) < 5:
            return 0.0
        
        # Calculate ln(H/L)
        hl_ratio = np.log(df_recent['high'] / df_recent['low'])
        
        # Parkinson formula
        n = len(hl_ratio)
        if n == 0:
            return 0.0
        
        sum_sq = np.sum(hl_ratio ** 2)
        daily_vol = np.sqrt(sum_sq / (4 * n * np.log(2)))
        
        # Annualize
        annualized_vol = daily_vol * np.sqrt(252)
        
        return float(annualized_vol)
    
    def compute_garch_volatility(
        self, 
        returns: np.ndarray
    ) -> Optional[Dict[str, float]]:
        """
        Fit GARCH(1,1) model and get volatility forecast
        
        Args:
            returns: Array of log returns
            
        Returns:
            Dict with GARCH parameters and forecasts, or None if fitting fails
        """
        if len(returns) < 500:
            log.debug(f"Insufficient data for GARCH ({len(returns)} < 500)")
            return None
        
        try:
            # Try to use arch library
            from arch import arch_model
            
            # Remove NaN values
            returns_clean = returns[~np.isnan(returns)]
            
            if len(returns_clean) < 500:
                return None
            
            # Fit GARCH(1,1)
            am = arch_model(returns_clean * 100, vol='GARCH', p=1, q=1, rescale=False)
            res = am.fit(disp='off')
            
            # Extract parameters
            omega = res.params.get('omega', 0.0)
            alpha = res.params.get('alpha[1]', 0.0)
            beta = res.params.get('beta[1]', 0.0)
            
            # Get conditional volatility (last value)
            cond_vol = res.conditional_volatility
            
            # 1-day ahead forecast (in percent, convert back)
            forecast_1d = res.forecast(horizon=1).mean.iloc[-1, 0] / 100
            
            log.info(
                f"GARCH fit successful: ω={omega:.6f}, α={alpha:.3f}, β={beta:.3f}"
            )
            
            return {
                "omega": float(omega),
                "alpha": float(alpha),
                "beta": float(beta),
                "garch_vol_1d_ahead": float(forecast_1d) if not np.isnan(forecast_1d) else None,
                "last_conditional_vol": float(cond_vol[-1]) / np.sqrt(252) if len(cond_vol) > 0 else None,
            }
            
        except ImportError:
            log.warning("arch library not installed, skipping GARCH")
            return None
        except Exception as e:
            log.warning(f"GARCH fitting failed: {str(e)}")
            return None
    
    def compute_beta(
        self, 
        stock_returns: np.ndarray, 
        spy_returns: np.ndarray
    ) -> float:
        """
        Compute beta vs SPY over 252 days
        
        Beta = Cov(R_stock, R_market) / Var(R_market)
        
        Args:
            stock_returns: Stock log returns
            spy_returns: SPY log returns
            
        Returns:
            Beta coefficient
        """
        # Align lengths
        min_len = min(len(stock_returns), len(spy_returns))
        if min_len < 21:
            return 1.0
        
        stock_ret = stock_returns[-min_len:]
        spy_ret = spy_returns[-min_len:]
        
        # Remove NaN values
        mask = ~(np.isnan(stock_ret) | np.isnan(spy_ret))
        stock_ret = stock_ret[mask]
        spy_ret = spy_ret[mask]
        
        if len(stock_ret) < 21:
            return 1.0
        
        # Calculate covariance and variance
        cov = np.cov(stock_ret, spy_ret)[0, 1]
        var_spy = np.var(spy_ret, ddof=1)
        
        if var_spy == 0:
            return 1.0
        
        beta = cov / var_spy
        
        return float(beta)
    
    def classify_vol_regime(
        self, 
        current_vol: float, 
        vix_percentile: float
    ) -> VolRegime:
        """
        Classify volatility regime
        
        Rules:
        - Crisis: VIX percentile > 80 OR realized vol > 40%
        - Elevated: VIX percentile > 60 OR realized vol > 25%
        - Low: Otherwise
        
        Args:
            current_vol: Current realized volatility
            vix_percentile: VIX 52-week percentile
            
        Returns:
            VolRegime enum value
        """
        if vix_percentile > 80 or current_vol > 0.40:
            return VolRegime.CRISIS
        
        if vix_percentile > 60 or current_vol > 0.25:
            return VolRegime.ELEVATED
        
        return VolRegime.LOW
    
    def analyze(
        self, 
        ticker: str, 
        as_of_date: date,
        price_df: Optional[pd.DataFrame] = None
    ) -> VolatilityEngineOutput:
        """
        Main analysis method - executes all Volatility Engine tasks
        
        Args:
            ticker: Stock ticker symbol
            as_of_date: Analysis anchor date
            price_df: Optional pre-fetched price DataFrame
            
        Returns:
            VolatilityEngineOutput with all computed metrics
        """
        log.info(f"Starting Volatility Engine analysis for {ticker} as of {as_of_date}")
        
        # Step 1: Fetch price data if not provided
        if price_df is None:
            start_date = as_of_date - timedelta(days=365)
            end_date = as_of_date
            price_df = self.data_source.fetch_price_data(ticker, start_date, end_date)
        
        if price_df.empty:
            log.error("No price data for volatility analysis")
            return self._create_empty_output()
        
        # Step 2: Calculate log returns
        closes = price_df['close'].values
        returns = []
        for i in range(1, len(closes)):
            if closes[i-1] > 0 and closes[i] > 0:
                lr = np.log(closes[i] / closes[i-1])
            else:
                lr = 0.0
            returns.append(lr)
        
        returns = np.array(returns)
        
        # Step 3: Compute realized volatility for each window
        realized_vols = {}
        for window in self.vol_windows:
            vol = self.compute_realized_volatility(returns, window)
            realized_vols[f'realized_vol_{window}d'] = vol
        
        # Step 4: Compute Parkinson volatility
        parkinson_vol = self.compute_parkinson_volatility(price_df, window=21)
        
        # Step 5: Compute GARCH volatility (if enough data)
        garch_results = self.compute_garch_volatility(returns)
        garch_vol_1d = None
        if garch_results:
            garch_vol_1d = garch_results.get('garch_vol_1d_ahead')
        
        # Step 6: Fetch VIX data
        vix_data = self.data_source.fetch_vix_data()
        vix_current = vix_data.get('vix_current', 20.0)
        vix_percentile = vix_data.get('vix_percentile_52w', 50.0)
        
        # Step 7: Fetch SPY data and compute beta
        start_date = as_of_date - timedelta(days=365)
        spy_df = self.data_source.fetch_spy_data(start_date, as_of_date)
        
        if not spy_df.empty and 'close' in spy_df.columns:
            spy_closes = spy_df['close'].values
            spy_returns = []
            for i in range(1, len(spy_closes)):
                if spy_closes[i-1] > 0 and spy_closes[i] > 0:
                    lr = np.log(spy_closes[i] / spy_closes[i-1])
                else:
                    lr = 0.0
                spy_returns.append(lr)
            spy_returns = np.array(spy_returns)
            
            beta = self.compute_beta(returns, spy_returns)
        else:
            beta = 1.0
            log.warning("Could not fetch SPY data, using beta=1.0")
        
        # Step 8: Classify volatility regime
        current_vol = realized_vols.get('realized_vol_21d', 0.20)
        vol_regime = self.classify_vol_regime(current_vol, vix_percentile)
        
        log.info(
            f"Volatility Engine complete: 21d vol={current_vol:.2%}, "
            f"VIX={vix_current:.1f}, beta={beta:.2f}, regime={vol_regime.value}"
        )
        
        return VolatilityEngineOutput(
            realized_vol_10d=realized_vols.get('realized_vol_10d', 0.0),
            realized_vol_21d=realized_vols.get('realized_vol_21d', 0.0),
            realized_vol_63d=realized_vols.get('realized_vol_63d', 0.0),
            realized_vol_252d=realized_vols.get('realized_vol_252d', 0.0),
            parkinson_vol=parkinson_vol,
            garch_vol_1d_ahead=garch_vol_1d,
            vix_current=vix_current,
            vix_percentile_52w=vix_percentile,
            beta_252d=beta,
            vol_regime=vol_regime,
        )
    
    def _create_empty_output(self) -> VolatilityEngineOutput:
        """Create empty output when no data is available"""
        return VolatilityEngineOutput(
            realized_vol_10d=0.0,
            realized_vol_21d=0.0,
            realized_vol_63d=0.0,
            realized_vol_252d=0.0,
            parkinson_vol=0.0,
            garch_vol_1d_ahead=None,
            vix_current=20.0,
            vix_percentile_52w=50.0,
            beta_252d=1.0,
            vol_regime=VolRegime.LOW,
        )


# Convenience function
def analyze_volatility(
    ticker: str, 
    as_of_date: date,
    data_source: Optional[DataSource] = None
) -> VolatilityEngineOutput:
    """
    Convenience function to run Volatility Engine analysis
    
    Args:
        ticker: Stock ticker symbol
        as_of_date: Analysis anchor date
        data_source: Optional custom data source
        
    Returns:
        VolatilityEngineOutput
    """
    engine = VolatilityEngine(data_source)
    return engine.analyze(ticker, as_of_date)
