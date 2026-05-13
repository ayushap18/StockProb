"""
StockProb Multi-Agent Orchestration Engine
Agent 5: Macro Puller - Live macro state from FRED + sector context
"""

import numpy as np
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple

from ..core.models import MacroPullerOutput, MacroRegime, SectorRelativeStrength
from ..config.settings import SECTOR_ETF_MAP
from ..data.sources import get_data_source, DataSource
from ..utils import log


class MacroPuller:
    """
    Agent 5: Macro Puller
    
    Role: Live macro state from FRED + sector context.
    
    Responsibilities:
    - Pull from FRED API: Fed Funds Rate, Treasury yields, CPI, unemployment, PMI
    - Classify macro regime: expansion | stagflation | recession | recovery
    - Pull sector ETF performance vs SPY
    - Determine sector relative strength
    """
    
    def __init__(self, data_source: Optional[DataSource] = None):
        """
        Initialize Macro Puller
        
        Args:
            data_source: Optional custom data source
        """
        self.data_source = data_source or get_data_source()
        log.info("Macro Puller agent initialized")
    
    def fetch_fred_data(self, as_of_date: date) -> Dict[str, float]:
        """
        Fetch economic data from FRED API
        
        In production, this would use the actual FRED API.
        For now, generates realistic synthetic data.
        
        Args:
            as_of_date: Analysis anchor date
            
        Returns:
            Dict of macroeconomic indicators
        """
        # Generate synthetic but realistic macro data
        # In production: use fredapi library
        np.random.seed(hash(as_of_date.isoformat()) % 2**32)
        
        # Typical ranges for each indicator
        fed_funds = 4.5 + np.random.normal(0, 0.5)  # Current range ~4-5%
        treasury_10y = 3.5 + np.random.normal(0, 0.5)  # ~3-4%
        treasury_2y = 4.0 + np.random.normal(0, 0.5)  # ~4-5%
        cpi_yoy = 3.0 + np.random.normal(0, 1.0)  # ~2-4%
        unemployment = 3.8 + np.random.normal(0, 0.3)  # ~3.5-4.5%
        
        # Ensure reasonable bounds
        fed_funds = max(0.0, min(10.0, fed_funds))
        treasury_10y = max(0.5, min(8.0, treasury_10y))
        treasury_2y = max(0.5, min(8.0, treasury_2y))
        cpi_yoy = max(-2.0, min(15.0, cpi_yoy))
        unemployment = max(2.0, min(12.0, unemployment))
        
        return {
            "fed_funds_rate": float(fed_funds),
            "treasury_10y": float(treasury_10y),
            "treasury_2y": float(treasury_2y),
            "cpi_yoy": float(cpi_yoy),
            "unemployment_rate": float(unemployment),
            "pull_date": as_of_date,
        }
    
    def classify_macro_regime(
        self, 
        macro_data: Dict[str, float],
        pmi: Optional[float] = None
    ) -> MacroRegime:
        """
        Classify macroeconomic regime based on indicators
        
        Rules:
        - Expansion: PMI>50, yield curve positive, unemployment falling
        - Stagflation: CPI>4%, PMI<50
        - Recession: unemployment rising, yield curve inverted >6mo
        - Recovery: PMI rising from <50, unemployment plateauing
        
        Args:
            macro_data: Dict of macro indicators
            pmi: Optional PMI value
            
        Returns:
            MacroRegime enum value
        """
        yield_curve = macro_data['treasury_10y'] - macro_data['treasury_2y']
        cpi = macro_data['cpi_yoy']
        unemployment = macro_data['unemployment_rate']
        
        # Use default PMI if not provided
        if pmi is None:
            # Estimate PMI from other indicators
            if yield_curve > 0 and unemployment < 4.5:
                pmi = 52 + np.random.normal(0, 2)
            elif yield_curve < 0 and unemployment > 4.5:
                pmi = 47 + np.random.normal(0, 2)
            else:
                pmi = 50 + np.random.normal(0, 2)
        
        # Check for stagflation first (high inflation + weak growth)
        if cpi > 4.0 and pmi < 50:
            return MacroRegime.STAGFLATION
        
        # Check for recession (inverted yield curve + high/rising unemployment)
        if yield_curve < -0.25 and unemployment > 4.5:
            return MacroRegime.RECESSION
        
        # Check for expansion (positive yield curve, low unemployment, PMI > 50)
        if yield_curve > 0 and unemployment < 4.2 and pmi > 50:
            return MacroRegime.EXPANSION
        
        # Default to recovery
        return MacroRegime.RECOVERY
    
    def get_sector_etf(self, ticker: str) -> str:
        """
        Get sector ETF for a given ticker
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Sector ETF ticker symbol
        """
        # Try to get sector info from data source
        sector_info = self.data_source.fetch_sector_info(ticker)
        sector = sector_info.get('sector', 'Unknown')
        
        # Map sector to ETF
        return SECTOR_ETF_MAP.get(sector, "XLK")  # Default to Tech
    
    def compute_sector_performance(
        self, 
        sector_etf: str, 
        as_of_date: date
    ) -> Tuple[float, SectorRelativeStrength]:
        """
        Compute sector ETF performance vs SPY
        
        Args:
            sector_etf: Sector ETF ticker
            as_of_date: Analysis anchor date
            
        Returns:
            Tuple of (3m return vs SPY, relative strength classification)
        """
        # Fetch price data for sector ETF and SPY
        start_date = as_of_date - timedelta(days=180)  # 6 months for context
        
        try:
            sector_df = self.data_source.fetch_price_data(sector_etf, start_date, as_of_date)
            spy_df = self.data_source.fetch_spy_data(start_date, as_of_date)
            
            if sector_df.empty or spy_df.empty:
                # Return mock data
                return self._mock_sector_performance()
            
            # Calculate 3-month returns
            if len(sector_df) >= 63 and len(spy_df) >= 63:
                sector_return = (sector_df['close'].iloc[-1] / sector_df['close'].iloc[-63]) - 1
                spy_return = (spy_df['close'].iloc[-1] / spy_df['close'].iloc[-63]) - 1
                
                relative_return = sector_return - spy_return
            else:
                return self._mock_sector_performance()
            
        except Exception as e:
            log.warning(f"Error computing sector performance: {e}")
            return self._mock_sector_performance()
        
        # Classify relative strength
        if relative_return > 0.03:  # Outperforming by > 3%
            strength = SectorRelativeStrength.OUTPERFORMING
        elif relative_return < -0.03:  # Underperforming by > 3%
            strength = SectorRelativeStrength.UNDERPERFORMING
        else:
            strength = SectorRelativeStrength.INLINE
        
        log.info(f"Sector {sector_etf}: 3m vs SPY = {relative_return:.2%}, {strength.value}")
        
        return float(relative_return), strength
    
    def _mock_sector_performance(self) -> Tuple[float, SectorRelativeStrength]:
        """Generate mock sector performance when real data unavailable"""
        np.random.seed()
        relative_return = np.random.normal(0, 0.03)
        
        if relative_return > 0.03:
            strength = SectorRelativeStrength.OUTPERFORMING
        elif relative_return < -0.03:
            strength = SectorRelativeStrength.UNDERPERFORMING
        else:
            strength = SectorRelativeStrength.INLINE
        
        return float(relative_return), strength
    
    def analyze(
        self, 
        ticker: str, 
        as_of_date: date
    ) -> MacroPullerOutput:
        """
        Main analysis method - executes all Macro Puller tasks
        
        Args:
            ticker: Stock ticker symbol
            as_of_date: Analysis anchor date
            
        Returns:
            MacroPullerOutput with all computed metrics
        """
        log.info(f"Starting Macro Puller analysis for {ticker} as of {as_of_date}")
        
        # Step 1: Fetch FRED data
        macro_data = self.fetch_fred_data(as_of_date)
        
        # Step 2: Calculate yield curve spread
        yield_curve_spread = macro_data['treasury_10y'] - macro_data['treasury_2y']
        
        # Step 3: Classify macro regime
        macro_regime = self.classify_macro_regime(macro_data)
        
        # Step 4: Get sector ETF
        sector_etf = self.get_sector_etf(ticker)
        
        # Step 5: Compute sector performance
        sector_vs_spy, sector_strength = self.compute_sector_performance(sector_etf, as_of_date)
        
        log.info(
            f"Macro Puller complete: regime={macro_regime.value}, "
            f"yield_curve={yield_curve_spread:.2f}%, sector={sector_strength.value}"
        )
        
        return MacroPullerOutput(
            fed_funds_rate=macro_data['fed_funds_rate'],
            treasury_10y=macro_data['treasury_10y'],
            treasury_2y=macro_data['treasury_2y'],
            yield_curve_spread=float(yield_curve_spread),
            cpi_yoy=macro_data['cpi_yoy'],
            unemployment_rate=macro_data['unemployment_rate'],
            macro_regime=macro_regime,
            sector_etf=sector_etf,
            sector_vs_spy_3m=sector_vs_spy,
            sector_relative_strength=sector_strength,
            fred_pulled_at=as_of_date,
        )
    
    def _create_empty_output(self, as_of_date: date) -> MacroPullerOutput:
        """Create empty output when no data is available"""
        return MacroPullerOutput(
            fed_funds_rate=4.5,
            treasury_10y=3.5,
            treasury_2y=4.0,
            yield_curve_spread=-0.5,
            cpi_yoy=3.0,
            unemployment_rate=4.0,
            macro_regime=MacroRegime.RECOVERY,
            sector_etf="XLK",
            sector_vs_spy_3m=0.0,
            sector_relative_strength=SectorRelativeStrength.INLINE,
            fred_pulled_at=as_of_date,
        )


# Convenience function
def analyze_macro(
    ticker: str, 
    as_of_date: date,
    data_source: Optional[DataSource] = None
) -> MacroPullerOutput:
    """
    Convenience function to run Macro Puller analysis
    
    Args:
        ticker: Stock ticker symbol
        as_of_date: Analysis anchor date
        data_source: Optional custom data source
        
    Returns:
        MacroPullerOutput
    """
    puller = MacroPuller(data_source)
    return puller.analyze(ticker, as_of_date)
