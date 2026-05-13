"""
StockProb Multi-Agent Orchestration Engine
Core configuration and constants
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
from enum import Enum


class RiskTolerance(Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"


class VolRegime(Enum):
    LOW = "low"
    ELEVATED = "elevated"
    CRISIS = "crisis"


class PriceRegime(Enum):
    BULL = "bull"
    BEAR = "bear"
    TRANSITIONAL = "transitional"


class MacroRegime(Enum):
    EXPANSION = "expansion"
    STAGFLATION = "stagflation"
    RECESSION = "recession"
    RECOVERY = "recovery"


class SectorRelativeStrength(Enum):
    OUTPERFORMING = "outperforming"
    INLINE = "inline"
    UNDERPERFORMING = "underperforming"


@dataclass
class Config:
    """Main configuration for StockProb engine"""
    
    # Monte Carlo settings
    MONTE_CARLO_PATHS: int = 10000
    DEFAULT_WINDOW_DAYS: int = 30
    MAX_WINDOW_DAYS: int = 252
    
    # Data sources
    YFINANCE_ENABLED: bool = True
    POLYGON_API_KEY: Optional[str] = None
    NEWSAPI_KEY: Optional[str] = None
    FRED_API_KEY: Optional[str] = None
    
    # Cache settings
    CACHE_ENABLED: bool = True
    CACHE_TTL_SECONDS: int = 3600  # 1 hour
    
    # Sentiment analysis
    SENTIMENT_LOOKBACK_DAYS: int = 30
    INSIDER_LOOKBACK_DAYS: int = 90
    
    # Volatility windows (in days)
    VOL_WINDOWS: List[int] = None
    
    # Rolling window periods
    ROLLING_WINDOWS: List[int] = None
    
    # GARCH settings
    GARCH_MIN_SESSIONS: int = 500
    
    # Confidence score adjustments
    CONFIDENCE_PENALTIES: Dict = None
    CONFIDENCE_BONUSES: Dict = None
    
    # Macro regime thresholds
    PMI_EXPANSION_THRESHOLD: float = 50.0
    CPI_STAGFLATION_THRESHOLD: float = 4.0
    YIELD_CURVE_INVERSION_MONTHS: int = 6
    
    def __post_init__(self):
        if self.VOL_WINDOWS is None:
            self.VOL_WINDOWS = [10, 21, 63, 252]
        if self.ROLLING_WINDOWS is None:
            self.ROLLING_WINDOWS = [10, 21, 63, 252]
        if self.CONFIDENCE_PENALTIES is None:
            self.CONFIDENCE_PENALTIES = {
                "insufficient_history": -20,
                "earnings_in_window": -10,
                "vol_crisis": -15,
                "macro_recession": -10,
                "negative_sentiment": -5,
            }
        if self.CONFIDENCE_BONUSES is None:
            self.CONFIDENCE_BONUSES = {
                "extensive_history": +10,
                "earnings_beat_streak": +10,
                "sector_outperforming": +5,
                "insider_buying": +5,
            }


# Default configuration instance
DEFAULT_CONFIG = Config()

# Sector ETF mapping (GICS sectors)
SECTOR_ETF_MAP = {
    "Technology": "XLK",
    "Health Care": "XLV",
    "Financials": "XLF",
    "Consumer Discretionary": "XLY",
    "Consumer Staples": "XLP",
    "Energy": "XLE",
    "Materials": "XLB",
    "Industrials": "XLI",
    "Utilities": "XLU",
    "Real Estate": "XLRE",
    "Communication Services": "XLC",
}

# FRED economic data series IDs
FRED_SERIES = {
    "fed_funds_rate": "FEDFUNDS",
    "treasury_10y": "DGS10",
    "treasury_2y": "DGS2",
    "cpi_yoy": "CPIAUCSL",
    "unemployment_rate": "UNRATE",
    "ism_manufacturing": "MANEMP",  # Alternative to PMI
}

# API endpoints
POLYGON_BASE_URL = "https://api.polygon.io/v2"
NEWSAPI_BASE_URL = "https://newsapi.org/v2"
FRED_BASE_URL = "https://api.stlouisfed.org/fred"
