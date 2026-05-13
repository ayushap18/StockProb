"""
StockProb Multi-Agent Orchestration Engine
Main package initialization
"""

from .config import (
    Config,
    DEFAULT_CONFIG,
    RiskTolerance,
    VolRegime,
    PriceRegime,
    MacroRegime,
    SectorRelativeStrength,
)

from .core import (
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
)

from .agents import (
    PriceHistorian,
    VolatilityEngine,
)

from .utils import (
    log,
    calculate_log_returns,
    percentile,
    max_drawdown,
)

__version__ = "1.0.0"
__author__ = "StockProb Team"

__all__ = [
    # Config
    "Config",
    "DEFAULT_CONFIG",
    "RiskTolerance",
    "VolRegime",
    "PriceRegime",
    "MacroRegime",
    "SectorRelativeStrength",
    
    # Core models
    "AnalysisRequest",
    "AnalysisResponse",
    "PriceHistorianOutput",
    "VolatilityEngineOutput",
    "EarningsAnalystOutput",
    "SentimentScannerOutput",
    "MacroPullerOutput",
    "MonteCarloOutput",
    "SignalDashboard",
    "AgentStatus",
    
    # Agents
    "PriceHistorian",
    "VolatilityEngine",
    
    # Utils
    "log",
    "calculate_log_returns",
    "percentile",
    "max_drawdown",
]
