"""
StockProb Multi-Agent Orchestration Engine
Utility functions for logging, caching, and common operations
"""

import logging
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Callable
from functools import wraps
import hashlib
import json
import os

from loguru import logger


# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

def setup_logger(
    name: str = "stockprob",
    level: str = "INFO",
    log_file: Optional[str] = None,
    rotation: str = "500 MB",
    retention: str = "7 days"
) -> logger:
    """
    Configure structured logging for StockProb
    
    Args:
        name: Logger name
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Path to log file (optional)
        rotation: Log rotation size
        retention: How long to keep old logs
        
    Returns:
        Configured logger instance
    """
    # Remove default handler
    logger.remove()
    
    # Add console handler with structured format
    logger.add(
        sink=lambda msg: print(msg, end=""),
        level=level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        ),
        colorize=True,
    )
    
    # Add file handler if specified
    if log_file:
        logger.add(
            log_file,
            level=level,
            rotation=rotation,
            retention=retention,
            format=(
                "{time:YYYY-MM-DD HH:mm:ss} | "
                "{level: <8} | "
                "{name}:{function}:{line} | "
                "{message}"
            ),
        )
    
    return logger


# Initialize default logger
log = setup_logger()


# ============================================================================
# CACHING UTILITIES
# ============================================================================

class SimpleCache:
    """
    In-memory cache with TTL support
    Thread-safe implementation using basic locking
    """
    
    def __init__(self, ttl_seconds: int = 3600):
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._ttl = ttl_seconds
        self._lock = False  # Simplified lock flag
    
    def _generate_key(self, func_name: str, args: tuple, kwargs: dict) -> str:
        """Generate unique cache key from function arguments"""
        key_data = {
            "func": func_name,
            "args": args,
            "kwargs": kwargs,
        }
        key_string = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache if not expired"""
        if key not in self._cache:
            return None
        
        entry = self._cache[key]
        if datetime.now() > entry["expires_at"]:
            del self._cache[key]
            return None
        
        return entry["value"]
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in cache with TTL"""
        expires_at = datetime.now() + timedelta(seconds=ttl or self._ttl)
        self._cache[key] = {
            "value": value,
            "expires_at": expires_at,
        }
    
    def invalidate(self, key: str) -> bool:
        """Invalidate a specific cache key"""
        if key in self._cache:
            del self._cache[key]
            return True
        return False
    
    def clear(self) -> None:
        """Clear all cached values"""
        self._cache.clear()
    
    def cleanup_expired(self) -> int:
        """Remove all expired entries, return count removed"""
        now = datetime.now()
        expired_keys = [
            k for k, v in self._cache.items()
            if now > v["expires_at"]
        ]
        for key in expired_keys:
            del self._cache[key]
        return len(expired_keys)


# Global cache instance
cache = SimpleCache()


def cached(ttl: Optional[int] = None, key_prefix: str = ""):
    """
    Decorator for caching function results
    
    Args:
        ttl: Time-to-live in seconds (uses default if None)
        key_prefix: Optional prefix for cache keys
    """
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            cache_key = f"{key_prefix}{func.__name__}:{hash(str(args) + str(kwargs))}"
            
            # Try to get from cache
            result = cache.get(cache_key)
            if result is not None:
                log.debug(f"Cache hit for {func.__name__}")
                return result
            
            # Execute function and cache result
            log.debug(f"Cache miss for {func.__name__}, executing...")
            result = func(*args, **kwargs)
            cache.set(cache_key, result, ttl)
            
            return result
        return wrapper
    return decorator


# ============================================================================
# DATE/TIME UTILITIES
# ============================================================================

def parse_date(date_str: str) -> date:
    """Parse date string in various formats"""
    formats = [
        "%Y-%m-%d",
        "%Y/%m/%d",
        "%d-%m-%Y",
        "%d/%m/%Y",
        "%m-%d-%Y",
        "%m/%d/%Y",
    ]
    
    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt).date()
        except ValueError:
            continue
    
    raise ValueError(f"Unable to parse date: {date_str}")


def is_trading_day(d: date) -> bool:
    """
    Check if a date is a trading day (not weekend or major holiday)
    Simplified version - doesn't account for all market holidays
    """
    # Weekend check
    if d.weekday() >= 5:  # Saturday=5, Sunday=6
        return False
    
    # Major US market holidays (simplified, dates vary by year)
    holidays = [
        # New Year's Day
        date(d.year, 1, 1),
        # Martin Luther King Jr. Day (3rd Monday of January)
        # Memorial Day (last Monday of May)
        # Independence Day
        date(d.year, 7, 4),
        # Labor Day (1st Monday of September)
        # Thanksgiving (4th Thursday of November)
        # Christmas
        date(d.year, 12, 25),
    ]
    
    if d in holidays:
        return False
    
    return True


def get_trading_days_between(start: date, end: date) -> List[date]:
    """Get list of trading days between two dates"""
    trading_days = []
    current = start
    
    while current <= end:
        if is_trading_day(current):
            trading_days.append(current)
        current += timedelta(days=1)
    
    return trading_days


def business_days_ago(days: int, from_date: Optional[date] = None) -> date:
    """Calculate date N business days ago"""
    if from_date is None:
        from_date = date.today()
    
    count = 0
    result = from_date
    
    while count < days:
        result -= timedelta(days=1)
        if is_trading_day(result):
            count += 1
    
    return result


# ============================================================================
# STATISTICAL UTILITIES
# ============================================================================

def calculate_log_returns(prices: List[float]) -> List[float]:
    """
    Calculate log returns from price series
    
    Args:
        prices: List of prices in chronological order
        
    Returns:
        List of log returns (length = len(prices) - 1)
    """
    import math
    
    if len(prices) < 2:
        return []
    
    returns = []
    for i in range(1, len(prices)):
        if prices[i-1] > 0 and prices[i] > 0:
            log_ret = math.log(prices[i] / prices[i-1])
            returns.append(log_ret)
    
    return returns


def annualize_volatility(daily_vol: float, trading_days: int = 252) -> float:
    """Convert daily volatility to annualized volatility"""
    import math
    return daily_vol * math.sqrt(trading_days)


def percentile(data: List[float], p: float) -> float:
    """
    Calculate percentile of data
    
    Args:
        data: List of values
        p: Percentile (0-100)
        
    Returns:
        Value at percentile p
    """
    if not data:
        raise ValueError("Cannot calculate percentile of empty list")
    
    sorted_data = sorted(data)
    n = len(sorted_data)
    
    if p <= 0:
        return sorted_data[0]
    if p >= 100:
        return sorted_data[-1]
    
    # Linear interpolation
    k = (n - 1) * (p / 100.0)
    f = int(k)
    c = f + 1
    
    if c >= n:
        return sorted_data[-1]
    
    return sorted_data[f] + (k - f) * (sorted_data[c] - sorted_data[f])


def max_drawdown(returns: List[float]) -> float:
    """
    Calculate maximum drawdown from return series
    
    Args:
        returns: List of returns
        
    Returns:
        Maximum drawdown as negative value
    """
    if not returns:
        return 0.0
    
    # Convert returns to cumulative wealth
    wealth = [1.0]
    for r in returns:
        wealth.append(wealth[-1] * (1 + r))
    
    # Track peak and max drawdown
    peak = wealth[0]
    max_dd = 0.0
    
    for w in wealth:
        if w > peak:
            peak = w
        dd = (w - peak) / peak
        if dd < max_dd:
            max_dd = dd
    
    return max_dd


# ============================================================================
# DATA VALIDATION UTILITIES
# ============================================================================

def validate_ticker(ticker: str) -> bool:
    """Basic ticker validation"""
    if not ticker:
        return False
    
    # Tickers are typically 1-5 uppercase letters, sometimes with dots
    import re
    pattern = r'^[A-Z]{1,5}(\.[A-Z]{1,2})?$'
    return bool(re.match(pattern, ticker.upper()))


def sanitize_input(value: Any, expected_type: type) -> Any:
    """Sanitize and convert input to expected type"""
    if value is None:
        return None
    
    try:
        if expected_type == str:
            return str(value).strip()
        elif expected_type == int:
            return int(float(value))
        elif expected_type == float:
            return float(value)
        elif expected_type == date:
            if isinstance(value, date):
                return value
            return parse_date(str(value))
        else:
            return expected_type(value)
    except (ValueError, TypeError):
        return None


# ============================================================================
# FILE SYSTEM UTILITIES
# ============================================================================

def ensure_directory(path: str) -> str:
    """Ensure directory exists, create if necessary"""
    os.makedirs(path, exist_ok=True)
    return path


def get_project_root() -> str:
    """Get project root directory"""
    import os
    current = os.path.dirname(os.path.abspath(__file__))
    # Navigate up to stockprob directory
    while not current.endswith('stockprob'):
        parent = os.path.dirname(current)
        if parent == current:
            break
        current = parent
    return current


def get_data_directory() -> str:
    """Get data storage directory"""
    return os.path.join(get_project_root(), "data")


# ============================================================================
# JSON SERIALIZATION UTILITIES
# ============================================================================

class EnhancedJSONEncoder(json.JSONEncoder):
    """JSON encoder that handles additional types"""
    
    def default(self, obj):
        if isinstance(obj, date):
            return obj.isoformat()
        if isinstance(obj, datetime):
            return obj.isoformat()
        if hasattr(obj, 'to_dict'):
            return obj.to_dict()
        if hasattr(obj, '__dict__'):
            return obj.__dict__
        return super().default(obj)


def to_json(obj: Any, indent: int = 2) -> str:
    """Convert object to JSON string with enhanced encoding"""
    return json.dumps(obj, cls=EnhancedJSONEncoder, indent=indent)


def from_json(json_str: str) -> Any:
    """Parse JSON string"""
    return json.loads(json_str)
