"""
StockProb Multi-Agent Orchestration Engine
Agent 4: Sentiment Scanner - Quantified news + SEC filing sentiment
"""

import numpy as np
from datetime import date, timedelta, datetime
from typing import Dict, List, Optional, Tuple

from ..core.models import SentimentScannerOutput, SentimentTrend, InsiderDirection
from ..data.sources import get_data_source, DataSource
from ..utils import log


class SentimentScanner:
    """
    Agent 4: Sentiment Scanner
    
    Role: Quantified news + SEC filing sentiment (NOT qualitative).
    
    Responsibilities:
    - Pull last 30 days of news headlines
    - Score each headline: +1 (positive), 0 (neutral), -1 (negative)
    - Compute net sentiment score = (pos - neg) / total
    - Pull last 10-K and 10-Q filing dates; flag recency
    - Detect: insider buying/selling (SEC Form 4, last 90d)
    - Compute short interest ratio (days to cover) if available
    """
    
    def __init__(self, data_source: Optional[DataSource] = None):
        """
        Initialize Sentiment Scanner
        
        Args:
            data_source: Optional custom data source
        """
        self.data_source = data_source or get_data_source()
        self.sentiment_lookback_days = 30
        self.insider_lookback_days = 90
        log.info("Sentiment Scanner agent initialized")
    
    def fetch_news_headlines(
        self, 
        ticker: str, 
        as_of_date: date,
        lookback_days: int = 30
    ) -> List[Dict]:
        """
        Fetch news headlines for the ticker
        
        In production, this would use NewsAPI or similar service.
        For now, generates synthetic headlines with sentiment scores.
        
        Args:
            ticker: Stock ticker symbol
            as_of_date: Analysis anchor date
            lookback_days: Days to look back
            
        Returns:
            List of headline dictionaries with sentiment scores
        """
        # Generate synthetic headlines
        # In production: call newsapi.org or similar
        np.random.seed(hash(ticker) % 2**32)
        
        n_headlines = np.random.randint(5, 30)
        headlines = []
        
        positive_phrases = [
            "beats earnings expectations",
            "announces new product launch",
            "receives analyst upgrade",
            "reports strong revenue growth",
            "expands into new market",
        ]
        
        negative_phrases = [
            "misses earnings estimates",
            "faces regulatory scrutiny",
            "receives analyst downgrade",
            "warns on future guidance",
            "experiences supply chain issues",
        ]
        
        neutral_phrases = [
            "holds annual shareholder meeting",
            "announces dividend payment",
            "files routine SEC document",
            "trades sideways amid market uncertainty",
            "maintains market position",
        ]
        
        start_date = as_of_date - timedelta(days=lookback_days)
        
        for i in range(n_headlines):
            # Random date in range
            days_ago = np.random.randint(0, lookback_days)
            pub_date = as_of_date - timedelta(days=days_ago)
            
            # Determine sentiment
            sentiment_roll = np.random.random()
            if sentiment_roll < 0.35:
                # Positive
                phrase = np.random.choice(positive_phrases)
                score = np.random.uniform(0.3, 1.0)
            elif sentiment_roll < 0.65:
                # Negative
                phrase = np.random.choice(negative_phrases)
                score = np.random.uniform(-1.0, -0.3)
            else:
                # Neutral
                phrase = np.random.choice(neutral_phrases)
                score = np.random.uniform(-0.2, 0.2)
            
            headlines.append({
                "title": f"{ticker} {phrase}",
                "source": np.random.choice(["Reuters", "Bloomberg", "CNBC", "WSJ"]),
                "published_at": datetime.combine(pub_date, datetime.min.time()),
                "sentiment_score": float(score),
                "url": None,
            })
        
        # Sort by date
        headlines.sort(key=lambda x: x['published_at'], reverse=True)
        
        log.info(f"Generated {len(headlines)} news headlines for {ticker}")
        return headlines
    
    def compute_net_sentiment(
        self, 
        headlines: List[Dict]
    ) -> Tuple[float, SentimentTrend]:
        """
        Compute net sentiment score from headlines
        
        Net sentiment = (positive - negative) / total
        
        Args:
            headlines: List of headline dictionaries
            
        Returns:
            Tuple of (net_sentiment_score, sentiment_trend)
        """
        if not headlines:
            return 0.0, SentimentTrend.FLAT
        
        scores = [h['sentiment_score'] for h in headlines]
        
        # Net sentiment: average of all scores
        net_sentiment = float(np.mean(scores))
        
        # Determine trend by comparing recent vs older headlines
        n = len(headlines)
        if n >= 6:
            recent_avg = np.mean(scores[:n//2])
            older_avg = np.mean(scores[n//2:])
            
            if recent_avg > older_avg + 0.1:
                trend = SentimentTrend.IMPROVING
            elif recent_avg < older_avg - 0.1:
                trend = SentimentTrend.DETERIORATING
            else:
                trend = SentimentTrend.FLAT
        else:
            trend = SentimentTrend.FLAT
        
        return net_sentiment, trend
    
    def fetch_sec_filings(
        self, 
        ticker: str, 
        as_of_date: date
    ) -> Dict[str, Optional[date]]:
        """
        Fetch latest SEC filing dates (10-K, 10-Q)
        
        In production, this would use SEC EDGAR API.
        
        Args:
            ticker: Stock ticker symbol
            as_of_date: Analysis anchor date
            
        Returns:
            Dict with filing dates
        """
        # Generate synthetic filing dates based on typical patterns
        # Most companies file 10-K within 60-90 days after fiscal year end
        # 10-Q within 40 days after quarter end
        
        np.random.seed(hash(ticker + "_filings") % 2**32)
        
        # Approximate last 10-K (within last 12 months)
        days_since_10k = np.random.randint(30, 300)
        last_10k = as_of_date - timedelta(days=days_since_10k)
        
        # Approximate last 10-Q (within last 3 months)
        days_since_10q = np.random.randint(10, 80)
        last_10q = as_of_date - timedelta(days=days_since_10q)
        
        return {
            "last_10k_date": last_10k.isoformat(),
            "last_10q_date": last_10q.isoformat(),
        }
    
    def analyze_insider_trading(
        self, 
        ticker: str, 
        as_of_date: date,
        lookback_days: int = 90
    ) -> Tuple[InsiderDirection, List[Dict]]:
        """
        Analyze insider trading activity from SEC Form 4
        
        Args:
            ticker: Stock ticker symbol
            as_of_date: Analysis anchor date
            lookback_days: Days to look back
            
        Returns:
            Tuple of (net_direction, list_of_trades)
        """
        # Fetch insider trades from data source
        trades = self.data_source.fetch_insider_trades(ticker)
        
        if not trades:
            return InsiderDirection.NEUTRAL, []
        
        # Filter to lookback period
        cutoff_date = as_of_date - timedelta(days=lookback_days)
        recent_trades = [
            t for t in trades 
            if t.get('transaction_date') and t['transaction_date'] >= cutoff_date
        ]
        
        if not recent_trades:
            return InsiderDirection.NEUTRAL, []
        
        # Calculate net buying/selling
        buys = 0
        sells = 0
        buy_value = 0.0
        sell_value = 0.0
        
        for trade in recent_trades:
            tx_type = trade.get('transaction_type', '')
            value = trade.get('total_value', trade.get('shares', 0) * trade.get('price_per_share', 0))
            
            if tx_type in ['P', 'Purchase', 'A']:  # Purchase or Award
                buys += 1
                buy_value += value
            elif tx_type in ['S', 'Sale', 'D']:  # Sale or Disposal
                sells += 1
                sell_value += value
        
        # Determine net direction
        if buy_value > sell_value * 1.5:  # Buying must be significantly higher
            direction = InsiderDirection.BUYING
        elif sell_value > buy_value * 1.5:
            direction = InsiderDirection.SELLING
        else:
            direction = InsiderDirection.NEUTRAL
        
        log.info(f"Insider trading: {buys} buys, {sells} sells -> {direction.value}")
        
        return direction, recent_trades
    
    def fetch_short_interest(self, ticker: str) -> Optional[float]:
        """
        Fetch short interest ratio (days to cover)
        
        In production, this would use exchange data or financial APIs.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Short interest ratio or None
        """
        # Generate synthetic short interest
        # Typical range: 1-10 days to cover
        # High short interest: > 10 days
        
        np.random.seed(hash(ticker + "_short") % 2**32)
        
        # Most stocks have moderate short interest
        short_ratio = np.random.exponential(3.0) + 1.0
        
        # Cap at reasonable level
        short_ratio = min(short_ratio, 25.0)
        
        return float(short_ratio)
    
    def analyze(
        self, 
        ticker: str, 
        as_of_date: date
    ) -> SentimentScannerOutput:
        """
        Main analysis method - executes all Sentiment Scanner tasks
        
        Args:
            ticker: Stock ticker symbol
            as_of_date: Analysis anchor date
            
        Returns:
            SentimentScannerOutput with all computed metrics
        """
        log.info(f"Starting Sentiment Scanner analysis for {ticker} as of {as_of_date}")
        
        # Step 1: Fetch and analyze news headlines
        headlines = self.fetch_news_headlines(ticker, as_of_date, self.sentiment_lookback_days)
        headline_count = len(headlines)
        
        # Step 2: Compute net sentiment
        if headlines:
            net_sentiment, sentiment_trend = self.compute_net_sentiment(headlines)
        else:
            net_sentiment = 0.0
            sentiment_trend = SentimentTrend.FLAT
        
        # Step 3: Fetch SEC filing dates
        filings = self.fetch_sec_filings(ticker, as_of_date)
        
        # Step 4: Analyze insider trading
        insider_direction, insider_trades = self.analyze_insider_trading(
            ticker, as_of_date, self.insider_lookback_days
        )
        
        # Step 5: Fetch short interest
        short_ratio = self.fetch_short_interest(ticker)
        
        log.info(
            f"Sentiment Scanner complete: score={net_sentiment:.2f}, "
            f"trend={sentiment_trend.value}, insiders={insider_direction.value}"
        )
        
        return SentimentScannerOutput(
            headline_count=headline_count,
            net_sentiment_score=net_sentiment,
            sentiment_trend=sentiment_trend,
            last_10k_date=filings.get('last_10k_date'),
            last_10q_date=filings.get('last_10q_date'),
            insider_net_direction=insider_direction,
            short_interest_ratio=short_ratio,
        )
    
    def _create_empty_output(self) -> SentimentScannerOutput:
        """Create empty output when no data is available"""
        return SentimentScannerOutput(
            headline_count=0,
            net_sentiment_score=0.0,
            sentiment_trend=SentimentTrend.FLAT,
            last_10k_date=None,
            last_10q_date=None,
            insider_net_direction=InsiderDirection.NEUTRAL,
            short_interest_ratio=None,
        )


# Convenience function
def analyze_sentiment(
    ticker: str, 
    as_of_date: date,
    data_source: Optional[DataSource] = None
) -> SentimentScannerOutput:
    """
    Convenience function to run Sentiment Scanner analysis
    
    Args:
        ticker: Stock ticker symbol
        as_of_date: Analysis anchor date
        data_source: Optional custom data source
        
    Returns:
        SentimentScannerOutput
    """
    scanner = SentimentScanner(data_source)
    return scanner.analyze(ticker, as_of_date)
