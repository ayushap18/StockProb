"""
StockProb CLI Runner - Terminal Interface
==========================================
Quick command-line interface for running StockProb analysis.
"""

import argparse
import json
from datetime import date, timedelta
from typing import Optional

def main():
    parser = argparse.ArgumentParser(
        description="StockProb Quantitative Stock Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_cli.py --ticker AAPL --window 30
  python run_cli.py --ticker TSLA --window 7 --risk high
  python run_cli.py --ticker MSFT --window 90 --real-data
        """
    )
    
    parser.add_argument(
        "--ticker", "-t",
        type=str,
        required=True,
        help="Stock ticker symbol (e.g., AAPL, MSFT, TSLA)"
    )
    
    parser.add_argument(
        "--window", "-w",
        type=int,
        default=30,
        help="Time horizon in days (default: 30)"
    )
    
    parser.add_argument(
        "--date", "-d",
        type=str,
        default=None,
        help="Analysis date in YYYY-MM-DD format (default: today)"
    )
    
    parser.add_argument(
        "--risk", "-r",
        type=str,
        choices=["low", "moderate", "high"],
        default="moderate",
        help="Risk tolerance level (default: moderate)"
    )
    
    parser.add_argument(
        "--real-data",
        action="store_true",
        help="Use real market data instead of mock data"
    )
    
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output raw JSON only"
    )
    
    args = parser.parse_args()
    
    # Import here to avoid issues if package not installed
    try:
        from stockprob.services.orchestrator import run_analysis
        from stockprob.core.models import RiskTolerance
    except ImportError as e:
        print(f"❌ Error: StockProb package not found.")
        print(f"   Make sure you're running this from /workspace directory")
        print(f"   Details: {e}")
        return 1
    
    # Parse date
    if args.date:
        try:
            as_of_date = date.fromisoformat(args.date)
        except ValueError:
            print(f"❌ Invalid date format. Use YYYY-MM-DD")
            return 1
    else:
        as_of_date = date.today()
    
    # Run analysis
    use_mock = not args.real_data
    
    if not args.json:
        print(f"\n📊 StockProb Analysis")
        print(f"   Ticker: {args.ticker.upper()}")
        print(f"   Window: {args.window} days")
        print(f"   Date: {as_of_date}")
        print(f"   Risk: {args.risk}")
        print(f"   Data: {'Mock' if use_mock else 'Real'}")
        print(f"\n⏳ Running 6-agent analysis...")
    
    try:
        result = run_analysis(
            ticker=args.ticker.upper(),
            window_days=args.window,
            as_of_date=as_of_date,
            risk_tolerance=RiskTolerance(args.risk),
            use_mock_data=use_mock
        )
    except Exception as e:
        print(f"\n❌ Analysis failed: {e}")
        return 1
    
    # Output
    if args.json:
        print(json.dumps(result.model_dump(), indent=2, default=str))
    else:
        # Pretty print results
        print(f"\n{'='*60}")
        print(f"RESULTS")
        print(f"{'='*60}")
        
        prob = result.probability_up
        color_code = "\033[92m" if prob >= 0.6 else "\033[93m" if prob >= 0.4 else "\033[91m"
        reset = "\033[0m"
        
        print(f"\n{color_code}📈 Probability UP: {prob:.1%}{reset}")
        print(f"   over {args.window} days from {as_of_date}")
        
        print(f"\n🎯 Confidence Score: {result.confidence_score}/100")
        
        # Signal Dashboard
        signal = result.signal_dashboard
        print(f"\n📡 Signal Dashboard:")
        print(f"   Risk:    {signal.risk.get('vol_regime', 'N/A'):12} (Beta: {signal.risk.get('beta', 0):.2f})")
        print(f"   Trend:   {signal.trend.get('regime', 'N/A'):12} (vs MA200: {signal.trend.get('vs_ma200', 'N/A')})")
        earnings_status = '⚠️ In Window' if signal.earnings.get('in_window', False) else '✅ Clear'
        print(f"   Earnings:{earnings_status:12} ({signal.earnings.get('trend', 'N/A')})")
        print(f"   Sentiment:{signal.sentiment.get('score', 0):+.2f}      (Insiders: {signal.sentiment.get('insiders', 'N/A')})")
        print(f"   Macro:   {signal.macro.get('regime', 'N/A'):12} (Yield Curve: {signal.macro.get('yield_curve', 0):.2f}%)")
        
        # Monte Carlo
        mc = result.monte_carlo
        print(f"\n📈 Return Distribution (10,000 paths):")
        print(f"   P10: {mc.p10_return:7.1%}  |  P25: {mc.p25_return:7.1%}  |  P50: {mc.p50_return:7.1%}  |  P75: {mc.p75_return:7.1%}  |  P90: {mc.p90_return:7.1%}")
        
        print(f"\n⚠️ Risk Metrics:")
        print(f"   Prob Loss > 10%:  {mc.probability_loss_gt_10pct:.1%}")
        print(f"   Prob Gain > 20%:  {mc.probability_gain_gt_20pct:.1%}")
        print(f"   Median Max DD:    {mc.median_max_drawdown:.1%}")
        
        # Risk Flags
        if result.risk_flags:
            print(f"\n🚩 Risk Flags:")
            for flag in result.risk_flags:
                print(f"   ⚠️ {flag}")
        else:
            print(f"\n✅ No significant risk flags")
        
        # Agent Status
        print(f"\n🤖 Agent Status:")
        status_icons = {"ok": "✅", "degraded": "⚠️", "failed": "❌"}
        agent_names = ["Price Historian", "Volatility Engine", "Earnings Analyst", 
                      "Sentiment Scanner", "Macro Puller", "Monte Carlo"]
        agent_attrs = ["PRICE_HISTORIAN", "VOLATILITY_ENGINE", "EARNINGS_ANALYST",
                     "SENTIMENT_SCANNER", "MACRO_PULLER", "MONTE_CARLO"]
        
        for name, attr in zip(agent_names, agent_attrs):
            status = getattr(result.agent_status, attr, "unknown")
            icon = status_icons.get(status, "❓")
            print(f"   {icon} {name:18}: {status}")
        
        print(f"\n{'='*60}")
        print(f"⚠️ {result.disclaimer}")
        print(f"{'='*60}\n")
    
    return 0


if __name__ == "__main__":
    exit(main())
