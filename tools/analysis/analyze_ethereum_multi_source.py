#!/usr/bin/env python3
"""
Ethereum Price Analysis Script with Multi-Source Data
Uses Colin Trading Bot v2.0 multi-source market data system to analyze ETH
"""

import sys
import os
import asyncio
import argparse
from datetime import datetime
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
from loguru import logger

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import v2 components
try:
    from v2.data_sources.market_data_manager import MarketDataManager
    from v2.data_sources.config import get_market_data_config
    from v2.data_sources.models import DataSource
    V2_AVAILABLE = True
except ImportError as e:
    logger.error(f"V2 components not available: {e}")
    V2_AVAILABLE = False
    sys.exit(1)


class MultiSourceEthereumAnalyzer:
    """Ethereum analyzer using multi-source market data."""

    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize analyzer.

        Args:
            config_file: Optional market data configuration file
        """
        self.config = get_market_data_config(config_file)
        self.market_data_manager: Optional[MarketDataManager] = None

    async def initialize(self):
        """Initialize the market data manager."""
        try:
            self.market_data_manager = MarketDataManager(self.config)
            await self.market_data_manager.initialize()
            logger.info("Multi-source Ethereum analyzer initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize analyzer: {e}")
            raise

    async def close(self):
        """Clean up resources."""
        if self.market_data_manager:
            await self.market_data_manager.close()

    async def get_ethereum_market_data(self, max_sources: int = 2) -> Dict[str, Any]:
        """
        Get Ethereum market data from multiple sources.

        Args:
            max_sources: Maximum number of sources to query

        Returns:
            Market data analysis results
        """
        if not self.market_data_manager:
            raise RuntimeError("Analyzer not initialized")

        try:
            # Get market data from multiple sources
            logger.info("Fetching ETH market data from multiple sources...")
            market_summary = await self.market_data_manager.get_market_data(
                symbol="ETH",
                max_sources=max_sources
            )

            # Get sentiment data
            logger.info("Fetching market sentiment data...")
            sentiment = await self.market_data_manager.get_sentiment_data()

            # Analyze the data
            analysis = self._analyze_market_data(market_summary, sentiment)

            return {
                "timestamp": datetime.now().isoformat(),
                "symbol": "ETH",
                "market_summary": {
                    "primary_price": market_summary.primary_price,
                    "consensus_price": market_summary.consensus_price,
                    "price_variance": market_summary.price_variance,
                    "data_quality_score": market_summary.data_quality_score,
                    "sources_count": len(market_summary.price_sources),
                    "available_sources": [s.value for s in market_summary.available_sources],
                    "failed_sources": [s.value for s in market_summary.failed_sources]
                },
                "source_details": [
                    {
                        "source": source.source.value,
                        "price": source.price,
                        "volume_24h": source.volume_24h,
                        "change_24h": source.change_24h,
                        "change_pct_24h": source.change_pct_24h,
                        "confidence": source.confidence,
                        "data_quality": source.data_quality.value,
                        "high_24h": source.high_24h,
                        "low_24h": source.low_24h,
                        "market_cap": source.market_cap,
                        "timestamp": source.timestamp.isoformat()
                    }
                    for source in market_summary.price_sources
                ],
                "sentiment": {
                    "value": sentiment.value if sentiment else None,
                    "classification": sentiment.value_classification if sentiment else None,
                    "timestamp": sentiment.timestamp.isoformat() if sentiment else None
                } if sentiment else None,
                "analysis": analysis,
                "system_stats": self.market_data_manager.get_system_stats()
            }

        except Exception as e:
            logger.error(f"Failed to get Ethereum market data: {e}")
            raise

    def _analyze_market_data(self, market_summary, sentiment) -> Dict[str, Any]:
        """Analyze market data and generate insights."""
        analysis = {}

        # Price analysis
        if market_summary.consensus_price:
            analysis["price_analysis"] = {
                "current_price": market_summary.consensus_price,
                "price_level": self._get_price_level(market_summary.consensus_price),
                "price_trend": self._analyze_price_trend(market_summary.price_sources)
            }

        # Data quality assessment
        analysis["data_quality"] = {
            "overall_score": market_summary.data_quality_score,
            "quality_rating": self._get_quality_rating(market_summary.data_quality_score),
            "source_diversity": len(market_summary.price_sources),
            "data_consistency": self._assess_data_consistency(market_summary)
        }

        # Sentiment analysis
        if sentiment:
            analysis["sentiment_analysis"] = {
                "fear_greed_index": sentiment.value,
                "sentiment_classification": sentiment.value_classification,
                "sentiment_signal": self._get_sentiment_signal(sentiment.value)
            }

        # Trading signals
        analysis["trading_signals"] = self._generate_trading_signals(market_summary, sentiment)

        # Risk assessment
        analysis["risk_assessment"] = self._assess_risk(market_summary, sentiment)

        return analysis

    def _get_price_level(self, price: float) -> str:
        """Categorize price level."""
        if price >= 4000:
            return "Very High"
        elif price >= 3000:
            return "High"
        elif price >= 2000:
            return "Medium-High"
        elif price >= 1500:
            return "Medium"
        elif price >= 1000:
            return "Medium-Low"
        else:
            return "Low"

    def _analyze_price_trend(self, price_sources) -> Dict[str, Any]:
        """Analyze price trend from multiple sources."""
        if not price_sources:
            return {"trend": "Unknown", "confidence": 0.0}

        # Get 24h changes from all sources
        changes = [source.change_pct_24h for source in price_sources if source.change_pct_24h is not None]
        if not changes:
            return {"trend": "Unknown", "confidence": 0.0}

        avg_change = np.mean(changes)
        change_std = np.std(changes) if len(changes) > 1 else 0

        # Determine trend
        if avg_change > 2:
            trend = "Strong Bullish"
        elif avg_change > 0.5:
            trend = "Bullish"
        elif avg_change > -0.5:
            trend = "Neutral"
        elif avg_change > -2:
            trend = "Bearish"
        else:
            trend = "Strong Bearish"

        # Confidence based on consistency
        confidence = max(0, 1 - (change_std / max(abs(avg_change), 0.1)))

        return {
            "trend": trend,
            "avg_change_24h": avg_change,
            "confidence": confidence,
            "consistency": "High" if change_std < 0.5 else "Medium" if change_std < 1.0 else "Low"
        }

    def _get_quality_rating(self, score: float) -> str:
        """Get quality rating from score."""
        if score >= 0.9:
            return "Excellent"
        elif score >= 0.7:
            return "Good"
        elif score >= 0.5:
            return "Fair"
        else:
            return "Poor"

    def _assess_data_consistency(self, market_summary) -> Dict[str, Any]:
        """Assess consistency of data across sources."""
        if len(market_summary.price_sources) < 2:
            return {"consistency": "Unknown", "price_variance": 0}

        prices = [source.price for source in market_summary.price_sources]
        price_variance = np.var(prices)
        price_std = np.std(prices)
        avg_price = np.mean(prices)

        # Coefficient of variation
        cv = (price_std / avg_price) * 100 if avg_price > 0 else 0

        if cv < 0.5:
            consistency = "Excellent"
        elif cv < 1.0:
            consistency = "Good"
        elif cv < 2.0:
            consistency = "Fair"
        else:
            consistency = "Poor"

        return {
            "consistency": consistency,
            "price_variance": price_variance,
            "price_std": price_std,
            "coefficient_of_variation": cv
        }

    def _get_sentiment_signal(self, fear_greed_value: int) -> str:
        """Get trading signal from fear & greed index."""
        if fear_greed_value <= 25:
            return "Strong Buy (Extreme Fear)"
        elif fear_greed_value <= 40:
            return "Buy (Fear)"
        elif fear_greed_value <= 60:
            return "Hold (Neutral)"
        elif fear_greed_value <= 75:
            return "Sell (Greed)"
        else:
            return "Strong Sell (Extreme Greed)"

    def _generate_trading_signals(self, market_summary, sentiment) -> Dict[str, Any]:
        """Generate comprehensive trading signals."""
        signals = []

        # Price-based signals
        if market_summary.consensus_price:
            # Get 24h changes
            changes = [s.change_pct_24h for s in market_summary.price_sources if s.change_pct_24h is not None]
            if changes:
                avg_change = np.mean(changes)

                if avg_change > 3:
                    signals.append({
                        "type": "momentum",
                        "signal": "BULLISH",
                        "strength": "STRONG",
                        "reasoning": f"Strong upward momentum: {avg_change:.2f}% in 24h"
                    })
                elif avg_change > 1:
                    signals.append({
                        "type": "momentum",
                        "signal": "BULLISH",
                        "strength": "MODERATE",
                        "reasoning": f"Positive momentum: {avg_change:.2f}% in 24h"
                    })
                elif avg_change < -3:
                    signals.append({
                        "type": "momentum",
                        "signal": "BEARISH",
                        "strength": "STRONG",
                        "reasoning": f"Strong downward momentum: {avg_change:.2f}% in 24h"
                    })

        # Sentiment-based signals
        if sentiment:
            if sentiment.value <= 25:
                signals.append({
                    "type": "sentiment",
                    "signal": "BULLISH",
                    "strength": "MODERATE",
                    "reasoning": f"Extreme Fear ({sentiment.value}) often signals buying opportunity"
                })
            elif sentiment.value >= 75:
                signals.append({
                    "type": "sentiment",
                    "signal": "BEARISH",
                    "strength": "MODERATE",
                    "reasoning": f"Extreme Greed ({sentiment.value}) often signals market top"
                })

        # Data quality signals
        if market_summary.data_quality_score >= 0.8:
            signals.append({
                "type": "data_quality",
                "signal": "HIGH_CONFIDCE",
                "strength": "WEAK",
                "reasoning": f"High data quality score: {market_summary.data_quality_score:.2f}"
            })

        return {
            "signals": signals,
            "overall_signal": self._get_overall_signal(signals),
            "signal_count": len(signals)
        }

    def _get_overall_signal(self, signals: list) -> str:
        """Get overall trading signal from individual signals."""
        if not signals:
            return "NEUTRAL"

        bullish_count = sum(1 for s in signals if s["signal"] == "BULLISH")
        bearish_count = sum(1 for s in signals if s["signal"] == "BEARISH")

        if bullish_count > bearish_count:
            return "BULLISH"
        elif bearish_count > bullish_count:
            return "BEARISH"
        else:
            return "NEUTRAL"

    def _assess_risk(self, market_summary, sentiment) -> Dict[str, Any]:
        """Assess risk levels."""
        risk_factors = []

        # Price volatility risk
        if market_summary.price_variance and market_summary.price_variance > 1000:
            risk_factors.append({
                "type": "price_volatility",
                "level": "HIGH",
                "description": "High price variance across data sources"
            })

        # Data quality risk
        if market_summary.data_quality_score < 0.6:
            risk_factors.append({
                "type": "data_quality",
                "level": "MEDIUM",
                "description": "Low data quality score may indicate unreliable data"
            })

        # Sentiment risk
        if sentiment and (sentiment.value <= 10 or sentiment.value >= 90):
            risk_factors.append({
                "type": "sentiment_extreme",
                "level": "HIGH",
                "description": "Extreme market sentiment may precede sharp reversals"
            })

        # Source concentration risk
        if len(market_summary.available_sources) < 2:
            risk_factors.append({
                "type": "source_concentration",
                "level": "MEDIUM",
                "description": "Limited data source diversity"
            })

        overall_risk = "LOW"
        if any(f["level"] == "HIGH" for f in risk_factors):
            overall_risk = "HIGH"
        elif any(f["level"] == "MEDIUM" for f in risk_factors):
            overall_risk = "MEDIUM"

        return {
            "overall_risk": overall_risk,
            "risk_factors": risk_factors,
            "risk_count": len(risk_factors)
        }


async def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Analyze Ethereum with multi-source data")
    parser.add_argument("--config", help="Market data configuration file")
    parser.add_argument("--sources", type=int, default=2, help="Maximum sources to query")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    # Configure logging
    log_level = "DEBUG" if args.verbose else "INFO"
    logger.remove()
    logger.add(sys.stderr, level=log_level)

    if not V2_AVAILABLE:
        logger.error("V2 components not available. Please check your installation.")
        return

    analyzer = MultiSourceEthereumAnalyzer(args.config)

    try:
        await analyzer.initialize()

        logger.info("🔍 Starting Ethereum multi-source analysis...")
        results = await analyzer.get_ethereum_market_data(max_sources=args.sources)

        # Display results
        print("\n" + "="*80)
        print("📊 ETHEREUM MULTI-SOURCE MARKET ANALYSIS")
        print("="*80)

        # Market summary
        print(f"\n💰 Price Information:")
        print(f"   Consensus Price: ${results['market_summary']['consensus_price']:.2f}")
        print(f"   Primary Source: ${results['market_summary']['primary_price']:.2f}")
        print(f"   Data Quality Score: {results['market_summary']['data_quality_score']:.2f}")
        print(f"   Sources Used: {results['market_summary']['sources_count']}")

        # Source details
        print(f"\n📡 Data Sources:")
        for source in results['source_details']:
            print(f"   {source['source'].upper()}: ${source['price']:.2f} "
                  f"({source['change_pct_24h']:+.2f}%) "
                  f"[Quality: {source['data_quality']}, Confidence: {source['confidence']:.2f}]")

        # Sentiment
        if results['sentiment']:
            print(f"\n😊 Market Sentiment:")
            print(f"   Fear & Greed Index: {results['sentiment']['value']} ({results['sentiment']['classification']})")

        # Analysis
        analysis = results['analysis']
        print(f"\n📈 Analysis:")

        if 'price_analysis' in analysis:
            price_analysis = analysis['price_analysis']
            print(f"   Price Level: {price_analysis['price_level']}")
            print(f"   Trend: {price_analysis['price_trend']['trend']} "
                  f"(Confidence: {price_analysis['price_trend']['confidence']:.2f})")

        if 'sentiment_analysis' in analysis:
            sentiment_analysis = analysis['sentiment_analysis']
            print(f"   Sentiment Signal: {sentiment_analysis['sentiment_signal']}")

        # Trading signals
        print(f"\n🎯 Trading Signals:")
        signals = analysis['trading_signals']
        print(f"   Overall Signal: {signals['overall_signal']}")
        for signal in signals['signals']:
            print(f"   • {signal['type'].title()}: {signal['signal']} "
                  f"({signal['strength'].title()}) - {signal['reasoning']}")

        # Risk assessment
        print(f"\n⚠️  Risk Assessment:")
        risk = analysis['risk_assessment']
        print(f"   Overall Risk: {risk['overall_risk']}")
        if risk['risk_factors']:
            for factor in risk['risk_factors']:
                print(f"   • {factor['type'].title()}: {factor['level']} - {factor['description']}")

        # System stats
        print(f"\n🔧 System Statistics:")
        stats = results['system_stats']
        print(f"   Total Requests: {stats['total_requests']}")
        print(f"   Cache Hit Rate: {stats['cache_hit_rate_percent']:.1f}%")
        print(f"   Available Sources: {len(stats['available_sources'])}")
        print(f"   Cache Size: {stats['cache_size']} items")

        print("\n" + "="*80)
        print(f"✅ Analysis completed at {results['timestamp']}")
        print("="*80)

    except KeyboardInterrupt:
        logger.info("Analysis interrupted by user")
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise
    finally:
        await analyzer.close()


if __name__ == "__main__":
    asyncio.run(main())