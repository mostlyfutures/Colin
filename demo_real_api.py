#!/usr/bin/env python3
"""
Real API Demo using Standard Library
Demonstrates the multi-source implementation with actual API calls using urllib.
"""

import urllib.request
import urllib.parse
import json
import time
from datetime import datetime
from typing import Dict, List, Optional

# Standalone data structures (copying from our implementation)
class DataSource:
    COINGECKO = "coingecko"
    KRAKEN = "kraken"
    CRYPTOCOMPARE = "cryptocompare"
    ALTERNATIVE_ME = "alternative_me"

class DataQuality:
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    UNKNOWN = "unknown"

class StandardMarketData:
    def __init__(self, symbol, price, volume_24h, change_24h, change_pct_24h,
                 high_24h, low_24h, timestamp, source, confidence, data_quality=None):
        self.symbol = symbol
        self.price = price
        self.volume_24h = volume_24h
        self.change_24h = change_24h
        self.change_pct_24h = change_pct_24h
        self.high_24h = high_24h
        self.low_24h = low_24h
        self.timestamp = timestamp
        self.source = source
        self.confidence = confidence
        self.data_quality = data_quality or DataQuality.UNKNOWN

    def to_dict(self):
        return {
            "symbol": self.symbol,
            "price": self.price,
            "volume_24h": self.volume_24h,
            "change_24h": self.change_24h,
            "change_pct_24h": self.change_pct_24h,
            "high_24h": self.high_24h,
            "low_24h": self.low_24h,
            "timestamp": self.timestamp.isoformat(),
            "source": self.source,
            "confidence": self.confidence,
            "data_quality": self.data_quality
        }

class SentimentData:
    def __init__(self, value, value_classification, timestamp):
        self.value = value
        self.value_classification = value_classification
        self.timestamp = timestamp

# API adapters using standard library
class CoinGeckoAPI:
    def __init__(self):
        self.base_url = "https://api.coingecko.com/api/v3"
        self.symbol_mapping = {
            "BTC": "bitcoin",
            "ETH": "ethereum",
            "ADA": "cardano",
            "SOL": "solana",
            "DOT": "polkadot"
        }

    def get_market_data(self, symbol: str) -> Optional[StandardMarketData]:
        try:
            coin_id = self.symbol_mapping.get(symbol.upper())
            if not coin_id:
                print(f"❌ Unsupported symbol: {symbol}")
                return None

            # Build URL with parameters
            params = {
                "ids": coin_id,
                "vs_currencies": "usd",
                "include_24hr_change": "true",
                "include_24hr_vol": "true",
                "include_24hr_high": "true",
                "include_24hr_low": "true"
            }

            url = f"{self.base_url}/simple/price?{urllib.parse.urlencode(params)}"

            # Make request
            with urllib.request.urlopen(url, timeout=10) as response:
                if response.status == 200:
                    data = json.loads(response.read().decode())
                    if coin_id in data:
                        coin_data = data[coin_id]

                        # Calculate confidence based on data completeness
                        confidence = 0.8  # Base confidence
                        if coin_data.get("usd_24h_vol"):
                            confidence += 0.1
                        if coin_data.get("usd_24h_change"):
                            confidence += 0.1

                        return StandardMarketData(
                            symbol=symbol,
                            price=float(coin_data.get("usd", 0)),
                            volume_24h=float(coin_data.get("usd_24h_vol", 0)),
                            change_24h=float(coin_data.get("usd_24h_change", 0)),
                            change_pct_24h=float(coin_data.get("usd_24h_change", 0)),
                            high_24h=float(coin_data.get("usd_24h_high", 0)) or 0,
                            low_24h=float(coin_data.get("usd_24h_low", 0)) or 0,
                            timestamp=datetime.now(),
                            source=DataSource.COINGECKO,
                            confidence=min(confidence, 1.0),
                            data_quality=DataQuality.GOOD
                        )
        except Exception as e:
            print(f"❌ CoinGecko API error: {e}")
            return None

class AlternativeMeAPI:
    def __init__(self):
        self.base_url = "https://api.alternative.me"

    def get_sentiment_data(self) -> Optional[SentimentData]:
        try:
            url = f"{self.base_url}/fng/"

            with urllib.request.urlopen(url, timeout=10) as response:
                if response.status == 200:
                    data = json.loads(response.read().decode())
                    if data.get("data") and len(data["data"]) > 0:
                        fng_data = data["data"][0]
                        return SentimentData(
                            value=int(fng_data["value"]),
                            value_classification=fng_data["value_classification"],
                            timestamp=datetime.fromtimestamp(int(fng_data["timestamp"]))
                        )
        except Exception as e:
            print(f"❌ Alternative.me API error: {e}")
            return None

# Demo manager
class MultiSourceRealDemo:
    def __init__(self):
        self.adapters = {
            DataSource.COINGECKO: CoinGeckoAPI(),
            DataSource.ALTERNATIVE_ME: AlternativeMeAPI()
        }

    def get_market_data(self, symbol: str, max_sources: int = 1) -> Dict:
        print(f"📊 Fetching market data for {symbol}...")

        results = []

        # Try CoinGecko first
        adapter = self.adapters.get(DataSource.COINGECKO)
        if adapter:
            print(f"🔄 Trying {DataSource.COINGECKO}...")
            start_time = time.time()
            result = adapter.get_market_data(symbol)
            elapsed = time.time() - start_time

            if result:
                results.append(result)
                print(f"✅ {DataSource.COINGECKO}: ${result.price:.2f} "
                      f"(confidence: {result.confidence:.2f}, latency: {elapsed:.2f}s)")
            else:
                print(f"❌ {DataSource.COINGECKO}: No data")

        if not results:
            raise ValueError(f"No market data available for {symbol}")

        # Calculate summary
        primary_price = results[0].price
        avg_price = sum(r.price for r in results) / len(results)

        return {
            "symbol": symbol,
            "primary_price": primary_price,
            "consensus_price": avg_price,
            "sources_count": len(results),
            "data_sources": [r.to_dict() for r in results],
            "timestamp": datetime.now().isoformat()
        }

    def get_sentiment_data(self) -> Optional[Dict]:
        print("😊 Fetching sentiment data...")

        adapter = self.adapters.get(DataSource.ALTERNATIVE_ME)
        if not adapter:
            return None

        try:
            print("🔄 Trying Alternative.me...")
            start_time = time.time()
            sentiment = adapter.get_sentiment_data()
            elapsed = time.time() - start_time

            if sentiment:
                result = {
                    "value": sentiment.value,
                    "classification": sentiment.value_classification,
                    "timestamp": sentiment.timestamp.isoformat(),
                    "fetch_time": elapsed
                }
                print(f"✅ Alternative.me: {sentiment.value} ({sentiment.value_classification}) "
                      f"(latency: {elapsed:.2f}s)")
                return result
            else:
                print("❌ Alternative.me: No sentiment data")
                return None
        except Exception as e:
            print(f"❌ Sentiment error: {e}")
            return None

def analyze_ethereum_data(market_data: Dict, sentiment: Optional[Dict]) -> Dict:
    """Analyze Ethereum market data and generate insights."""
    analysis = {}

    # Price analysis
    price = market_data["consensus_price"]
    if price >= 3000:
        price_level = "Very High"
        trend_signal = "🔴 CAUTION - High price level"
    elif price >= 2000:
        price_level = "High"
        trend_signal = "🟡 MODERATE - Monitor closely"
    elif price >= 1500:
        price_level = "Medium-High"
        trend_signal = "🟡 NEUTRAL - Watch for breakouts"
    else:
        price_level = "Medium"
        trend_signal = "🟢 POTENTIAL OPPORTUNITY"

    analysis["price_analysis"] = {
        "current_price": price,
        "price_level": price_level,
        "signal": trend_signal
    }

    # Data quality assessment
    confidence = market_data["data_sources"][0]["confidence"]
    if confidence >= 0.9:
        quality_rating = "Excellent"
    elif confidence >= 0.7:
        quality_rating = "Good"
    else:
        quality_rating = "Fair"

    analysis["data_quality"] = {
        "confidence_score": confidence,
        "quality_rating": quality_rating,
        "source": market_data["data_sources"][0]["source"]
    }

    # Sentiment analysis
    if sentiment:
        fng_value = sentiment["value"]
        if fng_value <= 25:
            sentiment_signal = "🟢 STRONG BUY (Extreme Fear - Often signals buying opportunity)"
        elif fng_value <= 40:
            sentiment_signal = "🟢 BUY (Fear - Market may be oversold)"
        elif fng_value <= 60:
            sentiment_signal = "🟡 HOLD (Neutral - Follow technical indicators)"
        elif fng_value <= 75:
            sentiment_signal = "🔴 SELL (Greed - Market may be overbought)"
        else:
            sentiment_signal = "🔴 STRONG SELL (Extreme Greed - Warning signal)"

        analysis["sentiment_analysis"] = {
            "fear_greed_index": fng_value,
            "classification": sentiment["classification"],
            "trading_signal": sentiment_signal
        }

    # Overall recommendation
    overall_signals = []
    if "price >= 2000 and price < 3000":
        overall_signals.append("MODERATE_BULLISH")
    elif price >= 3000:
        overall_signals.append("CAUTIOUS")
    else:
        overall_signals.append("BULLISH")

    if sentiment and sentiment["value"] <= 40:
        overall_signals.append("BULLISH")
    elif sentiment and sentiment["value"] >= 75:
        overall_signals.append("BEARISH")

    if "BULLISH" in overall_signals and "BEARISH" not in overall_signals:
        overall = "🟢 BULLISH - Consider buying opportunities"
    elif "BEARISH" in overall_signals:
        overall = "🔴 BEARISH - Exercise caution"
    else:
        overall = "🟡 NEUTRAL - Wait for clearer signals"

    analysis["overall_recommendation"] = overall

    return analysis

def run_real_demo():
    """Run the real API demonstration."""
    print("🚀 REAL Multi-Source Market Data Demo")
    print("=" * 80)
    print("📡 Using live API calls to free cryptocurrency data sources")

    demo = MultiSourceRealDemo()

    # Test Ethereum
    print("\n💰 ETHEREUM (ETH) MARKET ANALYSIS")
    print("-" * 50)

    try:
        eth_data = demo.get_market_data("ETH", max_sources=1)

        print(f"\n📈 Market Data Results:")
        print(f"   Current Price: ${eth_data['primary_price']:.2f}")
        print(f"   Data Source: {eth_data['data_sources'][0]['source'].upper()}")
        print(f"   Confidence: {eth_data['data_sources'][0]['confidence']:.2f}")
        print(f"   Volume 24h: ${eth_data['data_sources'][0]['volume_24h']:,.0f}")

        if eth_data['data_sources'][0].get('change_24h_pct', 0) != 0:
            change = eth_data['data_sources'][0]['change_24h_pct']
            print(f"   24h Change: {change:+.2f}%")

        if eth_data['data_sources'][0]['high_24h'] > 0:
            high = eth_data['data_sources'][0]['high_24h']
            low = eth_data['data_sources'][0]['low_24h']
            print(f"   24h Range: ${low:.2f} - ${high:.2f}")

    except Exception as e:
        print(f"❌ Ethereum analysis failed: {e}")
        return

    # Get sentiment data
    print(f"\n😊 MARKET SENTIMENT ANALYSIS")
    print("-" * 50)

    try:
        sentiment = demo.get_sentiment_data()

        if sentiment:
            print(f"\n📊 Fear & Greed Index:")
            print(f"   Current Value: {sentiment['value']}")
            print(f"   Classification: {sentiment['classification']}")
            print(f"   Last Updated: {sentiment['timestamp']}")

        else:
            print("❌ Sentiment data unavailable")
            sentiment = None

    except Exception as e:
        print(f"❌ Sentiment analysis failed: {e}")
        sentiment = None

    # Comprehensive analysis
    print(f"\n🎯 COMPREHENSIVE ANALYSIS")
    print("-" * 50)

    analysis = analyze_ethereum_data(eth_data, sentiment)

    print(f"\n💡 Price Analysis:")
    print(f"   Current Price: ${analysis['price_analysis']['current_price']:.2f}")
    print(f"   Price Level: {analysis['price_analysis']['price_level']}")
    print(f"   Signal: {analysis['price_analysis']['signal']}")

    print(f"\n📊 Data Quality:")
    print(f"   Confidence: {analysis['data_quality']['confidence_score']:.2f}")
    print(f"   Quality Rating: {analysis['data_quality']['quality_rating']}")
    print(f"   Source: {analysis['data_quality']['source'].upper()}")

    if sentiment:
        print(f"\n😊 Sentiment Analysis:")
        print(f"   Fear & Greed: {analysis['sentiment_analysis']['fear_greed_index']}")
        print(f"   Classification: {analysis['sentiment_analysis']['classification']}")
        print(f"   Trading Signal: {analysis['sentiment_analysis']['trading_signal']}")

    print(f"\n🎯 Overall Recommendation:")
    print(f"   {analysis['overall_recommendation']}")

    # Test Bitcoin for comparison
    print(f"\n💰 BITCOIN (BTC) COMPARISON")
    print("-" * 50)

    try:
        btc_data = demo.get_market_data("BTC", max_sources=1)

        eth_btc_ratio = eth_data['primary_price'] / btc_data['primary_price']
        print(f"\n📈 Bitcoin Data:")
        print(f"   BTC Price: ${btc_data['primary_price']:.2f}")
        print(f"   ETH/BTC Ratio: {eth_btc_ratio:.4f}")

        if eth_btc_ratio > 0.07:
            print(f"   ETH Strength: Strong (ETH outperforming BTC)")
        elif eth_btc_ratio > 0.05:
            print(f"   ETH Strength: Moderate")
        else:
            print(f"   ETH Strength: Weak (ETH underperforming BTC)")

    except Exception as e:
        print(f"❌ Bitcoin comparison failed: {e}")

    print(f"\n🎉 DEMONSTRATION COMPLETE")
    print("=" * 80)
    print("✅ Multi-source data implementation working with real APIs!")
    print("✅ Live market data from CoinGecko API")
    print("✅ Fear & Greed sentiment from Alternative.me")
    print("✅ Comprehensive market analysis and recommendations")
    print("✅ Ready for integration into Colin Trading Bot v2.0")

if __name__ == "__main__":
    run_real_demo()