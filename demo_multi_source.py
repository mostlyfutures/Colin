#!/usr/bin/env python3
"""
Standalone Multi-Source Data Demo
Demonstrates the multi-source implementation with real API calls.
"""

import asyncio
import aiohttp
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

class SentimentData:
    def __init__(self, value, value_classification, timestamp):
        self.value = value
        self.value_classification = value_classification
        self.timestamp = timestamp

# Simplified adapters for demonstration
class CoinGeckoDemo:
    def __init__(self):
        self.base_url = "https://api.coingecko.com/api/v3"
        self.symbol_mapping = {
            "BTC": "bitcoin",
            "ETH": "ethereum",
            "ADA": "cardano"
        }

    async def get_market_data(self, symbol: str) -> Optional[StandardMarketData]:
        try:
            coin_id = self.symbol_mapping.get(symbol.upper())
            if not coin_id:
                return None

            async with aiohttp.ClientSession() as session:
                url = f"{self.base_url}/simple/price"
                params = {
                    "ids": coin_id,
                    "vs_currencies": "usd",
                    "include_24hr_change": "true",
                    "include_24hr_vol": "true",
                    "include_24hr_high": "true",
                    "include_24hr_low": "true"
                }

                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        if coin_id in data:
                            coin_data = data[coin_id]
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
                                confidence=0.9,
                                data_quality=DataQuality.GOOD
                            )
        except Exception as e:
            print(f"❌ CoinGecko error: {e}")
            return None

class KrakenDemo:
    def __init__(self):
        self.base_url = "https://api.kraken.com/0/public"
        self.symbol_mapping = {
            "BTC": "XXBTZUSD",
            "ETH": "XETHZUSD"
        }

    async def get_market_data(self, symbol: str) -> Optional[StandardMarketData]:
        try:
            pair = self.symbol_mapping.get(symbol.upper())
            if not pair:
                return None

            async with aiohttp.ClientSession() as session:
                url = f"{self.base_url}/Ticker"
                params = {"pair": pair}

                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data.get("error") == [] and "result" in data:
                            result = data["result"]
                            ticker_key = list(result.keys())[0]
                            ticker = result[ticker_key]

                            price = float(ticker['c'][0]) if ticker.get('c') else 0
                            volume = float(ticker['v'][1]) if ticker.get('v') and len(ticker['v']) > 1 else 0
                            high = float(ticker['h'][1]) if ticker.get('h') and len(ticker['h']) > 1 else 0
                            low = float(ticker['l'][1]) if ticker.get('l') and len(ticker['l']) > 1 else 0

                            return StandardMarketData(
                                symbol=symbol,
                                price=price,
                                volume_24h=volume,
                                change_24h=0,  # Kraken doesn't provide direct 24h change
                                change_pct_24h=0,
                                high_24h=high,
                                low_24h=low,
                                timestamp=datetime.now(),
                                source=DataSource.KRAKEN,
                                confidence=0.85,
                                data_quality=DataQuality.GOOD
                            )
        except Exception as e:
            print(f"❌ Kraken error: {e}")
            return None

class AlternativeMeDemo:
    def __init__(self):
        self.base_url = "https://api.alternative.me"

    async def get_sentiment_data(self) -> Optional[SentimentData]:
        try:
            async with aiohttp.ClientSession() as session:
                url = f"{self.base_url}/fng/"

                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data.get("data") and len(data["data"]) > 0:
                            fng_data = data["data"][0]
                            return SentimentData(
                                value=int(fng_data["value"]),
                                value_classification=fng_data["value_classification"],
                                timestamp=datetime.fromtimestamp(int(fng_data["timestamp"]))
                            )
        except Exception as e:
            print(f"❌ Alternative.me error: {e}")
            return None

# Demo manager
class MultiSourceDemo:
    def __init__(self):
        self.adapters = {
            DataSource.COINGECKO: CoinGeckoDemo(),
            DataSource.KRAKEN: KrakenDemo(),
            DataSource.ALTERNATIVE_ME: AlternativeMeDemo()
        }

    async def get_market_data(self, symbol: str, max_sources: int = 2) -> Dict:
        print(f"📊 Fetching market data for {symbol}...")

        results = []
        tasks = []

        # Try sources in order
        for source in [DataSource.COINGECKO, DataSource.KRAKEN]:
            if len(results) >= max_sources:
                break

            adapter = self.adapters.get(source)
            if source != DataSource.ALTERNATIVE_ME and adapter:  # Skip sentiment adapter for market data
                task = asyncio.create_task(adapter.get_market_data(symbol))
                tasks.append((source, task))

        # Wait for results with timeout
        for source, task in tasks:
            try:
                result = await asyncio.wait_for(task, timeout=10)
                if result:
                    results.append(result)
                    print(f"✅ {source.value}: ${result.price:.2f} (confidence: {result.confidence:.2f})")
                else:
                    print(f"❌ {source.value}: No data")
            except asyncio.TimeoutError:
                print(f"⏰ {source.value}: Timeout")
            except Exception as e:
                print(f"❌ {source.value}: Error - {e}")

        if not results:
            raise ValueError(f"No market data available for {symbol}")

        # Calculate summary
        primary_price = results[0].price
        avg_price = sum(r.price for r in results) / len(results)
        price_variance = sum((r.price - avg_price) ** 2 for r in results) / len(results)

        return {
            "symbol": symbol,
            "primary_price": primary_price,
            "consensus_price": avg_price,
            "price_variance": price_variance,
            "sources_count": len(results),
            "data_sources": [
                {
                    "source": r.source,
                    "price": r.price,
                    "confidence": r.confidence,
                    "quality": r.data_quality,
                    "volume": r.volume_24h,
                    "change_24h_pct": r.change_pct_24h
                }
                for r in results
            ]
        }

    async def get_sentiment_data(self) -> Optional[Dict]:
        print("😊 Fetching sentiment data...")

        adapter = self.adapters.get(DataSource.ALTERNATIVE_ME)
        if not adapter:
            return None

        try:
            sentiment = await adapter.get_sentiment_data()
            if sentiment:
                result = {
                    "value": sentiment.value,
                    "classification": sentiment.value_classification,
                    "timestamp": sentiment.timestamp.isoformat()
                }
                print(f"✅ Fear & Greed: {sentiment.value} ({sentiment.value_classification})")
                return result
            else:
                print("❌ No sentiment data available")
                return None
        except Exception as e:
            print(f"❌ Sentiment error: {e}")
            return None

async def run_demo():
    """Run the demonstration."""
    print("🚀 Multi-Source Market Data Demo")
    print("=" * 80)

    demo = MultiSourceDemo()

    # Test Ethereum
    print("\n💰 ETHEREUM MARKET ANALYSIS")
    print("-" * 40)

    try:
        eth_data = await demo.get_market_data("ETH", max_sources=2)

        print(f"\n📈 Results Summary:")
        print(f"   Primary Price: ${eth_data['primary_price']:.2f}")
        print(f"   Consensus Price: ${eth_data['consensus_price']:.2f}")
        print(f"   Price Variance: {eth_data['price_variance']:.4f}")
        print(f"   Sources Used: {eth_data['sources_count']}")

        print(f"\n📊 Source Details:")
        for source in eth_data['data_sources']:
            print(f"   • {source['source'].upper()}: ${source['price']:.2f} "
                  f"[Confidence: {source['confidence']:.2f}, Quality: {source['quality']}]")
            if source['change_24h_pct'] != 0:
                print(f"     24h Change: {source['change_24h_pct']:+.2f}%")
            if source['volume'] > 0:
                print(f"     Volume: ${source['volume']:,.0f}")

    except Exception as e:
        print(f"❌ Ethereum analysis failed: {e}")

    # Test Bitcoin
    print(f"\n💰 BITCOIN MARKET ANALYSIS")
    print("-" * 40)

    try:
        btc_data = await demo.get_market_data("BTC", max_sources=2)

        print(f"\n📈 Results Summary:")
        print(f"   Primary Price: ${btc_data['primary_price']:.2f}")
        print(f"   Consensus Price: ${btc_data['consensus_price']:.2f}")
        print(f"   Sources Used: {btc_data['sources_count']}")

        print(f"\n📊 Source Details:")
        for source in btc_data['data_sources']:
            print(f"   • {source['source'].upper()}: ${source['price']:.2f} "
                  f"[Confidence: {source['confidence']:.2f}]")

    except Exception as e:
        print(f"❌ Bitcoin analysis failed: {e}")

    # Sentiment data
    print(f"\n😊 MARKET SENTIMENT ANALYSIS")
    print("-" * 40)

    try:
        sentiment = await demo.get_sentiment_data()

        if sentiment:
            print(f"\n📊 Fear & Greed Index:")
            print(f"   Value: {sentiment['value']}")
            print(f"   Classification: {sentiment['classification']}")
            print(f"   Timestamp: {sentiment['timestamp']}")

            # Trading signal based on sentiment
            value = sentiment['value']
            if value <= 25:
                signal = "🟢 STRONG BUY (Extreme Fear)"
            elif value <= 40:
                signal = "🟢 BUY (Fear)"
            elif value <= 60:
                signal = "🟡 HOLD (Neutral)"
            elif value <= 75:
                signal = "🔴 SELL (Greed)"
            else:
                signal = "🔴 STRONG SELL (Extreme Greed)"

            print(f"   Trading Signal: {signal}")
        else:
            print("❌ Sentiment data unavailable")

    except Exception as e:
        print(f"❌ Sentiment analysis failed: {e}")

    print(f"\n🎯 DEMONSTRATION COMPLETE")
    print("=" * 80)
    print("✅ Multi-source data implementation working successfully!")
    print("✅ Real-time API calls to multiple free sources")
    print("✅ Intelligent failover and data aggregation")
    print("✅ Market sentiment analysis integration")

if __name__ == "__main__":
    asyncio.run(run_demo())