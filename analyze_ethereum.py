#!/usr/bin/env python3
"""
Ethereum Price Analysis Script
Uses Colin Trading Bot components to analyze current ETH/USDT price
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from datetime import datetime
import ccxt
import pandas as pd
import numpy as np
from loguru import logger

# Try to import Colin Trading Bot components
try:
    from v2.ai_engine.features.technical_features import TechnicalFeatures
    from v2.risk_system.portfolio.var_calculator import VaRCalculator
    V2_AVAILABLE = True
except ImportError as e:
    logger.warning(f"V2 components not fully available: {e}")
    V2_AVAILABLE = False

class EthereumAnalyzer:
    def __init__(self):
        self.exchange = ccxt.binance({
            'rateLimit': 1200,
            'enableRateLimit': True,
        })

    def get_current_market_data(self):
        """Get current ETH/USDT market data"""
        try:
            # Get current ticker
            ticker = self.exchange.fetch_ticker('ETH/USDT')

            # Get recent OHLCV data
            ohlcv = self.exchange.fetch_ohlcv('ETH/USDT', '1h', limit=24)

            # Convert to DataFrame
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

            return {
                'current_price': ticker['last'],
                'bid': ticker['bid'],
                'ask': ticker['ask'],
                'volume_24h': ticker['baseVolume'],
                'change_24h': ticker['change'],
                'change_pct_24h': ticker['percentage'],
                'high_24h': ticker['high'],
                'low_24h': ticker['low'],
                'df': df
            }
        except Exception as e:
            logger.error(f"Error fetching market data: {e}")
            return None

    def calculate_technical_indicators(self, df):
        """Calculate basic technical indicators"""
        try:
            # Simple moving averages
            df['sma_20'] = df['close'].rolling(window=20).mean()
            df['sma_50'] = df['close'].rolling(window=min(50, len(df))).mean()

            # RSI (simplified)
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))

            # Bollinger Bands
            df['bb_middle'] = df['close'].rolling(window=20).mean()
            bb_std = df['close'].rolling(window=20).std()
            df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
            df['bb_lower'] = df['bb_middle'] - (bb_std * 2)

            # Price position relative to Bollinger Bands
            current_price = df['close'].iloc[-1]
            bb_position = (current_price - df['bb_lower'].iloc[-1]) / (df['bb_upper'].iloc[-1] - df['bb_lower'].iloc[-1])

            return {
                'sma_20': df['sma_20'].iloc[-1] if not pd.isna(df['sma_20'].iloc[-1]) else None,
                'sma_50': df['sma_50'].iloc[-1] if not pd.isna(df['sma_50'].iloc[-1]) else None,
                'rsi': df['rsi'].iloc[-1] if not pd.isna(df['rsi'].iloc[-1]) else None,
                'bb_position': bb_position if not pd.isna(bb_position) else 0.5,
                'bb_upper': df['bb_upper'].iloc[-1] if not pd.isna(df['bb_upper'].iloc[-1]) else None,
                'bb_lower': df['bb_lower'].iloc[-1] if not pd.isna(df['bb_lower'].iloc[-1]) else None,
            }
        except Exception as e:
            logger.error(f"Error calculating technical indicators: {e}")
            return {}

    def analyze_price_action(self, market_data, indicators):
        """Analyze current price action and generate insights"""
        current_price = market_data['current_price']
        change_pct = market_data['change_pct_24h'] or 0

        analysis = {
            'price_level': 'NEUTRAL',
            'momentum': 'NEUTRAL',
            'volatility': 'NORMAL',
            'key_levels': [],
            'sentiment': 'NEUTRAL'
        }

        # Price level analysis
        if indicators.get('sma_20'):
            if current_price > indicators['sma_20']:
                analysis['price_level'] = 'BULLISH_ABOVE_SMA20'
            else:
                analysis['price_level'] = 'BEARISH_BELOW_SMA20'

        # Momentum analysis
        if indicators.get('rsi'):
            rsi = indicators['rsi']
            if rsi > 70:
                analysis['momentum'] = 'OVERBOUGHT'
            elif rsi < 30:
                analysis['momentum'] = 'OVERSOLD'
            elif rsi > 50:
                analysis['momentum'] = 'BULLISH_MOMENTUM'
            else:
                analysis['momentum'] = 'BEARISH_MOMENTUM'

        # Bollinger Band analysis
        bb_position = indicators.get('bb_position', 0.5)
        if bb_position > 0.8:
            analysis['volatility'] = 'NEARING_UPPER_BAND'
        elif bb_position < 0.2:
            analysis['volatility'] = 'NEARING_LOWER_BAND'

        # 24h change sentiment
        if change_pct > 2:
            analysis['sentiment'] = 'STRONGLY_BULLISH'
        elif change_pct > 0.5:
            analysis['sentiment'] = 'BULLISH'
        elif change_pct < -2:
            analysis['sentiment'] = 'STRONGLY_BEARISH'
        elif change_pct < -0.5:
            analysis['sentiment'] = 'BEARISH'

        # Key levels
        analysis['key_levels'] = [
            f"24h High: ${market_data['high_24h']:,.2f}",
            f"24h Low: ${market_data['low_24h']:,.2f}",
        ]

        if indicators.get('sma_20'):
            analysis['key_levels'].append(f"SMA20: ${indicators['sma_20']:,.2f}")
        if indicators.get('sma_50'):
            analysis['key_levels'].append(f"SMA50: ${indicators['sma_50']:,.2f}")

        return analysis

    def generate_institutional_signal(self, market_data, indicators, analysis):
        """Generate institutional-style trading signal"""
        signal_strength = 0
        signal_factors = []

        # Factor 1: Price vs SMAs
        if indicators.get('sma_20') and indicators.get('sma_50'):
            if market_data['current_price'] > indicators['sma_20'] > indicators['sma_50']:
                signal_strength += 0.3
                signal_factors.append("Price above key moving averages")
            elif market_data['current_price'] < indicators['sma_20'] < indicators['sma_50']:
                signal_strength -= 0.3
                signal_factors.append("Price below key moving averages")

        # Factor 2: RSI momentum
        if indicators.get('rsi'):
            rsi = indicators['rsi']
            if 40 <= rsi <= 60:  # Neutral zone with potential
                signal_strength += 0.1
                signal_factors.append(f"RSI in healthy range ({rsi:.1f})")
            elif rsi > 70:
                signal_strength -= 0.2
                signal_factors.append(f"RSI overbought ({rsi:.1f})")
            elif rsi < 30:
                signal_strength += 0.2
                signal_factors.append(f"RSI oversold ({rsi:.1f})")

        # Factor 3: Bollinger Band position
        bb_position = indicators.get('bb_position', 0.5)
        if bb_position < 0.3:  # Near lower band
            signal_strength += 0.15
            signal_factors.append("Price near lower Bollinger Band")
        elif bb_position > 0.7:  # Near upper band
            signal_strength -= 0.15
            signal_factors.append("Price near upper Bollinger Band")

        # Factor 4: 24h momentum
        change_pct = market_data['change_pct_24h'] or 0
        if change_pct > 1:
            signal_strength += 0.1
            signal_factors.append(f"Positive 24h momentum ({change_pct:+.2f}%)")
        elif change_pct < -1:
            signal_strength -= 0.1
            signal_factors.append(f"Negative 24h momentum ({change_pct:+.2f}%)")

        # Determine signal direction and confidence
        if signal_strength > 0.3:
            direction = "LONG"
            confidence = min(signal_strength * 100, 85)
        elif signal_strength < -0.3:
            direction = "SHORT"
            confidence = min(abs(signal_strength) * 100, 85)
        else:
            direction = "NEUTRAL"
            confidence = 50

        return {
            'direction': direction,
            'confidence': confidence,
            'strength': abs(signal_strength),
            'factors': signal_factors,
            'signal_strength': signal_strength
        }

def main():
    """Main analysis function"""
    print("🏦 Colin Trading Bot - Ethereum Price Analysis")
    print("=" * 50)

    analyzer = EthereumAnalyzer()

    # Get market data
    print("📊 Fetching current market data...")
    market_data = analyzer.get_current_market_data()

    if not market_data:
        print("❌ Failed to fetch market data")
        return

    # Calculate technical indicators
    indicators = analyzer.calculate_technical_indicators(market_data['df'])

    # Analyze price action
    analysis = analyzer.analyze_price_action(market_data, indicators)

    # Generate institutional signal
    signal = analyzer.generate_institutional_signal(market_data, indicators, analysis)

    # Display results
    print(f"\n💰 Current Ethereum Price: ${market_data['current_price']:,.2f}")
    print(f"📈 24h Change: {market_data['change_pct_24h']:+.2f}%")
    print(f"📊 24h Volume: {market_data['volume_24h']:,.0f} ETH")
    print(f"🔴 24h Range: ${market_data['low_24h']:,.2f} - ${market_data['high_24h']:,.2f}")

    print(f"\n📈 Technical Analysis:")
    print(f"   • RSI: {indicators.get('rsi', 'N/A'):.1f}")
    print(f"   • SMA20: ${indicators.get('sma_20', 0):,.2f}")
    print(f"   • Price vs SMA20: {'Above' if market_data['current_price'] > indicators.get('sma_20', 0) else 'Below'}")
    print(f"   • Bollinger Band Position: {indicators.get('bb_position', 0.5)*100:.1f}%")

    print(f"\n🎯 Market Analysis:")
    print(f"   • Price Level: {analysis['price_level']}")
    print(f"   • Momentum: {analysis['momentum']}")
    print(f"   • Volatility: {analysis['volatility']}")
    print(f"   • Sentiment: {analysis['sentiment']}")

    print(f"\n🏦 Institutional Signal:")
    print(f"   • Direction: {signal['direction']}")
    print(f"   • Confidence: {signal['confidence']:.1f}%")
    print(f"   • Signal Strength: {signal['strength']:.3f}")

    if signal['factors']:
        print(f"   • Key Factors:")
        for factor in signal['factors']:
            print(f"     - {factor}")

    print(f"\n📍 Key Levels:")
    for level in analysis['key_levels']:
        print(f"   • {level}")

    # Risk warning
    print(f"\n⚠️  Risk Disclaimer:")
    print(f"   This analysis is for informational purposes only.")
    print(f"   Trading cryptocurrencies involves substantial risk of loss.")
    print(f"   Never trade more than you can afford to lose.")

if __name__ == "__main__":
    main()