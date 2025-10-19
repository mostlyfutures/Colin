"""
Sentiment Analyzer for High-Frequency Trading

Analyzes social media sentiment, forum discussions, and community
engagement to gauge market sentiment and identify trading opportunities.
"""

import asyncio
import aiohttp
import re
from typing import Dict, List, Optional, Set
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from enum import Enum
import logging
from collections import deque, defaultdict
import json

from ..utils.data_structures import TradingSignal, SignalDirection, SignalStrength


class SocialSource(Enum):
    """Social media and forum sources."""
    TWITTER = "twitter"
    REDDIT = "reddit"
    TELEGRAM = "telegram"
    DISCORD = "discord"
    BITCOINTALK = "bitcointalk"
    GITHUB = "github"


class SentimentScore(Enum):
    """Sentiment score classifications."""
    EXTREME_FEAR = "extreme_fear"
    FEAR = "fear"
    NEUTRAL = "neutral"
    GREED = "greed"
    EXTREME_GREED = "extreme_greed"


@dataclass
class SentimentData:
    """Sentiment data point."""
    source: SocialSource
    content: str
    author: str
    timestamp: datetime
    engagement_metrics: Dict[str, int] = field(default_factory=dict)
    sentiment_score: float = 0.0  # -1.0 to 1.0
    sentiment_classification: SentimentScore = SentimentScore.NEUTRAL
    relevant_symbols: Set[str] = field(default_factory=set)
    confidence: float = 0.0
    metadata: Dict = field(default_factory=dict)


@dataclass
class SentimentSignal:
    """Trading signal derived from sentiment analysis."""
    symbol: str
    direction: SignalDirection
    strength: SignalStrength
    confidence: float
    sentiment_sources: List[SocialSource] = field(default_factory=list)
    aggregate_sentiment: float = 0.0
    source_count: int = 0
    reasoning: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class SentimentAnalyzer:
    """
    Real-time sentiment analyzer for cryptocurrency markets.

    Monitors social media platforms, forums, and community channels
    to gauge market sentiment and identify trading opportunities.
    """

    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)

        # Configuration
        self.api_keys = self.config.get('api_keys', {})
        self.update_interval = self.config.get('update_interval', 60.0)  # 1 minute
        self.sentiment_window_hours = self.config.get('sentiment_window_hours', 6)
        self.min_sources_for_signal = self.config.get('min_sources_for_signal', 3)

        # Social sources to monitor
        self.active_sources = [
            SocialSource.TWITTER,
            SocialSource.REDDIT,
            SocialSource.TELEGRAM
        ]

        # Symbol detection patterns
        self.symbol_patterns = {
            'BTC': [r'\bBTC\b', r'\bBitcoin\b', r'\bBTC\b', r'₿'],
            'ETH': [r'\bETH\b', r'\bEthereum\b', r'\bEther\b', r'Ξ'],
            'SOL': [r'\bSOL\b', r'\bSolana\b'],
            'AVAX': [r'\bAVAX\b', r'\bAvalanche\b'],
            'DOT': [r'\bDOT\b', r'\bPolkadot\b'],
            'MATIC': [r'\bMATIC\b', r'\bPolygon\b'],
            'LINK': [r'\bLINK\b', r'\bChainlink\b'],
            'UNI': [r'\bUNI\b', r'\bUniswap\b']
        }

        # Sentiment keywords and weights
        self.bullish_keywords = {
            'moon': 1.0, 'pump': 0.9, 'bullish': 0.8, 'buy': 0.7,
            'hold': 0.3, 'hodl': 0.5, 'rocket': 1.0, 'lambo': 0.9,
            'dip': 0.6, 'accumulating': 0.7, 'long': 0.6, 'upgrade': 0.8,
            'partnership': 0.7, 'adoption': 0.8, 'breakout': 0.9
        }

        self.bearish_keywords = {
            'dump': 1.0, 'bearish': 0.8, 'sell': 0.7, 'crash': 1.0,
            'scam': 1.0, 'hack': 0.9, 'vulnerable': 0.7, 'liquidation': 0.8,
            'fud': 0.6, 'concern': 0.5, 'warning': 0.6, 'avoid': 0.7,
            'short': 0.6, 'overvalued': 0.5, 'bubble': 0.8
        }

        # Data storage
        self.sentiment_data: deque = deque(maxlen=5000)
        self.sentiment_signals: deque = deque(maxlen=1000)
        self.symbol_sentiment_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))

        # Performance tracking
        self.data_points_processed = 0
        self.signals_generated = 0

        # Subscribers
        self.signal_subscribers: Set[callable] = set()

        # Rate limiting
        self.api_call_timestamps: Dict[SocialSource, deque] = defaultdict(lambda: deque(maxlen=100))
        self.rate_limits = {
            SocialSource.TWITTER: 100,  # calls per minute
            SocialSource.REDDIT: 60,
            SocialSource.TELEGRAM: 30,
            SocialSource.DISCORD: 30
        }

    async def start_monitoring(self):
        """Start sentiment monitoring."""
        self.logger.info("Starting sentiment monitoring")
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())

    async def stop_monitoring(self):
        """Stop sentiment monitoring."""
        if hasattr(self, 'monitoring_task'):
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        self.logger.info("Sentiment monitoring stopped")

    async def _monitoring_loop(self):
        """Main monitoring loop for sentiment analysis."""
        while True:
            try:
                # Fetch sentiment data from all sources
                await self._fetch_sentiment_data()

                # Process recent sentiment data
                await self._process_sentiment_data()

                # Calculate aggregate sentiment
                await self._calculate_aggregate_sentiment()

                # Generate trading signals
                await self._generate_trading_signals()

                # Wait for next update
                await asyncio.sleep(self.update_interval)

            except Exception as e:
                self.logger.error(f"Error in sentiment monitoring loop: {e}")
                await asyncio.sleep(30)

    async def _fetch_sentiment_data(self):
        """Fetch sentiment data from social media sources."""
        for source in self.active_sources:
            if await self._check_rate_limit(source):
                try:
                    if source == SocialSource.TWITTER:
                        await self._fetch_twitter_sentiment()
                    elif source == SocialSource.REDDIT:
                        await self._fetch_reddit_sentiment()
                    elif source == SocialSource.TELEGRAM:
                        await self._fetch_telegram_sentiment()
                except Exception as e:
                    self.logger.error(f"Error fetching sentiment from {source.value}: {e}")

    async def _check_rate_limit(self, source: SocialSource) -> bool:
        """Check if API call is within rate limits."""
        now = datetime.now(timezone.utc)
        recent_calls = self.api_call_timestamps[source]

        # Remove calls older than 1 minute
        cutoff = now - timedelta(minutes=1)
        while recent_calls and recent_calls[0] < cutoff:
            recent_calls.popleft()

        # Check if under rate limit
        if len(recent_calls) < self.rate_limits.get(source, 30):
            recent_calls.append(now)
            return True

        return False

    async def _fetch_twitter_sentiment(self):
        """Fetch sentiment data from Twitter."""
        # Mock implementation - replace with Twitter API
        mock_tweets = [
            {
                'content': 'Bitcoin is going to the moon! 🚀 Hold strong!',
                'author': 'crypto_enthusiast',
                'timestamp': datetime.now(timezone.utc) - timedelta(minutes=5),
                'engagement': {'likes': 150, 'retweets': 45, 'replies': 12}
            },
            {
                'content': 'Concerned about the recent ETH price action. Might see a correction.',
                'author': 'trader_joe',
                'timestamp': datetime.now(timezone.utc) - timedelta(minutes=10),
                'engagement': {'likes': 89, 'retweets': 23, 'replies': 34}
            }
        ]

        for tweet in mock_tweets:
            sentiment_data = SentimentData(
                source=SocialSource.TWITTER,
                content=tweet['content'],
                author=tweet['author'],
                timestamp=tweet['timestamp'],
                engagement_metrics=tweet['engagement']
            )

            await self._analyze_sentiment_data(sentiment_data)

    async def _fetch_reddit_sentiment(self):
        """Fetch sentiment data from Reddit."""
        # Mock implementation - replace with Reddit API
        mock_posts = [
            {
                'content': 'Just bought more SOL at the dip. This project has amazing fundamentals!',
                'author': 'reddit_investor',
                'timestamp': datetime.now(timezone.utc) - timedelta(minutes=15),
                'engagement': {'upvotes': 234, 'comments': 45}
            },
            {
                'content': 'Anyone else worried about the recent network congestion on Ethereum?',
                'author': 'eth_holder',
                'timestamp': datetime.now(timezone.utc) - timedelta(minutes=20),
                'engagement': {'upvotes': 156, 'comments': 78}
            }
        ]

        for post in mock_posts:
            sentiment_data = SentimentData(
                source=SocialSource.REDDIT,
                content=post['content'],
                author=post['author'],
                timestamp=post['timestamp'],
                engagement_metrics=post['engagement']
            )

            await self._analyze_sentiment_data(sentiment_data)

    async def _fetch_telegram_sentiment(self):
        """Fetch sentiment data from Telegram channels."""
        # Mock implementation
        mock_messages = [
            {
                'content': 'Whales are accumulating BTC. Big moves coming soon! 🐋',
                'author': 'crypto_signals',
                'timestamp': datetime.now(timezone.utc) - timedelta(minutes=8),
                'engagement': {'views': 1500, 'forwards': 89}
            }
        ]

        for message in mock_messages:
            sentiment_data = SentimentData(
                source=SocialSource.TELEGRAM,
                content=message['content'],
                author=message['author'],
                timestamp=message['timestamp'],
                engagement_metrics=message['engagement']
            )

            await self._analyze_sentiment_data(sentiment_data)

    async def _analyze_sentiment_data(self, sentiment_data: SentimentData):
        """Analyze sentiment data for content and symbols."""
        try:
            # Extract relevant symbols
            sentiment_data.relevant_symbols = self._extract_symbols(sentiment_data.content)

            # Calculate sentiment score
            sentiment_data.sentiment_score = self._calculate_sentiment_score(sentiment_data.content)
            sentiment_data.sentiment_classification = self._classify_sentiment(sentiment_data.sentiment_score)

            # Calculate confidence based on engagement and content quality
            sentiment_data.confidence = self._calculate_confidence(sentiment_data)

            # Store sentiment data
            self.sentiment_data.append(sentiment_data)
            self.data_points_processed += 1

            # Update symbol-specific history
            for symbol in sentiment_data.relevant_symbols:
                self.symbol_sentiment_history[symbol].append(sentiment_data)

        except Exception as e:
            self.logger.error(f"Error analyzing sentiment data: {e}")

    def _extract_symbols(self, text: str) -> Set[str]:
        """Extract relevant cryptocurrency symbols from text."""
        text_lower = text.lower()
        found_symbols = set()

        for symbol, patterns in self.symbol_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    found_symbols.add(symbol)
                    break

        return found_symbols

    def _calculate_sentiment_score(self, text: str) -> float:
        """Calculate sentiment score from text content."""
        text_lower = text.lower()
        words = text_lower.split()

        bullish_score = 0
        bearish_score = 0

        # Calculate bullish score
        for word in words:
            if word in self.bullish_keywords:
                bullish_score += self.bullish_keywords[word]

        # Calculate bearish score
        for word in words:
            if word in self.bearish_keywords:
                bearish_score += self.bearish_keywords[word]

        # Normalize scores
        total_score = bullish_score + bearish_score
        if total_score == 0:
            return 0.0

        # Return normalized sentiment score (-1 to 1)
        return (bullish_score - bearish_score) / max(total_score, 1.0)

    def _classify_sentiment(self, sentiment_score: float) -> SentimentScore:
        """Classify sentiment score into categories."""
        if sentiment_score > 0.6:
            return SentimentScore.EXTREME_GREED
        elif sentiment_score > 0.2:
            return SentimentScore.GREED
        elif sentiment_score < -0.6:
            return SentimentScore.EXTREME_FEAR
        elif sentiment_score < -0.2:
            return SentimentScore.FEAR
        else:
            return SentimentScore.NEUTRAL

    def _calculate_confidence(self, sentiment_data: SentimentData) -> float:
        """Calculate confidence score for sentiment data."""
        confidence = 0.5  # Base confidence

        # Engagement-based confidence
        total_engagement = sum(sentiment_data.engagement_metrics.values())
        if total_engagement > 1000:
            confidence += 0.3
        elif total_engagement > 100:
            confidence += 0.2
        elif total_engagement > 10:
            confidence += 0.1

        # Content length-based confidence
        content_length = len(sentiment_data.content.split())
        if content_length > 20:
            confidence += 0.1
        elif content_length < 5:
            confidence -= 0.2

        # Source reliability
        source_reliability = {
            SocialSource.TWITTER: 0.7,
            SocialSource.REDDIT: 0.6,
            SocialSource.TELEGRAM: 0.5,
            SocialSource.DISCORD: 0.5,
            SocialSource.BITCOINTALK: 0.8,
            SocialSource.GITHUB: 0.9
        }

        source_multiplier = source_reliability.get(sentiment_data.source, 0.5)
        confidence *= source_multiplier

        return min(1.0, max(0.0, confidence))

    async def _process_sentiment_data(self):
        """Process recent sentiment data for trends."""
        # Process sentiment data in batches
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=self.sentiment_window_hours)
        recent_data = [
            data for data in self.sentiment_data
            if data.timestamp > cutoff_time
        ]

        # Group by symbol and calculate trends
        symbol_sentiment = defaultdict(list)
        for data in recent_data:
            for symbol in data.relevant_symbols:
                symbol_sentiment[symbol].append(data)

        # Analyze sentiment trends for each symbol
        for symbol, data_points in symbol_sentiment.items():
            await self._analyze_sentiment_trend(symbol, data_points)

    async def _analyze_sentiment_trend(self, symbol: str, data_points: List[SentimentData]):
        """Analyze sentiment trend for a specific symbol."""
        if len(data_points) < 5:  # Minimum data points for trend analysis
            return

        try:
            # Sort by timestamp
            data_points.sort(key=lambda x: x.timestamp)

            # Calculate moving average of sentiment
            recent_sentiment = [dp.sentiment_score for dp in data_points[-10:]]
            avg_sentiment = sum(recent_sentiment) / len(recent_sentiment)

            # Calculate sentiment momentum
            if len(data_points) >= 20:
                older_sentiment = [dp.sentiment_score for dp in data_points[-20:-10]]
                older_avg = sum(older_sentiment) / len(older_sentiment)
                momentum = avg_sentiment - older_avg
            else:
                momentum = 0

            # Store trend analysis
            trend_data = {
                'symbol': symbol,
                'avg_sentiment': avg_sentiment,
                'momentum': momentum,
                'data_points': len(data_points),
                'timestamp': datetime.now(timezone.utc)
            }

            # Use trend data for signal generation
            if abs(avg_sentiment) > 0.3 or abs(momentum) > 0.2:
                await self._generate_sentiment_signal(symbol, trend_data, data_points)

        except Exception as e:
            self.logger.error(f"Error analyzing sentiment trend for {symbol}: {e}")

    async def _calculate_aggregate_sentiment(self):
        """Calculate aggregate sentiment across all sources."""
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=1)
        recent_data = [
            data for data in self.sentiment_data
            if data.timestamp > cutoff_time
        ]

        if not recent_data:
            return

        # Calculate overall market sentiment
        total_weighted_sentiment = 0
        total_weight = 0

        for data in recent_data:
            weight = data.confidence * sum(data.engagement_metrics.values())
            total_weighted_sentiment += data.sentiment_score * weight
            total_weight += weight

        if total_weight > 0:
            aggregate_sentiment = total_weighted_sentiment / total_weight
            self.logger.debug(f"Aggregate market sentiment: {aggregate_sentiment:.2f}")

    async def _generate_sentiment_signal(self, symbol: str, trend_data: Dict, data_points: List[SentimentData]):
        """Generate trading signal from sentiment analysis."""
        try:
            avg_sentiment = trend_data['avg_sentiment']
            momentum = trend_data['momentum']

            # Determine signal direction
            if avg_sentiment > 0.4 and momentum > 0.1:
                direction = SignalDirection.LONG
                strength = SignalStrength.STRONG if avg_sentiment > 0.6 else SignalStrength.MODERATE
            elif avg_sentiment < -0.4 and momentum < -0.1:
                direction = SignalDirection.SHORT
                strength = SignalStrength.STRONG if avg_sentiment < -0.6 else SignalStrength.MODERATE
            else:
                return  # No clear signal

            # Calculate confidence
            base_confidence = abs(avg_sentiment) * 0.7
            momentum_confidence = abs(momentum) * 0.3
            confidence = min(base_confidence + momentum_confidence, 0.9)

            # Get unique sources
            sources = list(set(data.source for data in data_points))

            # Generate reasoning
            reasoning_parts = [
                f"Sentiment: {avg_sentiment:.2f}",
                f"Momentum: {momentum:.2f}",
                f"Sources: {len(sources)}",
                f"Data points: {trend_data['data_points']}"
            ]
            reasoning = "; ".join(reasoning_parts)

            # Create sentiment signal
            sentiment_signal = SentimentSignal(
                symbol=symbol,
                direction=direction,
                strength=strength,
                confidence=confidence,
                sentiment_sources=sources,
                aggregate_sentiment=avg_sentiment,
                source_count=len(sources),
                reasoning=reasoning
            )

            self.sentiment_signals.append(sentiment_signal)
            self.signals_generated += 1

            # Notify subscribers
            await self._notify_signal_subscribers(sentiment_signal)

        except Exception as e:
            self.logger.error(f"Error generating sentiment signal for {symbol}: {e}")

    async def _generate_trading_signals(self):
        """Generate additional trading signals from sentiment analysis."""
        # This is handled in _process_sentiment_data
        pass

    async def _notify_signal_subscribers(self, signal: SentimentSignal):
        """Notify subscribers of new sentiment signals."""
        for callback in self.signal_subscribers:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(signal)
                else:
                    callback(signal)
            except Exception as e:
                self.logger.error(f"Error notifying signal subscriber: {e}")

    def subscribe_to_signals(self, callback: callable):
        """Subscribe to sentiment-based trading signals."""
        self.signal_subscribers.add(callback)

    def unsubscribe_from_signals(self, callback: callable):
        """Unsubscribe from sentiment-based trading signals."""
        self.signal_subscribers.discard(callback)

    def get_sentiment_history(self, symbol: str, hours: int = 24) -> List[SentimentData]:
        """Get sentiment history for a symbol."""
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
        return [
            data for data in self.symbol_sentiment_history.get(symbol, [])
            if data.timestamp > cutoff_time
        ]

    def get_recent_signals(self, symbol: str = None, hours: int = 24) -> List[SentimentSignal]:
        """Get recent sentiment-based trading signals."""
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
        recent_signals = [
            signal for signal in self.sentiment_signals
            if signal.timestamp > cutoff_time
        ]

        if symbol:
            recent_signals = [s for s in recent_signals if s.symbol == symbol]

        return recent_signals

    def get_market_sentiment_overview(self) -> Dict:
        """Get overview of current market sentiment."""
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=6)
        recent_data = [
            data for data in self.sentiment_data
            if data.timestamp > cutoff_time
        ]

        if not recent_data:
            return {}

        # Calculate sentiment by source
        sentiment_by_source = defaultdict(list)
        for data in recent_data:
            sentiment_by_source[data.source].append(data.sentiment_score)

        source_sentiment = {}
        for source, scores in sentiment_by_source.items():
            if scores:
                source_sentiment[source.value] = sum(scores) / len(scores)

        # Calculate sentiment by symbol
        symbol_sentiment = defaultdict(list)
        for data in recent_data:
            for symbol in data.relevant_symbols:
                symbol_sentiment[symbol].append(data.sentiment_score)

        symbol_overview = {}
        for symbol, scores in symbol_sentiment.items():
            if scores:
                symbol_overview[symbol] = {
                    'avg_sentiment': sum(scores) / len(scores),
                    'data_points': len(scores),
                    'classification': self._classify_sentiment(sum(scores) / len(scores)).value
                }

        return {
            'overall_sentiment': sum(source_sentiment.values()) / len(source_sentiment) if source_sentiment else 0,
            'sentiment_by_source': source_sentiment,
            'symbol_sentiment': symbol_overview,
            'total_data_points': len(recent_data),
            'active_sources': list(source_sentiment.keys())
        }

    def get_statistics(self) -> Dict:
        """Get sentiment analyzer statistics."""
        return {
            'data_points_processed': self.data_points_processed,
            'signals_generated': self.signals_generated,
            'total_sentiment_data': len(self.sentiment_data),
            'total_signals': len(self.sentiment_signals),
            'active_subscribers': len(self.signal_subscribers),
            'monitored_symbols': len(self.symbol_sentiment_history),
            'recent_signals_24h': len(self.get_recent_signals(hours=24))
        }

    def reset(self):
        """Reset sentiment analyzer state."""
        self.sentiment_data.clear()
        self.sentiment_signals.clear()
        self.symbol_sentiment_history.clear()
        self.data_points_processed = 0
        self.signals_generated = 0