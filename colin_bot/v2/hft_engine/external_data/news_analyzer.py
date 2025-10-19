"""
News Analyzer for High-Frequency Trading

Analyzes news articles and social media content in real-time to extract
trading signals and market sentiment information.
"""

import asyncio
import aiohttp
import re
from typing import Dict, List, Optional, Set
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from enum import Enum
import logging
from collections import deque
import json

from ..utils.data_structures import TradingSignal, SignalDirection, SignalStrength
from ..utils.performance import profile_async_hft_operation, LatencyTracker


class NewsSentiment(Enum):
    """News sentiment classifications."""
    VERY_POSITIVE = "very_positive"
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"
    VERY_NEGATIVE = "very_negative"


class NewsCategory(Enum):
    """News categories for classification."""
    MARKET_NEWS = "market_news"
    REGULATORY = "regulatory"
    TECHNOLOGY = "technology"
    PARTNERSHIP = "partnership"
    SECURITY = "security"
    ADOPTION = "adoption"
    MACROECONOMIC = "macroeconomic"
    COMPETITION = "competition"


@dataclass
class NewsEvent:
    """News event data structure."""
    id: str
    title: str
    content: str
    source: str
    timestamp: datetime
    sentiment: NewsSentiment
    sentiment_score: float  # -1.0 to 1.0
    category: NewsCategory
    relevant_symbols: Set[str] = field(default_factory=set)
    keywords: List[str] = field(default_factory=list)
    urgency: float = 0.0  # 0.0 to 1.0
    reliability_score: float = 0.0  # 0.0 to 1.0
    metadata: Dict = field(default_factory=dict)


@dataclass
class NewsSignal:
    """Trading signal derived from news analysis."""
    symbol: str
    direction: SignalDirection
    strength: SignalStrength
    confidence: float
    news_events: List[NewsEvent] = field(default_factory=list)
    reasoning: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class NewsAnalyzer:
    """
    Real-time news analyzer for extracting trading signals.

    Processes news articles, social media posts, and other text sources
    to identify market-moving information and generate trading signals.
    """

    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)

        # News sources configuration
        self.news_sources = self.config.get('news_sources', [])
        self.api_keys = self.config.get('api_keys', {})
        self.update_interval = self.config.get('update_interval', 30.0)  # seconds

        # Symbol detection
        self.crypto_symbols = {
            'BTC': ['bitcoin', 'btc', 'bitcoin', 'Bitcoin'],
            'ETH': ['ethereum', 'eth', 'ether', 'Ethereum'],
            'SOL': ['solana', 'sol', 'Solana'],
            'AVAX': ['avalanche', 'avax', 'Avalanche'],
            'DOT': ['polkadot', 'dot', 'Polkadot'],
            'MATIC': ['polygon', 'matic', 'Polygon'],
            'LINK': ['chainlink', 'link', 'Chainlink'],
            'UNI': ['uniswap', 'uni', 'Uniswap']
        }

        # Sentiment keywords
        self.positive_keywords = [
            'bullish', 'surge', 'rally', 'growth', 'breakthrough', 'adoption',
            'partnership', 'launch', 'upgrade', 'expansion', 'milestone',
            'record', 'high', 'success', 'positive', 'optimistic'
        ]

        self.negative_keywords = [
            'bearish', 'crash', 'fall', 'decline', 'hack', 'breach',
            'regulation', 'ban', 'restriction', 'concern', 'warning',
            'low', 'drop', 'plunge', 'negative', 'pessimistic', 'fear'
        ]

        # Data storage
        self.news_events: deque = deque(maxlen=1000)
        self.news_signals: deque = deque(maxlen=500)
        self.processed_urls: Set[str] = set()

        # Performance tracking
        self.latency_tracker = LatencyTracker()
        self.articles_processed = 0
        self.signals_generated = 0

        # Subscribers
        self.signal_subscribers: Set[callable] = set()

    async def start_monitoring(self):
        """Start news monitoring."""
        self.logger.info("Starting news monitoring")
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())

    async def stop_monitoring(self):
        """Stop news monitoring."""
        if hasattr(self, 'monitoring_task'):
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        self.logger.info("News monitoring stopped")

    async def _monitoring_loop(self):
        """Main monitoring loop for news processing."""
        while True:
            try:
                # Fetch news from all configured sources
                await self._fetch_news_from_sources()

                # Process recent news events
                await self._process_recent_news()

                # Generate trading signals
                await self._generate_trading_signals()

                # Wait for next update
                await asyncio.sleep(self.update_interval)

            except Exception as e:
                self.logger.error(f"Error in news monitoring loop: {e}")
                await asyncio.sleep(60)  # Wait longer on error

    async def _fetch_news_from_sources(self):
        """Fetch news from configured sources."""
        for source in self.news_sources:
            try:
                if source == 'twitter':
                    await self._fetch_twitter_news()
                elif source == 'reddit':
                    await self._fetch_reddit_news()
                elif source == 'news_api':
                    await self._fetch_news_api()
                elif source == 'coindesk':
                    await self._fetch_coindesk_news()
                elif source == 'cointelegraph':
                    await self._fetch_cointelegraph_news()
            except Exception as e:
                self.logger.error(f"Error fetching news from {source}: {e}")

    async def _fetch_news_api(self):
        """Fetch news from News API."""
        api_key = self.api_keys.get('news_api')
        if not api_key:
            return

        url = "https://newsapi.org/v2/everything"
        params = {
            'q': 'bitcoin OR ethereum OR cryptocurrency OR blockchain',
            'language': 'en',
            'sortBy': 'publishedAt',
            'pageSize': 20,
            'apiKey': api_key
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        await self._process_news_api_response(data)
        except Exception as e:
            self.logger.error(f"Error fetching from News API: {e}")

    async def _fetch_coindesk_news(self):
        """Fetch news from CoinDesk."""
        try:
            url = "https://www.coindesk.com/arc/api/v2/news"
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        await self._process_coindesk_response(data)
        except Exception as e:
            self.logger.error(f"Error fetching from CoinDesk: {e}")

    async def _fetch_cointelegraph_news(self):
        """Fetch news from Cointelegraph."""
        try:
            url = "https://cointelegraph.com/api/v1/news"
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        await self._process_cointelegraph_response(data)
        except Exception as e:
            self.logger.error(f"Error fetching from Cointelegraph: {e}")

    async def _process_news_api_response(self, data: Dict):
        """Process News API response."""
        for article in data.get('articles', []):
            if article.get('url') in self.processed_urls:
                continue

            news_event = NewsEvent(
                id=f"newsapi_{hash(article.get('url', ''))}",
                title=article.get('title', ''),
                content=article.get('description', ''),
                source='newsapi',
                timestamp=self._parse_timestamp(article.get('publishedAt')),
                sentiment=NewsSentiment.NEUTRAL,  # Will be analyzed
                sentiment_score=0.0,
                category=NewsCategory.MARKET_NEWS,
                keywords=[]
            )

            await self._analyze_news_event(news_event)
            self.processed_urls.add(article.get('url'))

    async def _process_coindesk_response(self, data: Dict):
        """Process CoinDesk response."""
        for article in data.get('data', []):
            if article.get('url') in self.processed_urls:
                continue

            news_event = NewsEvent(
                id=f"coindesk_{hash(article.get('url', ''))}",
                title=article.get('title', {}).get('text', ''),
                content=article.get('description', {}).get('text', ''),
                source='coindesk',
                timestamp=self._parse_iso_timestamp(article.get('publishedAt')),
                sentiment=NewsSentiment.NEUTRAL,
                sentiment_score=0.0,
                category=NewsCategory.MARKET_NEWS,
                keywords=[]
            )

            await self._analyze_news_event(news_event)
            self.processed_urls.add(article.get('url'))

    async def _process_cointelegraph_response(self, data: Dict):
        """Process Cointelegraph response."""
        for article in data.get('data', []):
            if article.get('url') in self.processed_urls:
                continue

            news_event = NewsEvent(
                id=f"cointelegraph_{hash(article.get('url', ''))}",
                title=article.get('title', ''),
                content=article.get('excerpt', ''),
                source='cointelegraph',
                timestamp=self._parse_iso_timestamp(article.get('publishedAt')),
                sentiment=NewsSentiment.NEUTRAL,
                sentiment_score=0.0,
                category=NewsCategory.MARKET_NEWS,
                keywords=[]
            )

            await self._analyze_news_event(news_event)
            self.processed_urls.add(article.get('url'))

    @profile_async_hft_operation("news_analysis", LatencyTracker())
    async def _analyze_news_event(self, news_event: NewsEvent):
        """Analyze news event for sentiment and relevance."""
        try:
            # Extract relevant symbols
            news_event.relevant_symbols = self._extract_symbols(news_event.title + " " + news_event.content)

            # Analyze sentiment
            news_event.sentiment, news_event.sentiment_score = self._analyze_sentiment(
                news_event.title + " " + news_event.content
            )

            # Categorize news
            news_event.category = self._categorize_news(news_event.title + " " + news_event.content)

            # Calculate urgency and reliability
            news_event.urgency = self._calculate_urgency(news_event)
            news_event.reliability_score = self._calculate_reliability(news_event.source)

            # Extract keywords
            news_event.keywords = self._extract_keywords(news_event.title + " " + news_event.content)

            # Store news event
            self.news_events.append(news_event)
            self.articles_processed += 1

        except Exception as e:
            self.logger.error(f"Error analyzing news event {news_event.id}: {e}")

    def _extract_symbols(self, text: str) -> Set[str]:
        """Extract relevant cryptocurrency symbols from text."""
        text_lower = text.lower()
        found_symbols = set()

        for symbol, keywords in self.crypto_symbols.items():
            for keyword in keywords:
                if keyword in text_lower:
                    found_symbols.add(symbol)
                    break

        return found_symbols

    def _analyze_sentiment(self, text: str) -> tuple[NewsSentiment, float]:
        """Analyze sentiment of text."""
        text_lower = text.lower()
        positive_count = sum(1 for word in self.positive_keywords if word in text_lower)
        negative_count = sum(1 for word in self.negative_keywords if word in text_lower)

        total_words = len(text_lower.split())
        if total_words == 0:
            return NewsSentiment.NEUTRAL, 0.0

        # Calculate sentiment score
        sentiment_score = (positive_count - negative_count) / max(positive_count + negative_count, 1)
        sentiment_score = max(-1.0, min(1.0, sentiment_score))

        # Classify sentiment
        if sentiment_score > 0.6:
            return NewsSentiment.VERY_POSITIVE, sentiment_score
        elif sentiment_score > 0.2:
            return NewsSentiment.POSITIVE, sentiment_score
        elif sentiment_score < -0.6:
            return NewsSentiment.VERY_NEGATIVE, sentiment_score
        elif sentiment_score < -0.2:
            return NewsSentiment.NEGATIVE, sentiment_score
        else:
            return NewsSentiment.NEUTRAL, sentiment_score

    def _categorize_news(self, text: str) -> NewsCategory:
        """Categorize news based on content."""
        text_lower = text.lower()

        category_keywords = {
            NewsCategory.REGULATORY: ['regulation', 'sec', 'cftc', 'ban', 'legal', 'compliance', 'law'],
            NewsCategory.TECHNOLOGY: ['technology', 'upgrade', 'development', 'innovation', 'protocol'],
            NewsCategory.PARTNERSHIP: ['partnership', 'collaboration', 'integration', 'alliance'],
            NewsCategory.SECURITY: ['hack', 'security', 'breach', 'vulnerability', 'exploit'],
            NewsCategory.ADOPTION: ['adoption', 'integration', 'implementation', 'usage'],
            NewsCategory.MACROECONOMIC: ['inflation', 'fed', 'economy', 'market', 'financial'],
            NewsCategory.COMPETITION: ['competition', 'competitor', 'rival', 'alternative']
        }

        for category, keywords in category_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                return category

        return NewsCategory.MARKET_NEWS

    def _calculate_urgency(self, news_event: NewsEvent) -> float:
        """Calculate urgency score for news event."""
        urgency = 0.5  # Base urgency

        # Time-based urgency
        age_hours = (datetime.now(timezone.utc) - news_event.timestamp).total_seconds() / 3600
        if age_hours < 1:
            urgency += 0.3
        elif age_hours < 6:
            urgency += 0.2
        elif age_hours < 24:
            urgency += 0.1

        # Sentiment-based urgency
        if news_event.sentiment in [NewsSentiment.VERY_POSITIVE, NewsSentiment.VERY_NEGATIVE]:
            urgency += 0.2
        elif news_event.sentiment in [NewsSentiment.POSITIVE, NewsSentiment.NEGATIVE]:
            urgency += 0.1

        # Category-based urgency
        if news_event.category in [NewsCategory.REGULATORY, NewsCategory.SECURITY]:
            urgency += 0.2

        return min(1.0, urgency)

    def _calculate_reliability(self, source: str) -> float:
        """Calculate reliability score based on source."""
        source_reliability = {
            'reuters': 0.95,
            'bloomberg': 0.95,
            'coindesk': 0.85,
            'cointelegraph': 0.80,
            'newsapi': 0.70,
            'twitter': 0.60,
            'reddit': 0.55
        }
        return source_reliability.get(source.lower(), 0.5)

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract relevant keywords from text."""
        # Simple keyword extraction - in production, use more sophisticated NLP
        common_crypto_words = [
            'bitcoin', 'ethereum', 'blockchain', 'crypto', 'defi', 'nft', 'dao',
            'mining', 'staking', 'trading', 'exchange', 'wallet', 'smart contract'
        ]

        text_lower = text.lower()
        keywords = []

        for word in common_crypto_words:
            if word in text_lower:
                keywords.append(word)

        return keywords[:10]  # Limit to top 10 keywords

    async def _process_recent_news(self):
        """Process recent news events for signal generation."""
        if not self.news_events:
            return

        # Get recent news events (last hour)
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=1)
        recent_news = [
            event for event in self.news_events
            if event.timestamp > cutoff_time and event.urgency > 0.5
        ]

        # Group by symbol
        news_by_symbol = {}
        for event in recent_news:
            for symbol in event.relevant_symbols:
                if symbol not in news_by_symbol:
                    news_by_symbol[symbol] = []
                news_by_symbol[symbol].append(event)

        # Analyze each symbol's news
        for symbol, events in news_by_symbol.items():
            if len(events) >= 2:  # Minimum events for signal generation
                await self._analyze_symbol_news(symbol, events)

    async def _analyze_symbol_news(self, symbol: str, events: List[NewsEvent]):
        """Analyze news events for a specific symbol."""
        try:
            # Calculate aggregate sentiment
            total_weight = sum(event.urgency * event.reliability_score for event in events)
            if total_weight == 0:
                return

            weighted_sentiment = sum(
                event.sentiment_score * event.urgency * event.reliability_score
                for event in events
            ) / total_weight

            # Determine signal direction and strength
            if weighted_sentiment > 0.3:
                direction = SignalDirection.LONG
                strength = SignalStrength.STRONG if weighted_sentiment > 0.6 else SignalStrength.MODERATE
            elif weighted_sentiment < -0.3:
                direction = SignalDirection.SHORT
                strength = SignalStrength.STRONG if weighted_sentiment < -0.6 else SignalStrength.MODERATE
            else:
                return  # No clear signal

            # Calculate confidence
            confidence = min(abs(weighted_sentiment) * 1.5, 0.9)
            confidence *= (len(events) / 10)  # More events = higher confidence
            confidence = min(confidence, 0.9)

            # Generate reasoning
            reasoning = self._generate_signal_reasoning(symbol, events, weighted_sentiment)

            # Create news signal
            news_signal = NewsSignal(
                symbol=symbol,
                direction=direction,
                strength=strength,
                confidence=confidence,
                news_events=events,
                reasoning=reasoning
            )

            self.news_signals.append(news_signal)
            self.signals_generated += 1

            # Notify subscribers
            await self._notify_signal_subscribers(news_signal)

        except Exception as e:
            self.logger.error(f"Error analyzing news for {symbol}: {e}")

    def _generate_signal_reasoning(self, symbol: str, events: List[NewsEvent], sentiment_score: float) -> str:
        """Generate reasoning for news-based trading signal."""
        if not events:
            return ""

        # Count by sentiment
        positive_events = sum(1 for e in events if e.sentiment in [NewsSentiment.POSITIVE, NewsSentiment.VERY_POSITIVE])
        negative_events = sum(1 for e in events if e.sentiment in [NewsSentiment.NEGATIVE, NewsSentiment.VERY_NEGATIVE])

        # Count by category
        categories = {}
        for event in events:
            categories[event.category] = categories.get(event.category, 0) + 1

        # Generate reasoning text
        reasoning_parts = []

        if sentiment_score > 0.3:
            reasoning_parts.append(f"Positive news sentiment ({sentiment_score:.2f})")
            if positive_events > 0:
                reasoning_parts.append(f"{positive_events} positive news items")
        elif sentiment_score < -0.3:
            reasoning_parts.append(f"Negative news sentiment ({sentiment_score:.2f})")
            if negative_events > 0:
                reasoning_parts.append(f"{negative_events} negative news items")

        # Add category information
        if categories:
            top_category = max(categories.items(), key=lambda x: x[1])
            reasoning_parts.append(f"Primarily {top_category[0].value.replace('_', ' ')} news")

        # Add source diversity
        sources = set(event.source for event in events)
        if len(sources) > 1:
            reasoning_parts.append(f"Multiple sources ({len(sources)})")

        return "; ".join(reasoning_parts)

    async def _generate_trading_signals(self):
        """Generate trading signals from processed news."""
        # This is handled in _process_recent_news
        pass

    async def _notify_signal_subscribers(self, signal: NewsSignal):
        """Notify subscribers of new trading signals."""
        for callback in self.signal_subscribers:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(signal)
                else:
                    callback(signal)
            except Exception as e:
                self.logger.error(f"Error notifying signal subscriber: {e}")

    def subscribe_to_signals(self, callback: callable):
        """Subscribe to news-based trading signals."""
        self.signal_subscribers.add(callback)

    def unsubscribe_from_signals(self, callback: callable):
        """Unsubscribe from news-based trading signals."""
        self.signal_subscribers.discard(callback)

    def get_recent_signals(self, symbol: str = None, hours: int = 24) -> List[NewsSignal]:
        """Get recent news-based trading signals."""
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
        recent_signals = [
            signal for signal in self.news_signals
            if signal.timestamp > cutoff_time
        ]

        if symbol:
            recent_signals = [s for s in recent_signals if s.symbol == symbol]

        return recent_signals

    def get_news_events(self, symbol: str = None, hours: int = 24) -> List[NewsEvent]:
        """Get recent news events."""
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
        recent_events = [
            event for event in self.news_events
            if event.timestamp > cutoff_time
        ]

        if symbol:
            recent_events = [e for e in recent_events if symbol in e.relevant_symbols]

        return recent_events

    def get_statistics(self) -> Dict:
        """Get news analyzer statistics."""
        return {
            'articles_processed': self.articles_processed,
            'signals_generated': self.signals_generated,
            'news_events_count': len(self.news_events),
            'active_subscribers': len(self.signal_subscribers),
            'processed_urls_count': len(self.processed_urls),
            'recent_signals_24h': len(self.get_recent_signals(hours=24)),
            'recent_events_24h': len(self.get_news_events(hours=24))
        }

    def _parse_timestamp(self, timestamp_str: str) -> datetime:
        """Parse timestamp from various formats."""
        if not timestamp_str:
            return datetime.now(timezone.utc)

        try:
            # Try ISO format first
            return datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        except:
            try:
                # Try common formats
                return datetime.strptime(timestamp_str, '%Y-%m-%dT%H:%M:%S.%fZ').replace(tzinfo=timezone.utc)
            except:
                return datetime.now(timezone.utc)

    def _parse_iso_timestamp(self, timestamp_str: str) -> datetime:
        """Parse ISO timestamp."""
        if not timestamp_str:
            return datetime.now(timezone.utc)

        try:
            # Handle various ISO formats
            if timestamp_str.endswith('Z'):
                return datetime.fromisoformat(timestamp_str[:-1] + '+00:00')
            else:
                return datetime.fromisoformat(timestamp_str)
        except:
            return datetime.now(timezone.utc)

    def reset(self):
        """Reset news analyzer state."""
        self.news_events.clear()
        self.news_signals.clear()
        self.processed_urls.clear()
        self.articles_processed = 0
        self.signals_generated = 0