"""
Market Data Manager

Multi-source market data manager with intelligent failover, caching,
and circuit breaker capabilities.
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Any
from loguru import logger

from .models import (
    StandardMarketData, MarketDataSummary, DataSourceHealth,
    DataSource, SentimentData
)
from .config import MarketDataConfig
from .adapters import (
    CoinGeckoAdapter, KrakenAdapter, CryptoCompareAdapter,
    AlternativeMeAdapter
)


class CircuitBreaker:
    """Circuit breaker pattern for failover management."""

    def __init__(self, failure_threshold: int = 5, timeout_seconds: int = 300):
        """
        Initialize circuit breaker.

        Args:
            failure_threshold: Number of failures before opening circuit
            timeout_seconds: Time to wait before trying again
        """
        self.failure_threshold = failure_threshold
        self.timeout_seconds = timeout_seconds
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN

    def is_open(self) -> bool:
        """Check if circuit is open."""
        if self.state == "CLOSED":
            return False
        elif self.state == "OPEN":
            if time.time() - self.last_failure_time >= self.timeout_seconds:
                self.state = "HALF_OPEN"
                logger.info("Circuit breaker moved to HALF_OPEN state")
                return False
            return True
        else:  # HALF_OPEN
            return False

    def record_success(self):
        """Record successful operation."""
        self.failure_count = 0
        if self.state == "HALF_OPEN":
            self.state = "CLOSED"
            logger.info("Circuit breaker closed after successful operation")

    def record_failure(self):
        """Record failed operation."""
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"
            logger.warning(f"Circuit breaker opened after {self.failure_count} failures")


class DataCache:
    """Simple in-memory cache for market data."""

    def __init__(self, ttl_seconds: int = 300, max_size: int = 1000):
        """
        Initialize data cache.

        Args:
            ttl_seconds: Time to live for cached items
            max_size: Maximum number of cached items
        """
        self.ttl_seconds = ttl_seconds
        self.max_size = max_size
        self.cache: Dict[str, Dict[str, Any]] = {}

    def get(self, key: str) -> Optional[Any]:
        """
        Get item from cache.

        Args:
            key: Cache key

        Returns:
            Cached item or None if not found/expired
        """
        if key not in self.cache:
            return None

        item = self.cache[key]
        if time.time() - item["timestamp"] > self.ttl_seconds:
            del self.cache[key]
            return None

        return item["data"]

    def put(self, key: str, data: Any):
        """
        Put item in cache.

        Args:
            key: Cache key
            data: Data to cache
        """
        # Remove oldest items if cache is full
        if len(self.cache) >= self.max_size:
            oldest_key = min(self.cache.keys(),
                           key=lambda k: self.cache[k]["timestamp"])
            del self.cache[oldest_key]

        self.cache[key] = {
            "data": data,
            "timestamp": time.time()
        }

    def clear(self):
        """Clear all cached items."""
        self.cache.clear()

    def size(self) -> int:
        """Get number of cached items."""
        return len(self.cache)


class MarketDataManager:
    """Multi-source market data manager with failover capabilities."""

    def __init__(self, config: MarketDataConfig):
        """
        Initialize market data manager.

        Args:
            config: Market data configuration
        """
        self.config = config
        self.adapters: Dict[DataSource, Any] = {}
        self.circuit_breakers: Dict[DataSource, CircuitBreaker] = {}
        self.cache = DataCache(
            ttl_seconds=config.cache_ttl_seconds,
            max_size=config.cache_max_size
        )
        self.source_health: Dict[DataSource, DataSourceHealth] = {}

        # Initialize adapters
        self._initialize_adapters()

        # Statistics
        self.request_count = 0
        self.cache_hits = 0
        self.failover_count = 0

        logger.info(f"MarketDataManager initialized with {len(self.adapters)} data sources")

    def _initialize_adapters(self):
        """Initialize all configured adapters."""
        # Initialize adapters based on configuration
        source_configs = self.config.sources

        for source_type, source_config in source_configs.items():
            if not source_config.enabled:
                continue

            try:
                if source_type == DataSource.COINGECKO:
                    adapter = CoinGeckoAdapter(source_config)
                elif source_type == DataSource.KRAKEN:
                    adapter = KrakenAdapter(source_config)
                elif source_type == DataSource.CRYPTOCOMPARE:
                    adapter = CryptoCompareAdapter(source_config)
                elif source_type == DataSource.ALTERNATIVE_ME:
                    adapter = AlternativeMeAdapter(source_config)
                else:
                    logger.warning(f"Unknown data source type: {source_type}")
                    continue

                self.adapters[source_type] = adapter
                self.circuit_breakers[source_type] = CircuitBreaker(
                    failure_threshold=self.config.circuit_breaker_threshold,
                    timeout_seconds=self.config.circuit_breaker_timeout_seconds
                )

                # Initialize health tracking
                self.source_health[source_type] = DataSourceHealth(
                    source=source_type,
                    is_healthy=True
                )

                logger.info(f"Initialized adapter for {source_type}")

            except Exception as e:
                logger.error(f"Failed to initialize adapter for {source_type}: {e}")

    async def initialize(self):
        """Initialize all adapters."""
        initialization_tasks = []
        for source_type, adapter in self.adapters.items():
            initialization_tasks.append(self._initialize_adapter(source_type, adapter))

        if initialization_tasks:
            await asyncio.gather(*initialization_tasks, return_exceptions=True)

    async def _initialize_adapter(self, source_type: DataSource, adapter):
        """Initialize individual adapter."""
        try:
            await adapter.initialize()
            logger.info(f"Successfully initialized {source_type} adapter")
        except Exception as e:
            logger.error(f"Failed to initialize {source_type} adapter: {e}")
            self.source_health[source_type].is_healthy = False

    async def close(self):
        """Close all adapters."""
        close_tasks = []
        for adapter in self.adapters.values():
            close_tasks.append(adapter.close())

        if close_tasks:
            await asyncio.gather(*close_tasks, return_exceptions=True)

        logger.info("All market data adapters closed")

    async def get_market_data(
        self,
        symbol: str,
        use_cache: bool = True,
        max_sources: int = 2
    ) -> MarketDataSummary:
        """
        Get market data for a symbol with intelligent failover.

        Args:
            symbol: Trading symbol (e.g., 'ETH', 'BTC')
            use_cache: Whether to use cached data
            max_sources: Maximum number of sources to try

        Returns:
            Market data summary from one or more sources
        """
        self.request_count += 1
        cache_key = f"market_data_{symbol}"

        # Check cache first
        if use_cache and self.config.cache_enabled:
            cached_data = self.cache.get(cache_key)
            if cached_data:
                self.cache_hits += 1
                logger.debug(f"Cache hit for {symbol}")
                return cached_data

        # Get sources in priority order
        sources = self._get_sources_by_priority()

        # Try to fetch from available sources
        successful_data = []
        failed_sources = []
        sources_tried = 0

        for source_type in sources:
            if sources_tried >= max_sources:
                break

            if not self._is_source_available(source_type):
                failed_sources.append(source_type)
                continue

            try:
                data = await self._fetch_from_source(source_type, symbol)
                if data:
                    successful_data.append(data)
                    sources_tried += 1

                    # Record success
                    self.circuit_breakers[source_type].record_success()
                    self.source_health[source_type].consecutive_failures = 0

                    # If this is the first successful source and we only need one, break
                    if max_sources == 1:
                        break

            except Exception as e:
                logger.warning(f"Failed to fetch {symbol} from {source_type}: {e}")
                failed_sources.append(source_type)

                # Record failure
                self.circuit_breakers[source_type].record_failure()
                self.source_health[source_type].consecutive_failures += 1

        if not successful_data:
            logger.error(f"Failed to fetch market data for {symbol} from all sources")
            raise ValueError(f"No market data available for {symbol}")

        # Create market data summary
        summary = MarketDataSummary(
            symbol=symbol,
            primary_price=successful_data[0].price,
            price_sources=successful_data,
            available_sources=[s for s in sources if s not in failed_sources],
            failed_sources=failed_sources
        )

        # Cross-reference if multiple sources
        if len(successful_data) > 1 and self.config.cross_reference_sources:
            summary = self._cross_reference_data(summary)

        # Cache the result
        if use_cache and self.config.cache_enabled:
            self.cache.put(cache_key, summary)

        logger.info(f"Fetched market data for {symbol} from {len(successful_data)} source(s)")
        return summary

    async def get_sentiment_data(self) -> Optional[SentimentData]:
        """
        Get sentiment data from Alternative.me.

        Returns:
            Sentiment data or None if unavailable
        """
        if DataSource.ALTERNATIVE_ME not in self.adapters:
            logger.warning("Alternative.me adapter not available for sentiment data")
            return None

        if not self._is_source_available(DataSource.ALTERNATIVE_ME):
            logger.warning("Alternative.me source is not healthy")
            return None

        try:
            adapter = self.adapters[DataSource.ALTERNATIVE_ME]
            sentiment = await adapter.get_fear_and_greed_index()

            # Record success
            self.circuit_breakers[DataSource.ALTERNATIVE_ME].record_success()

            return sentiment

        except Exception as e:
            logger.error(f"Failed to fetch sentiment data: {e}")
            self.circuit_breakers[DataSource.ALTERNATIVE_ME].record_failure()
            return None

    async def get_supported_symbols(self) -> Set[str]:
        """
        Get all supported symbols across all sources.

        Returns:
            Set of supported symbol names
        """
        all_symbols = set()
        for adapter in self.adapters.values():
            try:
                symbols = await adapter.get_supported_symbols()
                all_symbols.update(symbols)
            except Exception as e:
                logger.warning(f"Failed to get supported symbols from adapter: {e}")

        return all_symbols

    def _get_sources_by_priority(self) -> List[DataSource]:
        """Get data sources ordered by priority and health."""
        available_sources = []

        for source_type in self.config.get_sources_by_priority():
            if self._is_source_available(source_type):
                available_sources.append(source_type)

        return available_sources

    def _is_source_available(self, source_type: DataSource) -> bool:
        """Check if a data source is available for use."""
        if source_type not in self.adapters:
            return False

        # Check circuit breaker
        if self.circuit_breakers[source_type].is_open():
            return False

        # Check health
        health = self.source_health.get(source_type)
        if health and not health.is_available:
            return False

        return True

    async def _fetch_from_source(self, source_type: DataSource, symbol: str) -> Optional[StandardMarketData]:
        """Fetch data from a specific source."""
        adapter = self.adapters[source_type]

        # Skip Alternative.me for market data requests
        if source_type == DataSource.ALTERNATIVE_ME:
            return None

        try:
            data = await adapter.get_market_data(symbol)
            return data
        except Exception as e:
            logger.error(f"Error fetching {symbol} from {source_type}: {e}")
            raise

    def _cross_reference_data(self, summary: MarketDataSummary) -> MarketDataSummary:
        """Cross-reference data from multiple sources for validation."""
        if len(summary.price_sources) < 2:
            return summary

        prices = [source.price for source in summary.price_sources]
        avg_price = sum(prices) / len(prices)

        # Calculate variance
        variance = sum((p - avg_price) ** 2 for p in prices) / len(prices)
        summary.price_variance = variance

        # Check for outliers
        max_deviation = self.config.max_price_deviation_percent / 100
        valid_sources = []

        for source_data in summary.price_sources:
            deviation = abs(source_data.price - avg_price) / avg_price
            if deviation <= max_deviation:
                valid_sources.append(source_data)
            else:
                logger.warning(
                    f"Price outlier detected for {summary.symbol} from {source_data.source}: "
                    f"{source_data.price} (deviation: {deviation:.2%})"
                )

        # If we have valid sources, use only those
        if valid_sources:
            summary.price_sources = valid_sources
            summary.consensus_price = sum(s.price for s in valid_sources) / len(valid_sources)

        return summary

    async def health_check(self) -> Dict[DataSource, bool]:
        """
        Perform health check on all data sources.

        Returns:
            Dictionary mapping sources to health status
        """
        health_status = {}

        for source_type, adapter in self.adapters.items():
            try:
                is_healthy = await adapter.health_check()
                health_status[source_type] = is_healthy
                self.source_health[source_type].is_healthy = is_healthy
            except Exception as e:
                logger.warning(f"Health check failed for {source_type}: {e}")
                health_status[source_type] = False
                self.source_health[source_type].is_healthy = False

        return health_status

    def get_system_stats(self) -> Dict[str, Any]:
        """Get system performance statistics."""
        cache_hit_rate = (self.cache_hits / self.request_count * 100) if self.request_count > 0 else 0

        return {
            "total_requests": self.request_count,
            "cache_hits": self.cache_hits,
            "cache_hit_rate_percent": round(cache_hit_rate, 2),
            "cache_size": self.cache.size(),
            "failover_count": self.failover_count,
            "available_sources": [
                source for source in self.adapters.keys()
                if self._is_source_available(source)
            ],
            "unhealthy_sources": [
                source for source, health in self.source_health.items()
                if not health.is_healthy
            ],
            "circuit_breakers": {
                source.value: {
                    "state": cb.state,
                    "failure_count": cb.failure_count
                }
                for source, cb in self.circuit_breakers.items()
            }
        }

    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()