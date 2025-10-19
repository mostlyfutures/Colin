"""
Base Adapter Class

Abstract base class for all data source adapters.
"""

import asyncio
import aiohttp
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Optional, Any
from loguru import logger

from ..models import StandardMarketData, DataSourceHealth, DataSource
from ..config import DataSourceConfig


class BaseAdapter(ABC):
    """Abstract base class for all data source adapters."""

    def __init__(self, config: DataSourceConfig, source_type: DataSource):
        """
        Initialize base adapter.

        Args:
            config: Data source configuration
            source_type: Type of data source
        """
        self.config = config
        self.source_type = source_type
        self.session: Optional[aiohttp.ClientSession] = None
        self._last_request_time = 0
        self._request_count = 0
        self._initialized = False

        # Health tracking
        self.health = DataSourceHealth(
            source=source_type,
            is_healthy=True,
            consecutive_failures=0,
            total_requests=0,
            successful_requests=0
        )

    async def initialize(self) -> None:
        """Initialize the HTTP session."""
        if self._initialized:
            return

        try:
            timeout = aiohttp.ClientTimeout(total=self.config.timeout_seconds)
            headers = {
                "User-Agent": "Colin-Trading-Bot/2.0",
                "Accept": "application/json"
            }

            # Add API key if available
            if self.config.api_key:
                headers["Authorization"] = f"Bearer {self.config.api_key}"

            self.session = aiohttp.ClientSession(
                timeout=timeout,
                headers=headers
            )
            self._initialized = True
            logger.info(f"{self.config.name} adapter initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize {self.config.name} adapter: {e}")
            await self._record_failure(str(e))
            raise

    async def close(self) -> None:
        """Close the HTTP session."""
        if self.session:
            await self.session.close()
            self.session = None
            self._initialized = False
            logger.info(f"{self.config.name} connection closed")

    async def _make_request(self, endpoint: str, params: Optional[Dict] = None) -> Dict:
        """
        Make HTTP request with rate limiting and error handling.

        Args:
            endpoint: API endpoint
            params: Query parameters

        Returns:
            JSON response data

        Raises:
            Exception: If request fails
        """
        if not self._initialized:
            await self.initialize()

        # Rate limiting
        await self._enforce_rate_limit()

        url = f"{self.config.base_url}{endpoint}"
        start_time = datetime.now()

        try:
            self.health.total_requests += 1

            async with self.session.get(url, params=params) as response:
                latency_ms = (datetime.now() - start_time).total_seconds() * 1000

                if response.status == 200:
                    data = await response.json()
                    await self._record_success(latency_ms)
                    return data
                else:
                    error_msg = f"HTTP {response.status}: {await response.text()}"
                    await self._record_failure(error_msg, latency_ms)
                    raise aiohttp.ClientResponseError(
                        response.request_info,
                        response.history,
                        status=response.status,
                        message=error_msg
                    )

        except asyncio.TimeoutError:
            error_msg = f"Request timeout after {self.config.timeout_seconds}s"
            await self._record_failure(error_msg)
            raise
        except aiohttp.ClientError as e:
            await self._record_failure(str(e))
            raise
        except Exception as e:
            await self._record_failure(f"Unexpected error: {e}")
            raise

    async def _enforce_rate_limit(self):
        """Enforce rate limiting."""
        if self.config.rate_limit_per_minute <= 0:
            return

        current_time = asyncio.get_event_loop().time()
        time_since_last = current_time - self._last_request_time
        min_interval = 60 / self.config.rate_limit_per_minute

        if time_since_last < min_interval:
            await asyncio.sleep(min_interval - time_since_last)

        self._last_request_time = asyncio.get_event_loop().time()

    async def _record_success(self, latency_ms: float):
        """Record successful request."""
        self.health.last_success_time = datetime.now()
        self.health.consecutive_failures = 0
        self.health.successful_requests += 1

        # Update average latency
        if self.health.average_latency_ms is None:
            self.health.average_latency_ms = latency_ms
        else:
            # Simple moving average
            self.health.average_latency_ms = (
                self.health.average_latency_ms * 0.9 + latency_ms * 0.1
            )

        # Reset health if previously unhealthy
        if not self.health.is_healthy:
            self.health.is_healthy = True
            logger.info(f"{self.config.name} adapter recovered and is now healthy")

    async def _record_failure(self, error_message: str, latency_ms: Optional[float] = None):
        """Record failed request."""
        self.health.last_failure_time = datetime.now()
        self.health.consecutive_failures += 1
        self.health.error_message = error_message

        # Update latency if available
        if latency_ms is not None and self.health.average_latency_ms is not None:
            self.health.average_latency_ms = (
                self.health.average_latency_ms * 0.9 + latency_ms * 0.1
            )

        # Mark as unhealthy if too many consecutive failures
        if self.health.consecutive_failures >= 5:
            self.health.is_healthy = False
            logger.warning(f"{self.config.name} adapter marked as unhealthy: {error_message}")

    @abstractmethod
    async def get_market_data(self, symbol: str) -> StandardMarketData:
        """
        Get market data for a symbol.

        Args:
            symbol: Trading symbol (e.g., 'ETH', 'BTC')

        Returns:
            Standardized market data

        Raises:
            Exception: If data fetch fails
        """
        pass

    @abstractmethod
    async def get_supported_symbols(self) -> List[str]:
        """
        Get list of supported symbols.

        Returns:
            List of supported symbol names
        """
        pass

    async def health_check(self) -> bool:
        """
        Perform health check.

        Returns:
            True if healthy, False otherwise
        """
        try:
            # Try to fetch data for a common symbol
            await self.get_market_data("BTC")
            return True
        except Exception as e:
            logger.warning(f"Health check failed for {self.config.name}: {e}")
            return False

    def get_health_status(self) -> DataSourceHealth:
        """Get current health status."""
        return self.health

    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()