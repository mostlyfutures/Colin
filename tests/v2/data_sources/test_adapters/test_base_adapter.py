"""
Tests for base adapter class.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from src.v2.data_sources.adapters.base_adapter import BaseAdapter
from src.v2.data_sources.config import DataSourceConfig
from src.v2.data_sources.models import DataSource


class MockAdapter(BaseAdapter):
    """Mock adapter for testing."""

    async def get_market_data(self, symbol: str):
        """Mock implementation."""
        from src.v2.data_sources.models import StandardMarketData, DataQuality
        return StandardMarketData(
            symbol=symbol,
            price=2000.0,
            volume_24h=1000000.0,
            change_24h=50.0,
            change_pct_24h=2.5,
            high_24h=2050.0,
            low_24h=1950.0,
            timestamp=asyncio.get_event_loop().time(),
            source=self.source_type,
            confidence=0.95,
            data_quality=DataQuality.GOOD
        )

    async def get_supported_symbols(self):
        """Mock implementation."""
        return ["BTC", "ETH", "ADA"]


class TestBaseAdapter:
    """Test BaseAdapter class."""

    @pytest.fixture
    def adapter_config(self):
        """Create adapter configuration for testing."""
        return DataSourceConfig(
            name="Test Source",
            base_url="https://api.test.com",
            rate_limit_per_minute=60,
            timeout_seconds=30,
            enabled=True,
            priority=1
        )

    @pytest.fixture
    def mock_adapter(self, adapter_config):
        """Create mock adapter for testing."""
        return MockAdapter(adapter_config, DataSource.COINGECKO)

    @pytest.mark.asyncio
    async def test_initialization(self, mock_adapter):
        """Test adapter initialization."""
        assert mock_adapter.config.name == "Test Source"
        assert mock_adapter.source_type == DataSource.COINGECKO
        assert mock_adapter._initialized is False
        assert mock_adapter.session is None

    @pytest.mark.asyncio
    async def test_initialize_session(self, mock_adapter):
        """Test session initialization."""
        await mock_adapter.initialize()
        assert mock_adapter._initialized is True
        assert mock_adapter.session is not None

    @pytest.mark.asyncio
    async def test_close_session(self, mock_adapter):
        """Test session closing."""
        await mock_adapter.initialize()
        assert mock_adapter._initialized is True

        await mock_adapter.close()
        assert mock_adapter._initialized is False
        assert mock_adapter.session is None

    @pytest.mark.asyncio
    async def test_rate_limiting(self, adapter_config):
        """Test rate limiting functionality."""
        # Create adapter with low rate limit for testing
        adapter_config.rate_limit_per_minute = 2
        adapter = MockAdapter(adapter_config, DataSource.COINGECKO)

        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value={"test": "data"})
            mock_get.return_value.__aenter__.return_value = mock_response

            await adapter.initialize()

            # Make multiple requests quickly
            start_time = asyncio.get_event_loop().time()
            await adapter._make_request("/test")
            await adapter._make_request("/test")
            end_time = asyncio.get_event_loop().time()

            # Should have taken at least 30 seconds due to rate limiting
            # (60 seconds / 2 requests per minute = 30 seconds between requests)
            assert end_time - start_time >= 25  # Allow some tolerance

            await adapter.close()

    @pytest.mark.asyncio
    async def test_record_success(self, mock_adapter):
        """Test recording successful request."""
        latency_ms = 100.0
        await mock_adapter._record_success(latency_ms)

        assert mock_adapter.health.consecutive_failures == 0
        assert mock_adapter.health.successful_requests == 1
        assert mock_adapter.health.is_healthy is True
        assert mock_adapter.health.average_latency_ms == 100.0

    @pytest.mark.asyncio
    async def test_record_failure(self, mock_adapter):
        """Test recording failed request."""
        error_message = "Test error"
        await mock_adapter._record_failure(error_message)

        assert mock_adapter.health.consecutive_failures == 1
        assert mock_adapter.health.last_failure_time is not None
        assert mock_adapter.health.error_message == error_message

    @pytest.mark.asyncio
    async def test_multiple_failures_triggers_unhealthy(self, mock_adapter):
        """Test that multiple failures mark adapter as unhealthy."""
        # Record 5 failures
        for i in range(5):
            await mock_adapter._record_failure(f"Error {i+1}")

        assert mock_adapter.health.consecutive_failures == 5
        assert mock_adapter.health.is_healthy is False

    @pytest.mark.asyncio
    async def test_success_after_failures_recovers(self, mock_adapter):
        """Test that success after failures recovers health."""
        # Make adapter unhealthy
        for i in range(5):
            await mock_adapter._record_failure(f"Error {i+1}")

        assert mock_adapter.health.is_healthy is False

        # Record success
        await mock_adapter._record_success(100.0)

        assert mock_adapter.health.consecutive_failures == 0
        assert mock_adapter.health.is_healthy is True

    @pytest.mark.asyncio
    async def test_get_health_status(self, mock_adapter):
        """Test getting health status."""
        health = mock_adapter.get_health_status()
        assert health.source == DataSource.COINGECKO
        assert health.is_healthy is True

    @pytest.mark.asyncio
    async def test_context_manager(self, adapter_config):
        """Test async context manager functionality."""
        adapter = MockAdapter(adapter_config, DataSource.COINGECKO)

        async with adapter as a:
            assert a._initialized is True
            assert a.session is not None

        # After context manager exits, session should be closed
        assert adapter._initialized is False
        assert adapter.session is None

    @pytest.mark.asyncio
    async def test_make_request_success(self, adapter_config):
        """Test successful HTTP request."""
        adapter = MockAdapter(adapter_config, DataSource.COINGECKO)

        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value={"test": "data"})
            mock_get.return_value.__aenter__.return_value = mock_response

            await adapter.initialize()
            result = await adapter._make_request("/test")

            assert result == {"test": "data"}
            assert adapter.health.total_requests == 1
            assert adapter.health.successful_requests == 1

            await adapter.close()

    @pytest.mark.asyncio
    async def test_make_request_http_error(self, adapter_config):
        """Test HTTP request with error status."""
        adapter = MockAdapter(adapter_config, DataSource.COINGECKO)

        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_response = AsyncMock()
            mock_response.status = 404
            mock_response.text = AsyncMock(return_value="Not Found")
            mock_get.return_value.__aenter__.return_value = mock_response

            await adapter.initialize()

            with pytest.raises(Exception):
                await adapter._make_request("/test")

            assert adapter.health.total_requests == 1
            assert adapter.health.successful_requests == 0
            assert adapter.health.consecutive_failures == 1

            await adapter.close()

    @pytest.mark.asyncio
    async def test_make_request_timeout(self, adapter_config):
        """Test HTTP request timeout."""
        adapter = MockAdapter(adapter_config, DataSource.COINGECKO)

        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_get.side_effect = asyncio.TimeoutError()

            await adapter.initialize()

            with pytest.raises(asyncio.TimeoutError):
                await adapter._make_request("/test")

            assert adapter.health.total_requests == 1
            assert adapter.health.successful_requests == 0
            assert adapter.health.consecutive_failures == 1

            await adapter.close()

    @pytest.mark.asyncio
    async def test_health_check_success(self, adapter_config):
        """Test successful health check."""
        adapter = MockAdapter(adapter_config, DataSource.COINGECKO)

        result = await adapter.health_check()
        assert result is True

    @pytest.mark.asyncio
    async def test_health_check_failure(self, adapter_config):
        """Test failed health check."""
        adapter = MockAdapter(adapter_config, DataSource.COINGECKO)

        # Mock get_market_data to raise an exception
        async def failing_get_market_data(symbol):
            raise Exception("Health check failed")

        adapter.get_market_data = failing_get_market_data

        result = await adapter.health_check()
        assert result is False