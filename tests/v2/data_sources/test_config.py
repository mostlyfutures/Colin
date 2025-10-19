"""
Tests for market data configuration.
"""

import pytest
import tempfile
import os
from src.v2.data_sources.config import (
    MarketDataConfig, DataSourceConfig, DataSource,
    get_market_data_config
)


class TestMarketDataConfig:
    """Test MarketDataConfig class."""

    def test_default_initialization(self):
        """Test default configuration initialization."""
        config = MarketDataConfig()

        assert config.primary_source == DataSource.COINGECKO
        assert len(config.fallback_sources) > 0
        assert config.cache_ttl_seconds == 300
        assert config.cache_enabled is True
        assert len(config.sources) > 0

    def test_source_config_initialization(self):
        """Test data source configuration initialization."""
        config = MarketDataConfig()

        # Check that all sources are initialized
        assert DataSource.COINGECKO in config.sources
        assert DataSource.KRAKEN in config.sources
        assert DataSource.CRYPTOCOMPARE in config.sources
        assert DataSource.ALTERNATIVE_ME in config.sources

        # Check individual source configurations
        coingecko_config = config.sources[DataSource.COINGECKO]
        assert coingecko_config.name == "CoinGecko"
        assert coingecko_config.base_url == "https://api.coingecko.com/api/v3"
        assert coingecko_config.enabled is True
        assert coingecko_config.priority == 1

    def test_get_enabled_sources(self):
        """Test getting enabled sources in priority order."""
        config = MarketDataConfig()

        enabled_sources = config.get_enabled_sources()
        assert len(enabled_sources) > 0

        # Check that they are in priority order
        priorities = [config.sources[source].priority for source in enabled_sources]
        assert priorities == sorted(priorities)

    def test_get_sources_by_priority(self):
        """Test getting all sources ordered by priority."""
        config = MarketDataConfig()

        sources = config.get_sources_by_priority()
        assert len(sources) == len(config.sources)

        # Check priority ordering
        priorities = [config.sources[source].priority for source in sources]
        assert priorities == sorted(priorities)

    def test_load_from_environment(self):
        """Test loading configuration from environment variables."""
        config = MarketDataConfig()

        # Set environment variables
        os.environ["COINGECKO_API_KEY"] = "test-key"
        os.environ["MARKET_DATA_CACHE_TTL"] = "600"
        os.environ["MARKET_DATA_CACHE_ENABLED"] = "false"
        os.environ["PRIMARY_MARKET_DATA_SOURCE"] = "kraken"

        config.load_from_environment()

        # Check that values were loaded
        assert config.sources[DataSource.COINGECKO].api_key == "test-key"
        assert config.cache_ttl_seconds == 600
        assert config.cache_enabled is False
        assert config.primary_source == DataSource.KRAKEN

        # Clean up
        del os.environ["COINGECKO_API_KEY"]
        del os.environ["MARKET_DATA_CACHE_TTL"]
        del os.environ["MARKET_DATA_CACHE_ENABLED"]
        del os.environ["PRIMARY_MARKET_DATA_SOURCE"]

    def test_save_to_file(self):
        """Test saving configuration to file."""
        config = MarketDataConfig()
        config.cache_ttl_seconds = 600
        config.primary_source = DataSource.KRAKEN

        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            config.save_to_file(f.name)

            # Check that file exists and has content
            assert os.path.exists(f.name)
            assert os.path.getsize(f.name) > 0

            # Clean up
            os.unlink(f.name)

    def test_load_from_file(self):
        """Test loading configuration from file."""
        # Create a test configuration file
        config = MarketDataConfig()
        config.cache_ttl_seconds = 600
        config.primary_source = DataSource.KRAKEN

        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            config.save_to_file(f.name)
            config_file = f.name

        # Load configuration from file
        new_config = MarketDataConfig()
        new_config.load_from_file(config_file)

        # Check that values were loaded
        assert new_config.cache_ttl_seconds == 600
        assert new_config.primary_source == DataSource.KRAKEN

        # Clean up
        os.unlink(config_file)

    def test_validate(self):
        """Test configuration validation."""
        config = MarketDataConfig()

        # Valid configuration should have no issues
        issues = config.validate()
        assert len(issues) == 0

        # Invalid configuration should have issues
        config.cache_ttl_seconds = -1
        issues = config.validate()
        assert len(issues) > 0
        assert any("Cache TTL must be positive" in issue for issue in issues)

    def test_get_market_data_config(self):
        """Test get_market_data_config function."""
        config = get_market_data_config()

        assert isinstance(config, MarketDataConfig)
        assert len(config.sources) > 0
        assert config.cache_enabled is True


class TestDataSourceConfig:
    """Test DataSourceConfig class."""

    def test_data_source_config_creation(self):
        """Test creating a data source configuration."""
        config = DataSourceConfig(
            name="Test Source",
            base_url="https://api.test.com",
            api_key="test-key",
            rate_limit_per_minute=100,
            timeout_seconds=30,
            enabled=True,
            priority=1
        )

        assert config.name == "Test Source"
        assert config.base_url == "https://api.test.com"
        assert config.api_key == "test-key"
        assert config.rate_limit_per_minute == 100
        assert config.timeout_seconds == 30
        assert config.enabled is True
        assert config.priority == 1