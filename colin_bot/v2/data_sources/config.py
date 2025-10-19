"""
Market Data Configuration

Configuration for multi-source market data management.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from pathlib import Path
import json
from loguru import logger

from .models import DataSource


@dataclass
class DataSourceConfig:
    """Configuration for a single data source."""

    name: str
    base_url: str
    api_key: Optional[str] = None
    rate_limit_per_minute: int = 60
    timeout_seconds: int = 30
    enabled: bool = True
    priority: int = 1  # Lower number = higher priority
    retry_attempts: int = 3
    retry_delay_seconds: float = 1.0


@dataclass
class MarketDataConfig:
    """Multi-source market data configuration."""

    # Source priorities and availability
    primary_source: DataSource = DataSource.COINGECKO
    fallback_sources: List[DataSource] = field(default_factory=lambda: [
        DataSource.KRAKEN,
        DataSource.CRYPTOCOMPARE,
        DataSource.ALTERNATIVE_ME
    ])

    # Data source configurations
    sources: Dict[DataSource, DataSourceConfig] = field(default_factory=dict)

    # Caching configuration
    cache_ttl_seconds: int = 300  # 5 minutes
    cache_max_size: int = 1000
    cache_enabled: bool = True

    # Circuit breaker configuration
    circuit_breaker_threshold: int = 5  # Failures before opening
    circuit_breaker_timeout_seconds: int = 300  # 5 minutes

    # Data validation and quality
    data_validation_enabled: bool = True
    cross_reference_sources: bool = True
    max_price_deviation_percent: float = 5.0  # Max deviation between sources
    min_confidence_threshold: float = 0.5

    # Performance settings
    max_concurrent_requests: int = 10
    request_timeout_seconds: int = 30
    failover_timeout_seconds: int = 5

    # Logging and monitoring
    detailed_logging: bool = False
    metrics_enabled: bool = True
    health_check_interval_seconds: int = 60

    def __post_init__(self):
        """Initialize default source configurations."""
        if not self.sources:
            self._initialize_default_sources()

    def _initialize_default_sources(self):
        """Initialize default data source configurations."""
        self.sources = {
            DataSource.COINGECKO: DataSourceConfig(
                name="CoinGecko",
                base_url="https://api.coingecko.com/api/v3",
                rate_limit_per_minute=50,
                timeout_seconds=30,
                enabled=True,
                priority=1,
                retry_attempts=3
            ),
            DataSource.KRAKEN: DataSourceConfig(
                name="Kraken",
                base_url="https://api.kraken.com/0/public",
                rate_limit_per_minute=60,
                timeout_seconds=30,
                enabled=True,
                priority=2,
                retry_attempts=3
            ),
            DataSource.CRYPTOCOMPARE: DataSourceConfig(
                name="CryptoCompare",
                base_url="https://min-api.cryptocompare.com/data",
                rate_limit_per_minute=100,
                timeout_seconds=30,
                enabled=True,
                priority=3,
                retry_attempts=3
            ),
            DataSource.ALTERNATIVE_ME: DataSourceConfig(
                name="Alternative.me",
                base_url="https://api.alternative.me",
                rate_limit_per_minute=5,
                timeout_seconds=30,
                enabled=True,
                priority=4,
                retry_attempts=2
            )
        }

    def get_enabled_sources(self) -> List[DataSource]:
        """Get list of enabled data sources in priority order."""
        enabled = [
            source for source, config in self.sources.items()
            if config.enabled
        ]
        # Sort by priority (lower number = higher priority)
        enabled.sort(key=lambda s: self.sources[s].priority)
        return enabled

    def get_source_config(self, source: DataSource) -> Optional[DataSourceConfig]:
        """Get configuration for a specific data source."""
        return self.sources.get(source)

    def get_sources_by_priority(self) -> List[DataSource]:
        """Get all sources ordered by priority."""
        sources = list(self.sources.keys())
        sources.sort(key=lambda s: self.sources[s].priority)
        return sources

    def load_from_environment(self):
        """Load configuration from environment variables."""
        # API keys
        if os.getenv("COINGECKO_API_KEY"):
            if DataSource.COINGECKO in self.sources:
                self.sources[DataSource.COINGECKO].api_key = os.getenv("COINGECKO_API_KEY")

        if os.getenv("CRYPTOCOMPARE_API_KEY"):
            if DataSource.CRYPTOCOMPARE in self.sources:
                self.sources[DataSource.CRYPTOCOMPARE].api_key = os.getenv("CRYPTOCOMPARE_API_KEY")

        # Cache settings
        if os.getenv("MARKET_DATA_CACHE_TTL"):
            self.cache_ttl_seconds = int(os.getenv("MARKET_DATA_CACHE_TTL"))

        if os.getenv("MARKET_DATA_CACHE_ENABLED"):
            self.cache_enabled = os.getenv("MARKET_DATA_CACHE_ENABLED").lower() == "true"

        # Circuit breaker settings
        if os.getenv("CIRCUIT_BREAKER_THRESHOLD"):
            self.circuit_breaker_threshold = int(os.getenv("CIRCUIT_BREAKER_THRESHOLD"))

        if os.getenv("CIRCUIT_BREAKER_TIMEOUT"):
            self.circuit_breaker_timeout_seconds = int(os.getenv("CIRCUIT_BREAKER_TIMEOUT"))

        # Primary source
        if os.getenv("PRIMARY_MARKET_DATA_SOURCE"):
            try:
                self.primary_source = DataSource(os.getenv("PRIMARY_MARKET_DATA_SOURCE"))
            except ValueError:
                logger.warning(f"Invalid primary source in environment: {os.getenv('PRIMARY_MARKET_DATA_SOURCE')}")

    def save_to_file(self, file_path: str):
        """Save configuration to JSON file."""
        try:
            config_dict = {
                "primary_source": self.primary_source.value,
                "fallback_sources": [s.value for s in self.fallback_sources],
                "cache_ttl_seconds": self.cache_ttl_seconds,
                "cache_max_size": self.cache_max_size,
                "cache_enabled": self.cache_enabled,
                "circuit_breaker_threshold": self.circuit_breaker_threshold,
                "circuit_breaker_timeout_seconds": self.circuit_breaker_timeout_seconds,
                "data_validation_enabled": self.data_validation_enabled,
                "cross_reference_sources": self.cross_reference_sources,
                "max_price_deviation_percent": self.max_price_deviation_percent,
                "min_confidence_threshold": self.min_confidence_threshold,
                "max_concurrent_requests": self.max_concurrent_requests,
                "request_timeout_seconds": self.request_timeout_seconds,
                "failover_timeout_seconds": self.failover_timeout_seconds,
                "detailed_logging": self.detailed_logging,
                "metrics_enabled": self.metrics_enabled,
                "health_check_interval_seconds": self.health_check_interval_seconds,
                "sources": {
                    source.value: {
                        "name": config.name,
                        "base_url": config.base_url,
                        "api_key": config.api_key,
                        "rate_limit_per_minute": config.rate_limit_per_minute,
                        "timeout_seconds": config.timeout_seconds,
                        "enabled": config.enabled,
                        "priority": config.priority,
                        "retry_attempts": config.retry_attempts,
                        "retry_delay_seconds": config.retry_delay_seconds
                    }
                    for source, config in self.sources.items()
                }
            }

            with open(file_path, 'w') as f:
                json.dump(config_dict, f, indent=2)

            logger.info(f"Market data configuration saved to {file_path}")

        except Exception as e:
            logger.error(f"Error saving market data configuration: {e}")
            raise

    def load_from_file(self, file_path: str):
        """Load configuration from JSON file."""
        try:
            if not Path(file_path).exists():
                logger.info(f"Configuration file {file_path} not found, using defaults")
                return

            with open(file_path, 'r') as f:
                config_dict = json.load(f)

            # Load basic settings
            if "primary_source" in config_dict:
                self.primary_source = DataSource(config_dict["primary_source"])

            if "fallback_sources" in config_dict:
                self.fallback_sources = [DataSource(s) for s in config_dict["fallback_sources"]]

            # Load numeric settings
            for key in [
                "cache_ttl_seconds", "cache_max_size", "cache_enabled",
                "circuit_breaker_threshold", "circuit_breaker_timeout_seconds",
                "data_validation_enabled", "cross_reference_sources",
                "max_price_deviation_percent", "min_confidence_threshold",
                "max_concurrent_requests", "request_timeout_seconds",
                "failover_timeout_seconds", "detailed_logging",
                "metrics_enabled", "health_check_interval_seconds"
            ]:
                if key in config_dict:
                    setattr(self, key, config_dict[key])

            # Load source configurations
            if "sources" in config_dict:
                self.sources = {}
                for source_str, source_data in config_dict["sources"].items():
                    source = DataSource(source_str)
                    self.sources[source] = DataSourceConfig(
                        name=source_data["name"],
                        base_url=source_data["base_url"],
                        api_key=source_data.get("api_key"),
                        rate_limit_per_minute=source_data.get("rate_limit_per_minute", 60),
                        timeout_seconds=source_data.get("timeout_seconds", 30),
                        enabled=source_data.get("enabled", True),
                        priority=source_data.get("priority", 1),
                        retry_attempts=source_data.get("retry_attempts", 3),
                        retry_delay_seconds=source_data.get("retry_delay_seconds", 1.0)
                    )

            logger.info(f"Market data configuration loaded from {file_path}")

        except Exception as e:
            logger.error(f"Error loading market data configuration: {e}")
            # Continue with defaults

    def validate(self) -> List[str]:
        """Validate configuration and return list of issues."""
        issues = []

        # Check primary source
        if self.primary_source not in self.sources:
            issues.append(f"Primary source {self.primary_source} not configured")

        # Check fallback sources
        for source in self.fallback_sources:
            if source not in self.sources:
                issues.append(f"Fallback source {source} not configured")

        # Check if any sources are enabled
        enabled_sources = self.get_enabled_sources()
        if not enabled_sources:
            issues.append("No data sources are enabled")

        # Check cache settings
        if self.cache_ttl_seconds <= 0:
            issues.append("Cache TTL must be positive")

        if self.cache_max_size <= 0:
            issues.append("Cache max size must be positive")

        # Check circuit breaker settings
        if self.circuit_breaker_threshold <= 0:
            issues.append("Circuit breaker threshold must be positive")

        if self.circuit_breaker_timeout_seconds <= 0:
            issues.append("Circuit breaker timeout must be positive")

        # Check validation settings
        if self.max_price_deviation_percent <= 0:
            issues.append("Max price deviation must be positive")

        if not 0 <= self.min_confidence_threshold <= 1:
            issues.append("Min confidence threshold must be between 0 and 1")

        return issues


# Default configuration instance
default_config = MarketDataConfig()


def get_market_data_config(config_file: Optional[str] = None) -> MarketDataConfig:
    """Get market data configuration instance."""
    config = MarketDataConfig()

    # Load from file if provided
    if config_file:
        config.load_from_file(config_file)

    # Load from environment variables
    config.load_from_environment()

    # Validate configuration
    issues = config.validate()
    if issues:
        logger.warning(f"Market data configuration issues: {issues}")

    return config