"""
Tests for market data models.
"""

import pytest
from datetime import datetime
from src.v2.data_sources.models import (
    StandardMarketData, MarketDataSummary, DataSourceHealth,
    SentimentData, DataSource, DataQuality
)


class TestStandardMarketData:
    """Test StandardMarketData class."""

    def test_market_data_creation(self):
        """Test creating market data."""
        timestamp = datetime.now()
        market_data = StandardMarketData(
            symbol="ETH",
            price=2000.0,
            volume_24h=1000000.0,
            change_24h=50.0,
            change_pct_24h=2.5,
            high_24h=2050.0,
            low_24h=1950.0,
            timestamp=timestamp,
            source=DataSource.COINGECKO,
            confidence=0.95
        )

        assert market_data.symbol == "ETH"
        assert market_data.price == 2000.0
        assert market_data.volume_24h == 1000000.0
        assert market_data.change_24h == 50.0
        assert market_data.change_pct_24h == 2.5
        assert market_data.high_24h == 2050.0
        assert market_data.low_24h == 1950.0
        assert market_data.timestamp == timestamp
        assert market_data.source == DataSource.COINGECKO
        assert market_data.confidence == 0.95
        assert market_data.data_quality == DataQuality.UNKNOWN

    def test_market_data_to_dict(self):
        """Test converting market data to dictionary."""
        timestamp = datetime.now()
        market_data = StandardMarketData(
            symbol="BTC",
            price=50000.0,
            volume_24h=2000000.0,
            change_24h=1000.0,
            change_pct_24h=2.0,
            high_24h=51000.0,
            low_24h=49000.0,
            timestamp=timestamp,
            source=DataSource.KRAKEN,
            confidence=0.90,
            data_quality=DataQuality.GOOD
        )

        data_dict = market_data.to_dict()

        assert data_dict["symbol"] == "BTC"
        assert data_dict["price"] == 50000.0
        assert data_dict["source"] == "kraken"
        assert data_dict["confidence"] == 0.90
        assert data_dict["data_quality"] == "good"
        assert data_dict["timestamp"] == timestamp.isoformat()

    def test_market_data_from_dict(self):
        """Test creating market data from dictionary."""
        timestamp = datetime.now()
        data_dict = {
            "symbol": "SOL",
            "price": 150.0,
            "volume_24h": 500000.0,
            "change_24h": 5.0,
            "change_pct_24h": 3.33,
            "high_24h": 155.0,
            "low_24h": 145.0,
            "timestamp": timestamp.isoformat(),
            "source": "cryptocompare",
            "confidence": 0.85,
            "data_quality": "excellent"
        }

        market_data = StandardMarketData.from_dict(data_dict)

        assert market_data.symbol == "SOL"
        assert market_data.price == 150.0
        assert market_data.source == DataSource.CRYPTOCOMPARE
        assert market_data.confidence == 0.85
        assert market_data.data_quality == DataQuality.EXCELLENT


class TestMarketDataSummary:
    """Test MarketDataSummary class."""

    def test_market_data_summary_creation(self):
        """Test creating market data summary."""
        timestamp = datetime.now()
        source1 = StandardMarketData(
            symbol="ETH", price=2000.0, volume_24h=1000000.0,
            change_24h=50.0, change_pct_24h=2.5, high_24h=2050.0,
            low_24h=1950.0, timestamp=timestamp, source=DataSource.COINGECKO,
            confidence=0.95
        )
        source2 = StandardMarketData(
            symbol="ETH", price=2005.0, volume_24h=950000.0,
            change_24h=55.0, change_pct_24h=2.8, high_24h=2060.0,
            low_24h=1955.0, timestamp=timestamp, source=DataSource.KRAKEN,
            confidence=0.90
        )

        summary = MarketDataSummary(
            symbol="ETH",
            primary_price=2000.0,
            price_sources=[source1, source2],
            available_sources=[DataSource.COINGECKO, DataSource.KRAKEN]
        )

        assert summary.symbol == "ETH"
        assert summary.primary_price == 2000.0
        assert len(summary.price_sources) == 2
        assert summary.consensus_price == 2002.5  # Average of 2000 and 2005
        assert summary.data_quality_score == 0.925  # Average of 0.95 and 0.90
        assert len(summary.available_sources) == 2

    def test_market_data_summary_single_source(self):
        """Test market data summary with single source."""
        timestamp = datetime.now()
        source = StandardMarketData(
            symbol="BTC", price=50000.0, volume_24h=2000000.0,
            change_24h=1000.0, change_pct_24h=2.0, high_24h=51000.0,
            low_24h=49000.0, timestamp=timestamp, source=DataSource.COINGECKO,
            confidence=0.95
        )

        summary = MarketDataSummary(
            symbol="BTC",
            primary_price=50000.0,
            price_sources=[source]
        )

        assert summary.symbol == "BTC"
        assert summary.primary_price == 50000.0
        assert len(summary.price_sources) == 1
        assert summary.consensus_price is None  # No consensus with single source
        assert summary.price_variance is None  # No variance with single source


class TestDataSourceHealth:
    """Test DataSourceHealth class."""

    def test_data_source_health_creation(self):
        """Test creating data source health."""
        health = DataSourceHealth(
            source=DataSource.COINGECKO,
            is_healthy=True,
            consecutive_failures=0,
            total_requests=100,
            successful_requests=95
        )

        assert health.source == DataSource.COINGECKO
        assert health.is_healthy is True
        assert health.consecutive_failures == 0
        assert health.total_requests == 100
        assert health.successful_requests == 95

    def test_success_rate_calculation(self):
        """Test success rate calculation."""
        health = DataSourceHealth(
            source=DataSource.KRAKEN,
            total_requests=100,
            successful_requests=85
        )

        assert health.success_rate == 0.85

    def test_success_rate_zero_requests(self):
        """Test success rate with zero requests."""
        health = DataSourceHealth(
            source=DataSource.CRYPTOCOMPARE,
            total_requests=0,
            successful_requests=0
        )

        assert health.success_rate == 0.0

    def test_is_available(self):
        """Test availability check."""
        # Healthy source should be available
        health1 = DataSourceHealth(
            source=DataSource.COINGECKO,
            is_healthy=True,
            consecutive_failures=0,
            total_requests=10,
            successful_requests=8
        )
        assert health1.is_available is True

        # Source with too many failures should not be available
        health2 = DataSourceHealth(
            source=DataSource.KRAKEN,
            is_healthy=True,
            consecutive_failures=5,
            total_requests=10,
            successful_requests=5
        )
        assert health2.is_available is False

        # Unhealthy source should not be available
        health3 = DataSourceHealth(
            source=DataSource.CRYPTOCOMPARE,
            is_healthy=False,
            consecutive_failures=0,
            total_requests=10,
            successful_requests=3
        )
        assert health3.is_available is False


class TestSentimentData:
    """Test SentimentData class."""

    def test_sentiment_data_creation(self):
        """Test creating sentiment data."""
        timestamp = datetime.now()
        sentiment = SentimentData(
            value=75,
            value_classification="Greed",
            timestamp=timestamp,
            time_until_update="2 hours"
        )

        assert sentiment.value == 75
        assert sentiment.value_classification == "Greed"
        assert sentiment.timestamp == timestamp
        assert sentiment.time_until_update == "2 hours"

    def test_sentiment_data_to_dict(self):
        """Test converting sentiment data to dictionary."""
        timestamp = datetime.now()
        sentiment = SentimentData(
            value=25,
            value_classification="Fear",
            timestamp=timestamp
        )

        data_dict = sentiment.to_dict()

        assert data_dict["value"] == 25
        assert data_dict["value_classification"] == "Fear"
        assert data_dict["timestamp"] == timestamp.isoformat()
        assert data_dict["time_until_update"] is None


class TestDataQuality:
    """Test DataQuality enum."""

    def test_data_quality_values(self):
        """Test data quality enum values."""
        assert DataQuality.EXCELLENT.value == "excellent"
        assert DataQuality.GOOD.value == "good"
        assert DataQuality.FAIR.value == "fair"
        assert DataQuality.POOR.value == "poor"
        assert DataQuality.UNKNOWN.value == "unknown"


class TestDataSource:
    """Test DataSource enum."""

    def test_data_source_values(self):
        """Test data source enum values."""
        assert DataSource.COINGECKO.value == "coingecko"
        assert DataSource.KRAKEN.value == "kraken"
        assert DataSource.CRYPTOCOMPARE.value == "cryptocompare"
        assert DataSource.ALTERNATIVE_ME.value == "alternative_me"