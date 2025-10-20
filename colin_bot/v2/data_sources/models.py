"""
Data Models for Market Data Sources

Standardized data structures for market data across all sources.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Any
from enum import Enum


class DataSource(str, Enum):
    """Available data sources."""
    COINGECKO = "coingecko"
    KRAKEN = "kraken"
    CRYPTOCOMPARE = "cryptocompare"
    ALTERNATIVE_ME = "alternative_me"
    HYPERLIQUID = "hyperliquid"


class DataQuality(str, Enum):
    """Data quality indicators."""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    UNKNOWN = "unknown"


@dataclass
class StandardMarketData:
    """Standardized market data format across all sources."""

    # Basic price data
    symbol: str
    price: float
    volume_24h: float
    change_24h: float
    change_pct_24h: float
    high_24h: float
    low_24h: float

    # Metadata
    timestamp: datetime
    source: DataSource
    confidence: float  # 0.0 to 1.0

    # Extended data (optional)
    market_cap: Optional[float] = None
    circulating_supply: Optional[float] = None
    total_supply: Optional[float] = None
    ath: Optional[float] = None  # All-time high
    atl: Optional[float] = None  # All-time low

    # Quality indicators
    data_quality: DataQuality = DataQuality.UNKNOWN
    latency_ms: Optional[float] = None
    update_frequency: Optional[str] = None

    # Source-specific data
    raw_data: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "symbol": self.symbol,
            "price": self.price,
            "volume_24h": self.volume_24h,
            "change_24h": self.change_24h,
            "change_pct_24h": self.change_pct_24h,
            "high_24h": self.high_24h,
            "low_24h": self.low_24h,
            "timestamp": self.timestamp.isoformat(),
            "source": self.source.value,
            "confidence": self.confidence,
            "market_cap": self.market_cap,
            "circulating_supply": self.circulating_supply,
            "total_supply": self.total_supply,
            "ath": self.ath,
            "atl": self.atl,
            "data_quality": self.data_quality.value,
            "latency_ms": self.latency_ms,
            "update_frequency": self.update_frequency,
            "raw_data": self.raw_data
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StandardMarketData":
        """Create from dictionary format."""
        return cls(
            symbol=data["symbol"],
            price=float(data["price"]),
            volume_24h=float(data["volume_24h"]),
            change_24h=float(data["change_24h"]),
            change_pct_24h=float(data["change_pct_24h"]),
            high_24h=float(data["high_24h"]),
            low_24h=float(data["low_24h"]),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            source=DataSource(data["source"]),
            confidence=float(data["confidence"]),
            market_cap=data.get("market_cap"),
            circulating_supply=data.get("circulating_supply"),
            total_supply=data.get("total_supply"),
            ath=data.get("ath"),
            atl=data.get("atl"),
            data_quality=DataQuality(data.get("data_quality", "unknown")),
            latency_ms=data.get("latency_ms"),
            update_frequency=data.get("update_frequency"),
            raw_data=data.get("raw_data")
        )


@dataclass
class MarketDataSummary:
    """Summary of market data across multiple sources."""

    symbol: str
    primary_price: float
    price_sources: List[StandardMarketData]
    consensus_price: Optional[float] = None
    price_variance: Optional[float] = None
    data_quality_score: float = 0.0

    # Source availability
    available_sources: List[DataSource] = None
    failed_sources: List[DataSource] = None

    # Timing
    timestamp: datetime = None

    def __post_init__(self):
        """Initialize derived fields."""
        if self.available_sources is None:
            self.available_sources = []
        if self.failed_sources is None:
            self.failed_sources = []
        if self.timestamp is None:
            self.timestamp = datetime.now()

        # Calculate consensus price and variance
        if len(self.price_sources) > 1:
            prices = [source.price for source in self.price_sources]
            self.consensus_price = sum(prices) / len(prices)
            self.price_variance = sum((p - self.consensus_price) ** 2 for p in prices) / len(prices)

        # Calculate data quality score
        if self.price_sources:
            self.data_quality_score = sum(source.confidence for source in self.price_sources) / len(self.price_sources)


@dataclass
class DataSourceHealth:
    """Health status of a data source."""

    source: DataSource
    is_healthy: bool
    last_success_time: Optional[datetime] = None
    last_failure_time: Optional[datetime] = None
    consecutive_failures: int = 0
    total_requests: int = 0
    successful_requests: int = 0
    average_latency_ms: Optional[float] = None
    error_message: Optional[str] = None

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_requests == 0:
            return 0.0
        return self.successful_requests / self.total_requests

    @property
    def is_available(self) -> bool:
        """Check if source is available for use."""
        return (self.is_healthy and
                self.consecutive_failures < 3 and
                self.success_rate > 0.5)


@dataclass
class SentimentData:
    """Market sentiment data from Alternative.me."""

    value: int  # Fear & Greed index value (0-100)
    value_classification: str  # "Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed"
    timestamp: datetime
    time_until_update: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "value": self.value,
            "value_classification": self.value_classification,
            "timestamp": self.timestamp.isoformat(),
            "time_until_update": self.time_until_update
        }