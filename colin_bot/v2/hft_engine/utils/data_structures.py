"""
Data structures for HFT engine.

Core data models and structures for high-frequency trading operations.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
from datetime import datetime, timezone
from enum import Enum
import numpy as np


class OrderSide(Enum):
    """Order side enumeration."""
    BUY = "buy"
    SELL = "sell"


class SignalDirection(Enum):
    """Signal direction enumeration."""
    LONG = "long"
    SHORT = "short"
    HOLD = "hold"


class EventType(Enum):
    """Market event type enumeration."""
    ORDER_BOOK_UPDATE = "order_book_update"
    TRADE = "trade"
    OPEN_INTEREST_UPDATE = "open_interest_update"
    NEWS_EVENT = "news_event"


@dataclass
class MarketEvent:
    """Market event data structure."""
    event_type: EventType
    symbol: str
    timestamp: datetime
    data: Dict[str, Any]
    exchange: str = "unknown"

    def __post_init__(self):
        """Validate event data."""
        if self.timestamp.tzinfo is None:
            self.timestamp = self.timestamp.replace(tzinfo=timezone.utc)


@dataclass
class OrderBookLevel:
    """Single order book level."""
    price: float
    size: float
    orders: int = 1

    def __post_init__(self):
        """Validate order book level."""
        if self.price <= 0:
            raise ValueError("Price must be positive")
        if self.size <= 0:
            raise ValueError("Size must be positive")


@dataclass
class OrderBook:
    """Order book data structure."""
    symbol: str
    timestamp: datetime
    exchange: str
    bids: List[OrderBookLevel] = field(default_factory=list)
    asks: List[OrderBookLevel] = field(default_factory=list)

    @property
    def best_bid(self) -> Optional[float]:
        """Get best bid price."""
        return self.bids[0].price if self.bids else None

    @property
    def best_ask(self) -> Optional[float]:
        """Get best ask price."""
        return self.asks[0].price if self.asks else None

    @property
    def mid_price(self) -> Optional[float]:
        """Get mid price."""
        if self.best_bid and self.best_ask:
            return (self.best_bid + self.best_ask) / 2
        return None

    @property
    def spread(self) -> Optional[float]:
        """Get bid-ask spread."""
        if self.best_bid and self.best_ask:
            return self.best_ask - self.best_bid
        return None

    @property
    def total_bid_size(self) -> float:
        """Get total bid size."""
        return sum(level.size for level in self.bids)

    @property
    def total_ask_size(self) -> float:
        """Get total ask size."""
        return sum(level.size for level in self.asks)

    def get_bids(self, levels: int = None) -> List[Tuple[float, float]]:
        """Get bids as price-size tuples."""
        bids = [(level.price, level.size) for level in self.bids]
        return bids[:levels] if levels else bids

    def get_asks(self, levels: int = None) -> List[Tuple[float, float]]:
        """Get asks as price-size tuples."""
        asks = [(level.price, level.size) for level in self.asks]
        return asks[:levels] if levels else asks


@dataclass
class Trade:
    """Trade data structure."""
    symbol: str
    timestamp: datetime
    price: float
    size: float
    side: OrderSide
    trade_id: str = ""
    exchange: str = "unknown"

    def __post_init__(self):
        """Validate trade data."""
        if self.price <= 0:
            raise ValueError("Price must be positive")
        if self.size <= 0:
            raise ValueError("Size must be positive")


@dataclass
class TradingSignal:
    """Trading signal data structure."""
    symbol: str
    direction: SignalDirection
    confidence: float  # 0.0 to 1.0
    strength: float    # Signal strength
    timestamp: datetime
    source: str = "unknown"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate signal data."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")
        if not -1.0 <= self.strength <= 1.0:
            raise ValueError("Strength must be between -1.0 and 1.0")

    @property
    def is_long(self) -> bool:
        """Check if signal is long."""
        return self.direction == SignalDirection.LONG

    @property
    def is_short(self) -> bool:
        """Check if signal is short."""
        return self.direction == SignalDirection.SHORT


@dataclass
class OFISignal:
    """Order Flow Imbalance signal structure."""
    symbol: str
    ofi_value: float
    forecast_direction: SignalDirection
    confidence: float
    timestamp: datetime
    hawkes_intensity: Dict[str, float] = field(default_factory=dict)

    @property
    def is_positive(self) -> bool:
        """Check if OFI is positive."""
        return self.ofi_value > 0

    @property
    def strength_category(self) -> str:
        """Get strength category."""
        abs_val = abs(self.ofi_value)
        if abs_val > 0.8:
            return "strong"
        elif abs_val > 0.4:
            return "moderate"
        else:
            return "weak"


@dataclass
class BookSkewSignal:
    """Order book skew signal structure."""
    symbol: str
    skew_value: float
    threshold: float
    signal_direction: SignalDirection
    confidence: float
    timestamp: datetime
    bid_size: float = 0.0
    ask_size: float = 0.0

    @property
    def is_skewed_positive(self) -> bool:
        """Check if skew is positive."""
        return self.skew_value > 0

    @property
    def significance(self) -> str:
        """Get skew significance."""
        ratio = abs(self.skew_value) / self.threshold if self.threshold > 0 else 0
        if ratio > 2.0:
            return "highly_significant"
        elif ratio > 1.0:
            return "significant"
        else:
            return "not_significant"


@dataclass
class LiquiditySignal:
    """Liquidity detection signal structure."""
    symbol: str
    liquidity_level: float  # 0.0 to 1.0
    thin_areas: List[Tuple[float, float]] = field(default_factory=list)  # (price, severity)
    accumulation_zones: List[Tuple[float, float]] = field(default_factory=list)  # (price, strength)
    distribution_zones: List[Tuple[float, float]] = field(default_factory=list)  # (price, strength)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def is_liquid(self) -> bool:
        """Check if market is liquid."""
        return self.liquidity_level > 0.7

    @property
    def has_thin_liquidity(self) -> bool:
        """Check if there are thin liquidity areas."""
        return len(self.thin_areas) > 0


@dataclass
class FusedSignal:
    """Fused trading signal structure."""
    symbol: str
    primary_direction: SignalDirection
    overall_confidence: float
    component_signals: Dict[str, TradingSignal] = field(default_factory=dict)
    weights: Dict[str, float] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def consensus_strength(self) -> float:
        """Get consensus strength among component signals."""
        if not self.component_signals:
            return 0.0

        directions = [s.direction for s in self.component_signals.values()]
        primary_count = directions.count(self.primary_direction)
        return primary_count / len(directions)

    @property
    def signal_quality(self) -> str:
        """Get signal quality assessment."""
        if self.overall_confidence > 0.8 and self.consensus_strength > 0.8:
            return "excellent"
        elif self.overall_confidence > 0.6 and self.consensus_strength > 0.6:
            return "good"
        elif self.overall_confidence > 0.4:
            return "moderate"
        else:
            return "poor"


@dataclass
class RiskMetrics:
    """Risk metrics structure."""
    symbol: str
    position_size: float
    max_position_size: float
    leverage: float
    var_95_1d: float  # Value at Risk 95% 1-day
    beta: float
    correlation_risk: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def utilization(self) -> float:
        """Get position size utilization."""
        return self.position_size / self.max_position_size if self.max_position_size > 0 else 0.0

    @property
    def risk_level(self) -> str:
        """Get risk level assessment."""
        if self.utilization > 0.9 or self.var_95_1d > 0.05:
            return "high"
        elif self.utilization > 0.7 or self.var_95_1d > 0.03:
            return "medium"
        else:
            return "low"


@dataclass
class PositionSize:
    """Position sizing recommendation structure."""
    symbol: str
    recommended_size: float
    max_size: float
    risk_adjusted_size: float
    liquidity_adjusted_size: float
    final_size: float
    reasoning: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def size_confidence(self) -> str:
        """Get size recommendation confidence."""
        if len(self.reasoning) >= 3:
            return "high"
        elif len(self.reasoning) >= 2:
            return "medium"
        else:
            return "low"