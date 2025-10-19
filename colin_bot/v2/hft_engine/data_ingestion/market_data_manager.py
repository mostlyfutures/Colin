"""
High-Frequency Trading Data Manager

Central manager for real-time market data ingestion and distribution.
Handles multiple data sources, data validation, and real-time processing.
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, List, Set, Optional, Callable, AsyncGenerator
from dataclasses import dataclass, field
from collections import defaultdict
import time

from ..utils.data_structures import OrderBook, Trade, MarketEvent, EventType
from ..utils.performance import LatencyTracker, PerformanceMonitor
from .connectors.mock_connector import MockDataConnector, MockMarketConfig


@dataclass
class DataFeedConfig:
    """Configuration for market data feeds."""
    symbols: List[str]
    data_sources: List[str] = field(default_factory=lambda: ['mock'])
    update_frequency: float = 100.0  # Hz
    enable_validation: bool = True
    enable_monitoring: bool = True
    max_lag_ms: float = 50.0  # Maximum acceptable data lag


@dataclass
class DataFeedStatus:
    """Status of a data feed."""
    feed_id: str
    symbol: str
    is_active: bool
    last_update: datetime
    messages_per_second: float
    latency_ms: float
    error_count: int = 0
    total_messages: int = 0


class HFTDataManager:
    """
    High-Frequency Trading Data Manager.

    Central hub for managing real-time market data from multiple sources,
    validating data quality, and distributing to downstream components.
    """

    def __init__(self, config: DataFeedConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Data storage
        self.order_books: Dict[str, OrderBook] = {}
        self.trades: Dict[str, List[Trade]] = defaultdict(list)
        self.events: List[MarketEvent] = []

        # Data connectors
        self.connectors: Dict[str, MockDataConnector] = {}
        self.connector_tasks: Dict[str, asyncio.Task] = {}

        # Subscribers
        self.order_book_subscribers: Dict[str, List[Callable]] = defaultdict(list)
        self.trade_subscribers: Dict[str, List[Callable]] = defaultdict(list)
        self.event_subscribers: List[Callable] = []

        # Performance monitoring
        self.latency_tracker = LatencyTracker()
        self.performance_monitor = PerformanceMonitor()
        self.feed_status: Dict[str, DataFeedStatus] = {}

        # State
        self.is_running = False
        self.start_time: Optional[datetime] = None

    async def initialize(self):
        """Initialize data manager and connectors."""
        self.logger.info("Initializing HFT Data Manager")

        # Initialize connectors for each symbol
        for symbol in self.config.symbols:
            await self._initialize_symbol_connector(symbol)

        self.logger.info(f"Initialized {len(self.connectors)} data connectors")

    async def _initialize_symbol_connector(self, symbol: str):
        """Initialize connector for a specific symbol."""
        # Create mock connector with realistic configuration
        mock_config = MockMarketConfig(
            symbol=symbol,
            base_price=self._get_base_price(symbol),
            volatility=0.001,
            tick_size=0.01,
            order_book_depth=20,
            trade_frequency=20.0,
            order_book_update_frequency=self.config.update_frequency
        )

        connector = MockDataConnector(mock_config)
        connector_id = f"{symbol}_mock"

        self.connectors[connector_id] = connector

        # Initialize feed status
        self.feed_status[connector_id] = DataFeedStatus(
            feed_id=connector_id,
            symbol=symbol,
            is_active=False,
            last_update=datetime.now(timezone.utc),
            messages_per_second=0.0,
            latency_ms=0.0
        )

        self.logger.info(f"Initialized connector for {symbol}")

    def _get_base_price(self, symbol: str) -> float:
        """Get base price for symbol."""
        # Realistic base prices for different assets
        base_prices = {
            'BTC/USDT': 67000.0,
            'ETH/USDT': 3980.0,
            'SOL/USDT': 180.0,
            'ADA/USDT': 0.38,
            'DOT/USDT': 7.5,
            'MATIC/USDT': 0.85,
            'AVAX/USDT': 42.0,
            'LINK/USDT': 15.5
        }
        return base_prices.get(symbol, 100.0)

    async def start(self):
        """Start all data feeds."""
        if self.is_running:
            self.logger.warning("Data manager is already running")
            return

        self.logger.info("Starting HFT Data Manager")
        self.is_running = True
        self.start_time = datetime.now(timezone.utc)

        # Start connectors
        for connector_id, connector in self.connectors.items():
            await connector.start()
            await self._start_connector_tasks(connector_id, connector)

        # Start monitoring
        if self.config.enable_monitoring:
            asyncio.create_task(self._monitor_feeds())

        self.logger.info("HFT Data Manager started successfully")

    async def stop(self):
        """Stop all data feeds."""
        if not self.is_running:
            return

        self.logger.info("Stopping HFT Data Manager")
        self.is_running = False

        # Stop connector tasks
        for task in self.connector_tasks.values():
            task.cancel()

        # Wait for tasks to finish
        await asyncio.gather(*self.connector_tasks.values(), return_exceptions=True)

        # Stop connectors
        for connector in self.connectors.values():
            await connector.stop()

        self.connector_tasks.clear()
        self.logger.info("HFT Data Manager stopped")

    async def _start_connector_tasks(self, connector_id: str, connector: MockDataConnector):
        """Start tasks for a connector."""
        symbol = connector.config.symbol

        # Order book streaming task
        orderbook_task = asyncio.create_task(
            self._stream_order_book_data(connector_id, connector)
        )
        self.connector_tasks[f"{connector_id}_orderbook"] = orderbook_task

        # Trade streaming task
        trade_task = asyncio.create_task(
            self._stream_trade_data(connector_id, connector)
        )
        self.connector_tasks[f"{connector_id}_trade"] = trade_task

        self.logger.debug(f"Started connector tasks for {symbol}")

    async def _stream_order_book_data(self, connector_id: str, connector: MockDataConnector):
        """Stream order book data from connector."""
        symbol = connector.config.symbol

        async for order_book in connector.stream_order_book_updates():
            if not self.is_running:
                break

            # Track latency
            update_time = datetime.now(timezone.utc)
            self.latency_tracker.record_latency(
                f"{symbol}_orderbook",
                order_book.timestamp,
                update_time
            )

            # Validate data
            if self.config.enable_validation:
                if not self._validate_order_book(order_book):
                    self.logger.warning(f"Invalid order book for {symbol}")
                    continue

            # Store order book
            self.order_books[symbol] = order_book

            # Update feed status
            status = self.feed_status[connector_id]
            status.last_update = update_time
            status.total_messages += 1

            # Notify subscribers
            await self._notify_order_book_subscribers(symbol, order_book)

            # Create market event
            event = MarketEvent(
                event_type=EventType.ORDER_BOOK_UPDATE,
                symbol=symbol,
                timestamp=update_time,
                data={'order_book': order_book, 'connector_id': connector_id}
            )
            await self._notify_event_subscribers(event)

    async def _stream_trade_data(self, connector_id: str, connector: MockDataConnector):
        """Stream trade data from connector."""
        symbol = connector.config.symbol

        async for trade in connector.stream_trades():
            if not self.is_running:
                break

            # Track latency
            update_time = datetime.now(timezone.utc)
            self.latency_tracker.record_latency(
                f"{symbol}_trade",
                trade.timestamp,
                update_time
            )

            # Validate data
            if self.config.enable_validation:
                if not self._validate_trade(trade):
                    self.logger.warning(f"Invalid trade for {symbol}")
                    continue

            # Store trade
            self.trades[symbol].append(trade)

            # Keep only recent trades (last 1000)
            if len(self.trades[symbol]) > 1000:
                self.trades[symbol] = self.trades[symbol][-1000:]

            # Update feed status
            status = self.feed_status[connector_id]
            status.last_update = update_time
            status.total_messages += 1

            # Notify subscribers
            await self._notify_trade_subscribers(symbol, trade)

            # Create market event
            event = MarketEvent(
                event_type=EventType.TRADE,
                symbol=symbol,
                timestamp=update_time,
                data={'trade': trade, 'connector_id': connector_id}
            )
            await self._notify_event_subscribers(event)

    def _validate_order_book(self, order_book: OrderBook) -> bool:
        """Validate order book data."""
        # Check if there are bids and asks
        if not order_book.bids or not order_book.asks:
            return False

        # Check price ordering
        if order_book.bids[0].price >= order_book.asks[0].price:
            return False

        # Check for negative prices or sizes
        for bid in order_book.bids[:5]:  # Check top 5 levels
            if bid.price <= 0 or bid.size <= 0:
                return False

        for ask in order_book.asks[:5]:  # Check top 5 levels
            if ask.price <= 0 or ask.size <= 0:
                return False

        return True

    def _validate_trade(self, trade: Trade) -> bool:
        """Validate trade data."""
        # Check price and size
        if trade.price <= 0 or trade.size <= 0:
            return False

        # Check timestamp (not too old)
        age = (datetime.now(timezone.utc) - trade.timestamp).total_seconds()
        if age > 60:  # Trade is more than 1 minute old
            return False

        return True

    async def _notify_order_book_subscribers(self, symbol: str, order_book: OrderBook):
        """Notify all order book subscribers for a symbol."""
        subscribers = self.order_book_subscribers.get(symbol, [])
        for subscriber in subscribers:
            try:
                if asyncio.iscoroutinefunction(subscriber):
                    await subscriber(order_book)
                else:
                    subscriber(order_book)
            except Exception as e:
                self.logger.error(f"Error notifying order book subscriber: {e}")

    async def _notify_trade_subscribers(self, symbol: str, trade: Trade):
        """Notify all trade subscribers for a symbol."""
        subscribers = self.trade_subscribers.get(symbol, [])
        for subscriber in subscribers:
            try:
                if asyncio.iscoroutinefunction(subscriber):
                    await subscriber(trade)
                else:
                    subscriber(trade)
            except Exception as e:
                self.logger.error(f"Error notifying trade subscriber: {e}")

    async def _notify_event_subscribers(self, event: MarketEvent):
        """Notify all event subscribers."""
        for subscriber in self.event_subscribers:
            try:
                if asyncio.iscoroutinefunction(subscriber):
                    await subscriber(event)
                else:
                    subscriber(event)
            except Exception as e:
                self.logger.error(f"Error notifying event subscriber: {e}")

    async def _monitor_feeds(self):
        """Monitor data feed health and performance."""
        while self.is_running:
            try:
                await self._update_feed_metrics()
                await self._check_feed_health()
                await asyncio.sleep(5.0)  # Monitor every 5 seconds
            except Exception as e:
                self.logger.error(f"Error in feed monitoring: {e}")
                await asyncio.sleep(5.0)

    async def _update_feed_metrics(self):
        """Update feed performance metrics."""
        current_time = datetime.now(timezone.utc)

        for connector_id, status in self.feed_status.items():
            # Calculate messages per second
            time_diff = (current_time - status.last_update).total_seconds()
            if time_diff > 0:
                # This is a simplified calculation
                status.messages_per_second = status.total_messages / max(1, time_diff)

            # Update latency metrics
            avg_latency = self.latency_tracker.get_average_latency(f"{status.symbol}_orderbook")
            status.latency_ms = avg_latency * 1000 if avg_latency else 0

            # Update performance monitor
            self.performance_monitor.record_metric(
                f"{status.symbol}_feed_latency_ms",
                status.latency_ms
            )
            self.performance_monitor.record_metric(
                f"{status.symbol}_messages_per_second",
                status.messages_per_second
            )

    async def _check_feed_health(self):
        """Check health of data feeds."""
        current_time = datetime.now(timezone.utc)

        for connector_id, status in self.feed_status.items():
            # Check for stale data
            age_seconds = (current_time - status.last_update).total_seconds()
            if age_seconds > 10:  # No update in 10 seconds
                self.logger.warning(f"Stale data detected for {status.symbol}")
                status.error_count += 1

            # Check for high latency
            if status.latency_ms > self.config.max_lag_ms:
                self.logger.warning(f"High latency for {status.symbol}: {status.latency_ms:.2f}ms")
                status.error_count += 1

            # Check for low message rate
            if status.messages_per_second < 1.0:
                self.logger.warning(f"Low message rate for {status.symbol}: {status.messages_per_second:.2f}/s")
                status.error_count += 1

    def subscribe_order_book(self, symbol: str, callback: Callable):
        """Subscribe to order book updates for a symbol."""
        self.order_book_subscribers[symbol].append(callback)
        self.logger.debug(f"Added order book subscriber for {symbol}")

    def subscribe_trades(self, symbol: str, callback: Callable):
        """Subscribe to trade updates for a symbol."""
        self.trade_subscribers[symbol].append(callback)
        self.logger.debug(f"Added trade subscriber for {symbol}")

    def subscribe_events(self, callback: Callable):
        """Subscribe to all market events."""
        self.event_subscribers.append(callback)
        self.logger.debug("Added event subscriber")

    def unsubscribe_order_book(self, symbol: str, callback: Callable):
        """Unsubscribe from order book updates."""
        if callback in self.order_book_subscribers[symbol]:
            self.order_book_subscribers[symbol].remove(callback)
            self.logger.debug(f"Removed order book subscriber for {symbol}")

    def unsubscribe_trades(self, symbol: str, callback: Callable):
        """Unsubscribe from trade updates."""
        if callback in self.trade_subscribers[symbol]:
            self.trade_subscribers[symbol].remove(callback)
            self.logger.debug(f"Removed trade subscriber for {symbol}")

    def unsubscribe_events(self, callback: Callable):
        """Unsubscribe from market events."""
        if callback in self.event_subscribers:
            self.event_subscribers.remove(callback)
            self.logger.debug("Removed event subscriber")

    def get_order_book(self, symbol: str) -> Optional[OrderBook]:
        """Get current order book for a symbol."""
        return self.order_books.get(symbol)

    def get_recent_trades(self, symbol: str, limit: int = 100) -> List[Trade]:
        """Get recent trades for a symbol."""
        trades = self.trades.get(symbol, [])
        return trades[-limit:] if trades else []

    def get_feed_status(self) -> Dict[str, DataFeedStatus]:
        """Get status of all data feeds."""
        return self.feed_status.copy()

    def get_performance_metrics(self) -> Dict[str, any]:
        """Get performance metrics."""
        return {
            'latency_stats': self.latency_tracker.get_statistics(),
            'performance_metrics': self.performance_monitor.get_metrics(),
            'uptime_seconds': (datetime.now(timezone.utc) - self.start_time).total_seconds() if self.start_time else 0
        }

    async def get_symbol_summary(self, symbol: str) -> Dict[str, any]:
        """Get summary data for a symbol."""
        order_book = self.get_order_book(symbol)
        recent_trades = self.get_recent_trades(symbol, limit=20)

        summary = {
            'symbol': symbol,
            'timestamp': datetime.now(timezone.utc),
            'order_book': None,
            'recent_trades_count': len(recent_trades),
            'trade_volume_24h': sum(trade.size for trade in recent_trades),
            'last_trade_price': recent_trades[-1].price if recent_trades else None,
            'price_change': 0.0,
            'spread': 0.0,
            'mid_price': 0.0
        }

        if order_book:
            summary['order_book'] = {
                'best_bid': order_book.best_bid,
                'best_ask': order_book.best_ask,
                'bid_size': order_book.total_bid_size,
                'ask_size': order_book.total_ask_size,
                'spread': order_book.spread,
                'mid_price': order_book.mid_price
            }
            summary['spread'] = order_book.spread or 0.0
            summary['mid_price'] = order_book.mid_price or 0.0

        return summary