"""
Real Data Connector for HFT Engine

This connector connects to live cryptocurrency exchanges to fetch real-time
market data for HFT signal generation.
"""

import asyncio
import aiohttp
import json
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from loguru import logger
import os
from decimal import Decimal

from .mock_connector import MockDataConnector, MockMarketConfig
from ...utils.data_structures import OrderBook, OrderBookLevel, Trade, MarketEvent, EventType


@dataclass
class RealMarketConfig:
    """Configuration for real market data connector."""
    exchange: str = "binance"
    symbols: List[str] = field(default_factory=lambda: ["BTCUSDT", "ETHUSDT", "SOLUSDT"])
    update_interval_ms: int = 100  # 100ms update interval for HFT
    max_order_book_levels: int = 20
    enable_websocket: bool = True
    enable_rest_fallback: bool = True
    rate_limit_per_second: int = 10


class RealDataConnector(MockDataConnector):
    """
    Real-time market data connector for cryptocurrency exchanges.

    Supports multiple exchanges and provides real-time order book and trade data.
    """

    def __init__(self, config_manager):
        """
        Initialize real data connector.

        Args:
            config_manager: Configuration manager instance
        """
        # Create a mock market config for the parent class
        mock_config = MockMarketConfig(
            symbol="BTCUSDT",  # Default symbol
            base_price=getattr(config_manager.config, 'base_price', 50000.0)
        )
        super().__init__(mock_config)

        # Real data configuration
        self.config = RealMarketConfig()
        self.api_base = self._get_api_base()

        # Session management
        self.session: Optional[aiohttp.ClientSession] = None
        self.ws_connection: Optional[aiohttp.ClientWebSocketResponse] = None

        # Data storage
        self.order_books: Dict[str, OrderBook] = {}
        self.last_update: Dict[str, float] = {}
        self.trade_history: Dict[str, List[Trade]] = {}

        # Rate limiting
        self.request_timestamps: List[float] = []
        self.rate_limit_per_second = self.config.rate_limit_per_second

        # Connection status
        self.is_connected = False
        self.connection_attempts = 0
        self.max_connection_attempts = 5

        logger.info(f"Initialized real data connector for {self.config.exchange}")

    def _get_api_base(self) -> str:
        """Get API base URL for the configured exchange."""
        apis = {
            "binance": "https://api.binance.com",
            "bybit": "https://api.bybit.com",
            "okx": "https://www.okx.com",
            "deribit": "https://api.deribit.com"
        }
        return apis.get(self.config.exchange, apis["binance"])

    async def initialize(self):
        """Initialize the real data connector."""
        try:
            # Create HTTP session
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=10),
                headers={
                    'User-Agent': 'ColinHFTBot/1.0'
                }
            )

            # Test API connectivity
            await self._test_api_connectivity()

            # Initialize order books
            await self._initialize_order_books()

            # Start WebSocket connection if enabled
            if self.config.enable_websocket:
                await self._start_websocket_connection()

            self.is_connected = True
            logger.info(f"✅ Real data connector initialized for {self.config.exchange}")

        except Exception as e:
            logger.error(f"❌ Failed to initialize real data connector: {e}")
            # Fall back to mock data if REST is enabled
            if self.config.enable_rest_fallback:
                logger.warning("⚠️ Falling back to REST API only")
                await super().initialize()
            else:
                raise

    async def _test_api_connectivity(self):
        """Test API connectivity."""
        try:
            # Use server time endpoint to test connectivity
            url = f"{self.api_base}/api/v3/time"

            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    logger.debug(f"✅ API connectivity test successful: {data}")
                else:
                    raise Exception(f"API test failed with status {response.status}")

        except Exception as e:
            raise Exception(f"API connectivity test failed: {e}")

    async def _initialize_order_books(self):
        """Initialize order books for all symbols."""
        for symbol in self.config.symbols:
            try:
                # Fetch initial order book snapshot
                order_book = await self._fetch_order_book_snapshot(symbol)
                if order_book:
                    self.order_books[symbol] = order_book
                    self.last_update[symbol] = time.time()
                    logger.debug(f"✅ Initialized order book for {symbol}")
                else:
                    logger.warning(f"⚠️ Failed to initialize order book for {symbol}")

            except Exception as e:
                logger.error(f"❌ Error initializing order book for {symbol}: {e}")

    async def _fetch_order_book_snapshot(self, symbol: str) -> Optional[OrderBook]:
        """Fetch order book snapshot via REST API."""
        try:
            await self._check_rate_limit()

            # Binance order book endpoint
            url = f"{self.api_base}/api/v3/depth"
            params = {
                'symbol': symbol,
                'limit': self.config.max_order_book_levels
            }

            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return self._parse_order_book_response(symbol, data)
                else:
                    logger.error(f"Order book request failed for {symbol}: {response.status}")
                    return None

        except Exception as e:
            logger.error(f"Error fetching order book for {symbol}: {e}")
            return None

    def _parse_order_book_response(self, symbol: str, data: Dict) -> OrderBook:
        """Parse order book response from exchange API."""
        try:
            # Parse bids and asks
            bids = [
                OrderBookLevel(
                    price=float(price),
                    size=float(quantity)
                )
                for price, quantity in data.get('bids', [])
            ]

            asks = [
                OrderBookLevel(
                    price=float(price),
                    size=float(quantity)
                )
                for price, quantity in data.get('asks', [])
            ]

            return OrderBook(
                symbol=symbol,
                timestamp=datetime.now(timezone.utc),
                exchange=self.config.exchange,
                bids=bids,
                asks=asks
            )

        except Exception as e:
            logger.error(f"Error parsing order book response for {symbol}: {e}")
            raise

    async def _start_websocket_connection(self):
        """Start WebSocket connection for real-time data."""
        try:
            # Binance WebSocket URL
            ws_url = "wss://stream.binance.com:9443/ws"

            # Create symbols stream
            streams = []
            for symbol in self.config.symbols:
                streams.append(f"{symbol.lower()}@depth")
                streams.append(f"{symbol.lower()}@trade")

            ws_url += f"/stream?streams={'/'.join(streams)}"

            self.ws_connection = await self.session.ws_connect(ws_url)

            # Start WebSocket message handler
            asyncio.create_task(self._handle_websocket_messages())

            logger.info(f"✅ WebSocket connection established for {len(self.config.symbols)} symbols")

        except Exception as e:
            logger.error(f"❌ WebSocket connection failed: {e}")
            # Continue with REST polling if WebSocket fails
            if self.config.enable_rest_fallback:
                logger.warning("⚠️ Using REST API polling fallback")
                asyncio.create_task(self._rest_polling_loop())

    async def _handle_websocket_messages(self):
        """Handle WebSocket messages for real-time updates."""
        try:
            while self.ws_connection and not self.ws_connection.closed:
                message = await self.ws_connection.receive_json()
                await self._process_websocket_message(message)

        except Exception as e:
            logger.error(f"WebSocket message handling error: {e}")
            # Attempt reconnection
            if self.connection_attempts < self.max_connection_attempts:
                self.connection_attempts += 1
                await asyncio.sleep(2 ** self.connection_attempts)  # Exponential backoff
                await self._start_websocket_connection()

    async def _process_websocket_message(self, message: Dict):
        """Process incoming WebSocket message."""
        try:
            if 'stream' in message and 'data' in message:
                stream = message['stream']
                data = message['data']

                if 'depth' in stream:
                    # Order book update
                    symbol = stream.split('@')[0].upper()
                    await self._update_order_book(symbol, data)

                elif 'trade' in stream:
                    # Trade update
                    symbol = stream.split('@')[0].upper()
                    await self._process_trade(symbol, data)

        except Exception as e:
            logger.error(f"Error processing WebSocket message: {e}")

    async def _update_order_book(self, symbol: str, data: Dict):
        """Update order book with WebSocket data."""
        try:
            if symbol in self.order_books:
                order_book = self.order_books[symbol]

                # Update bids
                if 'b' in data:
                    new_bids = [
                        OrderBookLevel(price=float(price), size=float(quantity))
                        for price, quantity in data['b']
                    ]
                    order_book.bids = new_bids[:self.config.max_order_book_levels]

                # Update asks
                if 'a' in data:
                    new_asks = [
                        OrderBookLevel(price=float(price), size=float(quantity))
                        for price, quantity in data['a']
                    ]
                    order_book.asks = new_asks[:self.config.max_order_book_levels]

                # Update timestamp
                order_book.timestamp = datetime.now(timezone.utc)
                self.last_update[symbol] = time.time()

                logger.debug(f"📊 Updated order book for {symbol}")

        except Exception as e:
            logger.error(f"Error updating order book for {symbol}: {e}")

    async def _process_trade(self, symbol: str, data: Dict):
        """Process trade data."""
        try:
            trade = Trade(
                symbol=symbol,
                timestamp=datetime.fromtimestamp(data['T'] / 1000, timezone.utc),
                price=float(data['p']),
                quantity=float(data['q']),
                side=data['m'].lower(),  # m is True for sell
                exchange=self.config.exchange,
                trade_id=str(data.get('t', ''))
            )

            # Add to trade history
            if symbol not in self.trade_history:
                self.trade_history[symbol] = []

            self.trade_history[symbol].append(trade)

            # Keep only last 1000 trades
            if len(self.trade_history[symbol]) > 1000:
                self.trade_history[symbol] = self.trade_history[symbol][-1000:]

            logger.debug(f"💰 Processed trade for {symbol}: {trade.price} {trade.quantity}")

        except Exception as e:
            logger.error(f"Error processing trade for {symbol}: {e}")

    async def _rest_polling_loop(self):
        """REST API polling loop as fallback."""
        while True:
            try:
                for symbol in self.config.symbols:
                    await self._fetch_order_book_snapshot(symbol)

                await asyncio.sleep(self.config.update_interval_ms / 1000)

            except Exception as e:
                logger.error(f"REST polling error: {e}")
                await asyncio.sleep(5)  # Wait before retrying

    async def _check_rate_limit(self):
        """Check and enforce rate limiting."""
        current_time = time.time()

        # Remove old timestamps (older than 1 second)
        self.request_timestamps = [
            ts for ts in self.request_timestamps
            if current_time - ts < 1.0
        ]

        # Check if we're at the rate limit
        if len(self.request_timestamps) >= self.rate_limit_per_second:
            sleep_time = 1.0 - (current_time - self.request_timestamps[0])
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)

        # Add current timestamp
        self.request_timestamps.append(current_time)

    async def generate_order_book(self, symbol: str) -> OrderBook:
        """
        Generate real order book data.

        Args:
            symbol: Trading symbol

        Returns:
            Real order book data
        """
        if not self.is_connected:
            logger.warning(f"⚠️ Connector not connected, using mock data for {symbol}")
            return await super().generate_order_book(symbol)

        try:
            # Return current order book if available
            if symbol in self.order_books:
                # Check if data is fresh (within last 5 seconds)
                if time.time() - self.last_update.get(symbol, 0) < 5:
                    return self.order_books[symbol]
                else:
                    # Fetch fresh data
                    fresh_book = await self._fetch_order_book_snapshot(symbol)
                    if fresh_book:
                        self.order_books[symbol] = fresh_book
                        self.last_update[symbol] = time.time()
                        return fresh_book
            else:
                # Fetch initial data
                order_book = await self._fetch_order_book_snapshot(symbol)
                if order_book:
                    self.order_books[symbol] = order_book
                    self.last_update[symbol] = time.time()
                    return order_book

            # Fall back to mock data if real data unavailable
            logger.warning(f"⚠️ Using mock data for {symbol}")
            return await super().generate_order_book(symbol)

        except Exception as e:
            logger.error(f"Error generating order book for {symbol}: {e}")
            # Fall back to mock data
            return await super().generate_order_book(symbol)

    async def generate_trade(self, symbol: str) -> Optional[Trade]:
        """
        Generate real trade data.

        Args:
            symbol: Trading symbol

        Returns:
            Real trade data or None
        """
        if not self.is_connected:
            return None

        try:
            # Return recent trade if available
            if symbol in self.trade_history and self.trade_history[symbol]:
                return self.trade_history[symbol][-1]  # Return most recent trade

        except Exception as e:
            logger.error(f"Error generating trade for {symbol}: {e}")

        return None

    async def generate_market_event(self, symbol: str) -> Optional[MarketEvent]:
        """
        Generate market event from real data.

        Args:
            symbol: Trading symbol

        Returns:
            Market event or None
        """
        if not self.is_connected:
            return None

        try:
            # Create market event from order book data
            order_book = await self.generate_order_book(symbol)
            if order_book:
                return MarketEvent(
                    symbol=symbol,
                    timestamp=order_book.timestamp,
                    event_type=EventType.ORDER_BOOK_UPDATE,
                    data={
                        'best_bid': order_book.best_bid,
                        'best_ask': order_book.best_ask,
                        'spread': order_book.spread,
                        'bid_volume': sum(level.size for level in order_book.bids[:5]),
                        'ask_volume': sum(level.size for level in order_book.asks[:5])
                    }
                )

        except Exception as e:
            logger.error(f"Error generating market event for {symbol}: {e}")

        return None

    async def cleanup(self):
        """Cleanup resources."""
        try:
            if self.ws_connection and not self.ws_connection.closed:
                await self.ws_connection.close()

            if self.session and not self.session.closed:
                await self.session.close()

            self.is_connected = False
            logger.info("✅ Real data connector cleaned up")

        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

    def get_connection_status(self) -> Dict[str, Any]:
        """Get connection status."""
        return {
            'is_connected': self.is_connected,
            'exchange': self.config.exchange,
            'symbols': self.config.symbols,
            'update_interval_ms': self.config.update_interval_ms,
            'websocket_enabled': self.config.enable_websocket,
            'rest_fallback': self.config.enable_rest_fallback,
            'connection_attempts': self.connection_attempts,
            'last_updates': self.last_update.copy()
        }