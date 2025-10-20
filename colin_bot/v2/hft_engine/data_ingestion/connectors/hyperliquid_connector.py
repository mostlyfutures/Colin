"""
Hyperliquid HFT Connector

High-frequency trading connector for Hyperliquid exchange.
Provides real-time market data streaming and WebSocket connectivity.
"""

import asyncio
import json
import websockets
import aiohttp
from datetime import datetime, timezone
from typing import List, Dict, AsyncGenerator, Optional, Any
from dataclasses import dataclass
from loguru import logger

from ...utils.data_structures import OrderBook, OrderBookLevel, Trade, MarketEvent, EventType, OrderSide


@dataclass
class HyperliquidConfig:
    """Configuration for Hyperliquid HFT connector."""
    symbols: List[str]
    websocket_url: str = "wss://api.hyperliquid.xyz/ws"
    rest_api_url: str = "https://api.hyperliquid.xyz/info"
    subscription_types: List[str] = None
    reconnect_attempts: int = 5
    reconnect_delay: float = 1.0
    ping_interval: int = 30
    max_reconnect_delay: float = 60.0

    def __post_init__(self):
        """Initialize default subscription types."""
        if self.subscription_types is None:
            self.subscription_types = ["trades", "l2Book"]


class HyperliquidConnector:
    """
    High-frequency trading connector for Hyperliquid.

    Provides real-time market data streaming via WebSocket and REST API fallback.
    Supports order book updates, trades, and other market events.
    """

    def __init__(self, config: HyperliquidConfig):
        """
        Initialize Hyperliquid connector.

        Args:
            config: Connector configuration
        """
        self.config = config
        self.websocket = None
        self.session = None
        self.is_running = False
        self.is_connected = False
        self._reconnect_count = 0
        self._subscriptions = {}
        self._message_handlers = {}
        self._last_ping_time = None
        self._connection_task = None

        # Symbol mapping
        self._symbol_map = {symbol: symbol.upper() for symbol in config.symbols}

        logger.info(f"Hyperliquid HFT connector initialized for symbols: {config.symbols}")

    async def initialize(self):
        """Initialize the connector."""
        try:
            # Create HTTP session for REST fallback
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=10),
                headers={"User-Agent": "Colin-HFT-Connector/1.0"}
            )
            logger.info("Hyperliquid connector HTTP session initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Hyperliquid connector: {e}")
            raise

    async def connect(self):
        """Connect to Hyperliquid WebSocket."""
        try:
            logger.info(f"Connecting to Hyperliquid WebSocket: {self.config.websocket_url}")

            self.websocket = await websockets.connect(
                self.config.websocket_url,
                ping_interval=self.config.ping_interval,
                ping_timeout=10,
                close_timeout=10
            )

            self.is_connected = True
            self._reconnect_count = 0
            logger.info("Hyperliquid WebSocket connected successfully")

            # Start connection monitoring
            self._connection_task = asyncio.create_task(self._monitor_connection())

        except Exception as e:
            logger.error(f"Failed to connect to Hyperliquid WebSocket: {e}")
            await self._handle_connection_error(e)

    async def disconnect(self):
        """Disconnect from Hyperliquid WebSocket."""
        try:
            self.is_running = False
            self.is_connected = False

            if self._connection_task:
                self._connection_task.cancel()
                try:
                    await self._connection_task
                except asyncio.CancelledError:
                    pass

            if self.websocket:
                await self.websocket.close()
                self.websocket = None

            if self.session:
                await self.session.close()
                self.session = None

            logger.info("Hyperliquid connector disconnected")

        except Exception as e:
            logger.error(f"Error during disconnect: {e}")

    async def stream_order_book_updates(self, symbol: str) -> AsyncGenerator[OrderBook, None]:
        """
        Stream order book updates for a symbol.

        Args:
            symbol: Trading symbol

        Yields:
            OrderBook: Updated order book data
        """
        if symbol not in self._symbol_map:
            raise ValueError(f"Symbol {symbol} not configured")

        hl_symbol = self._symbol_map[symbol]
        subscription_id = f"l2Book_{hl_symbol}"

        try:
            # Subscribe to order book updates
            await self._subscribe_to_order_book(hl_symbol)

            # Stream updates
            while self.is_running and self.is_connected:
                try:
                    message = await self._wait_for_subscription_data(subscription_id, timeout=1.0)
                    if message:
                        order_book = self._parse_order_book_message(message, symbol)
                        if order_book:
                            yield order_book
                except asyncio.TimeoutError:
                    # Timeout is expected, continue loop
                    continue
                except Exception as e:
                    logger.error(f"Error processing order book update for {symbol}: {e}")
                    await asyncio.sleep(0.1)

        except Exception as e:
            logger.error(f"Error in order book stream for {symbol}: {e}")
            raise

    async def stream_trades(self, symbol: str) -> AsyncGenerator[Trade, None]:
        """
        Stream trade data for a symbol.

        Args:
            symbol: Trading symbol

        Yields:
            Trade: Trade data
        """
        if symbol not in self._symbol_map:
            raise ValueError(f"Symbol {symbol} not configured")

        hl_symbol = self._symbol_map[symbol]
        subscription_id = f"trades_{hl_symbol}"

        try:
            # Subscribe to trades
            await self._subscribe_to_trades(hl_symbol)

            # Stream trades
            while self.is_running and self.is_connected:
                try:
                    message = await self._wait_for_subscription_data(subscription_id, timeout=1.0)
                    if message:
                        trades = self._parse_trades_message(message, symbol)
                        for trade in trades:
                            yield trade
                except asyncio.TimeoutError:
                    # Timeout is expected, continue loop
                    continue
                except Exception as e:
                    logger.error(f"Error processing trade update for {symbol}: {e}")
                    await asyncio.sleep(0.1)

        except Exception as e:
            logger.error(f"Error in trade stream for {symbol}: {e}")
            raise

    async def _subscribe_to_order_book(self, symbol: str):
        """Subscribe to order book updates."""
        subscription_msg = {
            "method": "subscribe",
            "subscription": {
                "type": "l2Book",
                "coin": symbol
            }
        }
        await self._send_message(subscription_msg)
        logger.info(f"Subscribed to order book updates for {symbol}")

    async def _subscribe_to_trades(self, symbol: str):
        """Subscribe to trades."""
        subscription_msg = {
            "method": "subscribe",
            "subscription": {
                "type": "trades",
                "coin": symbol
            }
        }
        await self._send_message(subscription_msg)
        logger.info(f"Subscribed to trades for {symbol}")

    async def _send_message(self, message: Dict[str, Any]):
        """Send message to WebSocket."""
        if self.websocket and self.is_connected:
            try:
                await self.websocket.send(json.dumps(message))
            except Exception as e:
                logger.error(f"Failed to send message: {e}")
                await self._handle_connection_error(e)

    async def _wait_for_subscription_data(self, subscription_id: str, timeout: float) -> Optional[Dict]:
        """Wait for data from specific subscription."""
        # This is a simplified implementation
        # In production, you'd want a more sophisticated message routing system
        if self.websocket and self.is_connected:
            try:
                message = await asyncio.wait_for(self.websocket.recv(), timeout=timeout)
                data = json.loads(message)

                # Check if message is for our subscription
                if self._is_subscription_message(data, subscription_id):
                    return data

            except asyncio.TimeoutError:
                return None
            except Exception as e:
                logger.error(f"Error waiting for subscription data: {e}")
                return None

        return None

    def _is_subscription_message(self, message: Dict, subscription_id: str) -> bool:
        """Check if message matches subscription."""
        # Simplified subscription matching
        # In production, implement proper subscription ID tracking
        return "data" in message

    def _parse_order_book_message(self, message: Dict, symbol: str) -> Optional[OrderBook]:
        """Parse order book message from Hyperliquid."""
        try:
            if "data" not in message:
                return None

            data = message["data"]
            if "levels" not in data:
                return None

            levels = data["levels"]
            bids = []
            asks = []

            # Parse bids
            if "bids" in levels:
                for level in levels["bids"]:
                    if len(level) >= 2:
                        price = float(level[0])
                        size = float(level[1])
                        bids.append(OrderBookLevel(price, size))

            # Parse asks
            if "asks" in levels:
                for level in levels["asks"]:
                    if len(level) >= 2:
                        price = float(level[0])
                        size = float(level[1])
                        asks.append(OrderBookLevel(price, size))

            return OrderBook(
                symbol=symbol,
                timestamp=datetime.now(timezone.utc),
                exchange="hyperliquid",
                bids=bids,
                asks=asks
            )

        except Exception as e:
            logger.error(f"Error parsing order book message: {e}")
            return None

    def _parse_trades_message(self, message: Dict, symbol: str) -> List[Trade]:
        """Parse trades message from Hyperliquid."""
        try:
            if "data" not in message:
                return []

            data = message["data"]
            if "trades" not in data:
                return []

            trades = []
            for trade_data in data["trades"]:
                if len(trade_data) >= 3:
                    price = float(trade_data[0])
                    size = float(trade_data[1])
                    side = OrderSide.BUY if trade_data[2] == "buy" else OrderSide.SELL

                    trade = Trade(
                        symbol=symbol,
                        price=price,
                        size=size,
                        side=side,
                        timestamp=datetime.now(timezone.utc),
                        exchange="hyperliquid",
                        trade_id=str(trade_data[3]) if len(trade_data) > 3 else None
                    )
                    trades.append(trade)

            return trades

        except Exception as e:
            logger.error(f"Error parsing trades message: {e}")
            return []

    async def _monitor_connection(self):
        """Monitor WebSocket connection health."""
        while self.is_running:
            try:
                if self.websocket:
                    # Send ping to keep connection alive
                    await self.websocket.ping()
                    self._last_ping_time = datetime.now()

                await asyncio.sleep(self.config.ping_interval)

            except Exception as e:
                logger.warning(f"Connection monitoring failed: {e}")
                await self._handle_connection_error(e)
                break

    async def _handle_connection_error(self, error: Exception):
        """Handle connection errors with reconnection logic."""
        self.is_connected = False
        logger.error(f"Connection error: {error}")

        if self._reconnect_count < self.config.reconnect_attempts:
            self._reconnect_count += 1
            delay = min(
                self.config.reconnect_delay * (2 ** (self._reconnect_count - 1)),
                self.config.max_reconnect_delay
            )

            logger.info(f"Attempting reconnection {self._reconnect_count}/{self.config.reconnect_attempts} in {delay}s")
            await asyncio.sleep(delay)

            try:
                await self.connect()
            except Exception as e:
                logger.error(f"Reconnection failed: {e}")
        else:
            logger.error("Max reconnection attempts reached")
            self.is_running = False

    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        await self.connect()
        self.is_running = True
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.disconnect()