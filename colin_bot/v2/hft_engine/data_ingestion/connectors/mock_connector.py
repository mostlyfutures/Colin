"""
Mock market data connector for testing and development.

Simulates real-time market data for HFT system testing.
"""

import asyncio
import random
import time
from datetime import datetime, timezone
from typing import List, Dict, AsyncGenerator, Optional
from dataclasses import dataclass

from ...utils.data_structures import OrderBook, OrderBookLevel, Trade, MarketEvent, EventType, OrderSide


@dataclass
class MockMarketConfig:
    """Configuration for mock market data generation."""
    symbol: str
    base_price: float
    volatility: float = 0.001
    tick_size: float = 0.01
    order_book_depth: int = 10
    trade_frequency: float = 10.0  # trades per second
    order_book_update_frequency: float = 100.0  # updates per second
    spread_bps: int = 5  # basis points


class MockDataConnector:
    """
    Mock market data connector for HFT testing.

    Generates realistic market data including order book updates and trades
    for testing HFT algorithms without requiring real market data feeds.
    """

    def __init__(self, config: MockMarketConfig):
        self.config = config
        self.is_running = False
        self.current_price = config.base_price
        self.order_book = self._initialize_order_book()
        self.trade_id_counter = 1000

    def _initialize_order_book(self) -> OrderBook:
        """Initialize order book with realistic levels."""
        bids = []
        asks = []

        # Generate order book levels
        for i in range(self.config.order_book_depth):
            # Bid levels
            bid_price = self.current_price - (i + 1) * self.config.tick_size
            bid_size = random.uniform(100, 1000) * (1 - i * 0.05)  # Decrease size with depth
            bids.append(OrderBookLevel(bid_price, bid_size))

            # Ask levels
            ask_price = self.current_price + (i + 1) * self.config.tick_size
            ask_size = random.uniform(100, 1000) * (1 - i * 0.05)  # Decrease size with depth
            asks.append(OrderBookLevel(ask_price, ask_size))

        return OrderBook(
            symbol=self.config.symbol,
            timestamp=datetime.now(timezone.utc),
            exchange="mock",
            bids=bids,
            asks=asks
        )

    async def stream_order_book_updates(self) -> AsyncGenerator[OrderBook, None]:
        """
        Stream order book updates.

        Yields:
            OrderBook: Updated order book
        """
        while self.is_running:
            # Update order book
            self._update_order_book()
            self.order_book.timestamp = datetime.now(timezone.utc)

            yield self.order_book

            # Sleep based on update frequency
            await asyncio.sleep(1.0 / self.config.order_book_update_frequency)

    async def stream_trades(self) -> AsyncGenerator[Trade, None]:
        """
        Stream trade data.

        Yields:
            Trade: Trade data
        """
        while self.is_running:
            # Generate trade
            trade = self._generate_trade()
            yield trade

            # Sleep based on trade frequency
            await asyncio.sleep(1.0 / self.config.trade_frequency)

    def _update_order_book(self):
        """Update order book with realistic modifications."""
        # Randomly modify existing levels
        if random.random() < 0.7:  # 70% chance to modify bids
            self._modify_order_book_side('bids')

        if random.random() < 0.7:  # 70% chance to modify asks
            self._modify_order_book_side('asks')

        # Occasionally add or remove levels
        if random.random() < 0.1:  # 10% chance to modify structure
            self._modify_order_book_structure()

        # Update mid price based on activity
        self._update_mid_price()

    def _modify_order_book_side(self, side: str):
        """Modify one side of the order book."""
        if side == 'bids':
            levels = self.order_book.bids
        else:
            levels = self.order_book.asks

        if not levels:
            return

        # Select random level to modify
        level_idx = random.randint(0, min(4, len(levels) - 1))  # Focus on top 5 levels
        level = levels[level_idx]

        # Random modification type
        modification_type = random.choice(['size_change', 'price_change', 'level_removal', 'level_addition'])

        if modification_type == 'size_change':
            # Modify size
            size_change = random.uniform(-0.3, 0.5)  # -30% to +50%
            new_size = max(level.size * (1 + size_change), 10)  # Minimum size
            level.size = new_size

        elif modification_type == 'price_change':
            # Slightly modify price (rare for top levels)
            if level_idx > 2:  # Only modify levels deeper than top 3
                price_change = random.uniform(-1, 1) * self.config.tick_size
                level.price += price_change

        elif modification_type == 'level_removal' and len(levels) > 5:
            # Remove level (if not top levels)
            if level_idx > 2:
                levels.pop(level_idx)

        elif modification_type == 'level_addition':
            # Add new level at the end
            if side == 'bids':
                last_price = levels[-1].price if levels else self.current_price
                new_price = last_price - self.config.tick_size
                new_size = random.uniform(50, 200)
                levels.append(OrderBookLevel(new_price, new_size))
            else:
                last_price = levels[-1].price if levels else self.current_price
                new_price = last_price + self.config.tick_size
                new_size = random.uniform(50, 200)
                levels.append(OrderBookLevel(new_price, new_size))

        # Keep order book sorted
        if side == 'bids':
            levels.sort(key=lambda x: x.price, reverse=True)
        else:
            levels.sort(key=lambda x: x.price)

        # Limit depth
        if len(levels) > self.config.order_book_depth * 2:
            levels[:] = levels[:self.config.order_book_depth * 2]

    def _modify_order_book_structure(self):
        """Modify order book structure (add/remove levels)."""
        # Randomly remove a level from middle
        if len(self.order_book.bids) > 5 and random.random() < 0.5:
            remove_idx = random.randint(2, min(5, len(self.order_book.bids) - 1))
            self.order_book.bids.pop(remove_idx)

        if len(self.order_book.asks) > 5 and random.random() < 0.5:
            remove_idx = random.randint(2, min(5, len(self.order_book.asks) - 1))
            self.order_book.asks.pop(remove_idx)

    def _update_mid_price(self):
        """Update mid price based on order book imbalance."""
        if not self.order_book.bids or not self.order_book.asks:
            return

        bid_pressure = sum(level.size for level in self.order_book.bids[:3])  # Top 3 bids
        ask_pressure = sum(level.size for level in self.order_book.asks[:3])  # Top 3 asks

        pressure_ratio = (bid_pressure - ask_pressure) / (bid_pressure + ask_pressure)

        # Update price based on pressure
        price_movement = pressure_ratio * self.config.volatility * self.current_price
        self.current_price += price_movement

        # Ensure price stays reasonable
        self.current_price = max(
            self.config.base_price * 0.8,
            min(self.config.base_price * 1.2, self.current_price)
        )

    def _generate_trade(self) -> Trade:
        """Generate a realistic trade."""
        # Determine trade side based on order book pressure
        if random.random() < 0.5:
            # More likely to trade on side with more size
            bid_size = self.order_book.total_bid_size
            ask_size = self.order_book.total_ask_size
            side = OrderSide.BUY if bid_size > ask_size else OrderSide.SELL
        else:
            side = random.choice([OrderSide.BUY, OrderSide.SELL])

        # Determine trade price (near mid price with some variation)
        mid_price = self.order_book.mid_price or self.current_price
        price_variation = random.uniform(-self.config.spread_bps * 0.5, self.config.spread_bps * 0.5) * 0.0001
        trade_price = mid_price + price_variation

        # Round to tick size
        trade_price = round(trade_price / self.config.tick_size) * self.config.tick_size

        # Determine trade size
        base_size = random.uniform(10, 100)
        size_multiplier = random.uniform(0.5, 2.0)
        trade_size = base_size * size_multiplier

        # Create trade
        trade = Trade(
            symbol=self.config.symbol,
            timestamp=datetime.now(timezone.utc),
            price=trade_price,
            size=trade_size,
            side=side,
            trade_id=f"mock_{self.trade_id_counter}",
            exchange="mock"
        )

        self.trade_id_counter += 1

        # Update order book to reflect trade
        self._apply_trade_to_order_book(trade)

        return trade

    def _apply_trade_to_order_book(self, trade: Trade):
        """Apply trade to order book (simulate execution)."""
        if trade.side == OrderSide.BUY:
            # Trade hits ask side
            self._consume_liquidity(self.order_book.asks, trade.size, trade.price)
        else:
            # Trade hits bid side
            self._consume_liquidity(self.order_book.bids, trade.size, trade.price)

    def _consume_liquidity(self, levels: List[OrderBookLevel], trade_size: float, trade_price: float):
        """Consume liquidity from order book levels."""
        remaining_size = trade_size

        for level in levels:
            if remaining_size <= 0:
                break

            # Check if price matches (within tick size tolerance)
            if abs(level.price - trade_price) <= self.config.tick_size:
                consumed = min(level.size, remaining_size)
                level.size -= consumed
                remaining_size -= consumed

        # Remove empty levels
        levels[:] = [level for level in levels if level.size > 0]

        # Replenish top levels if they get too thin
        if levels and levels[0].size < 50:
            levels[0].size = random.uniform(100, 500)

    async def start(self):
        """Start the mock data connector."""
        self.is_running = True

    async def stop(self):
        """Stop the mock data connector."""
        self.is_running = False

    def get_current_order_book(self) -> OrderBook:
        """Get current order book snapshot."""
        return self.order_book

    def get_status(self) -> Dict[str, any]:
        """Get connector status."""
        return {
            'is_running': self.is_running,
            'symbol': self.config.symbol,
            'current_price': self.current_price,
            'bid_size': self.order_book.total_bid_size,
            'ask_size': self.order_book.total_ask_size,
            'spread': self.order_book.spread,
            'mid_price': self.order_book.mid_price
        }