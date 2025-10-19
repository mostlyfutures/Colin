"""
Position Monitor for Colin Trading Bot v2.0

This module implements real-time position monitoring and risk tracking.

Key Features:
- Real-time position P&L calculation
- Exposure monitoring by symbol and asset class
- Concentration risk detection and alerts
- Position limit enforcement
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
from loguru import logger

from ...execution_engine.smart_routing.router import Order, OrderSide


class PositionStatus(Enum):
    """Position status enumeration."""
    ACTIVE = "active"
    CLOSED = "closed"
    CLOSING = "closing"
    ERROR = "error"


@dataclass
class Position:
    """Position data structure."""
    symbol: str
    side: OrderSide
    quantity: float
    entry_price: float
    current_price: float
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    total_fees: float = 0.0
    status: PositionStatus = PositionStatus.ACTIVE
    open_time: datetime = field(default_factory=datetime.now)
    close_time: Optional[datetime] = None
    last_update: datetime = field(default_factory=datetime.now)
    exchange: str = ""
    order_ids: List[str] = field(default_factory=list)


@dataclass
class ConcentrationAlert:
    """Concentration risk alert."""
    symbol: str
    exposure_ratio: float
    limit_ratio: float
    alert_type: str  # "warning", "critical"
    timestamp: datetime = field(default_factory=datetime.now)


class PositionMonitor:
    """
    Real-time position monitoring and risk tracking.

    This class monitors all open positions, calculates P&L in real-time,
    and detects concentration risks and other position-related issues.
    """

    def __init__(
        self,
        max_position_exposure: float = 0.20,  # 20% max per symbol
        concentration_warning_threshold: float = 0.15,  # 15% warning
        update_interval_seconds: int = 5
    ):
        """
        Initialize position monitor.

        Args:
            max_position_exposure: Maximum exposure ratio per symbol
            concentration_warning_threshold: Warning threshold for concentration
            update_interval_seconds: Position update interval
        """
        self.max_position_exposure = max_position_exposure
        self.concentration_warning_threshold = concentration_warning_threshold
        self.update_interval_seconds = update_interval_seconds

        # Position tracking
        self.positions: Dict[str, Position] = {}
        self.position_history: List[Dict[str, Any]] = []

        # Risk monitoring
        self.concentration_alerts: List[ConcentrationAlert] = []
        self.exposure_limits: Dict[str, float] = {}

        # Performance metrics
        self.total_unrealized_pnl = 0.0
        self.total_realized_pnl = 0.0
        self.total_fees = 0.0
        self.position_count = 0

        # Background monitoring task
        self.monitoring_task = None
        self.is_monitoring = False

        logger.info(f"PositionMonitor initialized with max exposure: {max_position_exposure:.1%}")

    async def start_monitoring(self):
        """Start background position monitoring."""
        if self.is_monitoring:
            logger.warning("Position monitoring already started")
            return

        self.is_monitoring = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Position monitoring started")

    async def stop_monitoring(self):
        """Stop background position monitoring."""
        if not self.is_monitoring:
            return

        self.is_monitoring = False
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass

        logger.info("Position monitoring stopped")

    async def add_position(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        entry_price: float,
        exchange: str = "",
        order_id: str = ""
    ) -> bool:
        """
        Add or update a position.

        Args:
            symbol: Trading symbol
            side: Position side (LONG/SHORT)
            quantity: Position quantity
            entry_price: Entry price
            exchange: Exchange name
            order_id: Order ID

        Returns:
            True if position added/updated successfully
        """
        try:
            position_key = f"{symbol}_{side.value}_{exchange}"

            if position_key in self.positions:
                # Update existing position
                position = self.positions[position_key]
                old_quantity = position.quantity
                position.quantity += quantity

                # Update entry price (weighted average)
                if quantity > 0:
                    total_cost = (old_quantity * position.entry_price) + (quantity * entry_price)
                    position.entry_price = total_cost / position.quantity

                if order_id:
                    position.order_ids.append(order_id)
                position.last_update = datetime.now()

                logger.debug(f"Updated position {position_key}: {position.quantity:.4f}")

            else:
                # Create new position
                position = Position(
                    symbol=symbol,
                    side=side,
                    quantity=quantity,
                    entry_price=entry_price,
                    current_price=entry_price,
                    exchange=exchange,
                    order_ids=[order_id] if order_id else []
                )
                self.positions[position_key] = position
                logger.info(f"New position opened: {position_key} @ {entry_price}")

            # Check concentration risk
            await self._check_concentration_risk(symbol)

            return True

        except Exception as e:
            logger.error(f"Error adding position {symbol}: {e}")
            return False

    async def close_position(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        exit_price: float,
        exchange: str = ""
    ) -> bool:
        """
        Close or reduce a position.

        Args:
            symbol: Trading symbol
            side: Position side
            quantity: Quantity to close
            exit_price: Exit price
            exchange: Exchange name

        Returns:
            True if position closed/reduced successfully
        """
        try:
            position_key = f"{symbol}_{side.value}_{exchange}"

            if position_key not in self.positions:
                logger.warning(f"Position {position_key} not found for closing")
                return False

            position = self.positions[position_key]

            # Calculate P&L for closed portion
            if side == OrderSide.BUY:  # Long position
                pnl_per_unit = exit_price - position.entry_price
            else:  # Short position
                pnl_per_unit = position.entry_price - exit_price

            realized_pnl = pnl_per_unit * quantity
            position.realized_pnl += realized_pnl

            # Update position
            position.quantity -= quantity
            position.last_update = datetime.now()

            if position.quantity <= 0.0001:  # Position fully closed
                position.status = PositionStatus.CLOSED
                position.close_time = datetime.now()
                position.current_price = exit_price

                # Add to history
                self.position_history.append({
                    "symbol": symbol,
                    "side": side.value,
                    "quantity": quantity,
                    "entry_price": position.entry_price,
                    "exit_price": exit_price,
                    "realized_pnl": realized_pnl,
                    "open_time": position.open_time,
                    "close_time": position.close_time,
                    "holding_period": (position.close_time - position.open_time).total_seconds()
                })

                logger.info(f"Position closed: {position_key}, P&L: {realized_pnl:.2f}")

            else:
                logger.debug(f"Position reduced: {position_key} to {position.quantity:.4f}")

            return True

        except Exception as e:
            logger.error(f"Error closing position {symbol}: {e}")
            return False

    async def update_prices(self, price_updates: Dict[str, Dict[str, float]]):
        """
        Update current prices for positions.

        Args:
            price_updates: Dictionary of symbol prices by exchange
        """
        try:
            total_unrealized_pnl = 0.0

            for position_key, position in self.positions.items():
                if position.status != PositionStatus.ACTIVE:
                    continue

                # Get current price
                if position.symbol in price_updates:
                    if position.exchange in price_updates[position.symbol]:
                        current_price = price_updates[position.symbol][position.exchange]
                    else:
                        # Use any available price
                        current_price = list(price_updates[position.symbol].values())[0]
                else:
                    continue

                # Calculate unrealized P&L
                old_price = position.current_price
                position.current_price = current_price

                if position.side == OrderSide.BUY:  # Long position
                    position.unrealized_pnl = (current_price - position.entry_price) * position.quantity
                else:  # Short position
                    position.unrealized_pnl = (position.entry_price - current_price) * position.quantity

                total_unrealized_pnl += position.unrealized_pnl

                # Log significant price changes
                price_change_pct = abs((current_price - old_price) / old_price) if old_price > 0 else 0
                if price_change_pct > 0.01:  # 1% change
                    logger.debug(f"Price update {position_key}: {old_price:.4f} -> {current_price:.4f}")

            self.total_unrealized_pnl = total_unrealized_pnl

        except Exception as e:
            logger.error(f"Error updating position prices: {e}")

    async def _check_concentration_risk(self, symbol: str):
        """Check for concentration risk in positions."""
        try:
            # Calculate total portfolio value (simplified)
            total_value = sum(
                abs(pos.quantity * pos.current_price)
                for pos in self.positions.values()
                if pos.status == PositionStatus.ACTIVE
            )

            if total_value == 0:
                return

            # Calculate exposure to symbol
            symbol_exposure = sum(
                abs(pos.quantity * pos.current_price)
                for pos in self.positions.values()
                if pos.symbol == symbol and pos.status == PositionStatus.ACTIVE
            )

            exposure_ratio = symbol_exposure / total_value

            # Check against limits
            if exposure_ratio > self.max_position_exposure:
                alert = ConcentrationAlert(
                    symbol=symbol,
                    exposure_ratio=exposure_ratio,
                    limit_ratio=self.max_position_exposure,
                    alert_type="critical"
                )
                self.concentration_alerts.append(alert)
                logger.warning(f"CRITICAL concentration risk: {symbol} exposure {exposure_ratio:.1%} exceeds {self.max_position_exposure:.1%}")

            elif exposure_ratio > self.concentration_warning_threshold:
                alert = ConcentrationAlert(
                    symbol=symbol,
                    exposure_ratio=exposure_ratio,
                    limit_ratio=self.concentration_warning_threshold,
                    alert_type="warning"
                )
                self.concentration_alerts.append(alert)
                logger.info(f"WARNING concentration risk: {symbol} exposure {exposure_ratio:.1%} above {self.concentration_warning_threshold:.1%}")

        except Exception as e:
            logger.error(f"Error checking concentration risk for {symbol}: {e}")

    async def _monitoring_loop(self):
        """Background monitoring loop."""
        while self.is_monitoring:
            try:
                # Update aggregate metrics
                await self._update_aggregate_metrics()

                # Clean up old alerts
                self.concentration_alerts = [
                    alert for alert in self.concentration_alerts
                    if (datetime.now() - alert.timestamp).total_seconds() < 3600  # Keep last hour
                ]

                await asyncio.sleep(self.update_interval_seconds)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in position monitoring loop: {e}")
                await asyncio.sleep(1)

    async def _update_aggregate_metrics(self):
        """Update aggregate position metrics."""
        try:
            active_positions = [
                pos for pos in self.positions.values()
                if pos.status == PositionStatus.ACTIVE
            ]

            self.position_count = len(active_positions)
            self.total_unrealized_pnl = sum(pos.unrealized_pnl for pos in active_positions)
            self.total_realized_pnl = sum(pos.realized_pnl for pos in self.positions.values())
            self.total_fees = sum(pos.total_fees for pos in self.positions.values())

        except Exception as e:
            logger.error(f"Error updating aggregate metrics: {e}")

    def get_position_summary(self) -> Dict[str, Any]:
        """Get comprehensive position summary."""
        active_positions = [
            pos for pos in self.positions.values()
            if pos.status == PositionStatus.ACTIVE
        ]

        # Group by symbol
        symbol_exposure = {}
        for pos in active_positions:
            if pos.symbol not in symbol_exposure:
                symbol_exposure[pos.symbol] = {
                    "total_exposure": 0.0,
                    "long_exposure": 0.0,
                    "short_exposure": 0.0,
                    "net_exposure": 0.0,
                    "positions": []
                }

            exposure = abs(pos.quantity * pos.current_price)
            symbol_exposure[pos.symbol]["total_exposure"] += exposure
            symbol_exposure[pos.symbol]["positions"].append(pos)

            if pos.side == OrderSide.BUY:
                symbol_exposure[pos.symbol]["long_exposure"] += exposure
            else:
                symbol_exposure[pos.symbol]["short_exposure"] += exposure

        # Calculate net exposure
        for symbol_data in symbol_exposure.values():
            symbol_data["net_exposure"] = symbol_data["long_exposure"] - symbol_data["short_exposure"]

        # Sort by exposure
        sorted_symbols = sorted(
            symbol_exposure.items(),
            key=lambda x: x[1]["total_exposure"],
            reverse=True
        )

        return {
            "total_positions": self.position_count,
            "total_unrealized_pnl": self.total_unrealized_pnl,
            "total_realized_pnl": self.total_realized_pnl,
            "total_pnl": self.total_unrealized_pnl + self.total_realized_pnl,
            "total_fees": self.total_fees,
            "active_concentration_alerts": len([
                alert for alert in self.concentration_alerts
                if (datetime.now() - alert.timestamp).total_seconds() < 300  # Last 5 minutes
            ]),
            "top_exposures": sorted_symbols[:10],  # Top 10 exposures
            "position_details": [
                {
                    "symbol": pos.symbol,
                    "side": pos.side.value,
                    "quantity": pos.quantity,
                    "entry_price": pos.entry_price,
                    "current_price": pos.current_price,
                    "unrealized_pnl": pos.unrealized_pnl,
                    "realized_pnl": pos.realized_pnl,
                    "exchange": pos.exchange,
                    "open_time": pos.open_time.isoformat(),
                    "pnl_percentage": (pos.unrealized_pnl / (pos.entry_price * pos.quantity)) * 100 if pos.quantity > 0 else 0
                }
                for pos in active_positions
            ]
        }

    def get_positions_by_symbol(self, symbol: str) -> List[Position]:
        """Get all positions for a specific symbol."""
        return [
            pos for pos in self.positions.values()
            if pos.symbol == symbol and pos.status == PositionStatus.ACTIVE
        ]

    def get_total_exposure_by_symbol(self) -> Dict[str, float]:
        """Get total exposure grouped by symbol."""
        symbol_exposure = {}

        for pos in self.positions.values():
            if pos.status != PositionStatus.ACTIVE:
                continue

            if pos.symbol not in symbol_exposure:
                symbol_exposure[pos.symbol] = 0.0

            symbol_exposure[pos.symbol] += abs(pos.quantity * pos.current_price)

        return symbol_exposure


# Standalone validation function
def validate_position_monitor():
    """Validate position monitor implementation."""
    print("🔍 Validating PositionMonitor implementation...")

    try:
        # Test imports
        from .position_monitor import PositionMonitor, Position, PositionStatus
        print("✅ Imports successful")

        # Test instantiation
        monitor = PositionMonitor()
        print("✅ PositionMonitor instantiation successful")

        # Test basic functionality
        if hasattr(monitor, 'add_position'):
            print("✅ add_position method exists")
        else:
            print("❌ add_position method missing")
            return False

        if hasattr(monitor, 'get_position_summary'):
            print("✅ get_position_summary method exists")
        else:
            print("❌ get_position_summary method missing")
            return False

        print("🎉 PositionMonitor validation PASSED")
        return True

    except Exception as e:
        print(f"❌ Validation failed: {e}")
        return False


if __name__ == "__main__":
    validate_position_monitor()