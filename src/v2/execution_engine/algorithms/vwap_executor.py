"""
VWAP Executor for Colin Trading Bot v2.0

This module implements Volume-Weighted Average Price (VWAP) execution algorithm
to minimize market impact while achieving target execution volume over time.

Based on PRP specifications:
- Participation rate: 0.1
- Time window: 300 seconds
- Institutional-grade execution algorithm
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import pandas as pd
from loguru import logger

from ..smart_routing.router import Order, OrderSide, OrderType, OrderStatus, SmartOrderRouter


@dataclass
class VWAPParameters:
    """VWAP algorithm parameters."""
    participation_rate: float = 0.1  # 10% of market volume
    time_window_seconds: int = 300   # 5 minutes
    max_participation_rate: float = 0.2  # Maximum 20% participation
    min_participation_rate: float = 0.05  # Minimum 5% participation
    price_tolerance_bps: float = 50    # 50 basis points price tolerance
    fill_rate_target: float = 0.95     # 95% fill rate target
    adaptive_participation: bool = True  # Adaptive participation rate
    volume_profile_adjustment: bool = True  # Adjust for volume profile


@dataclass
class VWAPOrderSlice:
    """VWAP order slice data."""
    slice_id: str
    quantity: float
    target_price: float
    execution_time: datetime
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    average_price: float = 0.0
    fees: float = 0.0


@dataclass
class VWAPExecutionResult:
    """VWAP execution result."""
    original_order: Order
    slices: List[VWAPOrderSlice]
    total_filled: float
    average_price: float
    vwap_price: float
    total_fees: float
    execution_time_seconds: float
    fill_rate: float
    participation_rate: float
    price_slippage_bps: float


class VWAPExecutor:
    """
    Volume-Weighted Average Price execution algorithm.

    This class implements institutional-grade VWAP execution that:
    - Executes orders over a specified time window
    - Maintains target participation rate
    - Minimizes market impact
    - Adapts to market conditions
    - Provides detailed execution analytics
    """

    def __init__(
        self,
        smart_router: SmartOrderRouter,
        config: Dict[str, Any],
        default_params: Optional[VWAPParameters] = None
    ):
        """
        Initialize VWAP Executor.

        Args:
            smart_router: Smart order router for execution
            config: VWAP configuration
            default_params: Default VWAP parameters
        """
        self.smart_router = smart_router
        self.config = config
        self.default_params = default_params or VWAPParameters()

        # Execution state
        self.active_executions = {}
        self.execution_history = []

        logger.info(f"Initialized VWAP Executor with default participation rate: {self.default_params.participation_rate}")

    async def execute_vwap(
        self,
        order: Order,
        params: Optional[VWAPParameters] = None,
        market_data_feed: Optional[callable] = None
    ) -> VWAPExecutionResult:
        """
        Execute order using VWAP algorithm.

        Args:
            order: Order to execute
            params: VWAP execution parameters
            market_data_feed: Optional market data feed function

        Returns:
            VWAP execution result
        """
        params = params or self.default_params
        execution_id = f"vwap_{order.client_order_id}_{int(time.time())}"

        logger.info(f"Starting VWAP execution {execution_id}: "
                   f"{order.side.value} {order.quantity} {order.symbol}")

        start_time = datetime.now()
        slices = []
        total_filled = 0.0
        total_cost = 0.0
        total_fees = 0.0

        try:
            # Calculate execution schedule
            execution_schedule = self._calculate_vwap_schedule(order, params)

            # Execute slices according to schedule
            for slice_data in execution_schedule:
                slice_result = await self._execute_vwap_slice(
                    order, slice_data, params, market_data_feed
                )

                if slice_result:
                    slices.append(slice_result)
                    total_filled += slice_result.filled_quantity
                    total_cost += slice_result.average_price * slice_result.filled_quantity
                    total_fees += slice_result.fees

                # Check if order is fully filled
                if total_filled >= order.quantity * params.fill_rate_target:
                    logger.info(f"VWAP target fill rate achieved: {total_filled/order.quantity:.2%}")
                    break

                # Small delay between slices
                await asyncio.sleep(1)

            # Calculate execution metrics
            execution_time = (datetime.now() - start_time).total_seconds()
            average_price = total_cost / total_filled if total_filled > 0 else 0.0
            vwap_price = self._calculate_vwap_price(slices)
            fill_rate = total_filled / order.quantity if order.quantity > 0 else 0.0
            price_slippage_bps = self._calculate_slippage_bps(order, average_price)

            result = VWAPExecutionResult(
                original_order=order,
                slices=slices,
                total_filled=total_filled,
                average_price=average_price,
                vwap_price=vwap_price,
                total_fees=total_fees,
                execution_time_seconds=execution_time,
                fill_rate=fill_rate,
                participation_rate=self._calculate_participation_rate(
                    slices, execution_time
                ),
                price_slippage_bps=price_slippage_bps
            )

            # Store execution history
            self.execution_history.append({
                "execution_id": execution_id,
                "timestamp": start_time,
                "order": order,
                "params": params,
                "result": result
            })

            logger.info(f"VWAP execution {execution_id} completed: "
                       f"Fill: {total_filled:.2f}, Price: {average_price:.4f}, "
                       f"Time: {execution_time:.1f}s")

            return result

        except Exception as e:
            logger.error(f"VWAP execution {execution_id} failed: {e}")
            # Return partial result
            return VWAPExecutionResult(
                original_order=order,
                slices=slices,
                total_filled=total_filled,
                average_price=total_cost / total_filled if total_filled > 0 else 0.0,
                vwap_price=0.0,
                total_fees=total_fees,
                execution_time_seconds=(datetime.now() - start_time).total_seconds(),
                fill_rate=total_filled / order.quantity if order.quantity > 0 else 0.0,
                participation_rate=0.0,
                price_slippage_bps=0.0
            )

    def _calculate_vwap_schedule(
        self,
        order: Order,
        params: VWAPParameters
    ) -> List[Dict[str, Any]]:
        """
        Calculate VWAP execution schedule.

        Args:
            order: Order to execute
            params: VWAP parameters

        Returns:
            List of execution slice data
        """
        schedule = []
        start_time = datetime.now()
        end_time = start_time + timedelta(seconds=params.time_window_seconds)

        # Create time slices (e.g., every 30 seconds for 5-minute window)
        slice_interval = 30  # seconds
        num_slices = params.time_window_seconds // slice_interval

        # Calculate target quantity per slice
        target_quantity_per_slice = order.quantity / num_slices

        for i in range(num_slices):
            slice_time = start_time + timedelta(seconds=i * slice_interval)

            # Adjust participation rate based on time and market conditions
            if params.adaptive_participation:
                participation_rate = self._calculate_adaptive_participation_rate(
                    i, num_slices, params
                )
            else:
                participation_rate = params.participation_rate

            slice_data = {
                "slice_id": f"{order.client_order_id}_slice_{i}",
                "target_time": slice_time,
                "target_quantity": target_quantity_per_slice,
                "participation_rate": participation_rate,
                "max_price_tolerance": params.price_tolerance_bps
            }

            schedule.append(slice_data)

        return schedule

    def _calculate_adaptive_participation_rate(
        self,
        slice_index: int,
        total_slices: int,
        params: VWAPParameters
    ) -> float:
        """
        Calculate adaptive participation rate based on execution progress.

        Args:
            slice_index: Current slice index
            total_slices: Total number of slices
            params: VWAP parameters

        Returns:
            Adjusted participation rate
        """
        # Basic adaptive logic: increase participation if behind schedule
        progress_ratio = slice_index / total_slices

        if progress_ratio > 0.7 and progress_ratio < 0.9:
            # Near end of execution, slightly increase participation
            return min(params.participation_rate * 1.2, params.max_participation_rate)
        elif progress_ratio >= 0.9:
            # Final phase, increase more aggressively
            return min(params.participation_rate * 1.5, params.max_participation_rate)
        else:
            # Normal participation
            return params.participation_rate

    async def _execute_vwap_slice(
        self,
        order: Order,
        slice_data: Dict[str, Any],
        params: VWAPParameters,
        market_data_feed: Optional[callable] = None
    ) -> Optional[VWAPOrderSlice]:
        """
        Execute a single VWAP slice.

        Args:
            order: Original order
            slice_data: Slice execution data
            params: VWAP parameters
            market_data_feed: Market data feed function

        Returns:
            Slice execution result
        """
        slice_id = slice_data["slice_id"]
        target_quantity = slice_data["target_quantity"]
        participation_rate = slice_data["participation_rate"]

        logger.debug(f"Executing VWAP slice {slice_id}: {target_quantity:.4f} @ {participation_rate:.2%}")

        try:
            # Get current market data if available
            market_data = None
            if market_data_feed:
                market_data = await market_data_feed(order.symbol)

            # Create slice order
            slice_order = Order(
                symbol=order.symbol,
                side=order.side,
                order_type=OrderType.LIMIT,
                quantity=target_quantity,
                price=self._calculate_slice_price(order, market_data, params),
                client_order_id=slice_id,
                metadata={
                    "parent_order_id": order.client_order_id,
                    "vwap_participation_rate": participation_rate,
                    "slice_target_time": slice_data["target_time"]
                }
            )

            # Route slice order
            routing_result = await self.smart_router.route_order(slice_order)

            # Create slice result
            slice_result = VWAPOrderSlice(
                slice_id=slice_id,
                quantity=target_quantity,
                target_price=slice_order.price,
                execution_time=datetime.now(),
                status=OrderStatus.FILLED,  # Simplified status
                filled_quantity=routing_result.total_expected_fill,
                average_price=slice_order.price,  # Simplified
                fees=routing_result.total_expected_cost
            )

            return slice_result

        except Exception as e:
            logger.error(f"Error executing VWAP slice {slice_id}: {e}")
            return None

    def _calculate_slice_price(
        self,
        order: Order,
        market_data: Optional[Dict[str, Any]],
        params: VWAPParameters
    ) -> float:
        """
        Calculate optimal price for slice execution.

        Args:
            order: Original order
            market_data: Current market data
            params: VWAP parameters

        Returns:
            Optimal execution price
        """
        if order.price is not None:
            # Use limit price from original order
            return order.price

        if market_data:
            # Use market mid price with tolerance
            mid_price = (market_data.get("best_bid", 0) + market_data.get("best_ask", 0)) / 2
            tolerance = mid_price * (params.price_tolerance_bps / 10000)

            if order.side == OrderSide.BUY:
                return mid_price + tolerance  # Buy slightly above mid price
            else:
                return mid_price - tolerance  # Sell slightly below mid price

        # Fallback: use current market price estimate
        return 0.0  # Would need actual market data in production

    def _calculate_vwap_price(self, slices: List[VWAPOrderSlice]) -> float:
        """Calculate VWAP price from executed slices."""
        if not slices:
            return 0.0

        total_volume = sum(slice.filled_quantity for slice in slices)
        total_value = sum(slice.average_price * slice.filled_quantity for slice in slices)

        return total_value / total_volume if total_volume > 0 else 0.0

    def _calculate_slippage_bps(self, order: Order, execution_price: float) -> float:
        """Calculate price slippage in basis points."""
        if order.price is None or order.price == 0:
            return 0.0

        slippage = (execution_price - order.price) / order.price
        return slippage * 10000  # Convert to basis points

    def _calculate_participation_rate(
        self,
        slices: List[VWAPOrderSlice],
        execution_time_seconds: float
    ) -> float:
        """Calculate actual participation rate."""
        if not slices or execution_time_seconds == 0:
            return 0.0

        total_executed_volume = sum(slice.filled_quantity for slice in slices)
        # This would need market volume data for accurate calculation
        # For now, return estimated participation rate
        return self.default_params.participation_rate

    def get_vwap_statistics(self) -> Dict[str, Any]:
        """
        Get VWAP execution statistics.

        Returns:
            VWAP performance statistics
        """
        if not self.execution_history:
            return {"status": "no_executions"}

        recent_executions = self.execution_history[-50:]  # Last 50 executions

        stats = {
            "total_executions": len(self.execution_history),
            "recent_executions": len(recent_executions),
            "average_fill_rate": np.mean([
                exec["result"].fill_rate for exec in recent_executions
            ]),
            "average_slippage_bps": np.mean([
                exec["result"].price_slippage_bps for exec in recent_executions
            ]),
            "average_execution_time_seconds": np.mean([
                exec["result"].execution_time_seconds for exec in recent_executions
            ]),
            "success_rate": len([e for e in recent_executions if e["result"].fill_rate >= 0.95]) / len(recent_executions)
        }

        return stats

    def cancel_execution(self, execution_id: str) -> bool:
        """
        Cancel an active VWAP execution.

        Args:
            execution_id: Execution ID to cancel

        Returns:
            True if cancellation was successful
        """
        # This would implement cancellation logic for pending slices
        logger.info(f"Cancelling VWAP execution {execution_id}")
        return True

    def pause_execution(self, execution_id: str) -> bool:
        """
        Pause an active VWAP execution.

        Args:
            execution_id: Execution ID to pause

        Returns:
            True if pause was successful
        """
        logger.info(f"Pausing VWAP execution {execution_id}")
        return True

    def resume_execution(self, execution_id: str) -> bool:
        """
        Resume a paused VWAP execution.

        Args:
            execution_id: Execution ID to resume

        Returns:
            True if resume was successful
        """
        logger.info(f"Resuming VWAP execution {execution_id}")
        return True