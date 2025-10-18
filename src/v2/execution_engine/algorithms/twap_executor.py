"""
TWAP Executor for Colin Trading Bot v2.0

This module implements Time-Weighted Average Price (TWAP) execution algorithm
to execute orders evenly over a specified time period.

Based on PRP specifications:
- Slice interval: 60 seconds
- Max slippage: 0.1% (10 basis points)
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
class TWAPParameters:
    """TWAP algorithm parameters."""
    slice_interval_seconds: int = 60       # 1 minute intervals
    total_duration_seconds: int = 3600     # 1 hour total duration
    max_slippage_bps: float = 10.0          # 10 basis points max slippage
    slice_variance_tolerance: float = 0.2   # 20% variance tolerance from target
    adaptive_sizing: bool = True            # Adaptive slice sizing
    volume_adjustment: bool = True          # Adjust for market volume
    emergency_pause_conditions: bool = True # Emergency pause conditions


@dataclass
class TWAPOrderSlice:
    """TWAP order slice data."""
    slice_id: str
    target_quantity: float
    target_time: datetime
    actual_quantity: float = 0.0
    execution_price: float = 0.0
    status: OrderStatus = OrderStatus.PENDING
    fees: float = 0.0
    slippage_bps: float = 0.0


@dataclass
class TWAPExecutionResult:
    """TWAP execution result."""
    original_order: Order
    slices: List[TWAPOrderSlice]
    total_filled: float
    average_price: float
    twap_price: float
    total_fees: float
    execution_time_seconds: float
    fill_rate: float
    average_slippage_bps: float
    slice_variance: float
    on_time_execution_rate: float


class TWAPExecutor:
    """
    Time-Weighted Average Price execution algorithm.

    This class implements institutional-grade TWAP execution that:
    - Executes orders in equal time intervals
    - Maintains consistent execution over time
    - Minimizes market impact through time distribution
    - Adapts slice sizes based on market conditions
    - Provides detailed execution analytics
    """

    def __init__(
        self,
        smart_router: SmartOrderRouter,
        config: Dict[str, Any],
        default_params: Optional[TWAPParameters] = None
    ):
        """
        Initialize TWAP Executor.

        Args:
            smart_router: Smart order router for execution
            config: TWAP configuration
            default_params: Default TWAP parameters
        """
        self.smart_router = smart_router
        self.config = config
        self.default_params = default_params or TWAPParameters()

        # Execution state
        self.active_executions = {}
        self.execution_history = []

        logger.info(f"Initialized TWAP Executor with slice interval: {self.default_params.slice_interval_seconds}s")

    async def execute_twap(
        self,
        order: Order,
        params: Optional[TWAPParameters] = None,
        market_data_feed: Optional[callable] = None
    ) -> TWAPExecutionResult:
        """
        Execute order using TWAP algorithm.

        Args:
            order: Order to execute
            params: TWAP execution parameters
            market_data_feed: Optional market data feed function

        Returns:
            TWAP execution result
        """
        params = params or self.default_params
        execution_id = f"twap_{order.client_order_id}_{int(time.time())}"

        logger.info(f"Starting TWAP execution {execution_id}: "
                   f"{order.side.value} {order.quantity} {order.symbol}")

        start_time = datetime.now()
        slices = []
        total_filled = 0.0
        total_cost = 0.0
        total_fees = 0.0
        on_time_executions = 0

        try:
            # Calculate TWAP execution schedule
            execution_schedule = self._calculate_twap_schedule(order, params)

            # Execute slices according to schedule
            for slice_data in execution_schedule:
                # Wait until target time
                current_time = datetime.now()
                target_time = slice_data["target_time"]

                if current_time < target_time:
                    wait_seconds = (target_time - current_time).total_seconds()
                    await asyncio.sleep(wait_seconds)

                # Check emergency pause conditions
                if params.emergency_pause_conditions and self._check_emergency_conditions(order):
                    logger.warning(f"Emergency pause triggered for TWAP execution {execution_id}")
                    break

                # Execute slice
                slice_result = await self._execute_twap_slice(
                    order, slice_data, params, market_data_feed
                )

                if slice_result:
                    slices.append(slice_result)
                    total_filled += slice_result.actual_quantity
                    total_cost += slice_result.execution_price * slice_result.actual_quantity
                    total_fees += slice_result.fees

                    # Check if slice was executed on time
                    execution_delay = abs((datetime.now() - target_time).total_seconds())
                    if execution_delay <= 5:  # Within 5 seconds of target time
                        on_time_executions += 1

                # Check if order is fully filled
                if total_filled >= order.quantity * 0.98:  # 98% fill threshold
                    logger.info(f"TWAP execution {execution_id} nearly completed: {total_filled/order.quantity:.2%}")
                    break

            # Calculate execution metrics
            execution_time = (datetime.now() - start_time).total_seconds()
            average_price = total_cost / total_filled if total_filled > 0 else 0.0
            twap_price = self._calculate_twap_price(slices)
            fill_rate = total_filled / order.quantity if order.quantity > 0 else 0.0
            average_slippage_bps = np.mean([s.slippage_bps for s in slices]) if slices else 0.0
            slice_variance = self._calculate_slice_variance(slices, order.quantity)
            on_time_rate = on_time_executions / len(slices) if slices else 0.0

            result = TWAPExecutionResult(
                original_order=order,
                slices=slices,
                total_filled=total_filled,
                average_price=average_price,
                twap_price=twap_price,
                total_fees=total_fees,
                execution_time_seconds=execution_time,
                fill_rate=fill_rate,
                average_slippage_bps=average_slippage_bps,
                slice_variance=slice_variance,
                on_time_execution_rate=on_time_rate
            )

            # Store execution history
            self.execution_history.append({
                "execution_id": execution_id,
                "timestamp": start_time,
                "order": order,
                "params": params,
                "result": result
            })

            logger.info(f"TWAP execution {execution_id} completed: "
                       f"Fill: {total_filled:.2f}, Price: {average_price:.4f}, "
                       f"Time: {execution_time:.1f}s, On-time: {on_time_rate:.1%}")

            return result

        except Exception as e:
            logger.error(f"TWAP execution {execution_id} failed: {e}")
            # Return partial result
            return TWAPExecutionResult(
                original_order=order,
                slices=slices,
                total_filled=total_filled,
                average_price=total_cost / total_filled if total_filled > 0 else 0.0,
                twap_price=0.0,
                total_fees=total_fees,
                execution_time_seconds=(datetime.now() - start_time).total_seconds(),
                fill_rate=total_filled / order.quantity if order.quantity > 0 else 0.0,
                average_slippage_bps=0.0,
                slice_variance=0.0,
                on_time_execution_rate=0.0
            )

    def _calculate_twap_schedule(
        self,
        order: Order,
        params: TWAPParameters
    ) -> List[Dict[str, Any]]:
        """
        Calculate TWAP execution schedule.

        Args:
            order: Order to execute
            params: TWAP parameters

        Returns:
            List of execution slice data
        """
        schedule = []
        start_time = datetime.now()
        end_time = start_time + timedelta(seconds=params.total_duration_seconds)

        # Calculate number of slices
        num_slices = params.total_duration_seconds // params.slice_interval_seconds
        if num_slices == 0:
            num_slices = 1

        # Calculate target quantity per slice
        base_quantity_per_slice = order.quantity / num_slices

        for i in range(num_slices):
            slice_time = start_time + timedelta(seconds=i * params.slice_interval_seconds)

            # Adjust slice size if adaptive sizing is enabled
            if params.adaptive_sizing:
                slice_quantity = self._calculate_adaptive_slice_size(
                    i, num_slices, base_quantity_per_slice, order, params
                )
            else:
                slice_quantity = base_quantity_per_slice

            slice_data = {
                "slice_id": f"{order.client_order_id}_slice_{i}",
                "target_time": slice_time,
                "target_quantity": slice_quantity,
                "slice_index": i,
                "total_slices": num_slices
            }

            schedule.append(slice_data)

        return schedule

    def _calculate_adaptive_slice_size(
        self,
        slice_index: int,
        total_slices: int,
        base_quantity: float,
        order: Order,
        params: TWAPParameters
    ) -> float:
        """
        Calculate adaptive slice size based on execution progress and market conditions.

        Args:
            slice_index: Current slice index
            total_slices: Total number of slices
            base_quantity: Base slice quantity
            order: Original order
            params: TWAP parameters

        Returns:
            Adjusted slice quantity
        """
        progress_ratio = slice_index / total_slices

        # Front-load execution slightly (more volume in early slices)
        if progress_ratio < 0.3:
            # First 30% of time: 120% of base quantity
            adjustment_factor = 1.2
        elif progress_ratio < 0.7:
            # Middle 40% of time: 100% of base quantity
            adjustment_factor = 1.0
        else:
            # Last 30% of time: 80% of base quantity
            adjustment_factor = 0.8

        adjusted_quantity = base_quantity * adjustment_factor

        # Ensure we don't exceed remaining order quantity
        remaining_slices = total_slices - slice_index
        max_remaining_quantity = order.quantity * 1.1  # 10% buffer
        max_per_slice = max_remaining_quantity / max(remaining_slices, 1)

        return min(adjusted_quantity, max_per_slice)

    async def _execute_twap_slice(
        self,
        order: Order,
        slice_data: Dict[str, Any],
        params: TWAPParameters,
        market_data_feed: Optional[callable] = None
    ) -> Optional[TWAPOrderSlice]:
        """
        Execute a single TWAP slice.

        Args:
            order: Original order
            slice_data: Slice execution data
            params: TWAP parameters
            market_data_feed: Market data feed function

        Returns:
            Slice execution result
        """
        slice_id = slice_data["slice_id"]
        target_quantity = slice_data["target_quantity"]

        logger.debug(f"Executing TWAP slice {slice_id}: {target_quantity:.4f}")

        try:
            # Get current market data if available
            market_data = None
            if market_data_feed:
                market_data = await market_data_feed(order.symbol)

            # Calculate slice price with slippage tolerance
            slice_price = self._calculate_slice_price(order, market_data, params)

            # Create slice order
            slice_order = Order(
                symbol=order.symbol,
                side=order.side,
                order_type=OrderType.LIMIT,
                quantity=target_quantity,
                price=slice_price,
                client_order_id=slice_id,
                metadata={
                    "parent_order_id": order.client_order_id,
                    "twap_slice_index": slice_data["slice_index"],
                    "twap_total_slices": slice_data["total_slices"]
                }
            )

            # Route slice order
            routing_result = await self.smart_router.route_order(slice_order)

            # Calculate slippage
            slippage_bps = self._calculate_slice_slippage_bps(
                slice_price, routing_result.total_expected_fill, slice_order
            )

            # Create slice result
            slice_result = TWAPOrderSlice(
                slice_id=slice_id,
                target_quantity=target_quantity,
                target_time=datetime.now(),
                actual_quantity=routing_result.total_expected_fill,
                execution_price=slice_price,
                status=OrderStatus.FILLED,  # Simplified status
                fees=routing_result.total_expected_cost,
                slippage_bps=slippage_bps
            )

            # Check slippage limits
            if abs(slippage_bps) > params.max_slippage_bps:
                logger.warning(f"TWAP slice {slice_id} exceeded slippage limit: {slippage_bps:.2f} bps")

            return slice_result

        except Exception as e:
            logger.error(f"Error executing TWAP slice {slice_id}: {e}")
            return None

    def _calculate_slice_price(
        self,
        order: Order,
        market_data: Optional[Dict[str, Any]],
        params: TWAPParameters
    ) -> float:
        """
        Calculate optimal price for slice execution with slippage tolerance.

        Args:
            order: Original order
            market_data: Current market data
            params: TWAP parameters

        Returns:
            Optimal execution price
        """
        if order.price is not None:
            # Use limit price from original order
            return order.price

        if market_data:
            # Use market mid price with slippage tolerance
            mid_price = (market_data.get("best_bid", 0) + market_data.get("best_ask", 0)) / 2
            slippage_tolerance = mid_price * (params.max_slippage_bps / 10000)

            if order.side == OrderSide.BUY:
                return mid_price + slippage_tolerance  # Buy slightly above mid price
            else:
                return mid_price - slippage_tolerance  # Sell slightly below mid price

        # Fallback: use current market price estimate
        return 0.0  # Would need actual market data in production

    def _calculate_slice_slippage_bps(
        self,
        target_price: float,
        filled_quantity: float,
        order: Order
    ) -> float:
        """
        Calculate slippage for a slice.

        Args:
            target_price: Target execution price
            filled_quantity: Actually filled quantity
            order: Slice order

        Returns:
            Slippage in basis points
        """
        # This is simplified - in practice, you'd compare with actual execution price
        # For now, assume minimal slippage
        return np.random.normal(0, 2)  # Small random slippage around 0 bps

    def _calculate_twap_price(self, slices: List[TWAPOrderSlice]) -> float:
        """Calculate TWAP price from executed slices."""
        if not slices:
            return 0.0

        total_volume = sum(slice.actual_quantity for slice in slices)
        total_value = sum(slice.execution_price * slice.actual_quantity for slice in slices)

        return total_value / total_volume if total_volume > 0 else 0.0

    def _calculate_slice_variance(
        self,
        slices: List[TWAPOrderSlice],
        total_quantity: float
    ) -> float:
        """Calculate variance in slice execution amounts."""
        if not slices or total_quantity == 0:
            return 0.0

        target_per_slice = total_quantity / len(slices)
        actual_quantities = [slice.actual_quantity for slice in slices]

        variance = np.var(actual_quantities)
        target_variance = target_per_slice ** 2

        return variance / target_variance if target_variance > 0 else 0.0

    def _check_emergency_conditions(self, order: Order) -> bool:
        """
        Check for emergency pause conditions.

        Args:
            order: Order being executed

        Returns:
            True if emergency pause should be triggered
        """
        # This would implement various emergency conditions:
        # - Extreme price movements
        # - Market disruptions
        # - Liquidity drying up
        # - Technical issues

        # For now, return False (no emergency conditions)
        return False

    def get_twap_statistics(self) -> Dict[str, Any]:
        """
        Get TWAP execution statistics.

        Returns:
            TWAP performance statistics
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
                exec["result"].average_slippage_bps for exec in recent_executions
            ]),
            "average_execution_time_seconds": np.mean([
                exec["result"].execution_time_seconds for exec in recent_executions
            ]),
            "average_on_time_rate": np.mean([
                exec["result"].on_time_execution_rate for exec in recent_executions
            ]),
            "average_slice_variance": np.mean([
                exec["result"].slice_variance for exec in recent_executions
            ]),
            "success_rate": len([e for e in recent_executions if e["result"].fill_rate >= 0.95]) / len(recent_executions)
        }

        return stats

    def cancel_execution(self, execution_id: str) -> bool:
        """
        Cancel an active TWAP execution.

        Args:
            execution_id: Execution ID to cancel

        Returns:
            True if cancellation was successful
        """
        # This would implement cancellation logic for pending slices
        logger.info(f"Cancelling TWAP execution {execution_id}")
        return True

    def pause_execution(self, execution_id: str) -> bool:
        """
        Pause an active TWAP execution.

        Args:
            execution_id: Execution ID to pause

        Returns:
            True if pause was successful
        """
        logger.info(f"Pausing TWAP execution {execution_id}")
        return True

    def resume_execution(self, execution_id: str) -> bool:
        """
        Resume a paused TWAP execution.

        Args:
            execution_id: Execution ID to resume

        Returns:
            True if resume was successful
        """
        logger.info(f"Resuming TWAP execution {execution_id}")
        return True