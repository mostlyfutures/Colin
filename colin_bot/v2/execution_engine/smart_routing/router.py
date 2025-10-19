"""
Smart Order Router for Colin Trading Bot v2.0

This module implements intelligent order routing across multiple exchanges
to achieve optimal execution with minimal market impact and transaction costs.

Based on PRP specifications:
- Multi-exchange liquidity aggregation
- Real-time fee optimization
- Latency-aware routing decisions
- Partial fill handling algorithms
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

from .liquidity_aggregator import LiquidityAggregator
from .fee_optimizer import FeeOptimizer


class OrderSide(Enum):
    """Order side enumeration."""
    BUY = "buy"
    SELL = "sell"


class OrderType(Enum):
    """Order type enumeration."""
    MARKET = "market"
    LIMIT = "limit"
    IOC = "immediate_or_cancel"
    FOK = "fill_or_kill"


class OrderStatus(Enum):
    """Order status enumeration."""
    PENDING = "pending"
    PARTIAL = "partial"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass
class Order:
    """Order data structure."""
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None
    time_in_force: str = "GTC"
    exchange: Optional[str] = None
    client_order_id: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    average_price: float = 0.0
    fees: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Route:
    """Route data structure."""
    exchange: str
    quantity: float
    price: Optional[float] = None
    expected_fee: float = 0.0
    expected_fill_ratio: float = 1.0
    latency_ms: float = 0.0
    liquidity_score: float = 0.0
    priority: int = 0


@dataclass
class RoutingResult:
    """Routing execution result."""
    routes: List[Route]
    total_expected_cost: float
    total_expected_fill: float
    routing_strategy: str
    execution_time_ms: float
    success_rate: float


class SmartOrderRouter:
    """
    Intelligent order routing across multiple exchanges.

    This class implements sophisticated routing algorithms that consider:
    - Real-time liquidity across exchanges
    - Transaction costs and fees
    - Latency and execution speed
    - Market impact minimization
    - Partial fill optimization
    """

    def __init__(
        self,
        exchanges: List[str],
        config: Dict[str, Any],
        liquidity_aggregator: LiquidityAggregator,
        fee_optimizer: FeeOptimizer
    ):
        """
        Initialize Smart Order Router.

        Args:
            exchanges: List of supported exchanges
            config: Router configuration
            liquidity_aggregator: Liquidity aggregation service
            fee_optimizer: Fee optimization service
        """
        self.exchanges = exchanges
        self.config = config
        self.liquidity_aggregator = liquidity_aggregator
        self.fee_optimizer = fee_optimizer

        # Routing parameters
        self.max_exchanges_per_order = config.get("max_exchanges_per_order", 3)
        self.min_liquidity_threshold = config.get("min_liquidity_threshold", 10000)
        self.max_latency_ms = config.get("max_latency_ms", 100)
        self.fill_ratio_threshold = config.get("fill_ratio_threshold", 0.8)

        # Performance tracking
        self.routing_history = []
        self.exchange_performance = {exchange: {} for exchange in exchanges}

        logger.info(f"Initialized SmartOrderRouter with {len(exchanges)} exchanges")

    async def route_order(self, order: Order) -> RoutingResult:
        """
        Calculate optimal routing strategy for an order.

        Args:
            order: Order to route

        Returns:
            Routing result with optimal routes
        """
        logger.info(f"Routing order: {order.side.value} {order.quantity} {order.symbol}")
        start_time = time.time()

        try:
            # Step 1: Get real-time liquidity across exchanges
            logger.debug("Step 1: Aggregating liquidity across exchanges")
            liquidity = await self.liquidity_aggregator.get_aggregated_liquidity(
                order.symbol,
                order.side,
                order.quantity
            )

            # Step 2: Calculate transaction costs
            logger.debug("Step 2: Calculating transaction costs")
            costs = await self.fee_optimizer.calculate_transaction_costs(
                liquidity,
                order.quantity,
                order.side
            )

            # Step 3: Optimize for minimal cost + maximal fill
            logger.debug("Step 3: Optimizing routing strategy")
            routes = await self.optimize_routes(liquidity, costs, order)

            # Step 4: Validate routing strategy
            logger.debug("Step 4: Validating routing strategy")
            validated_routes = await self.validate_routes(routes, order)

            # Calculate routing metrics
            total_expected_cost = sum(route.expected_fee for route in validated_routes)
            total_expected_fill = sum(route.quantity for route in validated_routes)
            execution_time = (time.time() - start_time) * 1000

            result = RoutingResult(
                routes=validated_routes,
                total_expected_cost=total_expected_cost,
                total_expected_fill=total_expected_fill,
                routing_strategy=self._determine_strategy(validated_routes),
                execution_time_ms=execution_time,
                success_rate=total_expected_fill / order.quantity if order.quantity > 0 else 0
            )

            # Log routing decision
            logger.info(f"Routing completed: {len(validated_routes)} routes, "
                       f"expected fill: {total_expected_fill:.2f}, "
                       f"expected cost: {total_expected_cost:.4f}")

            # Track routing history
            self.routing_history.append({
                "timestamp": datetime.now(),
                "order": order,
                "result": result,
                "liquidity_snapshot": liquidity
            })

            return result

        except Exception as e:
            logger.error(f"Error routing order {order.client_order_id}: {e}")
            # Return fallback routing
            return await self._fallback_routing(order)

    async def optimize_routes(
        self,
        liquidity: Dict[str, Dict],
        costs: Dict[str, Dict],
        order: Order
    ) -> List[Route]:
        """
        Optimize routing strategy based on liquidity and costs.

        Args:
            liquidity: Liquidity data by exchange
            costs: Cost data by exchange
            order: Original order

        Returns:
            Optimized list of routes
        """
        # Generate candidate routes
        candidate_routes = []

        for exchange in self.exchanges:
            if exchange not in liquidity or exchange not in costs:
                continue

            exchange_liquidity = liquidity[exchange]
            exchange_costs = costs[exchange]

            # Check if exchange has sufficient liquidity
            available_liquidity = exchange_liquidity.get("available_quantity", 0)
            if available_liquidity < self.min_liquidity_threshold:
                continue

            # Calculate route parameters
            route_quantity = min(order.quantity, available_liquidity)
            route_price = exchange_liquidity.get("best_price", order.price)
            route_fee = exchange_costs.get("fee_rate", 0.001) * route_quantity * route_price
            fill_ratio = exchange_liquidity.get("expected_fill_ratio", 1.0)
            latency = exchange_liquidity.get("latency_ms", 50)
            liquidity_score = exchange_liquidity.get("liquidity_score", 0.5)

            # Create route
            route = Route(
                exchange=exchange,
                quantity=route_quantity,
                price=route_price,
                expected_fee=route_fee,
                expected_fill_ratio=fill_ratio,
                latency_ms=latency,
                liquidity_score=liquidity_score,
                priority=self._calculate_route_priority(
                    route_fee, fill_ratio, latency, liquidity_score
                )
            )

            candidate_routes.append(route)

        # Sort routes by priority (higher priority = better)
        candidate_routes.sort(key=lambda r: r.priority, reverse=True)

        # Select optimal routes
        optimal_routes = await self._select_optimal_routes(candidate_routes, order)

        return optimal_routes

    async def validate_routes(self, routes: List[Route], order: Order) -> List[Route]:
        """
        Validate and filter routes.

        Args:
            routes: Candidate routes
            order: Original order

        Returns:
            Validated routes
        """
        validated_routes = []

        for route in routes:
            # Check latency constraints
            if route.latency_ms > self.max_latency_ms:
                logger.debug(f"Skipping {route.exchange} due to high latency: {route.latency_ms}ms")
                continue

            # Check fill ratio constraints
            if route.expected_fill_ratio < self.fill_ratio_threshold:
                logger.debug(f"Skipping {route.exchange} due to low fill ratio: {route.expected_fill_ratio}")
                continue

            # Check minimum quantity
            if route.quantity < self.config.get("min_route_quantity", 0.001):
                logger.debug(f"Skipping {route.exchange} due to insufficient quantity: {route.quantity}")
                continue

            validated_routes.append(route)

        # Ensure we don't exceed maximum exchanges per order
        if len(validated_routes) > self.max_exchanges_per_order:
            validated_routes = validated_routes[:self.max_exchanges_per_order]

        return validated_routes

    def _calculate_route_priority(
        self,
        fee: float,
        fill_ratio: float,
        latency: float,
        liquidity_score: float
    ) -> float:
        """
        Calculate route priority score.

        Args:
            fee: Expected fee
            fill_ratio: Expected fill ratio
            latency: Expected latency in ms
            liquidity_score: Liquidity quality score

        Returns:
            Priority score (higher is better)
        """
        # Normalize factors
        fee_score = max(0, 1 - fee / 100)  # Lower fee = higher score
        fill_score = fill_ratio  # Higher fill ratio = higher score
        latency_score = max(0, 1 - latency / self.max_latency_ms)  # Lower latency = higher score

        # Weighted combination
        weights = self.config.get("priority_weights", {
            "fee": 0.3,
            "fill_ratio": 0.4,
            "latency": 0.2,
            "liquidity": 0.1
        })

        priority = (
            weights["fee"] * fee_score +
            weights["fill_ratio"] * fill_score +
            weights["latency"] * latency_score +
            weights["liquidity"] * liquidity_score
        )

        return priority

    async def _select_optimal_routes(
        self,
        candidate_routes: List[Route],
        order: Order
    ) -> List[Route]:
        """
        Select optimal combination of routes.

        Args:
            candidate_routes: Sorted candidate routes
            order: Original order

        Returns:
            Selected routes
        """
        selected_routes = []
        remaining_quantity = order.quantity

        for route in candidate_routes:
            if remaining_quantity <= 0:
                break

            # Determine quantity for this route
            route_quantity = min(route.quantity, remaining_quantity)

            # Create route with adjusted quantity
            adjusted_route = Route(
                exchange=route.exchange,
                quantity=route_quantity,
                price=route.price,
                expected_fee=route.expected_fee * (route_quantity / route.quantity),
                expected_fill_ratio=route.expected_fill_ratio,
                latency_ms=route.latency_ms,
                liquidity_score=route.liquidity_score,
                priority=route.priority
            )

            selected_routes.append(adjusted_route)
            remaining_quantity -= route_quantity

        return selected_routes

    def _determine_strategy(self, routes: List[Route]) -> str:
        """Determine routing strategy name."""
        if not routes:
            return "no_routes"

        if len(routes) == 1:
            return f"single_exchange_{routes[0].exchange}"
        elif len(routes) <= 3:
            return f"multi_exchange_{len(routes)}_routes"
        else:
            return f"complex_routing_{len(routes)}_routes"

    async def _fallback_routing(self, order: Order) -> RoutingResult:
        """
        Fallback routing in case of errors.

        Args:
            order: Original order

        Returns:
            Fallback routing result
        """
        logger.warning(f"Using fallback routing for order {order.client_order_id}")

        # Simple fallback: route to first available exchange
        fallback_exchange = self.exchanges[0] if self.exchanges else "unknown"

        fallback_route = Route(
            exchange=fallback_exchange,
            quantity=order.quantity,
            price=order.price,
            expected_fee=order.quantity * (order.price or 0) * 0.002,  # 0.2% default fee
            expected_fill_ratio=0.9,
            latency_ms=100,
            liquidity_score=0.5,
            priority=0.5
        )

        return RoutingResult(
            routes=[fallback_route],
            total_expected_cost=fallback_route.expected_fee,
            total_expected_fill=fallback_route.quantity,
            routing_strategy="fallback",
            execution_time_ms=100,
            success_rate=0.9
        )

    def get_routing_statistics(self) -> Dict[str, Any]:
        """
        Get routing performance statistics.

        Returns:
            Routing statistics
        """
        if not self.routing_history:
            return {"status": "no_routing_history"}

        recent_routing = self.routing_history[-100:]  # Last 100 routes

        stats = {
            "total_routes": len(self.routing_history),
            "recent_routes": len(recent_routing),
            "average_execution_time_ms": np.mean([
                r["result"].execution_time_ms for r in recent_routing
            ]),
            "average_success_rate": np.mean([
                r["result"].success_rate for r in recent_routing
            ]),
            "exchange_usage": {},
            "strategy_distribution": {}
        }

        # Exchange usage statistics
        for routing in recent_routing:
            for route in routing["result"].routes:
                exchange = route.exchange
                if exchange not in stats["exchange_usage"]:
                    stats["exchange_usage"][exchange] = 0
                stats["exchange_usage"][exchange] += 1

        # Strategy distribution
        for routing in recent_routing:
            strategy = routing["result"].routing_strategy
            if strategy not in stats["strategy_distribution"]:
                stats["strategy_distribution"][strategy] = 0
            stats["strategy_distribution"][strategy] += 1

        return stats

    async def update_exchange_performance(self, exchange: str, metrics: Dict[str, float]) -> None:
        """
        Update exchange performance metrics.

        Args:
            exchange: Exchange name
            metrics: Performance metrics
        """
        if exchange not in self.exchange_performance:
            self.exchange_performance[exchange] = {}

        self.exchange_performance[exchange].update({
            "last_updated": datetime.now(),
            **metrics
        })

        logger.debug(f"Updated performance metrics for {exchange}")

    def get_exchange_performance(self, exchange: str) -> Dict[str, Any]:
        """
        Get performance metrics for an exchange.

        Args:
            exchange: Exchange name

        Returns:
            Performance metrics
        """
        return self.exchange_performance.get(exchange, {})