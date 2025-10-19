"""
Smart Order Routing for Colin Trading Bot v2.0

This module provides intelligent order routing capabilities across multiple
exchanges to optimize execution quality and minimize costs.

Components:
- Liquidity Aggregation: Combine liquidity from multiple sources
- Smart Router: Intelligent order routing decisions
- Fee Optimization: Minimize transaction costs
"""

from .liquidity_aggregator import LiquidityAggregator
from .router import SmartOrderRouter
from .fee_optimizer import FeeOptimizer

__all__ = [
    "LiquidityAggregator",
    "SmartOrderRouter",
    "FeeOptimizer"
]