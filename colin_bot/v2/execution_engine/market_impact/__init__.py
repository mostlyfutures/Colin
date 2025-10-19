"""
Market Impact Modeling for Colin Trading Bot v2.0

This module provides market impact analysis and cost optimization capabilities
to minimize trading costs and optimize execution quality.

Components:
- Market Impact Model: Predict execution impact on prices
- Cost Optimizer: Optimize execution parameters
"""

from .impact_model import MarketImpactModel
from .cost_optimizer import CostOptimizer

__all__ = [
    "MarketImpactModel",
    "CostOptimizer"
]