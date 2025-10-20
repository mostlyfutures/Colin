"""
Execution Engine for Colin Trading Bot v2.0

This module provides automated trading execution capabilities including:
- Smart order routing across multiple exchanges
- Execution algorithms (VWAP, TWAP, Impact-aware)
- Market impact modeling
- Liquidity aggregation and optimization
- Multi-exchange connectivity
"""

__version__ = "2.0.0"
__author__ = "Colin Trading Bot Team"

from .smart_routing import SmartOrderRouter

__all__ = [
    "SmartOrderRouter"
]