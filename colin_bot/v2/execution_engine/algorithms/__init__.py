"""
Execution Algorithms for Colin Trading Bot v2.0

This module provides institutional-grade execution algorithms for optimal
trade execution including VWAP, TWAP, and impact-aware execution.

Components:
- VWAP Executor: Volume-Weighted Average Price execution
- TWAP Executor: Time-Weighted Average Price execution
- Impact-Aware Executor: Market impact optimized execution
"""

from .vwap_executor import VWAPExecutor
from .twap_executor import TWAPExecutor
from .impact_aware_executor import ImpactAwareExecutor

__all__ = [
    "VWAPExecutor",
    "TWAPExecutor",
    "ImpactAwareExecutor"
]