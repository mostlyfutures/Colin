"""
Data Sources Package for Colin Trading Bot v2.0

This package provides multi-source market data capabilities with intelligent
failover, caching, and standardized data formats.
"""

from .market_data_manager import MarketDataManager
from .models import StandardMarketData, MarketDataSummary
from .config import MarketDataConfig

__all__ = [
    "MarketDataManager",
    "StandardMarketData",
    "MarketDataSummary",
    "MarketDataConfig"
]