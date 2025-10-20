"""
Data Source Adapters

Individual adapters for different cryptocurrency data sources.
"""

from .base_adapter import BaseAdapter
from .coingecko_adapter import CoinGeckoAdapter
from .kraken_adapter import KrakenAdapter
from .cryptocompare_adapter import CryptoCompareAdapter
from .alternative_me_adapter import AlternativeMeAdapter
from .hyperliquid_adapter import HyperliquidAdapter

__all__ = [
    "BaseAdapter",
    "CoinGeckoAdapter",
    "KrakenAdapter",
    "CryptoCompareAdapter",
    "AlternativeMeAdapter",
    "HyperliquidAdapter"
]