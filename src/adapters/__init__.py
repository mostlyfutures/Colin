"""
Data adapters module for Colin Trading Bot.

Contains adapters for various APIs and data sources.
"""

from .binance import BinanceAdapter
from .coinglass import CoinGlassAdapter

__all__ = ["BinanceAdapter", "CoinGlassAdapter"]