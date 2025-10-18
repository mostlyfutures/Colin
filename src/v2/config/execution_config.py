"""
Execution Engine Configuration for Colin Trading Bot v2.0

This module provides configuration for the execution engine components.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any


@dataclass
class ExecutionConfig:
    """Execution engine configuration."""
    # Exchange configuration
    supported_exchanges: List[str] = field(default_factory=lambda: [
        "binance", "bybit", "okx", "deribit"
    ])
    default_exchange: str = "binance"

    # Smart routing parameters
    max_exchanges_per_order: int = 3
    liquidity_threshold: float = 10000.0
    routing_strategy: str = "cost_optimized"
    max_slippage_bps: float = 10.0

    # Algorithm parameters
    vwap_participation_rate: float = 0.10
    vwap_time_window_minutes: int = 5
    twap_interval_seconds: int = 60
    twap_duration_seconds: int = 3600

    # Execution parameters
    order_timeout_seconds: int = 30
    retry_attempts: int = 3
    fill_rate_threshold: float = 0.95