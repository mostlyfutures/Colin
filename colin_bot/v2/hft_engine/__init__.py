"""
Colin Trading Bot v2.0 - High-Frequency Trading Engine

This module implements advanced high-frequency trading strategies based on
market microstructure analysis, order flow forecasting, and liquidity detection.

Key Features:
- Order Flow Imbalance (OFI) calculation using Hawkes processes
- Order book skew analysis with dynamic thresholds
- Liquidity detection and heatmap visualization
- Multi-signal fusion for enhanced accuracy
- Dynamic position sizing based on market conditions
- Circuit breakers for risk management
- Sub-50ms execution latency
"""

__version__ = "1.0.0"
__author__ = "Colin Trading Bot Team"

from .signal_processing.ofi_calculator import OFICalculator
from .signal_processing.book_skew_analyzer import BookSkewAnalyzer
from .signal_processing.liquidity_detector import LiquidityDetector
from .signal_processing.signal_fusion import SignalFusionEngine
from .risk_management.dynamic_sizing import DynamicPositionSizer
from .risk_management.circuit_breaker import CircuitBreakerSystem
from .data_ingestion.market_data_manager import HFTDataManager
from .execution.hft_executor import HFTExecutor

__all__ = [
    "OFICalculator",
    "BookSkewAnalyzer",
    "LiquidityDetector",
    "SignalFusionEngine",
    "DynamicPositionSizer",
    "CircuitBreakerSystem",
    "HFTDataManager",
    "HFTExecutor"
]