"""
Portfolio Risk Analytics Module for Colin Trading Bot v2.0

This module provides portfolio-level risk analysis capabilities including:
- Value-at-Risk (VaR) calculation
- Correlation analysis and monitoring
- Stress testing and scenario analysis
- Portfolio risk metrics and reporting
"""

__version__ = "2.0.0"
__author__ = "Colin Trading Bot Team"

from .var_calculator import VaRCalculator
from .correlation_analyzer import CorrelationAnalyzer
from .stress_tester import StressTester

__all__ = [
    "VaRCalculator",
    "CorrelationAnalyzer",
    "StressTester"
]