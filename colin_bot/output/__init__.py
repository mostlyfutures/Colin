"""
Output module for Colin Trading Bot.

Contains risk-aware output formatting and signal presentation.
"""

from .formatter import OutputFormatter, FormattedSignal, RiskMetrics

__all__ = ["OutputFormatter", "FormattedSignal", "RiskMetrics"]