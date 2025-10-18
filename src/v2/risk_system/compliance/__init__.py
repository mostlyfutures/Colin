"""
Compliance Engine Module for Colin Trading Bot v2.0

This module provides regulatory compliance capabilities including:
- Pre-trade compliance checking
- Regulatory reporting automation
- Compliance breach detection and alerts
- Audit trail maintenance and review
"""

__version__ = "2.0.0"
__author__ = "Colin Trading Bot Team"

from .pre_trade_check import PreTradeChecker
from .compliance_monitor import ComplianceMonitor

__all__ = [
    "PreTradeChecker",
    "ComplianceMonitor"
]