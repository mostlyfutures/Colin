"""
Risk Management System for Colin Trading Bot v2.0

This module provides comprehensive risk management capabilities including:
- Real-time risk monitoring and control
- Portfolio VaR calculation and stress testing
- Compliance engine with regulatory checks
- Position limits and concentration monitoring
- Circuit breakers and emergency controls
"""

__version__ = "2.0.0"
__author__ = "Colin Trading Bot Team"

from .real_time.risk_monitor import RealTimeRiskController, RiskDecision
from .real_time.position_monitor import PositionMonitor
from .real_time.drawdown_controller import DrawdownController
from .portfolio.var_calculator import VaRCalculator
from .portfolio.correlation_analyzer import CorrelationAnalyzer
from .portfolio.stress_tester import StressTester
from .compliance.pre_trade_check import PreTradeChecker
from .compliance.compliance_monitor import ComplianceMonitor

__all__ = [
    "RealTimeRiskController",
    "RiskDecision",
    "PositionMonitor",
    "DrawdownController",
    "VaRCalculator",
    "CorrelationAnalyzer",
    "StressTester",
    "PreTradeChecker",
    "ComplianceMonitor"
]