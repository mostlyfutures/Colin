"""
Real-Time Risk Monitoring Module for Colin Trading Bot v2.0

This module provides real-time risk monitoring capabilities including:
- Pre-trade risk validation
- Position monitoring and drawdown control
- Circuit breaker implementation
- Real-time risk decision making
"""

__version__ = "2.0.0"
__author__ = "Colin Trading Bot Team"

from .risk_monitor import RealTimeRiskController, RiskDecision
from .position_monitor import PositionMonitor
from .drawdown_controller import DrawdownController

__all__ = [
    "RealTimeRiskController",
    "RiskDecision",
    "PositionMonitor",
    "DrawdownController"
]