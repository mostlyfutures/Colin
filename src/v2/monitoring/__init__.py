"""
Monitoring System for Colin Trading Bot v2.0

This module provides comprehensive monitoring capabilities including:
- Real-time metrics collection
- System health monitoring
- Performance tracking
- Alert management
"""

__version__ = "2.0.0"
__author__ = "Colin Trading Bot Team"

from .metrics import MetricsCollector
from .alerts import AlertManager
from .dashboard import MonitoringDashboard

__all__ = [
    "MetricsCollector",
    "AlertManager",
    "MonitoringDashboard"
]