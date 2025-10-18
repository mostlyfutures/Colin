"""
Configuration Management for Colin Trading Bot v2.0

This module provides configuration management for all v2 components.
"""

from .main_config import MainConfigManager, get_main_config_manager, get_main_config
from .risk_config import RiskConfigManager, get_risk_config_manager, get_risk_config
from .ai_config import AIModelConfig
from .execution_config import ExecutionConfig

__all__ = [
    "MainConfigManager",
    "get_main_config_manager",
    "get_main_config",
    "RiskConfigManager",
    "get_risk_config_manager",
    "get_risk_config",
    "AIModelConfig",
    "ExecutionConfig"
]