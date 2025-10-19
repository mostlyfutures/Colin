"""
Core module for Colin Trading Bot.

Contains configuration management and shared utilities.
"""

from .config import Config, ConfigManager, config_manager

__all__ = ["Config", "ConfigManager", "config_manager"]