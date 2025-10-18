"""
Colin Trading Bot - Institutional-Grade Signal Scoring Bot

A sophisticated trading signal analysis system that incorporates
institutional-grade market structure and order flow principles
from the ICT framework and market microstructure research.
"""

__version__ = "1.0.0"
__author__ = "Colin Trading Bot Development Team"
__description__ = "Institutional-Grade Signal Scoring Bot for Crypto Perpetuals"

from .main import ColinTradingBot
from .core.config import ConfigManager
from .engine.institutional_scorer import InstitutionalScorer, InstitutionalSignal

__all__ = [
    "ColinTradingBot",
    "ConfigManager",
    "InstitutionalScorer",
    "InstitutionalSignal"
]