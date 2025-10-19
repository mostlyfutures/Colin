"""
Structure module for Colin Trading Bot.

Contains ICT structure detection and analysis.
"""

from .ict_detector import ICTDetector, ICTStructure, FairValueGap, OrderBlock, BreakOfStructure

__all__ = [
    "ICTDetector", "ICTStructure", "FairValueGap", "OrderBlock", "BreakOfStructure"
]