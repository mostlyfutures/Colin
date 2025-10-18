"""
Engine module for Colin Trading Bot.

Contains the main institutional signal scoring engine.
"""

from .institutional_scorer import InstitutionalScorer, InstitutionalSignal, SignalComponents

__all__ = ["InstitutionalScorer", "InstitutionalSignal", "SignalComponents"]