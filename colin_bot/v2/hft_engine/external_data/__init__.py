"""
External Data Integration Layer

Integrates external data sources including news, economic calendar,
social sentiment, and alternative data for enhanced trading signals.
"""

from .news_analyzer import NewsAnalyzer, NewsEvent, NewsSentiment
from .economic_calendar import EconomicCalendar, EconomicEvent, EventImpact
from .sentiment_analyzer import SentimentAnalyzer, SentimentScore, SocialSource
from .alternative_data import AlternativeDataManager, DataSourceType
from .data_fusion_engine import ExternalDataFusionEngine, DataSignal

__all__ = [
    "NewsAnalyzer",
    "NewsEvent",
    "NewsSentiment",
    "EconomicCalendar",
    "EconomicEvent",
    "EventImpact",
    "SentimentAnalyzer",
    "SentimentScore",
    "SocialSource",
    "AlternativeDataManager",
    "DataSourceType",
    "ExternalDataFusionEngine",
    "DataSignal"
]