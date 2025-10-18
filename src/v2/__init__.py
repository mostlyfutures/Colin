"""
Colin Trading Bot v2.0 - AI-Powered Institutional Trading System

This package transforms the original signal scoring bot into a comprehensive
AI-powered trading system with automated execution, real-time risk management,
and institutional-grade capabilities.

Key Features:
- AI-driven signal generation (LSTM, Transformer, Ensemble)
- Automated execution engine with smart order routing
- Real-time risk management and compliance
- Multi-exchange connectivity
- Sub-50ms execution latency
- >65% directional accuracy target

Architecture:
├── ai_engine/          # AI/ML components
├── execution_engine/   # Order execution and routing
├── risk_system/       # Risk management and compliance
├── market_access/     # Exchange connectivity
├── monitoring/        # System monitoring and alerting
└── config/           # Configuration management
"""

__version__ = "2.0.0"
__author__ = "Colin Trading Bot Team"

from .ai_engine import MLModelBase, FeatureEngineerBase, MLPipelineBase

__all__ = [
    "MLModelBase",
    "FeatureEngineerBase",
    "MLPipelineBase"
]