"""
AI Engine for Colin Trading Bot v2.0

This module provides AI-powered signal generation capabilities including:
- LSTM-based price prediction
- Transformer models for multi-timeframe analysis
- Ensemble model combinations
- Feature engineering pipelines
- Model training and validation
"""

__version__ = "2.0.0"
__author__ = "Colin Trading Bot Team"

from .base.ml_base import MLModelBase
from .base.feature_base import FeatureEngineerBase
from .base.pipeline_base import MLPipelineBase

__all__ = [
    "MLModelBase",
    "FeatureEngineerBase",
    "MLPipelineBase"
]