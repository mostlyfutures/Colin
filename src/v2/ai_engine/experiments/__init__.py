"""
Experiments and training for AI Engine.

This module provides training and evaluation capabilities:
- Model training pipelines
- Backtesting framework
- Model evaluation metrics
- Cross-validation utilities
"""

from .model_training import ModelTrainer
from .backtesting import Backtester
from .evaluation import ModelEvaluator

__all__ = [
    "ModelTrainer",
    "Backtester",
    "ModelEvaluator"
]