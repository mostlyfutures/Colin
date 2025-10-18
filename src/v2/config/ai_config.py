"""
AI Model Configuration for Colin Trading Bot v2.0

This module provides configuration for AI/ML components.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any


@dataclass
class AIModelConfig:
    """AI model configuration."""
    model_path: str = "models/"
    lstm_model_path: str = "models/lstm_model.pt"
    transformer_model_path: str = "models/transformer_model.pt"
    ensemble_model_path: str = "models/ensemble_model.pt"

    # Model parameters
    lstm_sequence_length: int = 60
    lstm_hidden_size: int = 128
    transformer_d_model: int = 256
    transformer_num_heads: int = 8

    # Training parameters
    batch_size: int = 32
    learning_rate: float = 0.001
    epochs: int = 100

    # Inference parameters
    confidence_threshold: float = 0.65
    max_signals_per_minute: int = 20
    feature_update_interval_seconds: int = 60