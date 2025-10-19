"""
Transformer Model for Multi-Timeframe Analysis - Colin Trading Bot v2.0

This module implements a Transformer-based model for multi-timeframe market analysis
with the following specifications from the PRP:
- Input: Multi-timeframe sequence (1m, 5m, 15m, 1h)
- Architecture: Multi-head attention (8 heads) + Position encoding
- Sequence Length: 256 tokens across timeframes
- Features: Market microstructure + technical indicators
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder
from math import sqrt
from loguru import logger
from typing import Dict, List, Tuple, Any, Optional

from ..base.ml_base import MLModelBase


class TransformerPredictor(MLModelBase):
    """
    Transformer-based model for multi-timeframe market analysis.

    This model uses Transformer architecture with multi-head attention to analyze
    market data across multiple timeframes simultaneously, providing superior
    pattern recognition capabilities compared to single-timeframe models.
    """

    def __init__(
        self,
        model_config: Dict[str, Any],
        training_config: Dict[str, Any],
        model_type: str = "transformer_predictor"
    ):
        """
        Initialize Transformer predictor with configuration.

        Args:
            model_config: Model-specific configuration
            training_config: Training-specific configuration
            model_type: Type identifier for the model
        """
        # Set default configurations
        default_model_config = {
            "sequence_length": 256,      # Total tokens across timeframes
            "d_model": 256,              # Model dimension
            "num_heads": 8,              # Multi-head attention heads
            "num_layers": 6,             # Number of transformer layers
            "d_ff": 1024,                # Feed-forward dimension
            "dropout": 0.1,              # Dropout rate
            "timeframes": ["1m", "5m", "15m", "1h"],  # Multiple timeframes
            "features_per_timeframe": 20, # Features per timeframe
            "max_sequence_length": 64,   # Max sequence per timeframe
            "num_classes": 3,            # Long/Short/Neutral
            "use_positional_encoding": True,
            "use_layer_norm": True
        }

        default_training_config = {
            "learning_rate": 0.0001,
            "batch_size": 16,
            "epochs": 100,
            "early_stopping_patience": 15,
            "validation_split": 0.2,
            "min_training_samples": 2000,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "weight_decay": 1e-4,
            "scheduler_patience": 8,
            "gradient_clipping": 1.0,
            "warmup_epochs": 5
        }

        # Merge with provided configs
        model_config = {**default_model_config, **model_config}
        training_config = {**default_training_config, **training_config}

        super().__init__(model_config, training_config, model_type)

        # Initialize device
        self.device = torch.device(self.training_config["device"])
        logger.info(f"Transformer model using device: {self.device}")

        # Label encoder for target variables
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(['long', 'short', 'neutral'])

        # Training history
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.best_epoch = 0

        # Multi-timeframe data processing
        self.timeframe_config = self._setup_timeframe_config()

    def _setup_timeframe_config(self) -> Dict[str, Any]:
        """Setup configuration for multi-timeframe processing."""
        timeframes = self.model_config["timeframes"]
        max_seq_len = self.model_config["max_sequence_length"]
        features_per_tf = self.model_config["features_per_timeframe"]

        return {
            "timeframes": timeframes,
            "sequence_per_timeframe": max_seq_len,
            "total_sequence_length": len(timeframes) * max_seq_len,
            "features_per_timeframe": features_per_tf,
            "input_dim": len(timeframes) * max_seq_len * features_per_tf
        }

    def build_model(self, input_shape: Tuple[int, ...]) -> nn.Module:
        """
        Build Transformer model architecture.

        Args:
            input_shape: Shape of input data

        Returns:
            Compiled Transformer model
        """
        # Validate input shape matches multi-timeframe config
        expected_length = self.timeframe_config["total_sequence_length"]
        expected_features = self.model_config["features_per_timeframe"]

        if input_shape[1] != expected_length * expected_features:
            logger.warning(f"Input shape {input_shape} doesn't match expected "
                         f"({expected_length * expected_features},)")

        model = TransformerNetwork(
            input_dim=input_shape[1],
            d_model=self.model_config["d_model"],
            num_heads=self.model_config["num_heads"],
            num_layers=self.model_config["num_layers"],
            d_ff=self.model_config["d_ff"],
            num_classes=self.model_config["num_classes"],
            dropout=self.model_config["dropout"],
            use_positional_encoding=self.model_config["use_positional_encoding"],
            use_layer_norm=self.model_config["use_layer_norm"],
            max_seq_len=expected_length
        )

        logger.info(f"Built Transformer model with {self._count_parameters()} parameters")
        logger.info(f"Input shape: {input_shape}, Output classes: {self.model_config['num_classes']}")

        return model.to(self.device)

    def preprocess_data(
        self,
        data: pd.DataFrame,
        target_column: str = "target"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess multi-timeframe data for Transformer training/inference.

        Args:
            data: Input multi-timeframe data DataFrame
            target_column: Name of target column

        Returns:
            Tuple of (sequences, targets)
        """
        logger.debug(f"Preprocessing multi-timeframe data with shape: {data.shape}")

        # Separate features and targets
        feature_columns = [col for col in data.columns if col != target_column]
        features = data[feature_columns].values
        targets = data[target_column].values

        # Encode targets
        targets_encoded = self.label_encoder.transform(targets)

        # Create multi-timeframe sequences
        sequences = self._create_multi_timeframe_sequences(features)

        # Align targets with sequences
        seq_length = self.timeframe_config["sequence_per_timeframe"]
        targets_aligned = targets_encoded[seq_length - 1:]

        logger.debug(f"Created multi-timeframe sequences: {sequences.shape}, "
                    f"Targets: {targets_aligned.shape}")

        return sequences, targets_aligned

    def _create_multi_timeframe_sequences(self, data: np.ndarray) -> np.ndarray:
        """
        Create sequences from multi-timeframe data.

        Args:
            data: Multi-timeframe data

        Returns:
            Array of sequences
        """
        seq_per_tf = self.timeframe_config["sequence_per_timeframe"]
        total_seq_len = self.timeframe_config["total_sequence_length"]
        features_per_tf = self.model_config["features_per_timeframe"]

        sequences = []

        for i in range(seq_per_tf - 1, len(data)):
            # Extract sequence data
            sequence_data = data[i - seq_per_tf + 1:i + 1]

            # Reshape for multi-timeframe processing
            # Expected shape: (num_timeframes, seq_per_timeframe, features_per_timeframe)
            num_timeframes = len(self.model_config["timeframes"])

            if len(sequence_data.shape) == 1:
                # Single row, reshape to multi-timeframe format
                sequence_data = sequence_data.reshape(1, -1)

            # Ensure we have enough data for all timeframes
            if sequence_data.shape[0] >= num_timeframes:
                # Take the most recent num_timeframes rows
                tf_data = sequence_data[-num_timeframes:]

                # Reshape to (sequence_length, features)
                reshaped_sequence = tf_data.flatten().reshape(total_seq_len, features_per_tf)

                # Flatten to (1, total_sequence_length * features_per_timeframe)
                flattened_sequence = reshaped_sequence.flatten()

                sequences.append(flattened_sequence)

        return np.array(sequences)

    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Make predictions on input features.

        Args:
            features: Input feature array

        Returns:
            Prediction array of class labels
        """
        if self.model is None:
            raise ValueError("Model must be built before making predictions")

        self.model.eval()
        with torch.no_grad():
            # Convert to tensor
            if len(features.shape) == 1:
                # Single sample, add batch dimension
                features = features.reshape(1, -1)

            features_tensor = torch.FloatTensor(features).to(self.device)

            # Get predictions
            outputs = self.model(features_tensor)
            _, predicted = torch.max(outputs.data, 1)

            return self.label_encoder.inverse_transform(predicted.cpu().numpy())

    def _get_prediction_probabilities(self, features: np.ndarray) -> np.ndarray:
        """
        Get prediction probabilities.

        Args:
            features: Input feature array

        Returns:
            Probability array
        """
        if self.model is None:
            raise ValueError("Model must be built before getting probabilities")

        self.model.eval()
        with torch.no_grad():
            if len(features.shape) == 1:
                features = features.reshape(1, -1)

            features_tensor = torch.FloatTensor(features).to(self.device)
            outputs = self.model(features_tensor)
            probabilities = torch.softmax(outputs, dim=1)

            return probabilities.cpu().numpy()

    def _execute_training(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray],
        y_val: Optional[np.ndarray]
    ) -> Dict[str, Any]:
        """
        Execute Transformer training with PyTorch.

        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets

        Returns:
            Training results
        """
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.LongTensor(y_train).to(self.device)

        if X_val is not None and y_val is not None:
            X_val_tensor = torch.FloatTensor(X_val).to(self.device)
            y_val_tensor = torch.LongTensor(y_val).to(self.device)
        else:
            X_val_tensor = None
            y_val_tensor = None

        # Create data loaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.training_config["batch_size"],
            shuffle=True
        )

        if X_val_tensor is not None:
            val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.training_config["batch_size"],
                shuffle=False
            )
        else:
            val_loader = None

        # Initialize optimizer and loss function
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.training_config["learning_rate"],
            weight_decay=self.training_config["weight_decay"]
        )

        criterion = nn.CrossEntropyLoss()

        # Learning rate scheduler with warmup
        warmup_epochs = self.training_config["warmup_epochs"]
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.training_config["epochs"] - warmup_epochs,
            eta_min=1e-6
        )

        # Training loop
        best_model_state = None
        patience_counter = 0

        for epoch in range(self.training_config["epochs"]):
            # Warmup phase
            if epoch < warmup_epochs:
                warmup_lr = self.training_config["learning_rate"] * (epoch + 1) / warmup_epochs
                for param_group in optimizer.param_groups:
                    param_group['lr'] = warmup_lr

            # Training phase
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for batch_features, batch_targets in train_loader:
                optimizer.zero_grad()

                outputs = self.model(batch_features)
                loss = criterion(outputs, batch_targets)

                # Gradient clipping
                if self.training_config["gradient_clipping"] > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.training_config["gradient_clipping"]
                    )

                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += batch_targets.size(0)
                train_correct += (predicted == batch_targets).sum().item()

            train_accuracy = 100 * train_correct / train_total
            avg_train_loss = train_loss / len(train_loader)

            # Learning rate scheduling (after warmup)
            if epoch >= warmup_epochs:
                scheduler.step()

            # Validation phase
            val_loss = 0.0
            val_accuracy = 0.0

            if val_loader:
                self.model.eval()
                val_correct = 0
                val_total = 0

                with torch.no_grad():
                    for batch_features, batch_targets in val_loader:
                        outputs = self.model(batch_features)
                        loss = criterion(outputs, batch_targets)

                        val_loss += loss.item()
                        _, predicted = torch.max(outputs.data, 1)
                        val_total += batch_targets.size(0)
                        val_correct += (predicted == batch_targets).sum().item()

                val_accuracy = 100 * val_correct / val_total
                avg_val_loss = val_loss / len(val_loader)

                # Early stopping check
                if avg_val_loss < self.best_val_loss:
                    self.best_val_loss = avg_val_loss
                    self.best_epoch = epoch
                    best_model_state = self.model.state_dict().copy()
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= self.training_config["early_stopping_patience"]:
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                    break

            # Log progress
            if epoch % 10 == 0 or epoch == self.training_config["epochs"] - 1:
                current_lr = optimizer.param_groups[0]['lr']
                logger.info(
                    f"Epoch {epoch + 1}/{self.training_config['epochs']} - "
                    f"LR: {current_lr:.6f} - "
                    f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}%"
                )
                if val_loader:
                    logger.info(
                        f"  Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%"
                    )

            # Store metrics
            self.train_losses.append(avg_train_loss)
            if val_loader:
                self.val_losses.append(avg_val_loss)

        # Load best model
        if best_model_state:
            self.model.load_state_dict(best_model_state)
            logger.info(f"Loaded best model from epoch {self.best_epoch + 1}")

        return {
            "final_train_loss": self.train_losses[-1],
            "final_train_accuracy": train_accuracy,
            "final_val_loss": self.val_losses[-1] if self.val_losses else None,
            "final_val_accuracy": val_accuracy if val_loader else None,
            "best_epoch": self.best_epoch,
            "best_val_loss": self.best_val_loss,
            "epochs_trained": epoch + 1
        }

    def _get_model_state(self) -> Dict[str, Any]:
        """Get model state for saving."""
        return {
            "model_state_dict": self.model.state_dict() if self.model else None,
            "label_encoder_classes": self.label_encoder.classes_.tolist(),
            "model_config": self.model_config,
            "training_config": self.training_config,
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "best_val_loss": self.best_val_loss,
            "best_epoch": self.best_epoch,
            "timeframe_config": self.timeframe_config
        }

    def _restore_model_state(self, state: Dict[str, Any]) -> None:
        """Restore model state from saved data."""
        # Rebuild model if needed
        if self.model is None and "model_config" in state:
            input_dim = state["model_config"]["features_per_timeframe"] * state["timeframe_config"]["total_sequence_length"]
            input_shape = (None, input_dim)
            self.model = self.build_model(input_shape)

        # Restore model weights
        if self.model and "model_state_dict" in state:
            self.model.load_state_dict(state["model_state_dict"])

        # Restore label encoder
        if "label_encoder_classes" in state:
            self.label_encoder.classes_ = np.array(state["label_encoder_classes"])

        # Restore other state
        self.model_config = state.get("model_config", self.model_config)
        self.training_config = state.get("training_config", self.training_config)
        self.train_losses = state.get("train_losses", [])
        self.val_losses = state.get("val_losses", [])
        self.best_val_loss = state.get("best_val_loss", float('inf'))
        self.best_epoch = state.get("best_epoch", 0)
        self.timeframe_config = state.get("timeframe_config", self.timeframe_config)

    def _count_parameters(self) -> int:
        """Count trainable parameters in the model."""
        if self.model is None:
            return 0
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)


class TransformerNetwork(nn.Module):
    """
    Transformer Network Architecture for Multi-Timeframe Analysis

    This implements the Transformer architecture with multi-head attention for
    analyzing market patterns across multiple timeframes simultaneously.
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
        d_ff: int,
        num_classes: int,
        dropout: float = 0.1,
        use_positional_encoding: bool = True,
        use_layer_norm: bool = True,
        max_seq_len: int = 256
    ):
        """
        Initialize Transformer network.

        Args:
            input_dim: Input feature dimension
            d_model: Model dimension
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            d_ff: Feed-forward dimension
            num_classes: Number of output classes
            dropout: Dropout rate
            use_positional_encoding: Whether to use positional encoding
            use_layer_norm: Whether to use layer normalization
            max_seq_len: Maximum sequence length for positional encoding
        """
        super(TransformerNetwork, self).__init__()

        self.d_model = d_model
        self.use_positional_encoding = use_positional_encoding

        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)

        # Positional encoding
        if use_positional_encoding:
            self.positional_encoding = PositionalEncoding(d_model, max_seq_len)

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation='relu',
            batch_first=True,
            norm_first=use_layer_norm
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # Output layers
        self.layer_norm = nn.LayerNorm(d_model) if use_layer_norm else nn.Identity()
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )

    def forward(self, x):
        """
        Forward pass through the Transformer network.

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        # Reshape for sequence processing
        # Add sequence dimension: (batch_size, 1, input_dim)
        x = x.unsqueeze(1)

        # Project to model dimension
        x = self.input_projection(x) * sqrt(self.d_model)

        # Add positional encoding
        if self.use_positional_encoding:
            x = self.positional_encoding(x)

        # Transformer encoder
        x = self.transformer_encoder(x)

        # Take the first (and only) sequence element
        x = x[:, 0, :]

        # Layer normalization and dropout
        x = self.layer_norm(x)
        x = self.dropout(x)

        # Classification
        output = self.classifier(x)

        return output


class PositionalEncoding(nn.Module):
    """
    Positional encoding for Transformer model.

    This adds positional information to the input embeddings, allowing the model
    to understand the sequence order of the input data.
    """

    def __init__(self, d_model: int, max_len: int = 5000):
        """
        Initialize positional encoding.

        Args:
            d_model: Model dimension
            max_len: Maximum sequence length
        """
        super(PositionalEncoding, self).__init__()

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                           (-np.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Apply positional encoding to input.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)

        Returns:
            Tensor with positional encoding added
        """
        return x + self.pe[:x.size(1), :].transpose(0, 1)