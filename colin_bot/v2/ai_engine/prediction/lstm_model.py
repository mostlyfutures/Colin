"""
LSTM Price Prediction Model for Colin Trading Bot v2.0

This module implements an LSTM-based price prediction model with the following
specifications from the PRP:
- Input: 60-minute windows with 50+ features
- Architecture: 2 LSTM layers (128 units) + Dense layers
- Output: 3-class probability distribution (Long/Short/Neutral)
- Training: Adam optimizer, learning rate 0.001
- Validation: Time series cross-validation
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from loguru import logger
from typing import Dict, List, Tuple, Any, Optional

from ..base.ml_base import MLModelBase


class LSTMPricePredictor(MLModelBase):
    """
    LSTM-based price prediction model for trading signals.

    This model uses LSTM networks to predict price direction (Long/Short/Neutral)
    based on historical market data and engineered features. It follows the
    specifications outlined in the PRP for institutional-grade performance.
    """

    def __init__(
        self,
        model_config: Dict[str, Any],
        training_config: Dict[str, Any],
        model_type: str = "lstm_predictor"
    ):
        """
        Initialize LSTM predictor with configuration.

        Args:
            model_config: Model-specific configuration
            training_config: Training-specific configuration
            model_type: Type identifier for the model
        """
        # Set default configurations
        default_model_config = {
            "sequence_length": 60,  # 60-minute windows
            "hidden_size": 128,     # LSTM hidden units
            "num_layers": 2,        # Number of LSTM layers
            "dropout": 0.2,         # Dropout rate
            "bidirectional": False, # Unidirectional LSTM
            "input_size": None,      # Will be set based on data
            "num_classes": 3,       # Long/Short/Neutral
            "use_batch_norm": True  # Batch normalization
        }

        default_training_config = {
            "learning_rate": 0.001,
            "batch_size": 32,
            "epochs": 100,
            "early_stopping_patience": 10,
            "validation_split": 0.2,
            "min_training_samples": 1000,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "weight_decay": 1e-5,
            "scheduler_patience": 5,
            "gradient_clipping": 1.0
        }

        # Merge with provided configs
        model_config = {**default_model_config, **model_config}
        training_config = {**default_training_config, **training_config}

        super().__init__(model_config, training_config, model_type)

        # Initialize device
        self.device = torch.device(self.training_config["device"])
        logger.info(f"LSTM model using device: {self.device}")

        # Label encoder for target variables
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(['long', 'short', 'neutral'])

        # Training history
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.best_epoch = 0

    def build_model(self, input_shape: Tuple[int, ...]) -> nn.Module:
        """
        Build LSTM model architecture.

        Args:
            input_shape: Shape of input data (sequence_length, features)

        Returns:
            Compiled LSTM model
        """
        sequence_length, input_size = input_shape
        self.model_config["input_size"] = input_size

        model = LSTMNetwork(
            input_size=input_size,
            hidden_size=self.model_config["hidden_size"],
            num_layers=self.model_config["num_layers"],
            num_classes=self.model_config["num_classes"],
            dropout=self.model_config["dropout"],
            bidirectional=self.model_config["bidirectional"],
            use_batch_norm=self.model_config["use_batch_norm"]
        )

        logger.info(f"Built LSTM model with {self._count_parameters()} parameters")
        logger.info(f"Input shape: {input_shape}, Output classes: {self.model_config['num_classes']}")

        return model.to(self.device)

    def preprocess_data(
        self,
        data: pd.DataFrame,
        target_column: str = "target"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess data for LSTM training/inference.

        Args:
            data: Input data DataFrame with features and target
            target_column: Name of target column

        Returns:
            Tuple of (sequences, targets)
        """
        logger.debug(f"Preprocessing data with shape: {data.shape}")

        # Separate features and targets
        feature_columns = [col for col in data.columns if col != target_column]
        features = data[feature_columns].values
        targets = data[target_column].values

        # Encode targets
        targets_encoded = self.label_encoder.transform(targets)

        # Create sequences
        sequences = self._create_sequences(features, self.model_config["sequence_length"])

        # Align targets with sequences
        targets_aligned = targets_encoded[self.model_config["sequence_length"] - 1:]

        logger.debug(f"Created sequences: {sequences.shape}, Targets: {targets_aligned.shape}")

        return sequences, targets_aligned

    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Make predictions on input features.

        Args:
            features: Input feature array of shape (samples, sequence_length, features)

        Returns:
            Prediction array of class labels
        """
        if self.model is None:
            raise ValueError("Model must be built before making predictions")

        self.model.eval()
        with torch.no_grad():
            # Convert to tensor
            if len(features.shape) == 2:
                # Single sample, add batch dimension
                features = features.reshape(1, features.shape[0], features.shape[1])

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
            Probability array of shape (samples, num_classes)
        """
        if self.model is None:
            raise ValueError("Model must be built before getting probabilities")

        self.model.eval()
        with torch.no_grad():
            if len(features.shape) == 2:
                features = features.reshape(1, features.shape[0], features.shape[1])

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
        Execute LSTM training with PyTorch.

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
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.training_config["learning_rate"],
            weight_decay=self.training_config["weight_decay"]
        )

        criterion = nn.CrossEntropyLoss()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=self.training_config["scheduler_patience"],
            verbose=True
        )

        # Training loop
        best_model_state = None
        patience_counter = 0

        for epoch in range(self.training_config["epochs"]):
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

                # Learning rate scheduling
                scheduler.step(avg_val_loss)

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
                logger.info(
                    f"Epoch {epoch + 1}/{self.training_config['epochs']} - "
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

    def _create_sequences(self, data: np.ndarray, sequence_length: int) -> np.ndarray:
        """
        Create sequences from time series data.

        Args:
            data: Input time series data
            sequence_length: Length of each sequence

        Returns:
            Array of sequences
        """
        sequences = []
        for i in range(sequence_length - 1, len(data)):
            sequences.append(data[i - sequence_length + 1:i + 1])
        return np.array(sequences)

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
            "best_epoch": self.best_epoch
        }

    def _restore_model_state(self, state: Dict[str, Any]) -> None:
        """Restore model state from saved data."""
        # Rebuild model if needed
        if self.model is None and "model_config" in state:
            # We need to infer input shape from saved config
            input_size = state["model_config"].get("input_size", 50)
            input_shape = (state["model_config"]["sequence_length"], input_size)
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

    def _count_parameters(self) -> int:
        """Count trainable parameters in the model."""
        if self.model is None:
            return 0
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)


class LSTMNetwork(nn.Module):
    """
    LSTM Network Architecture for Price Prediction

    This implements the LSTM architecture specified in the PRP:
    - 2 LSTM layers with 128 hidden units each
    - Dropout regularization
    - Dense output layer with softmax activation
    - Optional batch normalization
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        num_classes: int,
        dropout: float = 0.2,
        bidirectional: bool = False,
        use_batch_norm: bool = True
    ):
        """
        Initialize LSTM network.

        Args:
            input_size: Number of input features
            hidden_size: Number of hidden units in LSTM
            num_layers: Number of LSTM layers
            num_classes: Number of output classes
            dropout: Dropout rate
            bidirectional: Whether to use bidirectional LSTM
            use_batch_norm: Whether to use batch normalization
        """
        super(LSTMNetwork, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )

        # Calculate LSTM output size
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size

        # Batch normalization
        self.use_batch_norm = use_batch_norm
        if use_batch_norm:
            self.batch_norm = nn.BatchNorm1d(lstm_output_size)

        # Dropout layer
        self.dropout = nn.Dropout(dropout)

        # Dense layers
        self.fc1 = nn.Linear(lstm_output_size, 32)
        self.fc2 = nn.Linear(32, num_classes)

        # Activation functions
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)

        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)

        # Take the last time step output
        last_output = lstm_out[:, -1, :]

        # Batch normalization
        if self.use_batch_norm:
            last_output = self.batch_norm(last_output)

        # Dropout
        last_output = self.dropout(last_output)

        # Dense layers
        out = self.relu(self.fc1(last_output))
        out = self.dropout(out)
        out = self.fc2(out)

        return out