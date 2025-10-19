"""
Base ML Model class for Colin Trading Bot v2.0

This module provides the foundational ML model class that all prediction models
should inherit from. It includes common functionality for model training,
validation, prediction, and persistence.
"""

import abc
import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from loguru import logger
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import TimeSeriesSplit


class MLModelBase(abc.ABC):
    """
    Abstract base class for all ML models in Colin Trading Bot v2.0.

    This class provides common functionality for:
    - Model initialization and configuration
    - Training and validation workflows
    - Prediction and inference
    - Model persistence and versioning
    - Performance tracking and logging
    """

    def __init__(
        self,
        model_config: Dict[str, Any],
        training_config: Dict[str, Any],
        model_type: str = "base"
    ):
        """
        Initialize ML model with configuration.

        Args:
            model_config: Model-specific configuration parameters
            training_config: Training-specific configuration parameters
            model_type: Type identifier for the model
        """
        self.model_config = model_config
        self.training_config = training_config
        self.model_type = model_type
        self.model = None
        self.is_trained = False
        self.training_history = []
        self.performance_metrics = {}
        self.version = "1.0.0"

        # Performance tracking
        self.prediction_count = 0
        self.total_inference_time = 0.0
        self.last_prediction_time = None

        logger.info(f"Initialized {model_type} model with config: {model_config}")

    @abc.abstractmethod
    def build_model(self, input_shape: Tuple[int, ...]) -> Any:
        """
        Build and return the model architecture.

        Args:
            input_shape: Shape of input data

        Returns:
            Compiled model ready for training
        """
        pass

    @abc.abstractmethod
    def preprocess_data(
        self,
        data: pd.DataFrame,
        target_column: str = "target"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess data for model training/inference.

        Args:
            data: Input data DataFrame
            target_column: Name of target column

        Returns:
            Tuple of (features, targets)
        """
        pass

    @abc.abstractmethod
    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Make predictions on input features.

        Args:
            features: Input feature array

        Returns:
            Prediction array
        """
        pass

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        save_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Train the model with provided data.

        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            save_path: Path to save trained model (optional)

        Returns:
            Training metrics and results
        """
        logger.info(f"Starting training for {self.model_type} model")
        start_time = datetime.now()

        # Validate input data
        self._validate_training_data(X_train, y_train, X_val, y_val)

        # Build model if not already built
        if self.model is None:
            input_shape = X_train.shape[1:] if len(X_train.shape) > 2 else (X_train.shape[1],)
            self.model = self.build_model(input_shape)

        # Execute training
        training_results = self._execute_training(X_train, y_train, X_val, y_val)

        # Update model state
        self.is_trained = True
        self.training_history.append({
            "timestamp": start_time.isoformat(),
            "training_samples": len(X_train),
            "validation_samples": len(X_val) if X_val is not None else 0,
            "results": training_results
        })

        # Save model if path provided
        if save_path:
            self.save_model(save_path)

        # Calculate performance metrics
        if X_val is not None and y_val is not None:
            val_predictions = self.predict(X_val)
            self.performance_metrics = self._calculate_metrics(y_val, val_predictions)

        training_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"Training completed in {training_time:.2f} seconds")

        return {
            "training_time": training_time,
            "performance_metrics": self.performance_metrics,
            "model_parameters": self._count_parameters(),
            "training_results": training_results
        }

    def predict_with_confidence(
        self,
        features: np.ndarray,
        return_probabilities: bool = True
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Make predictions with confidence scores.

        Args:
            features: Input feature array
            return_probabilities: Whether to return probability distributions

        Returns:
            Predictions and optionally confidence probabilities
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        start_time = datetime.now()
        predictions = self.predict(features)
        inference_time = (datetime.now() - start_time).total_seconds()

        # Update performance tracking
        self.prediction_count += 1
        self.total_inference_time += inference_time
        self.last_prediction_time = datetime.now()

        if return_probabilities:
            probabilities = self._get_prediction_probabilities(features)
            return predictions, probabilities

        return predictions

    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        detailed: bool = False
    ) -> Dict[str, Any]:
        """
        Evaluate model performance on test data.

        Args:
            X_test: Test features
            y_test: Test targets
            detailed: Whether to return detailed metrics

        Returns:
            Performance metrics
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")

        predictions = self.predict(X_test)
        metrics = self._calculate_metrics(y_test, predictions)

        if detailed:
            metrics.update({
                "confusion_matrix": self._calculate_confusion_matrix(y_test, predictions),
                "classification_report": self._calculate_classification_report(y_test, predictions),
                "roc_auc": self._calculate_roc_auc(y_test, predictions) if hasattr(self, '_get_prediction_probabilities') else None
            })

        return metrics

    def save_model(self, path: str) -> None:
        """
        Save model to disk.

        Args:
            path: File path to save model
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")

        save_data = {
            "model_config": self.model_config,
            "training_config": self.training_config,
            "model_type": self.model_type,
            "version": self.version,
            "is_trained": self.is_trained,
            "training_history": self.training_history,
            "performance_metrics": self.performance_metrics,
            "model_state": self._get_model_state()
        }

        # Create directory if it doesn't exist
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        # Save using pickle for complex models
        with open(path, 'wb') as f:
            pickle.dump(save_data, f)

        logger.info(f"Model saved to {path}")

    def load_model(self, path: str) -> None:
        """
        Load model from disk.

        Args:
            path: File path to load model from
        """
        with open(path, 'rb') as f:
            save_data = pickle.load(f)

        # Restore model state
        self.model_config = save_data["model_config"]
        self.training_config = save_data["training_config"]
        self.model_type = save_data["model_type"]
        self.version = save_data["version"]
        self.is_trained = save_data["is_trained"]
        self.training_history = save_data["training_history"]
        self.performance_metrics = save_data["performance_metrics"]

        # Restore model architecture and weights
        self._restore_model_state(save_data["model_state"])

        logger.info(f"Model loaded from {path}")

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get comprehensive model information.

        Returns:
            Dictionary with model details
        """
        return {
            "model_type": self.model_type,
            "version": self.version,
            "is_trained": self.is_trained,
            "model_config": self.model_config,
            "training_config": self.training_config,
            "performance_metrics": self.performance_metrics,
            "training_history_count": len(self.training_history),
            "prediction_count": self.prediction_count,
            "average_inference_time": self.total_inference_time / max(1, self.prediction_count),
            "last_prediction_time": self.last_prediction_time.isoformat() if self.last_prediction_time else None,
            "parameter_count": self._count_parameters() if self.model else 0
        }

    def cross_validate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        cv_folds: int = 5,
        save_best: bool = True
    ) -> Dict[str, Any]:
        """
        Perform time series cross-validation.

        Args:
            X: Feature array
            y: Target array
            cv_folds: Number of cross-validation folds
            save_best: Whether to save the best model

        Returns:
            Cross-validation results
        """
        logger.info(f"Starting {cv_folds}-fold cross-validation")

        tscv = TimeSeriesSplit(n_splits=cv_folds)
        cv_scores = []
        best_score = -np.inf
        best_model_state = None

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            logger.info(f"Training fold {fold + 1}/{cv_folds}")

            X_train_fold, X_val_fold = X[train_idx], X[val_idx]
            y_train_fold, y_val_fold = y[train_idx], y[val_idx]

            # Clone model for this fold
            fold_model = self.__class__(self.model_config, self.training_config, self.model_type)

            # Train fold model
            fold_results = fold_model.train(X_train_fold, y_train_fold, X_val_fold, y_val_fold)
            fold_metrics = fold_model.evaluate(X_val_fold, y_val_fold)

            cv_scores.append(fold_metrics)

            # Track best model
            current_score = fold_metrics.get('accuracy', 0)
            if current_score > best_score:
                best_score = current_score
                best_model_state = fold_model._get_model_state()

        # Calculate average metrics
        avg_metrics = {}
        for key in cv_scores[0].keys():
            if isinstance(cv_scores[0][key], (int, float)):
                avg_metrics[f"avg_{key}"] = np.mean([score[key] for score in cv_scores])
                avg_metrics[f"std_{key}"] = np.std([score[key] for score in cv_scores])

        # Load best model if requested
        if save_best and best_model_state:
            self._restore_model_state(best_model_state)
            self.is_trained = True

        logger.info(f"Cross-validation completed. Best score: {best_score:.4f}")

        return {
            "cv_scores": cv_scores,
            "average_metrics": avg_metrics,
            "best_score": best_score,
            "fold_count": cv_folds
        }

    # Protected methods

    def _validate_training_data(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray],
        y_val: Optional[np.ndarray]
    ) -> None:
        """Validate training data dimensions and types."""
        if len(X_train) != len(y_train):
            raise ValueError("Training features and targets must have same length")

        if X_val is not None and y_val is not None:
            if len(X_val) != len(y_val):
                raise ValueError("Validation features and targets must have same length")

        if len(X_train) < self.training_config.get('min_training_samples', 100):
            raise ValueError(f"Insufficient training samples: {len(X_train)}")

    def _calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """Calculate standard classification metrics."""
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }

    def _get_prediction_probabilities(self, features: np.ndarray) -> np.ndarray:
        """Get prediction probabilities. Override in subclasses."""
        # Default implementation - return uniform probabilities
        n_classes = len(np.unique(self.predict(features)))
        return np.ones((len(features), n_classes)) / n_classes

    def _count_parameters(self) -> int:
        """Count model parameters."""
        if self.model is None:
            return 0

        if hasattr(self.model, 'parameters'):
            return sum(p.numel() for p in self.model.parameters())
        elif hasattr(self.model, 'coef_'):
            return self.model.coef_.size
        else:
            return 0

    def _get_model_state(self) -> Any:
        """Get model state for saving. Override in subclasses."""
        return self.model

    def _restore_model_state(self, state: Any) -> None:
        """Restore model state from saved data. Override in subclasses."""
        self.model = state

    @abc.abstractmethod
    def _execute_training(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray],
        y_val: Optional[np.ndarray]
    ) -> Dict[str, Any]:
        """Execute the actual training process. Override in subclasses."""
        pass

    def _calculate_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Calculate confusion matrix."""
        from sklearn.metrics import confusion_matrix
        return confusion_matrix(y_true, y_pred)

    def _calculate_classification_report(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """Calculate detailed classification report."""
        from sklearn.metrics import classification_report
        return classification_report(y_true, y_pred, output_dict=True)

    def _calculate_roc_auc(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate ROC AUC score."""
        try:
            from sklearn.metrics import roc_auc_score
            probabilities = self._get_prediction_probabilities(np.array([y_pred]))
            return roc_auc_score(y_true, probabilities, multi_class='ovr')
        except Exception:
            return 0.0