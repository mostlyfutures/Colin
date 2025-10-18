"""
Base ML Pipeline class for Colin Trading Bot v2.0

This module provides the foundational ML pipeline class that orchestrates
the complete machine learning workflow from data ingestion to model deployment.
"""

import abc
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from loguru import logger

from .ml_base import MLModelBase
from .feature_base import FeatureEngineerBase


class MLPipelineBase(abc.ABC):
    """
    Abstract base class for ML pipelines in Colin Trading Bot v2.0.

    This class orchestrates the complete ML workflow:
    - Data ingestion and validation
    - Feature engineering and preprocessing
    - Model training and validation
    - Prediction and inference
    - Performance monitoring and logging
    - Model versioning and deployment
    """

    def __init__(
        self,
        pipeline_config: Dict[str, Any],
        feature_engineer: FeatureEngineerBase,
        model: MLModelBase,
        pipeline_name: str = "base_pipeline"
    ):
        """
        Initialize ML pipeline with components.

        Args:
            pipeline_config: Pipeline configuration parameters
            feature_engineer: Feature engineering component
            model: ML model component
            pipeline_name: Name identifier for the pipeline
        """
        self.pipeline_config = pipeline_config
        self.feature_engineer = feature_engineer
        self.model = model
        self.pipeline_name = pipeline_name
        self.is_trained = False
        self.training_history = []
        self.performance_metrics = {}
        self.version = "1.0.0"

        # Performance tracking
        self.prediction_count = 0
        self.total_pipeline_time = 0.0
        self.last_prediction_time = None

        logger.info(f"Initialized {pipeline_name} pipeline")

    @abc.abstractmethod
    def load_data(self, data_source: Any, **kwargs) -> pd.DataFrame:
        """
        Load data from specified source.

        Args:
            data_source: Data source identifier or path
            **kwargs: Additional loading parameters

        Returns:
            Loaded data as DataFrame
        """
        pass

    @abc.abstractmethod
    def validate_data(self, data: pd.DataFrame) -> bool:
        """
        Validate loaded data meets requirements.

        Args:
            data: Data to validate

        Returns:
            True if data is valid
        """
        pass

    @abc.abstractmethod
    def create_target_variable(self, data: pd.DataFrame) -> pd.Series:
        """
        Create target variable for supervised learning.

        Args:
            data: Input data

        Returns:
            Target variable series
        """
        pass

    def train(
        self,
        data_source: Any,
        validation_split: float = 0.2,
        save_path: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Train the complete pipeline.

        Args:
            data_source: Data source for training
            validation_split: Fraction of data for validation
            save_path: Path to save trained pipeline (optional)
            **kwargs: Additional training parameters

        Returns:
            Training results and metrics
        """
        logger.info(f"Starting training for {self.pipeline_name} pipeline")
        start_time = datetime.now()

        # Load and validate data
        data = self.load_data(data_source, **kwargs)
        if not self.validate_data(data):
            raise ValueError("Data validation failed")

        # Create target variable
        target = self.create_target_variable(data)

        # Split data
        train_data, val_data = self._split_data(data, validation_split)
        train_target, val_target = self._split_data(target, validation_split)

        # Feature engineering
        logger.info("Performing feature engineering")
        feature_start = datetime.now()

        train_features = self.feature_engineer.fit_transform(train_data, train_target)
        val_features = self.feature_engineer.transform(val_data)

        feature_time = (datetime.now() - feature_start).total_seconds()
        logger.info(f"Feature engineering completed in {feature_time:.2f} seconds")

        # Convert to numpy arrays
        X_train = train_features.values
        y_train = train_target.values
        X_val = val_features.values
        y_val = val_target.values

        # Train model
        logger.info("Training model")
        model_start = datetime.now()

        model_results = self.model.train(X_train, y_train, X_val, y_val)

        model_time = (datetime.now() - model_start).total_seconds()
        logger.info(f"Model training completed in {model_time:.2f} seconds")

        # Update pipeline state
        self.is_trained = True
        total_time = (datetime.now() - start_time).total_seconds()

        # Record training history
        self.training_history.append({
            "timestamp": start_time.isoformat(),
            "data_source": str(data_source),
            "training_samples": len(X_train),
            "validation_samples": len(X_val),
            "feature_count": len(train_features.columns),
            "feature_engineering_time": feature_time,
            "model_training_time": model_time,
            "total_time": total_time,
            "model_results": model_results
        })

        # Save pipeline if path provided
        if save_path:
            self.save_pipeline(save_path)

        # Evaluate on validation set
        val_metrics = self.model.evaluate(X_val, y_val, detailed=True)
        self.performance_metrics = val_metrics

        logger.info(f"Pipeline training completed in {total_time:.2f} seconds")

        return {
            "total_time": total_time,
            "feature_engineering_time": feature_time,
            "model_training_time": model_time,
            "training_samples": len(X_train),
            "validation_samples": len(X_val),
            "feature_count": len(train_features.columns),
            "performance_metrics": val_metrics,
            "model_results": model_results
        }

    def predict(
        self,
        data_source: Any,
        return_probabilities: bool = True,
        **kwargs
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Make predictions using the trained pipeline.

        Args:
            data_source: Data source for prediction
            return_probabilities: Whether to return prediction probabilities
            **kwargs: Additional prediction parameters

        Returns:
            Predictions and optionally probabilities
        """
        if not self.is_trained:
            raise ValueError("Pipeline must be trained before making predictions")

        logger.debug(f"Making predictions with {self.pipeline_name} pipeline")
        start_time = datetime.now()

        # Load and validate data
        data = self.load_data(data_source, **kwargs)
        if not self.validate_data(data):
            raise ValueError("Data validation failed")

        # Feature engineering
        feature_start = datetime.now()
        features = self.feature_engineer.transform(data)
        feature_time = (datetime.now() - feature_start).total_seconds()

        # Make predictions
        prediction_start = datetime.now()
        predictions = self.model.predict_with_confidence(features.values, return_probabilities)
        prediction_time = (datetime.now() - prediction_start).total_seconds()

        # Update performance tracking
        total_time = (datetime.now() - start_time).total_seconds()
        self.prediction_count += 1
        self.total_pipeline_time += total_time
        self.last_prediction_time = datetime.now()

        logger.debug(f"Prediction completed in {total_time:.2f}s "
                    f"(features: {feature_time:.3f}s, model: {prediction_time:.3f}s)")

        return predictions

    def evaluate(
        self,
        data_source: Any,
        detailed: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Evaluate pipeline performance on test data.

        Args:
            data_source: Test data source
            detailed: Whether to return detailed metrics
            **kwargs: Additional evaluation parameters

        Returns:
            Performance metrics
        """
        logger.info(f"Evaluating {self.pipeline_name} pipeline")

        # Load and validate data
        data = self.load_data(data_source, **kwargs)
        if not self.validate_data(data):
            raise ValueError("Data validation failed")

        # Create target variable
        target = self.create_target_variable(data)

        # Feature engineering
        features = self.feature_engineer.transform(data)

        # Evaluate model
        metrics = self.model.evaluate(features.values, target.values, detailed)

        return metrics

    def cross_validate(
        self,
        data_source: Any,
        cv_folds: int = 5,
        validation_split: float = 0.2,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Perform cross-validation on the pipeline.

        Args:
            data_source: Data source for cross-validation
            cv_folds: Number of cross-validation folds
            validation_split: Fraction of data for validation in each fold
            **kwargs: Additional parameters

        Returns:
            Cross-validation results
        """
        logger.info(f"Starting {cv_folds}-fold cross-validation for {self.pipeline_name}")

        # Load and validate data
        data = self.load_data(data_source, **kwargs)
        if not self.validate_data(data):
            raise ValueError("Data validation failed")

        # Create target variable
        target = self.create_target_variable(data)

        # Combine data for cross-validation
        combined_data = pd.concat([data, target.rename('target')], axis=1)

        # Perform time series cross-validation
        cv_results = self._perform_cross_validation(
            combined_data,
            cv_folds,
            validation_split
        )

        # Update pipeline with best model
        if cv_results['best_model_state']:
            self.model._restore_model_state(cv_results['best_model_state'])
            self.is_trained = True

        return cv_results

    def save_pipeline(self, path: str) -> None:
        """
        Save complete pipeline to disk.

        Args:
            path: File path to save pipeline
        """
        if not self.is_trained:
            raise ValueError("Pipeline must be trained before saving")

        import pickle

        save_data = {
            "pipeline_config": self.pipeline_config,
            "pipeline_name": self.pipeline_name,
            "version": self.version,
            "is_trained": self.is_trained,
            "training_history": self.training_history,
            "performance_metrics": self.performance_metrics,
            "feature_engineer_state": self.feature_engineer._get_model_state(),
            "model_state": self.model._get_model_state()
        }

        # Create directory if it doesn't exist
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'wb') as f:
            pickle.dump(save_data, f)

        logger.info(f"Pipeline saved to {path}")

    def load_pipeline(self, path: str) -> None:
        """
        Load pipeline from disk.

        Args:
            path: File path to load pipeline from
        """
        import pickle

        with open(path, 'rb') as f:
            save_data = pickle.load(f)

        # Restore pipeline state
        self.pipeline_config = save_data["pipeline_config"]
        self.pipeline_name = save_data["pipeline_name"]
        self.version = save_data["version"]
        self.is_trained = save_data["is_trained"]
        self.training_history = save_data["training_history"]
        self.performance_metrics = save_data["performance_metrics"]

        # Restore component states
        self.feature_engineer._restore_model_state(save_data["feature_engineer_state"])
        self.model._restore_model_state(save_data["model_state"])

        logger.info(f"Pipeline loaded from {path}")

    def get_pipeline_info(self) -> Dict[str, Any]:
        """
        Get comprehensive pipeline information.

        Returns:
            Dictionary with pipeline details
        """
        return {
            "pipeline_name": self.pipeline_name,
            "version": self.version,
            "is_trained": self.is_trained,
            "pipeline_config": self.pipeline_config,
            "training_history_count": len(self.training_history),
            "performance_metrics": self.performance_metrics,
            "prediction_count": self.prediction_count,
            "total_pipeline_time": self.total_pipeline_time,
            "average_pipeline_time": self.total_pipeline_time / max(1, self.prediction_count),
            "last_prediction_time": self.last_prediction_time.isoformat() if self.last_prediction_time else None,
            "feature_engineer_info": self.feature_engineer.get_engineer_info(),
            "model_info": self.model.get_model_info()
        }

    def monitor_performance(self, recent_predictions: int = 100) -> Dict[str, Any]:
        """
        Monitor recent pipeline performance.

        Args:
            recent_predictions: Number of recent predictions to analyze

        Returns:
            Performance monitoring metrics
        """
        if not self.training_history:
            return {"status": "no_training_data"}

        latest_training = self.training_history[-1]

        return {
            "pipeline_status": "active" if self.is_trained else "not_trained",
            "latest_training": latest_training["timestamp"],
            "training_samples": latest_training["training_samples"],
            "feature_count": latest_training["feature_count"],
            "total_predictions": self.prediction_count,
            "average_pipeline_time": self.total_pipeline_time / max(1, self.prediction_count),
            "last_prediction": self.last_prediction_time.isoformat() if self.last_prediction_time else None,
            "performance_metrics": self.performance_metrics
        }

    # Protected methods

    def _split_data(
        self,
        data: Union[pd.DataFrame, pd.Series],
        validation_split: float
    ) -> Tuple[Union[pd.DataFrame, pd.Series], Union[pd.DataFrame, pd.Series]]:
        """Split data into training and validation sets."""
        split_idx = int(len(data) * (1 - validation_split))
        return data.iloc[:split_idx], data.iloc[split_idx:]

    def _perform_cross_validation(
        self,
        data: pd.DataFrame,
        cv_folds: int,
        validation_split: float
    ) -> Dict[str, Any]:
        """Perform time series cross-validation."""
        from sklearn.model_selection import TimeSeriesSplit

        # Prepare features and targets
        feature_cols = [col for col in data.columns if col != 'target']
        X = data[feature_cols]
        y = data['target']

        tscv = TimeSeriesSplit(n_splits=cv_folds)
        cv_scores = []
        best_score = -np.inf
        best_model_state = None

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            logger.info(f"Training fold {fold + 1}/{cv_folds}")

            X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
            y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]

            # Create fold-specific dataframes for feature engineering
            train_data = pd.concat([X_train_fold, y_train_fold], axis=1)
            val_data = pd.concat([X_val_fold, y_val_fold], axis=1)

            # Clone pipeline for this fold
            fold_pipeline = self.__class__(
                self.pipeline_config,
                self.feature_engineer.__class__(self.feature_engineer.feature_config),
                self.model.__class__(self.model.model_config, self.model.training_config),
                f"{self.pipeline_name}_fold_{fold}"
            )

            # Train fold pipeline
            try:
                fold_results = fold_pipeline.train(train_data, validation_split)
                fold_metrics = fold_pipeline.evaluate(val_data)

                cv_scores.append(fold_metrics)

                # Track best model
                current_score = fold_metrics.get('accuracy', 0)
                if current_score > best_score:
                    best_score = current_score
                    best_model_state = fold_pipeline.model._get_model_state()

            except Exception as e:
                logger.error(f"Fold {fold + 1} failed: {e}")
                continue

        # Calculate average metrics
        avg_metrics = {}
        if cv_scores:
            for key in cv_scores[0].keys():
                if isinstance(cv_scores[0][key], (int, float)):
                    avg_metrics[f"avg_{key}"] = np.mean([score[key] for score in cv_scores])
                    avg_metrics[f"std_{key}"] = np.std([score[key] for score in cv_scores])

        return {
            "cv_scores": cv_scores,
            "average_metrics": avg_metrics,
            "best_score": best_score,
            "best_model_state": best_model_state,
            "fold_count": cv_folds
        }