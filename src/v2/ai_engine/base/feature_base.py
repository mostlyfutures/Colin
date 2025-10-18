"""
Base Feature Engineering class for Colin Trading Bot v2.0

This module provides the foundational feature engineering class that all feature
engineers should inherit from. It includes common functionality for data
preprocessing, feature calculation, validation, and persistence.
"""

import abc
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from loguru import logger
import ta
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler


class FeatureEngineerBase(abc.ABC):
    """
    Abstract base class for all feature engineers in Colin Trading Bot v2.0.

    This class provides common functionality for:
    - Data validation and preprocessing
    - Feature calculation and transformation
    - Feature scaling and normalization
    - Feature selection and importance
    - Persistence and caching
    - Performance tracking
    """

    def __init__(
        self,
        feature_config: Dict[str, Any],
        scaling_config: Optional[Dict[str, Any]] = None,
        feature_type: str = "base"
    ):
        """
        Initialize feature engineer with configuration.

        Args:
            feature_config: Feature-specific configuration parameters
            scaling_config: Scaling and normalization configuration
            feature_type: Type identifier for the feature engineer
        """
        self.feature_config = feature_config
        self.scaling_config = scaling_config or {}
        self.feature_type = feature_type
        self.scaler = None
        self.feature_columns = []
        self.is_fitted = False
        self.feature_stats = {}
        self.computation_history = []

        # Performance tracking
        self.feature_count = 0
        self.total_computation_time = 0.0
        self.last_computation_time = None

        # Initialize scaler based on config
        self._initialize_scaler()

        logger.info(f"Initialized {feature_type} feature engineer")

    @abc.abstractmethod
    def calculate_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate features from input data.

        Args:
            data: Input OHLCV data

        Returns:
            DataFrame with calculated features
        """
        pass

    @abc.abstractmethod
    def get_feature_names(self) -> List[str]:
        """
        Get list of feature names this engineer produces.

        Returns:
            List of feature names
        """
        pass

    def fit(
        self,
        data: pd.DataFrame,
        target_column: Optional[str] = None
    ) -> 'FeatureEngineerBase':
        """
        Fit the feature engineer on training data.

        Args:
            data: Training data
            target_column: Target column name (optional)

        Returns:
            Self for method chaining
        """
        logger.info(f"Fitting {self.feature_type} feature engineer")
        start_time = datetime.now()

        # Validate input data
        self._validate_input_data(data)

        # Calculate features
        features = self.calculate_features(data)
        self.feature_columns = features.columns.tolist()

        # Fit scaler if configured
        if self.scaler is not None:
            self.scaler.fit(features)
            logger.info(f"Fitted {type(self.scaler).__name__} scaler on {len(features)} samples")

        # Calculate feature statistics
        self._calculate_feature_stats(features)

        # Update state
        self.is_fitted = True
        computation_time = (datetime.now() - start_time).total_seconds()
        self.total_computation_time += computation_time
        self.last_computation_time = datetime.now()

        # Track computation history
        self.computation_history.append({
            "timestamp": start_time.isoformat(),
            "operation": "fit",
            "samples": len(data),
            "features": len(self.feature_columns),
            "computation_time": computation_time
        })

        logger.info(f"Feature fitting completed in {computation_time:.2f} seconds")
        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data using fitted feature engineer.

        Args:
            data: Input data to transform

        Returns:
            Transformed feature data
        """
        if not self.is_fitted:
            raise ValueError("Feature engineer must be fitted before transformation")

        logger.debug(f"Transforming {len(data)} samples with {self.feature_type} features")
        start_time = datetime.now()

        # Calculate features
        features = self.calculate_features(data)

        # Apply scaling if fitted
        if self.scaler is not None:
            features = pd.DataFrame(
                self.scaler.transform(features),
                index=features.index,
                columns=features.columns
            )

        # Validate output
        self._validate_output_features(features)

        # Update performance tracking
        computation_time = (datetime.now() - start_time).total_seconds()
        self.total_computation_time += computation_time
        self.feature_count += len(features)
        self.last_computation_time = datetime.now()

        return features

    def fit_transform(self, data: pd.DataFrame, target_column: Optional[str] = None) -> pd.DataFrame:
        """
        Fit and transform data in one step.

        Args:
            data: Input data
            target_column: Target column name (optional)

        Returns:
            Transformed feature data
        """
        return self.fit(data, target_column).transform(data)

    def get_feature_importance(
        self,
        features: pd.DataFrame,
        targets: pd.Series,
        method: str = "correlation"
    ) -> pd.Series:
        """
        Calculate feature importance scores.

        Args:
            features: Feature data
            targets: Target data
            method: Importance calculation method ('correlation', 'mutual_info', 'variance')

        Returns:
            Series of feature importance scores
        """
        if method == "correlation":
            importance = features.corrwith(targets).abs()
        elif method == "mutual_info":
            from sklearn.feature_selection import mutual_info_regression
            importance = pd.Series(
                mutual_info_regression(features, targets),
                index=features.columns
            )
        elif method == "variance":
            importance = features.var()
        else:
            raise ValueError(f"Unknown importance method: {method}")

        return importance.sort_values(ascending=False)

    def select_features(
        self,
        features: pd.DataFrame,
        targets: pd.Series,
        n_features: int,
        method: str = "correlation"
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Select top N features based on importance.

        Args:
            features: Feature data
            targets: Target data
            n_features: Number of features to select
            method: Importance calculation method

        Returns:
            Tuple of (selected features, selected feature names)
        """
        importance = self.get_feature_importance(features, targets, method)
        selected_features = importance.head(n_features).index.tolist()

        return features[selected_features], selected_features

    def save_features(self, features: pd.DataFrame, path: str) -> None:
        """
        Save calculated features to disk.

        Args:
            features: Feature data to save
            path: File path to save features
        """
        # Create directory if it doesn't exist
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        # Save metadata
        metadata = {
            "feature_type": self.feature_type,
            "feature_config": self.feature_config,
            "feature_columns": self.feature_columns,
            "is_fitted": self.is_fitted,
            "feature_stats": self.feature_stats,
            "timestamp": datetime.now().isoformat()
        }

        # Save features and metadata
        features.to_csv(path.replace('.csv', '_features.csv'), index=False)
        with open(path.replace('.csv', '_metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2, default=str)

        logger.info(f"Features saved to {path}")

    def load_features(self, path: str) -> pd.DataFrame:
        """
        Load features from disk.

        Args:
            path: File path to load features from

        Returns:
            Loaded feature data
        """
        features_path = path.replace('.csv', '_features.csv')
        metadata_path = path.replace('.csv', '_metadata.json')

        # Load metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        # Restore state
        self.feature_type = metadata["feature_type"]
        self.feature_config = metadata["feature_config"]
        self.feature_columns = metadata["feature_columns"]
        self.is_fitted = metadata["is_fitted"]
        self.feature_stats = metadata["feature_stats"]

        # Load features
        features = pd.read_csv(features_path)

        logger.info(f"Features loaded from {path}")
        return features

    def save_engineer(self, path: str) -> None:
        """
        Save feature engineer state to disk.

        Args:
            path: File path to save engineer state
        """
        if not self.is_fitted:
            raise ValueError("Feature engineer must be fitted before saving")

        import pickle

        save_data = {
            "feature_config": self.feature_config,
            "scaling_config": self.scaling_config,
            "feature_type": self.feature_type,
            "feature_columns": self.feature_columns,
            "is_fitted": self.is_fitted,
            "feature_stats": self.feature_stats,
            "computation_history": self.computation_history,
            "scaler": self.scaler
        }

        # Create directory if it doesn't exist
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'wb') as f:
            pickle.dump(save_data, f)

        logger.info(f"Feature engineer saved to {path}")

    def load_engineer(self, path: str) -> None:
        """
        Load feature engineer state from disk.

        Args:
            path: File path to load engineer state from
        """
        import pickle

        with open(path, 'rb') as f:
            save_data = pickle.load(f)

        # Restore state
        self.feature_config = save_data["feature_config"]
        self.scaling_config = save_data["scaling_config"]
        self.feature_type = save_data["feature_type"]
        self.feature_columns = save_data["feature_columns"]
        self.is_fitted = save_data["is_fitted"]
        self.feature_stats = save_data["feature_stats"]
        self.computation_history = save_data["computation_history"]
        self.scaler = save_data["scaler"]

        logger.info(f"Feature engineer loaded from {path}")

    def get_engineer_info(self) -> Dict[str, Any]:
        """
        Get comprehensive feature engineer information.

        Returns:
            Dictionary with engineer details
        """
        return {
            "feature_type": self.feature_type,
            "feature_config": self.feature_config,
            "scaling_config": self.scaling_config,
            "is_fitted": self.is_fitted,
            "feature_columns": self.feature_columns,
            "feature_count": len(self.feature_columns),
            "feature_stats": self.feature_stats,
            "computation_history_count": len(self.computation_history),
            "total_computation_time": self.total_computation_time,
            "last_computation_time": self.last_computation_time.isoformat() if self.last_computation_time else None,
            "scaler_type": type(self.scaler).__name__ if self.scaler else None
        }

    # Protected methods

    def _initialize_scaler(self) -> None:
        """Initialize scaler based on configuration."""
        scaler_type = self.scaling_config.get('type', 'standard')

        if scaler_type == 'standard':
            self.scaler = StandardScaler(**self.scaling_config.get('params', {}))
        elif scaler_type == 'minmax':
            self.scaler = MinMaxScaler(**self.scaling_config.get('params', {}))
        elif scaler_type == 'robust':
            self.scaler = RobustScaler(**self.scaling_config.get('params', {}))
        elif scaler_type == 'none':
            self.scaler = None
        else:
            logger.warning(f"Unknown scaler type: {scaler_type}, using no scaling")
            self.scaler = None

    def _validate_input_data(self, data: pd.DataFrame) -> None:
        """Validate input data format and content."""
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Input data must be a pandas DataFrame")

        if len(data) == 0:
            raise ValueError("Input data cannot be empty")

        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        # Check for data quality issues
        if data.isnull().any().any():
            logger.warning("Input data contains null values")

        # Check price consistency
        invalid_prices = (
            (data['high'] < data['low']) |
            (data['high'] < data['open']) |
            (data['high'] < data['close']) |
            (data['low'] > data['open']) |
            (data['low'] > data['close'])
        )
        if invalid_prices.any():
            logger.warning(f"Found {invalid_prices.sum()} rows with invalid price relationships")

    def _validate_output_features(self, features: pd.DataFrame) -> None:
        """Validate output feature data."""
        if features.isnull().any().any():
            logger.warning("Output features contain null values")

        if np.isinf(features.values).any():
            logger.warning("Output features contain infinite values")

    def _calculate_feature_stats(self, features: pd.DataFrame) -> None:
        """Calculate feature statistics for monitoring."""
        self.feature_stats = {
            "mean": features.mean().to_dict(),
            "std": features.std().to_dict(),
            "min": features.min().to_dict(),
            "max": features.max().to_dict(),
            "null_count": features.isnull().sum().to_dict(),
            "zero_count": (features == 0).sum().to_dict()
        }

    def _calculate_returns(self, data: pd.DataFrame, periods: List[int] = [1, 5, 10, 20]) -> pd.DataFrame:
        """Calculate returns for different periods."""
        returns = pd.DataFrame(index=data.index)

        for period in periods:
            returns[f'return_{period}'] = data['close'].pct_change(period)

        return returns

    def _calculate_rolling_features(
        self,
        data: pd.Series,
        windows: List[int],
        functions: List[str] = ['mean', 'std', 'min', 'max']
    ) -> pd.DataFrame:
        """Calculate rolling window features."""
        features = pd.DataFrame(index=data.index)

        for window in windows:
            rolling = data.rolling(window=window)

            if 'mean' in functions:
                features[f'{data.name}_mean_{window}'] = rolling.mean()
            if 'std' in functions:
                features[f'{data.name}_std_{window}'] = rolling.std()
            if 'min' in functions:
                features[f'{data.name}_min_{window}'] = rolling.min()
            if 'max' in functions:
                features[f'{data.name}_max_{window}'] = rolling.max()

        return features

    def _calculate_lag_features(
        self,
        data: pd.Series,
        lags: List[int]
    ) -> pd.DataFrame:
        """Calculate lagged features."""
        features = pd.DataFrame(index=data.index)

        for lag in lags:
            features[f'{data.name}_lag_{lag}'] = data.shift(lag)

        return features