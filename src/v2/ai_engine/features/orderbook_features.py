"""
Order Book Feature Engineer for Colin Trading Bot v2.0

This module implements order book analysis features that provide insights into
market microstructure and liquidity dynamics. These features are crucial for
understanding order flow imbalance, market depth, and short-term price movements.

Features include:
- Order book imbalance metrics
- Market depth and liquidity measures
- Order flow and pressure indicators
- Spread and market impact features
- Volume profile analysis
- Microstructure patterns
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from loguru import logger

from ..base.feature_base import FeatureEngineerBase


class OrderBookFeatureEngineer(FeatureEngineerBase):
    """
    Order book feature engineer for market microstructure analysis.

    This class processes order book data to extract features that reveal
    market liquidity, order flow imbalance, and potential price movements.
    """

    def __init__(
        self,
        feature_config: Dict[str, Any],
        scaling_config: Optional[Dict[str, Any]] = None,
        feature_type: str = "orderbook"
    ):
        """
        Initialize order book feature engineer.

        Args:
            feature_config: Configuration for order book features
            scaling_config: Scaling configuration
            feature_type: Feature type identifier
        """
        # Default configuration
        default_config = {
            "depth_levels": 10,           # Number of price levels to analyze
            "imbalance_threshold": 0.7,   # Threshold for significant imbalance
            "spread_features": True,      # Include spread-related features
            "depth_features": True,       # Include market depth features
            "flow_features": True,        # Include order flow features
            "microstructure_features": True,  # Include microstructure features
            "time_windows": [1, 5, 10, 20],  # Time windows for aggregation
            "price_levels": 20,           # Number of price levels for volume profile
            "include_lagged_features": True,
            "lag_periods": [1, 2, 3, 5],
            "normalize_features": True
        }

        feature_config = {**default_config, **feature_config}

        # Default scaling configuration for order book features
        default_scaling = {
            "type": "standard",
            "params": {}
        }

        scaling_config = {**default_scaling, **(scaling_config or {})}

        super().__init__(feature_config, scaling_config, feature_type)

        logger.info(f"Initialized OrderBookFeatureEngineer with depth_levels={feature_config['depth_levels']}")

    def calculate_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate order book features.

        Args:
            data: DataFrame with order book data. Expected columns:
                  - bid_price_1, bid_volume_1, ask_price_1, ask_volume_1, etc.
                  - Or aggregated order book snapshots

        Returns:
            DataFrame with calculated order book features
        """
        logger.debug(f"Calculating order book features for {len(data)} samples")
        features = pd.DataFrame(index=data.index)

        # Basic spread features
        if self.feature_config["spread_features"]:
            features = self._calculate_spread_features(data, features)

        # Market depth features
        if self.feature_config["depth_features"]:
            features = self._calculate_depth_features(data, features)

        # Order flow features
        if self.feature_config["flow_features"]:
            features = self._calculate_flow_features(data, features)

        # Microstructure features
        if self.feature_config["microstructure_features"]:
            features = self._calculate_microstructure_features(data, features)

        # Lagged features
        if self.feature_config["include_lagged_features"]:
            features = self._calculate_lagged_features(data, features)

        # Normalized features
        if self.feature_config["normalize_features"]:
            features = self._calculate_normalized_features(data, features)

        logger.debug(f"Calculated {len(features.columns)} order book features")
        return features

    def get_feature_names(self) -> List[str]:
        """
        Get list of feature names this engineer produces.

        Returns:
            List of feature names
        """
        feature_names = []

        # Spread features
        if self.feature_config["spread_features"]:
            feature_names.extend([
                'bid_ask_spread', 'spread_pct', 'mid_price', 'spread_bps',
                'spread_volatility', 'spread_trend'
            ])

        # Depth features
        if self.feature_config["depth_features"]:
            depth_levels = self.feature_config["depth_levels"]
            for level in range(1, depth_levels + 1):
                feature_names.extend([
                    f'bid_depth_{level}', f'ask_depth_{level}',
                    f'total_depth_{level}', f'depth_imbalance_{level}'
                ])

            feature_names.extend([
                'cumulative_bid_depth', 'cumulative_ask_depth',
                'total_market_depth', 'depth_ratio'
            ])

        # Flow features
        if self.feature_config["flow_features"]:
            feature_names.extend([
                'order_flow_imbalance', 'volume_weighted_imbalance',
                'price_pressure', 'liquidity_ratio', 'market_impact_estimate'
            ])

        # Microstructure features
        if self.feature_config["microstructure_features"]:
            feature_names.extend([
                'order_book_slope', 'volume_profile_skew',
                'concentration_ratio', 'liquidity_gradient'
            ])

        return feature_names

    def _calculate_spread_features(self, data: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Calculate spread-related features."""
        try:
            # Get best bid and ask prices
            if 'bid_price_1' in data.columns and 'ask_price_1' in data.columns:
                bid_price = data['bid_price_1']
                ask_price = data['ask_price_1']
            elif 'best_bid' in data.columns and 'best_ask' in data.columns:
                bid_price = data['best_bid']
                ask_price = data['best_ask']
            else:
                # Estimate from OHLC data
                bid_price = data['low']  # Simplified
                ask_price = data['high']  # Simplified

            # Calculate spread
            features['bid_ask_spread'] = ask_price - bid_price
            features['spread_pct'] = features['bid_ask_spread'] / ((ask_price + bid_price) / 2)
            features['mid_price'] = (ask_price + bid_price) / 2
            features['spread_bps'] = features['spread_pct'] * 10000

            # Spread dynamics
            features['spread_volatility'] = features['bid_ask_spread'].rolling(window=10).std()
            features['spread_trend'] = features['bid_ask_spread'].diff()

        except Exception as e:
            logger.warning(f"Error calculating spread features: {e}")

        return features

    def _calculate_depth_features(self, data: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Calculate market depth features."""
        try:
            depth_levels = self.feature_config["depth_levels"]

            cumulative_bid_depth = 0
            cumulative_ask_depth = 0

            for level in range(1, depth_levels + 1):
                bid_vol_col = f'bid_volume_{level}'
                ask_vol_col = f'ask_volume_{level}'

                if bid_vol_col in data.columns and ask_vol_col in data.columns:
                    bid_volume = data[bid_vol_col]
                    ask_volume = data[ask_vol_col]
                else:
                    # Simulate depth levels from OHLCV
                    bid_volume = data['volume'] * (0.6 ** level)  # Decreasing volume
                    ask_volume = data['volume'] * (0.4 ** level)

                features[f'bid_depth_{level}'] = bid_volume
                features[f'ask_depth_{level}'] = ask_volume
                features[f'total_depth_{level}'] = bid_volume + ask_volume

                # Imbalance at this level
                total_volume = bid_volume + ask_volume
                features[f'depth_imbalance_{level}'] = np.where(
                    total_volume > 0,
                    (bid_volume - ask_volume) / total_volume,
                    0
                )

                # Cumulative depth
                cumulative_bid_depth += bid_volume
                cumulative_ask_depth += ask_volume

            features['cumulative_bid_depth'] = cumulative_bid_depth
            features['cumulative_ask_depth'] = cumulative_ask_depth
            features['total_market_depth'] = cumulative_bid_depth + cumulative_ask_depth
            features['depth_ratio'] = cumulative_bid_depth / (cumulative_ask_depth + 1e-8)

        except Exception as e:
            logger.warning(f"Error calculating depth features: {e}")

        return features

    def _calculate_flow_features(self, data: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Calculate order flow and pressure features."""
        try:
            # Order flow imbalance (best level)
            if 'bid_volume_1' in data.columns and 'ask_volume_1' in data.columns:
                bid_volume = data['bid_volume_1']
                ask_volume = data['ask_volume_1']
            else:
                # Estimate from price movement and volume
                price_change = data['close'].diff().fillna(0)
                bid_volume = np.where(price_change > 0, data['volume'], data['volume'] * 0.4)
                ask_volume = np.where(price_change < 0, data['volume'], data['volume'] * 0.6)

            total_volume = bid_volume + ask_volume
            features['order_flow_imbalance'] = np.where(
                total_volume > 0,
                (bid_volume - ask_volume) / total_volume,
                0
            )

            # Volume-weighted imbalance (considering depth)
            if 'cumulative_bid_depth' in features and 'cumulative_ask_depth' in features:
                total_depth = features['cumulative_bid_depth'] + features['cumulative_ask_depth']
                features['volume_weighted_imbalance'] = np.where(
                    total_depth > 0,
                    (features['cumulative_bid_depth'] - features['cumulative_ask_depth']) / total_depth,
                    0
                )

            # Price pressure
            if 'bid_ask_spread' in features:
                features['price_pressure'] = features['order_flow_imbalance'] / (features['bid_ask_spread'] + 1e-8)

            # Liquidity ratio
            features['liquidity_ratio'] = features['total_market_depth'] / (data['volume'] + 1e-8)

            # Market impact estimate
            if 'spread_pct' in features:
                features['market_impact_estimate'] = features['order_flow_imbalance'] * features['spread_pct']

        except Exception as e:
            logger.warning(f"Error calculating flow features: {e}")

        return features

    def _calculate_microstructure_features(self, data: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Calculate microstructure features."""
        try:
            # Order book slope (price vs volume relationship)
            if 'bid_price_1' in data.columns and 'bid_price_2' in data.columns:
                # Calculate slope from first few levels
                price_diff_bids = abs(data['bid_price_2'] - data['bid_price_1'])
                volume_sum_bids = data.get('bid_volume_1', 0) + data.get('bid_volume_2', 0)
                features['order_book_slope'] = np.where(volume_sum_bids > 0, price_diff_bids / volume_sum_bids, 0)
            else:
                # Estimate from price range and volume
                features['order_book_slope'] = (data['high'] - data['low']) / (data['volume'] + 1e-8)

            # Volume profile skew
            if 'depth_imbalance_1' in features and 'depth_imbalance_5' in features:
                features['volume_profile_skew'] = features['depth_imbalance_1'] - features['depth_imbalance_5']

            # Concentration ratio (how concentrated is volume at best levels)
            if 'total_depth_1' in features and 'total_market_depth' in features:
                features['concentration_ratio'] = features['total_depth_1'] / (features['total_market_depth'] + 1e-8)

            # Liquidity gradient (how quickly liquidity changes with price)
            if 'bid_ask_spread' in features and 'total_market_depth' in features:
                features['liquidity_gradient'] = features['bid_ask_spread'] / (features['total_market_depth'] + 1e-8)

        except Exception as e:
            logger.warning(f"Error calculating microstructure features: {e}")

        return features

    def _calculate_lagged_features(self, data: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Calculate lagged order book features."""
        try:
            lag_periods = self.feature_config["lag_periods"]

            for lag in lag_periods:
                if 'order_flow_imbalance' in features:
                    features[f'flow_imbalance_lag_{lag}'] = features['order_flow_imbalance'].shift(lag)

                if 'bid_ask_spread' in features:
                    features[f'spread_lag_{lag}'] = features['bid_ask_spread'].shift(lag)

                if 'depth_ratio' in features:
                    features[f'depth_ratio_lag_{lag}'] = features['depth_ratio'].shift(lag)

        except Exception as e:
            logger.warning(f"Error calculating lagged features: {e}")

        return features

    def _calculate_normalized_features(self, data: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Calculate normalized features."""
        try:
            # Normalize spread by price
            if 'bid_ask_spread' in features and 'mid_price' in features:
                features['spread_normalized'] = features['bid_ask_spread'] / features['mid_price']

            # Normalize depth by volume
            if 'total_market_depth' in features:
                features['depth_normalized'] = features['total_market_depth'] / (data['volume'].rolling(20).mean() + 1e-8)

            # Z-score normalization for key features
            for feature in ['order_flow_imbalance', 'depth_ratio', 'spread_pct']:
                if feature in features:
                    rolling_mean = features[feature].rolling(20).mean()
                    rolling_std = features[feature].rolling(20).std()
                    features[f'{feature}_zscore'] = (features[feature] - rolling_mean) / (rolling_std + 1e-8)

        except Exception as e:
            logger.warning(f"Error calculating normalized features: {e}")

        return features

    def process_order_book_snapshot(
        self,
        snapshot: Dict[str, Any],
        timestamp: Optional[pd.Timestamp] = None
    ) -> pd.DataFrame:
        """
        Process a single order book snapshot into features.

        Args:
            snapshot: Order book snapshot with bids and asks
            timestamp: Timestamp for the snapshot

        Returns:
            DataFrame with calculated features
        """
        # Convert snapshot to DataFrame format
        data = {}

        # Extract bids and asks
        if 'bids' in snapshot:
            for i, (price, volume) in enumerate(snapshot['bids'][:self.feature_config["depth_levels"]]):
                data[f'bid_price_{i+1}'] = price
                data[f'bid_volume_{i+1}'] = volume

        if 'asks' in snapshot:
            for i, (price, volume) in enumerate(snapshot['asks'][:self.feature_config["depth_levels"]]):
                data[f'ask_price_{i+1}'] = price
                data[f'ask_volume_{i+1}'] = volume

        # Create DataFrame
        df = pd.DataFrame([data])
        if timestamp:
            df.index = [timestamp]

        # Calculate features
        return self.calculate_features(df)

    def aggregate_features_over_window(
        self,
        features: pd.DataFrame,
        window_minutes: int = 5
    ) -> pd.DataFrame:
        """
        Aggregate order book features over time windows.

        Args:
            features: Order book features DataFrame
            window_minutes: Window size in minutes

        Returns:
            Aggregated features
        """
        try:
            # Resample and aggregate
            agg_features = features.resample(f'{window_minutes}T').agg({
                # Mean for continuous features
                **{col: 'mean' for col in features.columns if 'imbalance' in col or 'ratio' in col},
                # Sum for volume features
                **{col: 'sum' for col in features.columns if 'depth' in col and 'volume' not in col},
                # Std for volatility measures
                **{col: 'std' for col in features.columns if 'spread' in col or 'volatility' in col},
            })

            return agg_features

        except Exception as e:
            logger.warning(f"Error aggregating features: {e}")
            return features