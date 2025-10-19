"""
Liquidity Feature Engineer for Colin Trading Bot v2.0

This module implements liquidity analysis features that are crucial for understanding
market dynamics, especially in cryptocurrency markets. These features build upon
the existing liquidation analysis from the v1 system and provide comprehensive
liquidity metrics for AI model training.

Features include:
- Liquidation cluster analysis
- Liquidity proximity metrics
- Market depth and concentration
- Funding rate analysis
- Open interest dynamics
- Liquidity flow indicators
- Market maker activity measures
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from loguru import logger
from scipy import stats
from sklearn.cluster import DBSCAN

from ..base.feature_base import FeatureEngineerBase


class LiquidityFeatureEngineer(FeatureEngineerBase):
    """
    Liquidity feature engineer for comprehensive market liquidity analysis.

    This class processes various liquidity-related data sources to generate
    features that help predict price movements based on liquidity dynamics.
    """

    def __init__(
        self,
        feature_config: Dict[str, Any],
        scaling_config: Optional[Dict[str, Any]] = None,
        feature_type: str = "liquidity"
    ):
        """
        Initialize liquidity feature engineer.

        Args:
            feature_config: Configuration for liquidity features
            scaling_config: Scaling configuration
            feature_type: Feature type identifier
        """
        # Default configuration
        default_config = {
            "liquidation_features": True,
            "funding_rate_features": True,
            "open_interest_features": True,
            "depth_features": True,
            "concentration_features": True,
            "flow_features": True,
            "time_windows": [5, 15, 60, 240],  # minutes
            "proximity_threshold": 0.005,  # 0.5% price proximity
            "cluster_threshold": 1000000,   # $1M density threshold
            "min_liquidity_threshold": 50000,  # $50K minimum
            "volatility_adjustment": True,
            "include_lagged_features": True,
            "lag_periods": [1, 2, 3, 5]
        }

        feature_config = {**default_config, **feature_config}

        # Default scaling configuration for liquidity features
        default_scaling = {
            "type": "robust",
            "params": {
                "quantile_range": (10.0, 90.0)  # Wider range for liquidity data
            }
        }

        scaling_config = {**default_scaling, **(scaling_config or {})}

        super().__init__(feature_config, scaling_config, feature_type)

        logger.info(f"Initialized LiquidityFeatureEngineer with proximity_threshold={feature_config['proximity_threshold']}")

    def calculate_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate liquidity features.

        Args:
            data: DataFrame with OHLCV data plus liquidity columns:
                  - liquidation_long, liquidation_short (if available)
                  - funding_rate, open_interest (if available)
                  - Or will be estimated from price/volume

        Returns:
            DataFrame with calculated liquidity features
        """
        logger.debug(f"Calculating liquidity features for {len(data)} samples")
        features = pd.DataFrame(index=data.index)

        # Liquidation features
        if self.feature_config["liquidation_features"]:
            features = self._calculate_liquidation_features(data, features)

        # Funding rate features
        if self.feature_config["funding_rate_features"]:
            features = self._calculate_funding_rate_features(data, features)

        # Open interest features
        if self.feature_config["open_interest_features"]:
            features = self._calculate_open_interest_features(data, features)

        # Depth and concentration features
        if self.feature_config["depth_features"]:
            features = self._calculate_depth_features(data, features)

        # Concentration features
        if self.feature_config["concentration_features"]:
            features = self._calculate_concentration_features(data, features)

        # Flow features
        if self.feature_config["flow_features"]:
            features = self._calculate_flow_features(data, features)

        # Lagged features
        if self.feature_config["include_lagged_features"]:
            features = self._calculate_lagged_features(data, features)

        logger.debug(f"Calculated {len(features.columns)} liquidity features")
        return features

    def get_feature_names(self) -> List[str]:
        """
        Get list of feature names this engineer produces.

        Returns:
            List of feature names
        """
        feature_names = []

        # Liquidation features
        if self.feature_config["liquidation_features"]:
            feature_names.extend([
                'long_liquidation_pressure', 'short_liquidation_pressure',
                'liquidation_imbalance', 'nearest_liquidation_distance',
                'liquidation_density_score', 'liquidation_cluster_strength'
            ])

        # Funding rate features
        if self.feature_config["funding_rate_features"]:
            feature_names.extend([
                'funding_rate', 'funding_rate_zscore', 'funding_rate_trend',
                'funding_rate_momentum', 'funding_pressure', 'funding_mean_reversion'
            ])

        # Open interest features
        if self.feature_config["open_interest_features"]:
            feature_names.extend([
                'open_interest', 'oi_change', 'oi_zscore', 'oi_volume_ratio',
                'oi_price_correlation', 'oi_concentration', 'oi_flow_strength'
            ])

        # Depth features
        if self.feature_config["depth_features"]:
            feature_names.extend([
                'liquidity_depth', 'depth_imbalance', 'liquidity_gradient',
                'market_impact_estimate', 'depth_volatility'
            ])

        # Concentration features
        if self.feature_config["concentration_features"]:
            feature_names.extend([
                'liquidity_concentration', 'large_order_density',
                'whale_activity', 'institutional_flow'
            ])

        return feature_names

    def _calculate_liquidation_features(self, data: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Calculate liquidation-related features."""
        try:
            # Simulate liquidation data if not provided
            if 'liquidation_long' not in data.columns:
                # Estimate liquidation pressure from price movements and volume
                price_change = data['close'].pct_change()
                volatility = price_change.rolling(20).std()

                # Long liquidations happen when price drops sharply
                long_liquidation_pressure = np.where(
                    (price_change < -2 * volatility) & (data['volume'] > data['volume'].quantile(0.7)),
                    data['volume'] * abs(price_change),
                    0
                )

                # Short liquidations happen when price rises sharply
                short_liquidation_pressure = np.where(
                    (price_change > 2 * volatility) & (data['volume'] > data['volume'].quantile(0.7)),
                    data['volume'] * abs(price_change),
                    0
                )

                features['liquidation_long'] = long_liquidation_pressure
                features['liquidation_short'] = short_liquidation_pressure
            else:
                features['liquidation_long'] = data['liquidation_long']
                features['liquidation_short'] = data['liquidation_short']

            # Liquidation pressure metrics
            features['long_liquidation_pressure'] = features['liquidation_long'].rolling(10).mean()
            features['short_liquidation_pressure'] = features['liquidation_short'].rolling(10).mean()

            # Liquidation imbalance
            total_liquidation = features['liquidation_long'] + features['liquidation_short']
            features['liquidation_imbalance'] = np.where(
                total_liquidation > 0,
                (features['liquidation_long'] - features['liquidation_short']) / total_liquidation,
                0
            )

            # Distance to nearest liquidation levels
            features['nearest_liquidation_distance'] = self._calculate_liquidation_proximity(data)

            # Liquidation density score
            features['liquidation_density_score'] = self._calculate_liquidation_density(data)

            # Liquidation cluster strength
            features['liquidation_cluster_strength'] = self._calculate_liquidation_clusters(data)

        except Exception as e:
            logger.warning(f"Error calculating liquidation features: {e}")

        return features

    def _calculate_funding_rate_features(self, data: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Calculate funding rate related features."""
        try:
            # Simulate funding rate if not provided
            if 'funding_rate' not in data.columns:
                # Estimate funding rate from price momentum and volume
                price_momentum = data['close'].pct_change(8)  # 8-hour momentum
                volume_pressure = (data['volume'] / data['volume'].rolling(24).mean()) - 1

                # Funding rate tends to follow price momentum
                funding_rate = price_momentum * 0.1 + volume_pressure * 0.05
                funding_rate = funding_rate.clip(-0.01, 0.01)  # Typical funding rate range
            else:
                funding_rate = data['funding_rate']

            features['funding_rate'] = funding_rate

            # Funding rate statistics
            features['funding_rate_zscore'] = (funding_rate - funding_rate.rolling(48).mean()) / funding_rate.rolling(48).std()
            features['funding_rate_trend'] = funding_rate.diff()
            features['funding_rate_momentum'] = funding_rate.diff(3)

            # Funding pressure (extreme funding rates)
            funding_percentile = funding_rate.rolling(168).rank(pct=True)  # Weekly percentile
            features['funding_pressure'] = np.where(
                (funding_percentile > 0.9) | (funding_percentile < 0.1),
                np.abs(funding_rate) * 1000,  # Scale up for significance
                0
            )

            # Mean reversion indicator
            features['funding_mean_reversion'] = np.where(
                np.abs(funding_rate) > funding_rate.rolling(48).std() * 2,
                -np.sign(funding_rate),  # Expect reversal
                0
            )

        except Exception as e:
            logger.warning(f"Error calculating funding rate features: {e}")

        return features

    def _calculate_open_interest_features(self, data: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Calculate open interest related features."""
        try:
            # Simulate open interest if not provided
            if 'open_interest' not in data.columns:
                # Estimate OI from volume and price trends
                base_oi = data['volume'].rolling(24).sum() * 0.3  # Base OI as fraction of volume
                trend_adjustment = 1 + data['close'].pct_change(24).clip(-0.5, 0.5) * 2
                open_interest = base_oi * trend_adjustment
            else:
                open_interest = data['open_interest']

            features['open_interest'] = open_interest

            # OI dynamics
            features['oi_change'] = open_interest.pct_change()
            features['oi_zscore'] = (open_interest - open_interest.rolling(168).mean()) / open_interest.rolling(168).std()

            # OI to volume ratio
            features['oi_volume_ratio'] = open_interest / (data['volume'].rolling(24).sum() + 1e-8)

            # OI-price correlation
            features['oi_price_correlation'] = self._calculate_oi_price_correlation(data, open_interest)

            # OI concentration
            features['oi_concentration'] = open_interest.rolling(12).std() / (open_interest.rolling(12).mean() + 1e-8)

            # OI flow strength
            features['oi_flow_strength'] = np.abs(features['oi_change']) * features['oi_volume_ratio']

        except Exception as e:
            logger.warning(f"Error calculating open interest features: {e}")

        return features

    def _calculate_depth_features(self, data: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Calculate market depth features."""
        try:
            # Estimate liquidity depth from volume patterns
            volume_ma = data['volume'].rolling(20).mean()
            volume_std = data['volume'].rolling(20).std()

            # Liquidity depth (inverse of volatility relative to volume)
            price_volatility = data['close'].pct_change().rolling(20).std()
            features['liquidity_depth'] = volume_ma / (price_volatility * data['close'] + 1e-8)

            # Depth imbalance (using volume-weighted price movement)
            price_change = data['close'].pct_change()
            volume_weighted_change = price_change * data['volume']
            features['depth_imbalance'] = volume_weighted_change.rolling(10).sum() / (volume_ma + 1e-8)

            # Liquidity gradient (how depth changes with price movement)
            features['liquidity_gradient'] = features['liquidity_depth'].diff() / (data['close'].diff() + 1e-8)

            # Market impact estimate
            features['market_impact_estimate'] = np.abs(price_change) * (data['volume'] / (features['liquidity_depth'] + 1e-8))

            # Depth volatility
            features['depth_volatility'] = features['liquidity_depth'].rolling(20).std() / (features['liquidity_depth'].rolling(20).mean() + 1e-8)

        except Exception as e:
            logger.warning(f"Error calculating depth features: {e}")

        return features

    def _calculate_concentration_features(self, data: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Calculate liquidity concentration features."""
        try:
            # Liquidity concentration (how concentrated is liquidity at current price)
            price_deviation = abs(data['close'] - data['close'].rolling(20).mean()) / data['close'].rolling(20).mean()
            volume_at_price = data['volume'] * np.exp(-price_deviation * 10)  # Volume concentration

            features['liquidity_concentration'] = volume_at_price / (data['volume'].rolling(20).mean() + 1e-8)

            # Large order density (proxy for institutional activity)
            large_orders = data['volume'] > data['volume'].quantile(0.9)
            features['large_order_density'] = large_orders.rolling(10).sum() / 10

            # Whale activity indicator
            volume_surge = data['volume'] / data['volume'].rolling(24).mean()
            features['whale_activity'] = np.where(volume_surge > 3, np.log(volume_surge), 0)

            # Institutional flow (sustained high volume with price impact)
            sustained_volume = data['volume'].rolling(6).mean() > data['volume'].rolling(24).mean() * 2
            price_impact = abs(data['close'].pct_change(6))
            features['institutional_flow'] = sustained_volume * price_impact

        except Exception as e:
            logger.warning(f"Error calculating concentration features: {e}")

        return features

    def _calculate_flow_features(self, data: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Calculate liquidity flow features."""
        try:
            # Liquidity inflow/outflow
            volume_change = data['volume'].diff()
            features['liquidity_inflow'] = np.where(volume_change > 0, volume_change, 0)
            features['liquidity_outflow'] = np.where(volume_change < 0, -volume_change, 0)

            # Net liquidity flow
            features['net_liquidity_flow'] = features['liquidity_inflow'] - features['liquidity_outflow']

            # Flow momentum
            features['flow_momentum'] = features['net_liquidity_flow'].rolling(5).mean()

            # Flow volatility
            features['flow_volatility'] = features['net_liquidity_flow'].rolling(20).std()

        except Exception as e:
            logger.warning(f"Error calculating flow features: {e}")

        return features

    def _calculate_lagged_features(self, data: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Calculate lagged liquidity features."""
        try:
            lag_periods = self.feature_config["lag_periods"]

            for lag in lag_periods:
                if 'liquidation_imbalance' in features:
                    features[f'liquidation_imbalance_lag_{lag}'] = features['liquidation_imbalance'].shift(lag)

                if 'funding_rate' in features:
                    features[f'funding_rate_lag_{lag}'] = features['funding_rate'].shift(lag)

                if 'oi_change' in features:
                    features[f'oi_change_lag_{lag}'] = features['oi_change'].shift(lag)

                if 'liquidity_depth' in features:
                    features[f'liquidity_depth_lag_{lag}'] = features['liquidity_depth'].shift(lag)

        except Exception as e:
            logger.warning(f"Error calculating lagged features: {e}")

        return features

    def _calculate_liquidation_proximity(self, data: pd.DataFrame) -> pd.Series:
        """Calculate distance to nearest liquidation levels."""
        try:
            # Estimate liquidation levels based on price leverage assumptions
            # This is a simplified calculation - in practice, you'd use actual liquidation data

            current_price = data['close']
            volatility = current_price.pct_change().rolling(20).std()

            # Estimate liquidation distances (simplified)
            long_liquidation_distance = self.feature_config["proximity_threshold"] + np.random.normal(0, volatility)
            short_liquidation_distance = self.feature_config["proximity_threshold"] + np.random.normal(0, volatility)

            # Return minimum distance to any liquidation level
            return np.minimum(long_liquidation_distance, short_liquidation_distance)

        except:
            return pd.Series(self.feature_config["proximity_threshold"], index=data.index)

    def _calculate_liquidation_density(self, data: pd.DataFrame) -> pd.Series:
        """Calculate liquidation density score."""
        try:
            # Simulate liquidation density based on volume and price movement
            volume_pressure = data['volume'] / data['volume'].rolling(24).mean()
            price_acceleration = data['close'].pct_change().diff()

            density_score = volume_pressure * np.abs(price_acceleration) * 1000000
            return density_score.fillna(0)

        except:
            return pd.Series(0, index=data.index)

    def _calculate_liquidation_clusters(self, data: pd.DataFrame) -> pd.Series:
        """Calculate liquidation cluster strength."""
        try:
            # Simulate liquidation clusters using price levels
            price_levels = (data['close'] * 100).astype(int)  # Discretize price
            liquidation_intensity = data['volume'] * np.abs(data['close'].pct_change())

            # Cluster liquidation levels
            cluster_data = np.column_stack([price_levels.values, liquidation_intensity.values])
            clustering = DBSCAN(eps=5, min_samples=2).fit(cluster_data)

            # Calculate cluster strength
            unique_labels, counts = np.unique(clustering.labels_, return_counts=True)
            max_cluster_size = counts[counts > 0].max() if len(counts) > 1 else 0

            return pd.Series(max_cluster_size / len(data), index=data.index)

        except:
            return pd.Series(0, index=data.index)

    def _calculate_oi_price_correlation(self, data: pd.DataFrame, open_interest: pd.Series) -> pd.Series:
        """Calculate correlation between open interest and price."""
        try:
            # Rolling correlation between OI and price
            price_change = data['close'].pct_change()
            oi_change = open_interest.pct_change()

            rolling_corr = price_change.rolling(48).corr(oi_change)
            return rolling_corr.fillna(0)

        except:
            return pd.Series(0, index=data.index)

    def analyze_liquidation_heatmap(
        self,
        liquidation_data: pd.DataFrame,
        price_range: Tuple[float, float],
        granularity: float = 0.001
    ) -> pd.DataFrame:
        """
        Analyze liquidation heatmap data.

        Args:
            liquidation_data: DataFrame with liquidation levels
            price_range: (min_price, max_price) for analysis
            granularity: Price granularity for heatmap

        Returns:
            Processed heatmap features
        """
        try:
            # Create price bins
            price_bins = np.arange(price_range[0], price_range[1], granularity)
            heatmap = pd.DataFrame(index=price_bins, columns=['long_liquidation', 'short_liquidation'])
            heatmap.fillna(0, inplace=True)

            # Aggregate liquidations by price level
            if not liquidation_data.empty:
                for _, row in liquidation_data.iterrows():
                    price = row['price']
                    if price_range[0] <= price <= price_range[1]:
                        bin_idx = int((price - price_range[0]) / granularity)
                        if 0 <= bin_idx < len(heatmap):
                            if row.get('side') == 'long':
                                heatmap.iloc[bin_idx, 0] += row.get('size', 0)
                            elif row.get('side') == 'short':
                                heatmap.iloc[bin_idx, 1] += row.get('size', 0)

            # Calculate density features
            heatmap['total_liquidation'] = heatmap['long_liquidation'] + heatmap['short_liquidation']
            heatmap['liquidation_imbalance'] = (heatmap['long_liquidation'] - heatmap['short_liquidation']) / (heatmap['total_liquidation'] + 1e-8)

            return heatmap

        except Exception as e:
            logger.warning(f"Error analyzing liquidation heatmap: {e}")
            return pd.DataFrame()