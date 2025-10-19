"""
Alternative Data Feature Engineer for Colin Trading Bot v2.0

This module implements alternative data features that provide insights beyond
traditional market data. These features can include sentiment analysis,
on-chain metrics, social media signals, and macroeconomic indicators.

Features include:
- Sentiment analysis scores
- Social media metrics
- On-chain blockchain metrics
- Macroeconomic indicators
- Market sentiment indices
- Alternative data correlations
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from loguru import logger
from sklearn.preprocessing import StandardScaler

from ..base.feature_base import FeatureEngineerBase


class AlternativeFeatureEngineer(FeatureEngineerBase):
    """
    Alternative data feature engineer for comprehensive market analysis.

    This class processes various alternative data sources to generate features
    that complement traditional technical and fundamental analysis.
    """

    def __init__(
        self,
        feature_config: Dict[str, Any],
        scaling_config: Optional[Dict[str, Any]] = None,
        feature_type: str = "alternative"
    ):
        """
        Initialize alternative feature engineer.

        Args:
            feature_config: Configuration for alternative features
            scaling_config: Scaling configuration
            feature_type: Feature type identifier
        """
        # Default configuration
        default_config = {
            "sentiment_features": True,
            "social_media_features": True,
            "onchain_features": True,
            "macro_features": True,
            "market_sentiment_features": True,
            "time_windows": [1, 4, 24, 168],  # hours
            "sentiment_sources": ["twitter", "reddit", "news"],
            "social_metrics": ["mentions", "volume", "engagement"],
            "onchain_metrics": ["active_addresses", "transaction_volume", "gas_price"],
            "macro_indicators": ["vix", "dxy", "interest_rates"],
            "include_lagged_features": True,
            "lag_periods": [1, 2, 4, 8],
            "sentiment_threshold": 0.7,
            "trend_window": 12
        }

        feature_config = {**default_config, **feature_config}

        # Default scaling configuration for alternative features
        default_scaling = {
            "type": "standard",
            "params": {}
        }

        scaling_config = {**default_scaling, **(scaling_config or {})}

        super().__init__(feature_config, scaling_config, feature_type)

        logger.info(f"Initialized AlternativeFeatureEngineer with {len(feature_config)} config items")

    def calculate_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate alternative data features.

        Args:
            data: DataFrame with OHLCV data plus alternative data columns
                  (will be simulated if not provided)

        Returns:
            DataFrame with calculated alternative features
        """
        logger.debug(f"Calculating alternative features for {len(data)} samples")
        features = pd.DataFrame(index=data.index)

        # Sentiment features
        if self.feature_config["sentiment_features"]:
            features = self._calculate_sentiment_features(data, features)

        # Social media features
        if self.feature_config["social_media_features"]:
            features = self._calculate_social_media_features(data, features)

        # On-chain features
        if self.feature_config["onchain_features"]:
            features = self._calculate_onchain_features(data, features)

        # Macroeconomic features
        if self.feature_config["macro_features"]:
            features = self._calculate_macro_features(data, features)

        # Market sentiment features
        if self.feature_config["market_sentiment_features"]:
            features = self._calculate_market_sentiment_features(data, features)

        # Lagged features
        if self.feature_config["include_lagged_features"]:
            features = self._calculate_lagged_features(data, features)

        logger.debug(f"Calculated {len(features.columns)} alternative features")
        return features

    def get_feature_names(self) -> List[str]:
        """
        Get list of feature names this engineer produces.

        Returns:
            List of feature names
        """
        feature_names = []

        # Sentiment features
        if self.feature_config["sentiment_features"]:
            for source in self.feature_config["sentiment_sources"]:
                feature_names.extend([
                    f'{source}_sentiment', f'{source}_sentiment_zscore',
                    f'{source}_sentiment_trend', f'{source}_sentiment_volatility'
                ])

            feature_names.extend([
                'overall_sentiment', 'sentiment_disagreement', 'sentiment_momentum'
            ])

        # Social media features
        if self.feature_config["social_media_features"]:
            for metric in self.feature_config["social_metrics"]:
                feature_names.extend([
                    f'social_{metric}_volume', f'social_{metric}_growth',
                    f'social_{metric}_engagement', f'social_{metric}_reach'
                ])

        # On-chain features
        if self.feature_config["onchain_features"]:
            for metric in self.feature_config["onchain_metrics"]:
                feature_names.extend([
                    f'onchain_{metric}', f'onchain_{metric}_change',
                    f'onchain_{metric}_trend', f'onchain_{metric}_anomaly'
                ])

        # Macro features
        if self.feature_config["macro_features"]:
            for indicator in self.feature_config["macro_indicators"]:
                feature_names.extend([
                    f'macro_{indicator}', f'macro_{indicator}_change',
                    f'macro_{indicator}_trend', f'macro_{indicator}_impact'
                ])

        return feature_names

    def _calculate_sentiment_features(self, data: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Calculate sentiment analysis features."""
        try:
            # Simulate sentiment data if not provided
            sentiment_sources = self.feature_config["sentiment_sources"]

            for source in sentiment_sources:
                if f'{source}_sentiment' not in data.columns:
                    # Simulate sentiment based on price movements with noise
                    price_change = data['close'].pct_change()
                    base_sentiment = np.tanh(price_change * 10)  # Scale price change to [-1, 1]

                    # Add random noise and some trend persistence
                    noise = np.random.normal(0, 0.2, len(data))
                    sentiment = base_sentiment + noise
                    sentiment = sentiment.clip(-1, 1)

                    # Add some smoothing
                    sentiment = sentiment.rolling(3).mean().fillna(sentiment)
                else:
                    sentiment = data[f'{source}_sentiment']

                features[f'{source}_sentiment'] = sentiment

                # Sentiment statistics
                features[f'{source}_sentiment_zscore'] = (
                    (sentiment - sentiment.rolling(24).mean()) /
                    (sentiment.rolling(24).std() + 1e-8)
                )
                features[f'{source}_sentiment_trend'] = sentiment.diff()
                features[f'{source}_sentiment_volatility'] = sentiment.rolling(12).std()

            # Overall sentiment (weighted average)
            if len(sentiment_sources) > 0:
                sentiment_cols = [f'{source}_sentiment' for source in sentiment_sources]
                features['overall_sentiment'] = data[sentiment_cols].mean(axis=1)

                # Sentiment disagreement (standard deviation across sources)
                features['sentiment_disagreement'] = data[sentiment_cols].std(axis=1)

                # Sentiment momentum (change in overall sentiment)
                features['sentiment_momentum'] = features['overall_sentiment'].diff()

        except Exception as e:
            logger.warning(f"Error calculating sentiment features: {e}")

        return features

    def _calculate_social_media_features(self, data: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Calculate social media metrics features."""
        try:
            social_metrics = self.feature_config["social_metrics"]

            for metric in social_metrics:
                # Simulate social media data
                base_volume = data['volume'] / 1000  # Base social activity from trading volume

                if f'social_{metric}' not in data.columns:
                    # Add noise and trend
                    noise = np.random.lognormal(0, 0.5, len(data))
                    trend = np.linspace(1, 1.2, len(data))  # Upward trend
                    social_value = base_volume * noise * trend
                else:
                    social_value = data[f'social_{metric}']

                features[f'social_{metric}_volume'] = social_value
                features[f'social_{metric}_growth'] = social_value.pct_change()
                features[f'social_{metric}_engagement'] = social_value / (social_value.rolling(24).mean() + 1e-8)
                features[f'social_{metric}_reach'] = social_value.rolling(12).max() / (social_value + 1e-8)

        except Exception as e:
            logger.warning(f"Error calculating social media features: {e}")

        return features

    def _calculate_onchain_features(self, data: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Calculate on-chain blockchain metrics features."""
        try:
            onchain_metrics = self.feature_config["onchain_metrics"]

            for metric in onchain_metrics:
                # Simulate on-chain data based on trading activity
                base_activity = data['volume'].rolling(24).sum() / 1000

                if f'onchain_{metric}' not in data.columns:
                    if metric == "active_addresses":
                        # Active addresses correlate with volume
                        onchain_value = base_activity * np.random.uniform(0.1, 0.3)
                    elif metric == "transaction_volume":
                        # Transaction volume correlates with trading volume
                        onchain_value = base_activity * np.random.uniform(0.5, 2.0)
                    elif metric == "gas_price":
                        # Gas price correlates with network congestion
                        congestion = data['volume'] / data['volume'].rolling(24).mean()
                        onchain_value = 20 + congestion * 50  # Base gas price + congestion premium
                    else:
                        onchain_value = base_activity * np.random.uniform(0.5, 1.5)
                else:
                    onchain_value = data[f'onchain_{metric}']

                features[f'onchain_{metric}'] = onchain_value
                features[f'onchain_{metric}_change'] = onchain_value.pct_change()
                features[f'onchain_{metric}_trend'] = onchain_value.diff()

                # Anomaly detection (z-score based)
                zscore = (onchain_value - onchain_value.rolling(168).mean()) / (onchain_value.rolling(168).std() + 1e-8)
                features[f'onchain_{metric}_anomaly'] = np.abs(zscore) > 2

        except Exception as e:
            logger.warning(f"Error calculating onchain features: {e}")

        return features

    def _calculate_macro_features(self, data: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Calculate macroeconomic indicator features."""
        try:
            macro_indicators = self.feature_config["macro_indicators"]

            for indicator in macro_indicators:
                # Simulate macro data
                if f'macro_{indicator}' not in data.columns:
                    if indicator == "vix":
                        # VIX correlates with market volatility
                        market_vol = data['close'].pct_change().rolling(20).std() * np.sqrt(252)
                        vix_value = 15 + market_vol * 100  # Base VIX + volatility premium
                    elif indicator == "dxy":
                        # Dollar index with some trend and noise
                        trend = np.linspace(100, 105, len(data))
                        noise = np.random.normal(0, 1, len(data))
                        vix_value = trend + noise
                    elif indicator == "interest_rates":
                        # Interest rates with gradual changes
                        base_rate = 5.0
                        changes = np.random.normal(0, 0.01, len(data)).cumsum()
                        vix_value = base_rate + changes
                    else:
                        vix_value = 100 + np.random.normal(0, 5, len(data))
                else:
                    vix_value = data[f'macro_{indicator}']

                features[f'macro_{indicator}'] = vix_value
                features[f'macro_{indicator}_change'] = vix_value.diff()
                features[f'macro_{indicator}_trend'] = vix_value.diff(4)

                # Impact on crypto (correlation-based)
                if indicator == "vix":
                    # High VIX typically negative for crypto
                    features[f'macro_{indicator}_impact'] = -vix_value / 100
                elif indicator == "dxy":
                    # Strong dollar typically negative for crypto
                    features[f'macro_{indicator}_impact'] = -(vix_value - 100) / 100
                else:
                    features[f'macro_{indicator}_impact'] = (vix_value - vix_value.rolling(168).mean()) / vix_value.rolling(168).mean()

        except Exception as e:
            logger.warning(f"Error calculating macro features: {e}")

        return features

    def _calculate_market_sentiment_features(self, data: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Calculate market sentiment indices."""
        try:
            # Fear & Greed Index (composite)
            if 'overall_sentiment' in features:
                # Combine sentiment with price momentum
                price_momentum = data['close'].pct_change(24)
                fear_greed = (features['overall_sentiment'] + price_momentum * 5) / 2
                features['fear_greed_index'] = (fear_greed + 1) * 50  # Scale to 0-100
            else:
                features['fear_greed_index'] = 50  # Neutral

            # Crypto Fear & Greed (simplified)
            volatility_component = data['close'].pct_change().rolling(30).std() * 100
            volume_component = (data['volume'] / data['volume'].rolling(30).mean() - 1) * 50
            momentum_component = data['close'].pct_change(30) * 1000

            crypto_fear_greed = 50 - volatility_component + volume_component + momentum_component
            features['crypto_fear_greed'] = crypto_fear_greed.clip(0, 100)

            # Market sentiment divergence
            if 'sentiment_disagreement' in features:
                price_vol = data['close'].pct_change().rolling(20).std()
                features['sentiment_divergence'] = features['sentiment_disagreement'] / (price_vol + 1e-8)

            # Sentiment extremes
            if 'overall_sentiment' in features:
                sentiment_extreme = np.abs(features['overall_sentiment']) > self.feature_config["sentiment_threshold"]
                features['sentiment_extreme'] = sentiment_extreme.astype(int)

        except Exception as e:
            logger.warning(f"Error calculating market sentiment features: {e}")

        return features

    def _calculate_lagged_features(self, data: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Calculate lagged alternative features."""
        try:
            lag_periods = self.feature_config["lag_periods"]

            for lag in lag_periods:
                if 'overall_sentiment' in features:
                    features[f'sentiment_lag_{lag}'] = features['overall_sentiment'].shift(lag)

                if 'fear_greed_index' in features:
                    features[f'fear_greed_lag_{lag}'] = features['fear_greed_index'].shift(lag)

                if 'social_mentions_volume' in features:
                    features[f'social_mentions_lag_{lag}'] = features['social_mentions_volume'].shift(lag)

                if 'onchain_active_addresses' in features:
                    features[f'addresses_lag_{lag}'] = features['onchain_active_addresses'].shift(lag)

        except Exception as e:
            logger.warning(f"Error calculating lagged features: {e}")

        return features

    def process_sentiment_data(
        self,
        sentiment_data: pd.DataFrame,
        text_column: str = "text",
        timestamp_column: str = "timestamp"
    ) -> pd.DataFrame:
        """
        Process raw sentiment data into features.

        Args:
            sentiment_data: DataFrame with text sentiment data
            text_column: Column containing text to analyze
            timestamp_column: Column containing timestamps

        Returns:
            Processed sentiment features
        """
        try:
            # This would integrate with a sentiment analysis library like VADER or transformers
            # For now, return a placeholder
            processed_features = pd.DataFrame(index=sentiment_data[timestamp_column])

            # Simulate sentiment scores
            processed_features['sentiment_score'] = np.random.uniform(-1, 1, len(sentiment_data))
            processed_features['sentiment_confidence'] = np.random.uniform(0.5, 1.0, len(sentiment_data))

            return processed_features

        except Exception as e:
            logger.warning(f"Error processing sentiment data: {e}")
            return pd.DataFrame()

    def analyze_social_trends(
        self,
        social_data: pd.DataFrame,
        keywords: List[str],
        time_window: str = "1H"
    ) -> pd.DataFrame:
        """
        Analyze social media trends for specific keywords.

        Args:
            social_data: DataFrame with social media data
            keywords: List of keywords to track
            time_window: Time window for aggregation

        Returns:
            Trend analysis features
        """
        try:
            # This would analyze keyword frequency, sentiment, and engagement
            # For now, return a placeholder
            trend_features = pd.DataFrame()

            for keyword in keywords:
                # Simulate trend metrics
                trend_features[f'{keyword}_mentions'] = np.random.poisson(100, len(social_data))
                trend_features[f'{keyword}_sentiment'] = np.random.uniform(-1, 1, len(social_data))

            return trend_features

        except Exception as e:
            logger.warning(f"Error analyzing social trends: {e}")
            return pd.DataFrame()

    def calculate_alternative_correlations(
        self,
        market_data: pd.DataFrame,
        alternative_data: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Calculate correlations between market and alternative data.

        Args:
            market_data: Market price data
            alternative_data: Alternative data features

        Returns:
            Dictionary of correlation coefficients
        """
        try:
            correlations = {}
            price_returns = market_data['close'].pct_change()

            for column in alternative_data.columns:
                if column in alternative_data.columns:
                    correlation = price_returns.corr(alternative_data[column].pct_change())
                    correlations[column] = correlation if not np.isnan(correlation) else 0.0

            return correlations

        except Exception as e:
            logger.warning(f"Error calculating alternative correlations: {e}")
            return {}