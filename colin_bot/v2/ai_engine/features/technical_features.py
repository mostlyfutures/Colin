"""
Technical Feature Engineer for Colin Trading Bot v2.0

This module implements comprehensive technical analysis features for financial
time series data, building upon existing signal generation patterns from the v1 system.

Features include:
- Price-based indicators (moving averages, Bollinger Bands, etc.)
- Momentum indicators (RSI, MACD, Stochastic, etc.)
- Volume indicators (OBV, VWAP, Volume Profile, etc.)
- Volatility indicators (ATR, Keltner Channels, etc.)
- Pattern recognition features
- Multi-timeframe features
"""

import numpy as np
import pandas as pd
import ta
from talib import abstract
from loguru import logger
from typing import Dict, List, Tuple, Any, Optional
from sklearn.preprocessing import RobustScaler

from ..base.feature_base import FeatureEngineerBase


class TechnicalFeatureEngineer(FeatureEngineerBase):
    """
    Comprehensive technical analysis feature engineer.

    This class generates a wide range of technical indicators and features
    that are essential for sophisticated trading models. It builds upon
    the existing institutional signal framework from the v1 system.
    """

    def __init__(
        self,
        feature_config: Dict[str, Any],
        scaling_config: Optional[Dict[str, Any]] = None,
        feature_type: str = "technical"
    ):
        """
        Initialize technical feature engineer.

        Args:
            feature_config: Configuration for technical features
            scaling_config: Scaling configuration
            feature_type: Feature type identifier
        """
        # Default configuration
        default_config = {
            "price_indicators": True,
            "momentum_indicators": True,
            "volume_indicators": True,
            "volatility_indicators": True,
            "pattern_features": True,
            "multi_timeframe": True,
            "timeframes": [5, 15, 60, 240, 1440],  # minutes
            "lookback_periods": [5, 10, 20, 50, 100, 200],
            "include_lagged_features": True,
            "lag_periods": [1, 2, 3, 5, 10],
            "include_interaction_features": True,
            "custom_indicators": {}
        }

        feature_config = {**default_config, **feature_config}

        # Default scaling configuration for technical features
        default_scaling = {
            "type": "robust",  # Robust scaling works well for financial data
            "params": {
                "quantile_range": (25.0, 75.0)
            }
        }

        scaling_config = {**default_scaling, **(scaling_config or {})}

        super().__init__(feature_config, scaling_config, feature_type)

        logger.info(f"Initialized TechnicalFeatureEngineer with {len(feature_config)} config items")

    def calculate_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all technical features.

        Args:
            data: OHLCV data DataFrame

        Returns:
            DataFrame with calculated technical features
        """
        logger.debug(f"Calculating technical features for {len(data)} samples")
        features = pd.DataFrame(index=data.index)

        # Basic price features
        features = self._calculate_price_features(data, features)

        # Moving averages and trends
        if self.feature_config["price_indicators"]:
            features = self._calculate_price_indicators(data, features)

        # Momentum indicators
        if self.feature_config["momentum_indicators"]:
            features = self._calculate_momentum_indicators(data, features)

        # Volume indicators
        if self.feature_config["volume_indicators"]:
            features = self._calculate_volume_indicators(data, features)

        # Volatility indicators
        if self.feature_config["volatility_indicators"]:
            features = self._calculate_volatility_indicators(data, features)

        # Pattern recognition features
        if self.feature_config["pattern_features"]:
            features = self._calculate_pattern_features(data, features)

        # Lagged features
        if self.feature_config["include_lagged_features"]:
            features = self._calculate_lagged_features(data, features)

        # Interaction features
        if self.feature_config["include_interaction_features"]:
            features = self._calculate_interaction_features(data, features)

        # Multi-timeframe features
        if self.feature_config["multi_timeframe"]:
            features = self._calculate_multi_timeframe_features(data, features)

        # Custom indicators
        if self.feature_config["custom_indicators"]:
            features = self._calculate_custom_indicators(data, features)

        logger.debug(f"Calculated {len(features.columns)} technical features")
        return features

    def get_feature_names(self) -> List[str]:
        """
        Get list of feature names this engineer produces.

        Returns:
            List of feature names
        """
        feature_names = []

        # Price features
        feature_names.extend([
            'price_change', 'price_change_pct', 'high_low_ratio', 'open_close_ratio',
            'upper_shadow', 'lower_shadow', 'body_size', 'range_size'
        ])

        # Moving average features
        if self.feature_config["price_indicators"]:
            for period in self.feature_config["lookback_periods"]:
                feature_names.extend([
                    f'sma_{period}', f'ema_{period}', f'price_sma_ratio_{period}',
                    f'price_ema_ratio_{period}', f'sma_slope_{period}', f'ema_slope_{period}'
                ])

        # Momentum features
        if self.feature_config["momentum_indicators"]:
            feature_names.extend([
                'rsi_14', 'rsi_overbought', 'rsi_oversold', 'rsi_divergence',
                'macd', 'macd_signal', 'macd_histogram', 'macd_bullish',
                'stoch_k', 'stoch_d', 'stoch_overbought', 'stoch_oversold',
                'williams_r', 'cci_14', 'mfi_14', 'roc_10', 'roc_change'
            ])

        # Volume features
        if self.feature_config["volume_indicators"]:
            feature_names.extend([
                'volume_sma_20', 'volume_ratio', 'volume_price_trend',
                'obv', 'obv_sma', 'vwap', 'price_vwap_ratio',
                'volume_weighted_price', 'efficiency_ratio'
            ])

        # Volatility features
        if self.feature_config["volatility_indicators"]:
            feature_names.extend([
                'atr_14', 'atr_ratio', 'true_range', 'keltner_upper',
                'keltner_lower', 'keltner_width', 'bollinger_upper',
                'bollinger_lower', 'bollinger_width', 'bollinger_position'
            ])

        # Pattern features
        if self.feature_config["pattern_features"]:
            feature_names.extend([
                'doji', 'hammer', 'engulfing_bullish', 'engulfing_bearish',
                'morning_star', 'evening_star', 'harami_bullish', 'harami_bearish'
            ])

        # Lagged features
        if self.feature_config["include_lagged_features"]:
            for lag in self.feature_config["lag_periods"]:
                feature_names.extend([f'return_lag_{lag}', f'volume_lag_{lag}'])

        return feature_names

    def _calculate_price_features(self, data: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Calculate basic price-based features."""
        # Price changes
        features['price_change'] = data['close'].diff()
        features['price_change_pct'] = data['close'].pct_change()

        # Price ratios
        features['high_low_ratio'] = data['high'] / data['low']
        features['open_close_ratio'] = data['open'] / data['close']

        # Candlestick features
        features['upper_shadow'] = data['high'] - data[['open', 'close']].max(axis=1)
        features['lower_shadow'] = data[['open', 'close']].min(axis=1) - data['low']
        features['body_size'] = abs(data['close'] - data['open'])
        features['range_size'] = data['high'] - data['low']

        # Price position in range
        features['price_position'] = (data['close'] - data['low']) / (data['high'] - data['low'])

        return features

    def _calculate_price_indicators(self, data: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Calculate moving average and trend indicators."""
        for period in self.feature_config["lookback_periods"]:
            # Simple Moving Average
            sma = data['close'].rolling(window=period).mean()
            features[f'sma_{period}'] = sma
            features[f'price_sma_ratio_{period}'] = data['close'] / sma

            # Exponential Moving Average
            ema = data['close'].ewm(span=period).mean()
            features[f'ema_{period}'] = ema
            features[f'price_ema_ratio_{period}'] = data['close'] / ema

            # Moving average slopes
            features[f'sma_slope_{period}'] = sma.diff() / sma.shift(1)
            features[f'ema_slope_{period}'] = ema.diff() / ema.shift(1)

            # Price above/below moving averages
            features[f'price_above_sma_{period}'] = (data['close'] > sma).astype(int)
            features[f'price_above_ema_{period}'] = (data['close'] > ema).astype(int)

        # Moving average crossovers
        if len(self.feature_config["lookback_periods"]) >= 2:
            short_period = self.feature_config["lookback_periods"][0]
            long_period = self.feature_config["lookback_periods"][-1]

            features['ma_crossover_signal'] = np.where(
                features[f'sma_{short_period}'] > features[f'sma_{long_period}'], 1, -1
            )

        return features

    def _calculate_momentum_indicators(self, data: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Calculate momentum indicators."""
        # RSI
        rsi = ta.momentum.RSIIndicator(close=data['close'], window=14)
        features['rsi_14'] = rsi.rsi()
        features['rsi_overbought'] = (features['rsi_14'] > 70).astype(int)
        features['rsi_oversold'] = (features['rsi_14'] < 30).astype(int)
        features['rsi_divergence'] = self._calculate_rsi_divergence(data, features['rsi_14'])

        # MACD
        macd = ta.trend.MACD(close=data['close'])
        features['macd'] = macd.macd()
        features['macd_signal'] = macd.macd_signal()
        features['macd_histogram'] = macd.macd_diff()
        features['macd_bullish'] = (features['macd'] > features['macd_signal']).astype(int)

        # Stochastic Oscillator
        stoch = ta.momentum.StochasticOscillator(
            high=data['high'], low=data['low'], close=data['close']
        )
        features['stoch_k'] = stoch.stoch()
        features['stoch_d'] = stoch.stoch_signal()
        features['stoch_overbought'] = (features['stoch_k'] > 80).astype(int)
        features['stoch_oversold'] = (features['stoch_k'] < 20).astype(int)

        # Williams %R
        features['williams_r'] = ta.momentum.WilliamsRIndicator(
            high=data['high'], low=data['low'], close=data['close']
        ).williams_r()

        # Commodity Channel Index
        features['cci_14'] = ta.trend.CCIIndicator(
            high=data['high'], low=data['low'], close=data['close'], window=14
        ).cci()

        # Money Flow Index
        features['mfi_14'] = ta.volume.MFIIndicator(
            high=data['high'], low=data['low'], close=data['close'],
            volume=data['volume'], window=14
        ).money_flow_index()

        # Rate of Change
        features['roc_10'] = ta.momentum.ROCIndicator(close=data['close'], window=10).roc()
        features['roc_change'] = features['roc_10'].diff()

        return features

    def _calculate_volume_indicators(self, data: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Calculate volume-based indicators."""
        # Volume moving averages
        features['volume_sma_20'] = data['volume'].rolling(window=20).mean()
        features['volume_ratio'] = data['volume'] / features['volume_sma_20']

        # Volume-Price Trend
        features['volume_price_trend'] = ta.volume.VolumePriceTrendIndicator(
            close=data['close'], volume=data['volume']
        ).volume_price_trend()

        # On-Balance Volume
        features['obv'] = ta.volume.OnBalanceVolumeIndicator(
            close=data['close'], volume=data['volume']
        ).on_balance_volume()
        features['obv_sma'] = features['obv'].rolling(window=20).mean()

        # VWAP
        features['vwap'] = ta.volume.VolumeWeightedAveragePrice(
            high=data['high'], low=data['low'], close=data['close'], volume=data['volume']
        ).volume_weighted_average_price()
        features['price_vwap_ratio'] = data['close'] / features['vwap']

        # Volume Weighted Price
        typical_price = (data['high'] + data['low'] + data['close']) / 3
        features['volume_weighted_price'] = (typical_price * data['volume']).rolling(window=20).sum() / \
                                         data['volume'].rolling(window=20).sum()

        # Efficiency Ratio (Kaufman's Indicator)
        features['efficiency_ratio'] = ta.others.EfficiencyRatioIndicator(
            close=data['close'], window=14
        ).efficiency_ratio()

        return features

    def _calculate_volatility_indicators(self, data: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Calculate volatility indicators."""
        # Average True Range
        atr = ta.volatility.AverageTrueRange(
            high=data['high'], low=data['low'], close=data['close'], window=14
        )
        features['atr_14'] = atr.average_true_range()
        features['atr_ratio'] = features['atr_14'] / data['close']
        features['true_range'] = atr.true_range()

        # Keltner Channels
        keltner = ta.volatility.KeltnerChannel(
            high=data['high'], low=data['low'], close=data['close'], window=20
        )
        features['keltner_upper'] = keltner.keltner_channel_hband()
        features['keltner_lower'] = keltner.keltner_channel_lband()
        features['keltner_width'] = features['keltner_upper'] - features['keltner_lower']

        # Bollinger Bands
        bollinger = ta.volatility.BollingerBands(close=data['close'], window=20)
        features['bollinger_upper'] = bollinger.bollinger_hband()
        features['bollinger_lower'] = bollinger.bollinger_lband()
        features['bollinger_width'] = bollinger.bollinger_wband()
        features['bollinger_position'] = (data['close'] - features['bollinger_lower']) / features['bollinger_width']

        # Historical Volatility
        features['historical_volatility'] = features['price_change_pct'].rolling(window=20).std() * np.sqrt(252)

        return features

    def _calculate_pattern_features(self, data: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Calculate candlestick pattern recognition features."""
        try:
            # Doji patterns
            features['doji'] = (abs(data['close'] - data['open']) / (data['high'] - data['low']) < 0.1).astype(int)

            # Hammer patterns
            body_size = abs(data['close'] - data['open'])
            upper_shadow = data['high'] - data[['open', 'close']].max(axis=1)
            lower_shadow = data[['open', 'close']].min(axis=1) - data['low']
            features['hammer'] = ((lower_shadow > 2 * body_size) & (upper_shadow < 0.1 * body_size)).astype(int)

            # Engulfing patterns
            price_change = data['close'].diff()
            features['engulfing_bullish'] = ((price_change > 0) &
                                           (data['close'].shift(1) < data['open'].shift(1)) &
                                           (data['open'] < data['close'].shift(1)) &
                                           (data['close'] > data['open'].shift(1))).astype(int)

            features['engulfing_bearish'] = ((price_change < 0) &
                                           (data['close'].shift(1) > data['open'].shift(1)) &
                                           (data['open'] > data['close'].shift(1)) &
                                           (data['close'] < data['open'].shift(1))).astype(int)

            # Morning/Evening Star patterns (simplified)
            features['morning_star'] = ((price_change > 0) &
                                       (data['close'].shift(1) < data['open'].shift(1)) &
                                       (price_change.shift(1) < 0) &
                                       (abs(price_change.shift(1)) > 0.02)).astype(int)

            features['evening_star'] = ((price_change < 0) &
                                      (data['close'].shift(1) > data['open'].shift(1)) &
                                      (price_change.shift(1) > 0) &
                                      (abs(price_change.shift(1)) > 0.02)).astype(int)

        except Exception as e:
            logger.warning(f"Error calculating pattern features: {e}")

        return features

    def _calculate_lagged_features(self, data: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Calculate lagged features."""
        returns = features['price_change_pct']
        volume_change = data['volume'].pct_change()

        for lag in self.feature_config["lag_periods"]:
            features[f'return_lag_{lag}'] = returns.shift(lag)
            features[f'volume_lag_{lag}'] = volume_change.shift(lag)

        return features

    def _calculate_interaction_features(self, data: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Calculate interaction features between indicators."""
        try:
            # RSI and Volume interaction
            features['rsi_volume_interaction'] = features['rsi_14'] * features['volume_ratio']

            # Price and Volume interaction
            features['price_volume_trend'] = features['price_change_pct'] * features['volume_ratio']

            # Volatility and Volume interaction
            features['vol_volume_interaction'] = features['atr_ratio'] * features['volume_ratio']

            # MACD and RSI interaction
            features['macd_rsi_interaction'] = features['macd'] * features['rsi_14']

        except Exception as e:
            logger.warning(f"Error calculating interaction features: {e}")

        return features

    def _calculate_multi_timeframe_features(self, data: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Calculate multi-timeframe features."""
        # This is a simplified version - in production, you'd resample data
        # to different timeframes and calculate features for each

        try:
            # Simulate multi-timeframe features using different lookback periods
            for tf in [5, 15, 60]:  # Represent different timeframes
                # Resampled features would be calculated here
                # For now, using rolling windows to simulate
                features[f'tf_{tf}_return'] = data['close'].pct_change(tf)
                features[f'tf_{tf}_volatility'] = data['close'].pct_change().rolling(tf).std()
                features[f'tf_{tf}_trend'] = (data['close'] > data['close'].rolling(tf).mean()).astype(int)

        except Exception as e:
            logger.warning(f"Error calculating multi-timeframe features: {e}")

        return features

    def _calculate_custom_indicators(self, data: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Calculate custom indicators based on configuration."""
        for indicator_name, indicator_config in self.feature_config["custom_indicators"].items():
            try:
                # This would implement custom indicators based on config
                # For now, it's a placeholder
                logger.debug(f"Custom indicator {indicator_name} not implemented yet")
            except Exception as e:
                logger.warning(f"Error calculating custom indicator {indicator_name}: {e}")

        return features

    def _calculate_rsi_divergence(self, data: pd.DataFrame, rsi: pd.Series, window: int = 14) -> pd.Series:
        """Calculate RSI divergence."""
        try:
            price_highs = data['close'].rolling(window=window).max()
            price_lows = data['close'].rolling(window=window).min()
            rsi_highs = rsi.rolling(window=window).max()
            rsi_lows = rsi.rolling(window=window).min()

            # Bullish divergence: price makes lower low, RSI makes higher low
            bullish_divergence = ((price_lows.diff() < 0) & (rsi_lows.diff() > 0)).astype(int)

            # Bearish divergence: price makes higher high, RSI makes lower high
            bearish_divergence = ((price_highs.diff() > 0) & (rsi_highs.diff() < 0)).astype(int)

            return bullish_divergence - bearish_divergence
        except:
            return pd.Series(0, index=data.index)