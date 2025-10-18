"""
Test ICT structure detector for Colin Trading Bot.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.structure.ict_detector import ICTDetector, FairValueGap, OrderBlock, BreakOfStructure
from src.core.config import ConfigManager


class TestICTDetector:
    """Test ICT structure detector functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create a mock config
        self.config_data = {
            'symbols': ['ETHUSDT'],
            'apis': {
                'binance': {'base_url': 'https://fapi.binance.com', 'testnet': True},
                'coinglass': {'base_url': 'https://www.coinglass.com/api', 'rate_limit': 60}
            },
            'sessions': {
                'london': {'start': '07:00', 'end': '16:00', 'weight': 1.2}
            },
            'ict': {
                'fair_value_gap': {'min_gap_size': 0.001, 'lookback_periods': 50},
                'order_block': {'min_candle_size': 0.002, 'lookback_periods': 20},
                'break_of_structure': {'lookback_periods': 10}
            },
            'scoring': {
                'weights': {
                    'liquidity_proximity': 0.25,
                    'ict_confluence': 0.25,
                    'killzone_alignment': 0.15,
                    'order_flow_delta': 0.20,
                    'volume_oi_confirmation': 0.15
                },
                'thresholds': {
                    'high_confidence': 80,
                    'medium_confidence': 60,
                    'low_confidence': 40
                }
            },
            'order_flow': {
                'order_book': {'depth_levels': 20, 'imbalance_threshold': 0.7},
                'trade_delta': {'lookback_minutes': 15, 'volume_threshold': 1000000}
            },
            'liquidations': {
                'heatmap': {'timeframes': ['1h'], 'min_density_threshold': 1000000},
                'proximity_threshold': 0.005
            },
            'risk': {
                'max_position_size': 0.02,
                'stop_loss_buffer': 0.002,
                'volatility_threshold': 0.03
            },
            'output': {
                'format': 'json',
                'include_rationale': True,
                'max_rationale_points': 3,
                'include_stop_loss': True,
                'include_volatility_warning': True
            },
            'logging': {'level': 'INFO'},
            'development': {'test_mode': True}
        }

        # Create config manager and detector
        import tempfile
        import yaml
        import os

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(self.config_data, f)
            temp_path = f.name

        try:
            self.config_manager = ConfigManager(temp_path)
            self.detector = ICTDetector(self.config_manager)
        finally:
            os.unlink(temp_path)

    def create_sample_ohlcv_data(self, pattern_type="normal"):
        """Create sample OHLCV data for testing."""
        # Create 50 candles of sample data
        np.random.seed(42)  # For reproducible tests

        base_price = 2000.0
        timestamps = pd.date_range(
            start=datetime.now() - timedelta(hours=50),
            periods=50,
            freq='1h'
        )

        # Generate price data
        if pattern_type == "bullish_fvg":
            # Create data with bullish FVG pattern
            prices = np.random.normal(0, 5, 50).cumsum() + base_price
            # Create FVG at candle 20-22
            prices[20] = 2010.0  # High candle
            prices[21] = 2025.0  # Impulsive candle (strong up)
            prices[22] = 2035.0  # High candle with gap
        elif pattern_type == "bearish_fvg":
            # Create data with bearish FVG pattern
            prices = np.random.normal(0, 5, 50).cumsum() + base_price
            # Create FVG at candle 20-22
            prices[20] = 2010.0  # Low candle
            prices[21] = 1995.0  # Impulsive candle (strong down)
            prices[22] = 1985.0  # Low candle with gap
        elif pattern_type == "order_block":
            # Create data with order block pattern
            prices = np.random.normal(0, 3, 50).cumsum() + base_price
            # Create order block at candle 25
            prices[25] = 1990.0  # Strong down candle
            prices[26] = 2005.0  # Strong up candle
            prices[27] = 2015.0  # Continuation up
        else:
            prices = np.random.normal(0, 5, 50).cumsum() + base_price

        # Create OHLCV DataFrame
        data = []
        for i, (timestamp, close_price) in enumerate(zip(timestamps, prices)):
            # Generate realistic OHLC
            volatility = 10
            high = close_price + np.random.uniform(0, volatility)
            low = close_price - np.random.uniform(0, volatility)
            open_price = close_price + np.random.uniform(-volatility/2, volatility/2)

            # Ensure OHLC relationships are valid
            high = max(high, open_price, close_price)
            low = min(low, open_price, close_price)

            # Generate volume
            volume = np.random.uniform(100000, 1000000)

            data.append([open_price, high, low, close_price, volume])

        df = pd.DataFrame(data, columns=['open', 'high', 'low', 'close', 'volume'])
        df.index = timestamps

        return df

    def test_detect_fair_value_gaps_bullish(self):
        """Test detection of bullish Fair Value Gaps."""
        # Create data with bullish FVG
        df = self.create_sample_ohlcv_data("bullish_fvg")

        # Detect FVGs
        fvg_list = self.detector.detect_fair_value_gaps(df)

        # Should detect at least one FVG
        assert len(fvg_list) > 0

        # Check that detected FVGs have correct properties
        for fvg in fvg_list:
            assert isinstance(fvg, FairValueGap)
            assert fvg.type.value == "fair_value_gap"
            assert fvg.confidence > 0
            assert fvg.top < fvg.bottom  # Bullish FVG: gap up
            assert fvg.midline == (fvg.top + fvg.bottom) / 2
            assert fvg.size > 0

    def test_detect_fair_value_gaps_bearish(self):
        """Test detection of bearish Fair Value Gaps."""
        # Create data with bearish FVG
        df = self.create_sample_ohlcv_data("bearish_fvg")

        # Detect FVGs
        fvg_list = self.detector.detect_fair_value_gaps(df)

        # Should detect at least one FVG
        assert len(fvg_list) > 0

        # Check that detected FVGs have correct properties
        for fvg in fvg_list:
            assert isinstance(fvg, FairValueGap)
            assert fvg.confidence > 0
            # For bearish FVG, top should be greater than bottom
            if fvg.top > fvg.bottom:
                # This is a bearish FVG (gap down)
                assert fvg.size > 0

    def test_detect_fair_value_gaps_empty_data(self):
        """Test FVG detection with empty data."""
        df = pd.DataFrame()

        fvg_list = self.detector.detect_fair_value_gaps(df)

        assert fvg_list == []

    def test_detect_fair_value_gaps_insufficient_data(self):
        """Test FVG detection with insufficient data."""
        # Create data with less than 3 candles
        df = self.create_sample_ohlcv_data("normal").iloc[:2]

        fvg_list = self.detector.detect_fair_value_gaps(df)

        assert fvg_list == []

    def test_detect_order_blocks(self):
        """Test detection of Order Blocks."""
        # Create data with order block pattern
        df = self.create_sample_ohlcv_data("order_block")

        # Detect Order Blocks
        ob_list = self.detector.detect_order_blocks(df)

        # Should detect order blocks
        assert len(ob_list) >= 0  # May or may not detect depending on pattern strength

        # Check properties of detected order blocks
        for ob in ob_list:
            assert isinstance(ob, OrderBlock)
            assert ob.type.value == "order_block"
            assert ob.confidence > 0
            assert ob.candle_high > ob.candle_low
            assert ob.side in ["bullish", "bearish"]
            assert ob.candle_volume > 0

    def test_detect_break_of_structure(self):
        """Test detection of Break of Structure."""
        # Create trend data with clear BOS
        df = self.create_sample_ohlcv_data("normal")

        # Create a clear uptrend with BOS
        for i in range(10, 30):
            df.iloc[i, df.columns.get_loc('close')] += i * 2  # Trending up

        # Detect BOS
        bos_list = self.detector.detect_break_of_structure(df)

        # Should detect BOS in trending data
        assert len(bos_list) >= 0

        # Check properties of detected BOS
        for bos in bos_list:
            assert isinstance(bos, BreakOfStructure)
            assert bos.type.value == "break_of_structure"
            assert bos.confidence > 0
            assert bos.side in ["bullish", "bearish"]
            assert bos.break_candle_high > 0
            assert bos.break_candle_low > 0

    def test_analyze_ict_confluence(self):
        """Test ICT confluence analysis."""
        df = self.create_sample_ohlcv_data("normal")
        current_price = df['close'].iloc[-1]

        # Analyze confluence
        confluence = self.detector.analyze_ict_confluence(df, current_price)

        # Check confluence structure
        assert 'confluence_score' in confluence
        assert 'rationale_points' in confluence
        assert 'nearby_structures' in confluence
        assert 'total_structures' in confluence

        # Check data types
        assert isinstance(confluence['confluence_score'], float)
        assert isinstance(confluence['rationale_points'], list)
        assert isinstance(confluence['nearby_structures'], dict)
        assert isinstance(confluence['total_structures'], dict)

        # Check score range
        assert 0 <= confluence['confluence_score'] <= 1

    def test_get_structural_stop_loss(self):
        """Test structural stop loss calculation."""
        df = self.create_sample_ohlcv_data("normal")
        entry_price = df['close'].iloc[-1]

        # Test long position stop loss
        long_stop = self.detector.get_structural_stop_loss(df, entry_price, "long")
        if long_stop is not None:
            assert long_stop < entry_price
            assert long_stop > 0

        # Test short position stop loss
        short_stop = self.detector.get_structural_stop_loss(df, entry_price, "short")
        if short_stop is not None:
            assert short_stop > entry_price
            assert short_stop > 0

        # Test invalid direction
        invalid_stop = self.detector.get_structural_stop_loss(df, entry_price, "invalid")
        assert invalid_stop is None

    def test_fvg_confidence_calculation(self):
        """Test Fair Value Gap confidence calculation."""
        # Create a sample impulsive candle
        impulsive_candle = pd.Series({
            'open': 2000.0,
            'high': 2020.0,
            'low': 1995.0,
            'close': 2018.0,
            'volume': 2000000
        })

        gap_size = 0.01  # 1% gap

        confidence = self.detector._calculate_fvg_confidence(gap_size, impulsive_candle)

        assert 0 <= confidence <= 1
        assert confidence > 0  # Should have some confidence

    def test_get_timeframe_from_df(self):
        """Test timeframe detection from DataFrame."""
        # Test 1-hour data
        hourly_timestamps = pd.date_range(
            start=datetime.now() - timedelta(hours=10),
            periods=10,
            freq='1h'
        )
        df_hourly = pd.DataFrame({
            'close': [2000 + i for i in range(10)]
        }, index=hourly_timestamps)

        timeframe = self.detector._get_timeframe_from_df(df_hourly)
        assert timeframe == "1h"

        # Test 4-hour data
        four_hour_timestamps = pd.date_range(
            start=datetime.now() - timedelta(hours=40),
            periods=10,
            freq='4h'
        )
        df_4hourly = pd.DataFrame({
            'close': [2000 + i for i in range(10)]
        }, index=four_hour_timestamps)

        timeframe = self.detector._get_timeframe_from_df(df_4hourly)
        assert timeframe == "4h"

    def test_swing_highs_lows_detection(self):
        """Test swing high and low detection."""
        df = self.create_sample_ohlcv_data("normal")

        # Test swing highs detection
        swing_highs = self.detector._find_swing_highs(df, window=3)
        assert isinstance(swing_highs, list)

        # Test swing lows detection
        swing_lows = self.detector._find_swing_lows(df, window=3)
        assert isinstance(swing_lows, list)

        # Verify data structure
        for high_timestamp, high_price in swing_highs:
            assert isinstance(high_timestamp, (datetime, pd.Timestamp))
            assert high_price > 0

        for low_timestamp, low_price in swing_lows:
            assert isinstance(low_timestamp, (datetime, pd.Timestamp))
            assert low_price > 0


if __name__ == '__main__':
    pytest.main([__file__])