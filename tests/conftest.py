"""
Pytest configuration and fixtures for Colin Trading Bot tests.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock
import tempfile
import yaml
import os

from src.core.config import ConfigManager


@pytest.fixture
def sample_config_data():
    """Sample configuration data for testing."""
    return {
        'symbols': ['ETHUSDT', 'BTCUSDT'],
        'apis': {
            'binance': {
                'base_url': 'https://fapi.binance.com',
                'testnet': True,
                'rate_limit': 10
            },
            'coinglass': {
                'base_url': 'https://www.coinglass.com/api/futures',
                'rate_limit': 60
            }
        },
        'sessions': {
            'asian': {'start': '00:00', 'end': '09:00', 'weight': 1.0},
            'london': {'start': '07:00', 'end': '16:00', 'weight': 1.2},
            'new_york': {'start': '12:00', 'end': '22:00', 'weight': 1.2},
            'london_ny_overlap': {'start': '12:00', 'end': '16:00', 'weight': 1.5}
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
            'heatmap': {
                'timeframes': ['1h', '4h', '24h'],
                'min_density_threshold': 1000000
            },
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
        'logging': {
            'level': 'INFO',
            'file': 'logs/test_colin_bot.log'
        },
        'development': {
            'test_mode': True,
            'mock_api_responses': True,
            'save_intermediate_data': False
        }
    }


@pytest.fixture
def config_manager(sample_config_data):
    """Create a ConfigManager instance with sample data."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(sample_config_data, f)
        temp_path = f.name

    try:
        manager = ConfigManager(temp_path)
        manager.load_config()  # Load config immediately
        yield manager
    finally:
        os.unlink(temp_path)


@pytest.fixture
def sample_ohlcv_data():
    """Create sample OHLCV data for testing."""
    np.random.seed(42)  # For reproducible tests

    base_price = 2000.0
    timestamps = pd.date_range(
        start=datetime.now() - timedelta(hours=100),
        periods=100,
        freq='1h'
    )

    # Generate realistic price data with some trends
    price_changes = np.random.normal(0, 5, 100)
    prices = base_price + np.cumsum(price_changes)

    # Add some trend components
    trend = np.linspace(0, 50, 100)  # Gentle uptrend
    prices += trend

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

        # Generate volume with some correlation to price movement
        if i > 0:
            price_move = abs(close_price - prices[i-1]) / prices[i-1]
            base_volume = 500000
            volume = base_volume * (1 + price_move * 10) * np.random.uniform(0.5, 2.0)
        else:
            volume = 500000

        data.append([open_price, high, low, close_price, volume])

    df = pd.DataFrame(data, columns=['open', 'high', 'low', 'close', 'volume'])
    df.index = timestamps

    return df


@pytest.fixture
def sample_order_book_data():
    """Create sample order book data for testing."""
    # Generate bid orders (descending price)
    mid_price = 2000.0
    bid_prices = np.linspace(mid_price - 10, mid_price - 0.1, 20)
    bid_volumes = np.random.uniform(1, 10, 20) * 100  # 100-1000 units

    # Generate ask orders (ascending price)
    ask_prices = np.linspace(mid_price + 0.1, mid_price + 10, 20)
    ask_volumes = np.random.uniform(1, 10, 20) * 100  # 100-1000 units

    return {
        'bids': pd.DataFrame({
            'price': bid_prices,
            'volume': bid_volumes
        }),
        'asks': pd.DataFrame({
            'price': ask_prices,
            'volume': ask_volumes
        }),
        'timestamp': datetime.now()
    }


@pytest.fixture
def sample_trades_data():
    """Create sample trades data for testing."""
    np.random.seed(42)

    num_trades = 100
    timestamps = pd.date_range(
        start=datetime.now() - timedelta(minutes=30),
        periods=num_trades,
        freq='30s'
    )

    # Generate trade data
    data = []
    for timestamp in timestamps:
        side = np.random.choice(['buy', 'sell'])
        price = 2000.0 + np.random.uniform(-5, 5)
        amount = np.random.uniform(0.1, 2.0)

        data.append([timestamp, side, price, amount])

    df = pd.DataFrame(data, columns=['datetime', 'side', 'price', 'amount'])
    return df


@pytest.fixture
def mock_binance_adapter():
    """Create a mock Binance adapter."""
    adapter = Mock()
    adapter.initialize = AsyncMock()
    adapter.close = AsyncMock()
    adapter.get_ohlcv = AsyncMock()
    adapter.get_open_interest = AsyncMock()
    adapter.get_funding_rate = AsyncMock()
    adapter.get_order_book = AsyncMock()
    adapter.get_recent_trades = AsyncMock()
    adapter.get_symbol_info = AsyncMock()

    return adapter


@pytest.fixture
def mock_coinglass_adapter():
    """Create a mock CoinGlass adapter."""
    adapter = Mock()
    adapter.initialize = AsyncMock()
    adapter.close = AsyncMock()
    adapter.get_liquidation_heatmap = AsyncMock()
    adapter.get_liquidation_levels = AsyncMock()
    adapter.get_liquidation_history = AsyncMock()
    adapter.analyze_liquidation_density = AsyncMock()
    adapter.get_liquidation_indicators = AsyncMock()

    return adapter


@pytest.fixture
def mock_liquidation_data():
    """Create mock liquidation data."""
    return {
        'density_clusters': [
            {
                'price_level': 1990.0,
                'density_score': 5000000,
                'side': 'long',
                'distance': 0.005
            },
            {
                'price_level': 2010.0,
                'density_score': 3000000,
                'side': 'short',
                'distance': 0.005
            }
        ],
        'heatmap': pd.DataFrame(),
        'levels': {
            'long': pd.DataFrame({
                'price': [1990.0, 1985.0],
                'volume': [100, 80],
                'usd_value': [200000, 160000]
            }),
            'short': pd.DataFrame({
                'price': [2010.0, 2015.0],
                'volume': [120, 90],
                'usd_value': [240000, 180000]
            })
        },
        'indicators': {
            'long_liquidation_pressure': 5000000,
            'short_liquidation_pressure': 3000000,
            'liquidation_imbalance': 0.25,
            'nearest_liquidation_distance': 0.005
        }
    }


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    import asyncio
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()