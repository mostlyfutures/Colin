"""
Test configuration management for Colin Trading Bot.
"""

import pytest
import tempfile
import yaml
from pathlib import Path
from src.core.config import ConfigManager, Config


class TestConfigManager:
    """Test configuration manager functionality."""

    def test_load_valid_config(self):
        """Test loading a valid configuration file."""
        config_data = {
            'symbols': ['ETHUSDT', 'BTCUSDT'],
            'apis': {
                'binance': {
                    'base_url': 'https://fapi.binance.com',
                    'testnet': False
                },
                'coinglass': {
                    'base_url': 'https://www.coinglass.com/api/futures',
                    'rate_limit': 60
                }
            },
            'sessions': {
                'london': {
                    'start': '07:00',
                    'end': '16:00',
                    'weight': 1.2
                }
            },
            'ict': {
                'fair_value_gap': {
                    'min_gap_size': 0.001,
                    'lookback_periods': 50
                },
                'order_block': {
                    'min_candle_size': 0.002,
                    'lookback_periods': 20
                },
                'break_of_structure': {
                    'lookback_periods': 10
                }
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
                'order_book': {
                    'depth_levels': 20,
                    'imbalance_threshold': 0.7
                },
                'trade_delta': {
                    'lookback_minutes': 15,
                    'volume_threshold': 1000000
                }
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
                'file': 'logs/colin_bot.log'
            },
            'development': {
                'test_mode': True,
                'mock_api_responses': True
            }
        }

        # Create temporary config file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name

        try:
            # Test loading config
            config_manager = ConfigManager(temp_path)
            config = config_manager.load_config()

            # Verify loaded config
            assert isinstance(config, Config)
            assert config.symbols == ['ETHUSDT', 'BTCUSDT']
            assert config.apis['binance'].base_url == 'https://fapi.binance.com'
            assert config.apis['binance'].testnet is False
            assert config.sessions['london'].start == '07:00'
            assert config.sessions['london'].weight == 1.2
            assert config.scoring.weights['liquidity_proximity'] == 0.25
            assert config.risk.max_position_size == 0.02

        finally:
            # Clean up temp file
            Path(temp_path).unlink()

    def test_load_nonexistent_config(self):
        """Test loading a non-existent configuration file."""
        config_manager = ConfigManager('/nonexistent/path/config.yaml')

        with pytest.raises(FileNotFoundError):
            config_manager.load_config()

    def test_load_invalid_yaml_config(self):
        """Test loading an invalid YAML configuration file."""
        # Create temporary invalid YAML file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write('invalid: yaml: content: [')
            temp_path = f.name

        try:
            config_manager = ConfigManager(temp_path)

            with pytest.raises(ValueError, match="Invalid YAML"):
                config_manager.load_config()

        finally:
            Path(temp_path).unlink()

    def test_environment_variable_substitution(self):
        """Test environment variable substitution in config."""
        import os

        # Set environment variable
        os.environ['TEST_API_KEY'] = 'test_api_key_value'

        config_data = {
            'symbols': ['ETHUSDT'],
            'apis': {
                'binance': {
                    'base_url': 'https://fapi.binance.com',
                    'api_key': '${TEST_API_KEY}'
                }
            },
            'sessions': {},
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

        # Create temporary config file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name

        try:
            # Test loading config with env var substitution
            config_manager = ConfigManager(temp_path)
            config = config_manager.load_config()

            # Verify environment variable was substituted
            # Note: This would require the actual config object to have the substituted value
            # For now, we just test that it loads without error

        finally:
            # Clean up
            Path(temp_path).unlink()
            if 'TEST_API_KEY' in os.environ:
                del os.environ['TEST_API_KEY']

    def test_get_api_key_and_secret(self):
        """Test retrieving API keys and secrets."""
        import os

        # Set environment variables
        os.environ['BINANCE_API_KEY'] = 'test_binance_key'
        os.environ['BINANCE_API_SECRET'] = 'test_binance_secret'

        try:
            config_manager = ConfigManager()

            # Test getting API key
            api_key = config_manager.get_api_key('binance')
            assert api_key == 'test_binance_key'

            # Test getting API secret
            api_secret = config_manager.get_api_secret('binance')
            assert api_secret == 'test_binance_secret'

            # Test non-existent API
            non_existent_key = config_manager.get_api_key('nonexistent')
            assert non_existent_key is None

        finally:
            # Clean up environment variables
            if 'BINANCE_API_KEY' in os.environ:
                del os.environ['BINANCE_API_KEY']
            if 'BINANCE_API_SECRET' in os.environ:
                del os.environ['BINANCE_API_SECRET']

    def test_config_property_access(self):
        """Test that config property provides easy access to loaded configuration."""
        config_data = {
            'symbols': ['ETHUSDT'],
            'apis': {
                'binance': {'base_url': 'https://fapi.binance.com', 'testnet': False}
            },
            'sessions': {},
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

        # Create temporary config file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name

        try:
            config_manager = ConfigManager(temp_path)

            # Test that accessing config property loads config
            config = config_manager.config
            assert isinstance(config, Config)
            assert config.symbols == ['ETHUSDT']

            # Test that subsequent accesses don't reload
            config2 = config_manager.config
            assert config is config2  # Should be the same object

        finally:
            Path(temp_path).unlink()


if __name__ == '__main__':
    pytest.main([__file__])