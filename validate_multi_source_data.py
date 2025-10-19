#!/usr/bin/env python3
"""
Validation script for multi-source market data implementation.
"""

import sys
import os
import traceback
from pathlib import Path

def validate_imports():
    """Validate that all new modules can be imported."""
    print("🔍 Validating imports...")

    try:
        # Test basic module imports
        sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

        # Test v2 data sources imports
        from v2.data_sources.config import MarketDataConfig, DataSourceConfig
        print("✅ MarketDataConfig import successful")

        from v2.data_sources.models import StandardMarketData, MarketDataSummary
        print("✅ Market data models import successful")

        # Test main config integration
        from v2.config.main_config import MainV2Config
        config = MainV2Config()
        assert hasattr(config, 'market_data'), "market_data field missing from MainV2Config"
        print("✅ Main configuration integration successful")

        return True

    except Exception as e:
        print(f"❌ Import validation failed: {e}")
        traceback.print_exc()
        return False

def validate_file_structure():
    """Validate that all required files exist."""
    print("🔍 Validating file structure...")

    required_files = [
        "src/v2/data_sources/__init__.py",
        "src/v2/data_sources/config.py",
        "src/v2/data_sources/models.py",
        "src/v2/data_sources/market_data_manager.py",
        "src/v2/data_sources/adapters/__init__.py",
        "src/v2/data_sources/adapters/base_adapter.py",
        "src/v2/data_sources/adapters/coingecko_adapter.py",
        "src/v2/data_sources/adapters/kraken_adapter.py",
        "src/v2/data_sources/adapters/cryptocompare_adapter.py",
        "src/v2/data_sources/adapters/alternative_me_adapter.py",
        "analyze_ethereum_multi_source.py",
        "tests/v2/data_sources/test_config.py",
        "tests/v2/data_sources/test_models.py",
        "tests/v2/data_sources/test_adapters/test_base_adapter.py"
    ]

    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
        else:
            print(f"✅ {file_path}")

    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False

    return True

def validate_configuration():
    """Validate configuration functionality."""
    print("🔍 Validating configuration...")

    try:
        sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
        from v2.data_sources.config import MarketDataConfig, DataSource

        # Test default configuration
        config = MarketDataConfig()
        assert config.primary_source == DataSource.COINGECKO, "Default primary source incorrect"
        assert len(config.sources) >= 4, "Insufficient default data sources"

        # Test source priorities
        enabled_sources = config.get_enabled_sources()
        assert len(enabled_sources) >= 3, "Insufficient enabled sources"

        # Test configuration validation
        issues = config.validate()
        if issues:
            print(f"⚠️ Configuration validation issues: {issues}")
        else:
            print("✅ Configuration validation passed")

        print("✅ Configuration functionality validated")
        return True

    except Exception as e:
        print(f"❌ Configuration validation failed: {e}")
        traceback.print_exc()
        return False

def validate_market_data_models():
    """Validate market data model functionality."""
    print("🔍 Validating market data models...")

    try:
        sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
        from v2.data_sources.models import StandardMarketData, MarketDataSummary, DataSource, DataQuality
        from datetime import datetime

        # Test StandardMarketData creation
        market_data = StandardMarketData(
            symbol="ETH",
            price=2000.0,
            volume_24h=1000000.0,
            change_24h=50.0,
            change_pct_24h=2.5,
            high_24h=2050.0,
            low_24h=1950.0,
            timestamp=datetime.now(),
            source=DataSource.COINGECKO,
            confidence=0.95,
            data_quality=DataQuality.GOOD
        )

        # Test to_dict conversion
        data_dict = market_data.to_dict()
        assert data_dict["symbol"] == "ETH", "to_dict conversion failed"

        # Test from_dict conversion
        recovered_data = StandardMarketData.from_dict(data_dict)
        assert recovered_data.symbol == market_data.symbol, "from_dict conversion failed"

        # Test MarketDataSummary
        summary = MarketDataSummary(
            symbol="ETH",
            primary_price=2000.0,
            price_sources=[market_data]
        )
        assert summary.symbol == "ETH", "MarketDataSummary creation failed"

        print("✅ Market data models validated")
        return True

    except Exception as e:
        print(f"❌ Market data models validation failed: {e}")
        traceback.print_exc()
        return False

def validate_adapter_structure():
    """Validate adapter structure and inheritance."""
    print("🔍 Validating adapter structure...")

    try:
        sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
        from v2.data_sources.adapters.base_adapter import BaseAdapter
        from v2.data_sources.adapters.coingecko_adapter import CoinGeckoAdapter
        from v2.data_sources.config import DataSourceConfig, DataSource

        # Test that adapters inherit from BaseAdapter
        assert issubclass(CoinGeckoAdapter, BaseAdapter), "CoinGeckoAdapter doesn't inherit from BaseAdapter"

        # Test adapter configuration
        config = DataSourceConfig(
            name="Test",
            base_url="https://test.com",
            enabled=True
        )

        adapter = CoinGeckoAdapter(config)
        assert adapter.source_type == DataSource.COINGECKO, "Adapter source type incorrect"

        print("✅ Adapter structure validated")
        return True

    except Exception as e:
        print(f"❌ Adapter structure validation failed: {e}")
        traceback.print_exc()
        return False

def validate_integration():
    """Validate integration with existing system."""
    print("🔍 Validating system integration...")

    try:
        sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
        from v2.config.main_config import MainV2Config
        from v2.data_sources.config import MarketDataConfig

        # Test main configuration includes market data
        main_config = MainV2Config()
        assert hasattr(main_config, 'market_data'), "MainV2Config missing market_data field"
        assert isinstance(main_config.market_data, MarketDataConfig), "market_data field incorrect type"

        # Test that the configuration is properly initialized
        assert main_config.market_data.primary_source, "Primary source not initialized"
        assert len(main_config.market_data.sources) > 0, "Data sources not initialized"

        print("✅ System integration validated")
        return True

    except Exception as e:
        print(f"❌ System integration validation failed: {e}")
        traceback.print_exc()
        return False

def validate_test_structure():
    """Validate test structure."""
    print("🔍 Validating test structure...")

    test_files = [
        "tests/v2/data_sources/test_config.py",
        "tests/v2/data_sources/test_models.py",
        "tests/v2/data_sources/test_adapters/test_base_adapter.py"
    ]

    for test_file in test_files:
        if Path(test_file).exists():
            print(f"✅ {test_file}")
        else:
            print(f"❌ Missing test file: {test_file}")
            return False

    print("✅ Test structure validated")
    return True

def main():
    """Run all validation checks."""
    print("🚀 Starting Multi-Source Market Data Implementation Validation")
    print("=" * 80)

    validations = [
        ("File Structure", validate_file_structure),
        ("Configuration", validate_configuration),
        ("Market Data Models", validate_market_data_models),
        ("Adapter Structure", validate_adapter_structure),
        ("System Integration", validate_integration),
        ("Test Structure", validate_test_structure),
        ("Imports", validate_imports)
    ]

    results = []
    for name, validator in validations:
        try:
            result = validator()
            results.append((name, result))
        except Exception as e:
            print(f"❌ {name} validation crashed: {e}")
            results.append((name, False))
        print()

    # Summary
    print("=" * 80)
    print("📊 VALIDATION SUMMARY")
    print("=" * 80)

    passed = 0
    total = len(results)

    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {name}")
        if result:
            passed += 1

    print()
    print(f"Overall Result: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 ALL VALIDATIONS PASSED! Multi-source market data implementation is ready.")
        return 0
    else:
        print("⚠️ Some validations failed. Please review and fix the issues.")
        return 1

if __name__ == "__main__":
    sys.exit(main())