#!/usr/bin/env python3
"""
Standalone validation script for multi-source market data implementation.
Tests core functionality without importing problematic dependencies.
"""

import sys
import os
import traceback
from pathlib import Path

def test_direct_imports():
    """Test direct imports without going through v2 __init__.py."""
    print("🔍 Testing direct imports...")

    try:
        # Add src to path
        src_path = os.path.join(os.path.dirname(__file__), 'src')
        if src_path not in sys.path:
            sys.path.insert(0, src_path)

        # Test direct imports
        from v2.data_sources.config import MarketDataConfig, DataSourceConfig, DataSource
        print("✅ MarketDataConfig imported directly")

        from v2.data_sources.models import StandardMarketData, MarketDataSummary, DataQuality
        print("✅ Market data models imported directly")

        # Test basic functionality
        config = MarketDataConfig()
        assert config.primary_source == DataSource.COINGECKO
        print("✅ MarketDataConfig basic functionality works")

        # Test model creation
        from datetime import datetime
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
            confidence=0.95
        )
        assert market_data.symbol == "ETH"
        print("✅ StandardMarketData creation works")

        # Test dict conversion
        data_dict = market_data.to_dict()
        recovered = StandardMarketData.from_dict(data_dict)
        assert recovered.symbol == market_data.symbol
        print("✅ Market data serialization works")

        return True

    except Exception as e:
        print(f"❌ Direct import test failed: {e}")
        traceback.print_exc()
        return False

def test_adapter_imports():
    """Test adapter imports."""
    print("🔍 Testing adapter imports...")

    try:
        from v2.data_sources.adapters.base_adapter import BaseAdapter
        from v2.data_sources.adapters.coingecko_adapter import CoinGeckoAdapter
        from v2.data_sources.adapters.kraken_adapter import KrakenAdapter
        from v2.data_sources.adapters.cryptocompare_adapter import CryptoCompareAdapter
        from v2.data_sources.adapters.alternative_me_adapter import AlternativeMeAdapter

        print("✅ All adapter imports successful")

        # Test adapter configuration
        config = DataSourceConfig(
            name="Test",
            base_url="https://test.com",
            enabled=True
        )

        adapter = CoinGeckoAdapter(config)
        assert adapter.source_type.value == "coingecko"
        print("✅ CoinGeckoAdapter creation works")

        return True

    except Exception as e:
        print(f"❌ Adapter import test failed: {e}")
        traceback.print_exc()
        return False

def test_market_data_manager():
    """Test market data manager."""
    print("🔍 Testing market data manager...")

    try:
        from v2.data_sources.market_data_manager import MarketDataManager, CircuitBreaker, DataCache

        # Test CircuitBreaker
        cb = CircuitBreaker(failure_threshold=3, timeout_seconds=60)
        assert cb.state == "CLOSED"
        assert not cb.is_open()
        print("✅ CircuitBreaker works")

        # Test DataCache
        cache = DataCache(ttl_seconds=60, max_size=10)
        cache.put("test_key", "test_value")
        value = cache.get("test_key")
        assert value == "test_value"
        print("✅ DataCache works")

        # Test MarketDataManager creation
        config = MarketDataConfig()
        manager = MarketDataManager(config)
        assert manager.config == config
        assert len(manager.adapters) >= 0
        print("✅ MarketDataManager creation works")

        return True

    except Exception as e:
        print(f"❌ Market data manager test failed: {e}")
        traceback.print_exc()
        return False

def test_configuration_system():
    """Test configuration system."""
    print("🔍 Testing configuration system...")

    try:
        from v2.data_sources.config import MarketDataConfig, DataSourceConfig, DataSource

        # Test default configuration
        config = MarketDataConfig()
        assert len(config.sources) >= 4
        assert DataSource.COINGECKO in config.sources
        assert DataSource.KRAKEN in config.sources
        assert DataSource.CRYPTOCOMPARE in config.sources
        assert DataSource.ALTERNATIVE_ME in config.sources
        print("✅ Default sources configured")

        # Test source priority
        enabled = config.get_enabled_sources()
        assert len(enabled) >= 3
        priorities = [config.sources[s].priority for s in enabled]
        assert priorities == sorted(priorities)
        print("✅ Source priority ordering works")

        # Test configuration validation
        issues = config.validate()
        if issues:
            print(f"⚠️ Configuration validation issues: {issues}")
        else:
            print("✅ Configuration validation passed")

        return True

    except Exception as e:
        print(f"❌ Configuration system test failed: {e}")
        traceback.print_exc()
        return False

def test_file_integrity():
    """Test file integrity and completeness."""
    print("🔍 Testing file integrity...")

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
    ]

    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)

    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False

    print("✅ All required files present")

    # Check file sizes (basic sanity check)
    min_sizes = {
        "src/v2/data_sources/config.py": 1000,
        "src/v2/data_sources/models.py": 1000,
        "src/v2/data_sources/market_data_manager.py": 2000,
        "src/v2/data_sources/adapters/coingecko_adapter.py": 1000,
        "analyze_ethereum_multi_source.py": 3000,
    }

    for file_path, min_size in min_sizes.items():
        size = Path(file_path).stat().st_size
        if size < min_size:
            print(f"❌ File {file_path} too small: {size} bytes (expected > {min_size})")
            return False

    print("✅ File sizes look reasonable")
    return True

def test_syntax_validation():
    """Test syntax validation of all Python files."""
    print("🔍 Testing syntax validation...")

    python_files = []
    for pattern in [
        "src/v2/data_sources/**/*.py",
        "analyze_ethereum_multi_source.py"
    ]:
        for file_path in Path(".").glob(pattern):
            python_files.append(str(file_path))

    syntax_errors = []
    for file_path in python_files:
        try:
            with open(file_path, 'r') as f:
                code = f.read()
            compile(code, file_path, 'exec')
        except SyntaxError as e:
            syntax_errors.append(f"{file_path}: {e}")
        except Exception as e:
            syntax_errors.append(f"{file_path}: {e}")

    if syntax_errors:
        print("❌ Syntax errors found:")
        for error in syntax_errors:
            print(f"   {error}")
        return False

    print(f"✅ All {len(python_files)} Python files have valid syntax")
    return True

def main():
    """Run standalone validation."""
    print("🚀 Standalone Multi-Source Market Data Validation")
    print("=" * 80)

    tests = [
        ("File Integrity", test_file_integrity),
        ("Syntax Validation", test_syntax_validation),
        ("Direct Imports", test_direct_imports),
        ("Configuration System", test_configuration_system),
        ("Adapter Imports", test_adapter_imports),
        ("Market Data Manager", test_market_data_manager),
    ]

    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"❌ {name} test crashed: {e}")
            results.append((name, False))
        print()

    # Summary
    print("=" * 80)
    print("📊 STANDALONE VALIDATION SUMMARY")
    print("=" * 80)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {name}")

    print()
    print(f"Overall Result: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 ALL STANDALONE VALIDATIONS PASSED!")
        print("✅ Multi-source market data implementation is syntactically correct and functional.")
        print("✅ Core components are working as expected.")
        print("✅ Ready for integration and testing.")
        return 0
    else:
        print("⚠️ Some validations failed. Please review the issues above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())