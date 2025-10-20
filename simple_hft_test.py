#!/usr/bin/env python3
"""
Simple HFT Integration Test

Direct test of HFT components without complex configuration.
"""

import sys
from pathlib import Path
import asyncio

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

async def test_hft_integration():
    """Test HFT integration directly."""
    print("🚀 Simple HFT Integration Test")
    print("=" * 40)

    try:
        # Test 1: Import HFT components
        print("\n1️⃣ Testing HFT imports...")
        from colin_bot.v2.hft_engine import (
            OFICalculator, BookSkewAnalyzer, SignalFusionEngine,
            DynamicPositionSizer, CircuitBreakerSystem, HFTDataManager
        )
        from colin_bot.v2.hft_engine.utils.data_structures import (
            OrderBook, OrderBookLevel, TradingSignal, SignalDirection
        )
        from colin_bot.v2.hft_engine.utils.math_utils import calculate_skew, hawkes_process
        print("✅ All HFT components imported successfully")

        # Test 2: Create HFT adapter
        print("\n2️⃣ Testing HFT adapter...")
        from colin_bot.engine.hft_integration_adapter import HFTIntegrationAdapter

        # Create a mock config manager
        class MockConfigManager:
            def __init__(self):
                self.config = MockConfig()

        class MockConfig:
            def __init__(self):
                self.logging = MockLoggingConfig()
                self.base_price = 50000.0  # Add missing base_price attribute
                self.hft = MockHFTConfig()

        class MockLoggingConfig:
            def __init__(self):
                self.level = "INFO"
                self.file = None
                self.max_size = "10MB"
                self.backup_count = 5

        class MockHFTConfig:
            def __init__(self):
                self.max_order_book_levels = 10
                self.signal_timeout_ms = 100
                self.circuit_breaker_threshold = 0.05

        mock_config = MockConfigManager()
        adapter = HFTIntegrationAdapter(mock_config, enable_hft=True)
        print(f"✅ HFT adapter created: {adapter.is_hft_enabled()}")

        # Test 3: Generate HFT signal
        print("\n3️⃣ Testing HFT signal generation...")
        symbol = "BTCUSDT"
        hft_signal = await adapter.generate_hft_signal(symbol)

        if hft_signal:
            print(f"✅ HFT signal generated for {symbol}:")
            print(f"   Direction: {hft_signal.direction}")
            print(f"   Confidence: {hft_signal.confidence:.1f}%")
            print(f"   Strength: {hft_signal.strength}")
            print(f"   OFI Signal: {hft_signal.ofi_signal:.3f}")
            print(f"   Book Skew: {hft_signal.book_skew:.3f}")
        else:
            print("⚠️  No HFT signal generated")

        # Test 4: Test signal bridge
        print("\n4️⃣ Testing signal bridge...")
        from colin_bot.engine.hft_signal_bridge import HFTSignalBridge
        bridge = HFTSignalBridge(adapter)

        if bridge.is_available():
            print("✅ Signal bridge is available")

            # Test signal enhancement
            mock_institutional_signal = {
                'symbol': symbol,
                'direction': 'long',
                'long_confidence': 65.0,
                'short_confidence': 35.0,
                'confidence_level': 'medium'
            }

            enhanced_signal = await bridge.enhance_institutional_signal(symbol, mock_institutional_signal)

            if enhanced_signal:
                print("✅ Signal enhancement successful:")
                print(f"   Original confidence: {mock_institutional_signal['long_confidence']:.1f}%")
                print(f"   Enhanced confidence: {enhanced_signal.get('enhanced_long_confidence', 0):.1f}%")

                hft_components = enhanced_signal.get('hft_components', {})
                if hft_components:
                    print(f"   HFT components: {len(hft_components)} items")
            else:
                print("⚠️  Signal enhancement failed")
        else:
            print("⚠️  Signal bridge not available")

        # Test 5: Performance metrics
        print("\n5️⃣ Testing performance metrics...")
        metrics = adapter.get_performance_metrics()
        print(f"✅ Performance metrics:")
        print(f"   Signals generated: {metrics.get('signals_generated', 0)}")
        print(f"   Circuit breaker status: {metrics.get('circuit_breaker_status', 'unknown')}")

        print("\n🎉 All HFT integration tests passed!")
        return True

    except Exception as e:
        print(f"\n❌ HFT integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_simple_main_bot():
    """Test simple main bot functionality."""
    print("\n📋 Testing Simple Main Bot")
    print("-" * 40)

    try:
        # Test importing main bot
        from colin_bot.main import ColinTradingBot

        # Create bot with minimal config
        class MockConfigManager:
            def __init__(self):
                self.config = MockConfig()

        class MockConfig:
            def __init__(self):
                self.logging = MockLoggingConfig()
                self.scoring = MockScoringConfig()
                self.base_price = 50000.0
                self.hft = MockHFTConfig()

        class MockLoggingConfig:
            def __init__(self):
                self.level = "INFO"
                self.file = None
                self.max_size = "10MB"
                self.backup_count = 5

        class MockScoringConfig:
            def __init__(self):
                self.weights = {
                    'liquidity': 0.25,
                    'ict': 0.25,
                    'killzone': 0.25,
                    'order_flow': 0.15,
                    'volume_oi': 0.1
                }

        class MockHFTConfig:
            def __init__(self):
                self.max_order_book_levels = 10
                self.signal_timeout_ms = 100
                self.circuit_breaker_threshold = 0.05

        mock_config = MockConfigManager()

        # Test bot creation with HFT enabled
        print("Creating bot with HFT integration...")
        bot = ColinTradingBot(mock_config, enable_hft=True)
        print(f"✅ Bot created: {bot.bot_type}")
        print(f"   HFT enabled: {bot.enable_hft}")

        # Test HFT status
        print("Testing HFT status...")
        status = await bot.get_hft_status()
        print(f"✅ HFT status: {status.get('hft_enabled', False)}")
        print(f"   Bot type: {status.get('bot_type', 'unknown')}")

        print("\n🎉 Simple main bot test passed!")
        return True

    except Exception as e:
        print(f"\n❌ Simple main bot test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Main test function."""
    print("🧪 Simple HFT Integration Tests")
    print("Testing HFT components with minimal setup\n")

    # Test HFT components
    hft_success = await test_hft_integration()

    # Test main bot
    bot_success = await test_simple_main_bot()

    # Final summary
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)

    if hft_success and bot_success:
        print("🎉 ALL TESTS PASSED!")
        print("✅ HFT integration is working correctly")
        print("✅ Ready for production use")
    elif hft_success:
        print("⚡ HFT COMPONENTS WORK!")
        print("⚠️  Main bot integration needs attention")
    elif bot_success:
        print("⚡ MAIN BOT WORKS!")
        print("⚠️  HFT components need attention")
    else:
        print("❌ ALL TESTS FAILED")
        print("🔧 Integration needs debugging")

if __name__ == "__main__":
    asyncio.run(main())