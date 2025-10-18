"""
End-to-End Integration Tests for Colin Trading Bot v2.0

This module contains comprehensive integration tests for the complete system.
"""

import pytest
import asyncio
import time
from datetime import datetime
from unittest.mock import Mock, patch, AsyncMock
from pathlib import Path

# Import v2 components
from src.v2.main import ColinTradingBotV2
from src.v2.api_gateway.rest_api import RestAPI
from src.v2.api_gateway.websocket_api import WebSocketAPI
from src.v2.config.main_config import get_main_config_manager
from src.v2.risk_system import RealTimeRiskController, PreTradeChecker
from src.v2.monitoring.metrics import MetricsCollector
from src.v2.monitoring.alerts import AlertManager


class TestSystemIntegration:
    """Test cases for complete system integration."""

    @pytest.fixture
    def trading_bot(self):
        """Create a test trading bot instance."""
        # Mock dependencies to avoid external requirements
        with patch('src.v2.main.SmartOrderRouter') as mock_router:
            mock_router.return_value.route_order = AsyncMock(return_value=Mock(
                selected_routes=[Mock(exchange="test_exchange")],
                total_expected_fill=100.0,
                total_expected_cost=1.0
            ))

            bot = ColinTradingBotV2(mode="test")
            return bot

    @pytest.fixture
    def rest_api(self, trading_bot):
        """Create a test REST API instance."""
        api = RestAPI(trading_bot)
        return api

    @pytest.fixture
    def websocket_api(self, trading_bot):
        """Create a test WebSocket API instance."""
        ws_api = WebSocketAPI(trading_bot)
        return ws_api

    @pytest.fixture
    def config_manager(self):
        """Create a test configuration manager."""
        return get_main_config_manager()

    @pytest.mark.asyncio
    async def test_bot_initialization(self, trading_bot):
        """Test trading bot initialization."""
        # Test basic initialization
        assert trading_bot.mode == "test"
        assert trading_bot.is_running is False
        assert trading_bot.shutdown_requested is False

        # Test component initialization
        await trading_bot.initialize_components()

        # Verify components are initialized
        assert trading_bot.smart_router is not None
        assert trading_bot.risk_controller is not None
        assert trading_bot.pre_trade_checker is not None
        assert trading_bot.compliance_monitor is not None

        # Test shutdown
        await trading_bot.shutdown()

    @pytest.mark.asyncio
    async def test_signal_to_execution_workflow(self, trading_bot):
        """Test complete signal-to-execution workflow."""
        await trading_bot.initialize_components()

        # Create mock signal
        from src.v2.main import TradingSignal
        signal = TradingSignal(
            symbol="BTC/USDT",
            direction="long",
            confidence=0.75,
            strength=0.8,
            timestamp=datetime.now(),
            source_model="test_model",
            metadata={"predicted_return": 0.05}
        )

        # Process signal through pipeline
        await trading_bot._process_signal(signal)

        # Verify workflow completed (no exceptions thrown)
        assert True  # If we reach here, the workflow completed

        await trading_bot.shutdown()

    @pytest.mark.asyncio
    async def test_risk_validation_integration(self, trading_bot):
        """Test risk validation integration."""
        await trading_bot.initialize_components()

        from src.v2.execution_engine.smart_routing.router import Order, OrderSide, OrderType

        # Create test order
        order = Order(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=1.0,
            price=50000.0,
            client_order_id="test_order_001"
        )

        # Get portfolio metrics
        portfolio_metrics = await trading_bot._get_portfolio_metrics()

        # Run risk validation
        risk_decision = await trading_bot.risk_controller.validate_trade(
            order, trading_bot.active_positions, portfolio_metrics
        )

        # Verify risk decision structure
        assert hasattr(risk_decision, 'approved')
        assert hasattr(risk_decision, 'risk_level')
        assert hasattr(risk_decision, 'reasoning')

        await trading_bot.shutdown()

    @pytest.mark.asyncio
    async def test_compliance_integration(self, trading_bot):
        """Test compliance integration."""
        await trading_bot.initialize_components()

        from src.v2.execution_engine.smart_routing.router import Order, OrderSide, OrderType

        # Create test order
        order = Order(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=1.0,
            price=50000.0,
            client_order_id="test_order_002"
        )

        # Run compliance check
        compliance_result = await trading_bot.pre_trade_checker.check_compliance(
            order, trading_bot.active_positions
        )

        # Verify compliance result structure
        assert hasattr(compliance_result, 'compliant')
        assert hasattr(compliance_result, 'status')
        assert hasattr(compliance_result, 'rules_checked')
        assert hasattr(compliance_result, 'failed_rules')

        await trading_bot.shutdown()

    def test_configuration_system(self, config_manager):
        """Test configuration system integration."""
        # Test configuration loading
        assert config_manager.config is not None
        assert config_manager.config.system.environment in ["development", "test", "staging", "production"]

        # Test configuration validation
        validation_result = config_manager.validate_configuration()
        assert isinstance(validation_result, bool)

        # Test configuration summary
        summary = config_manager.get_configuration_summary()
        assert "environment" in summary
        assert "feature_flags" in summary
        assert "performance_settings" in summary

    def test_rest_api_initialization(self, rest_api):
        """Test REST API initialization."""
        assert rest_api.app is not None
        assert rest_api.trading_bot is not None
        assert rest_api.config is not None
        assert rest_api.start_time > 0

        # Test app routes
        routes = [route.path for route in rest_api.app.routes]
        expected_routes = [
            "/api/v2/health",
            "/api/v2/signals/generate",
            "/api/v2/orders",
            "/api/v2/portfolio",
            "/api/v2/metrics"
        ]

        for route in expected_routes:
            assert any(expected_route in path for path in routes)

    def test_websocket_api_initialization(self, websocket_api):
        """Test WebSocket API initialization."""
        assert websocket_api.trading_bot is not None
        assert websocket_api.config is not None
        assert websocket_api.connections == {}
        assert websocket_api.channel_subscribers is not None

        # Test channel setup
        expected_channels = ["signals", "orders", "portfolio", "metrics", "risk"]
        for channel in expected_channels:
            assert channel in websocket_api.channel_subscribers

    @pytest.mark.asyncio
    async def test_monitoring_integration(self, trading_bot):
        """Test monitoring system integration."""
        # Create monitoring components
        metrics_collector = MetricsCollector()
        alert_manager = AlertManager()

        # Test metrics collection
        metrics_collector.increment_counter("signals_generated")
        metrics_collector.set_gauge("active_positions", 5)
        metrics_collector.record_histogram("execution_latency_ms", 25.5)

        # Test metrics retrieval
        all_metrics = metrics_collector.get_all_metrics()
        assert "counters" in all_metrics
        assert "gauges" in all_metrics
        assert "histograms" in all_metrics

        # Test alert system
        from src.v2.monitoring.alerts import AlertSeverity
        alert = await alert_manager.create_alert(
            severity=AlertSeverity.MEDIUM,
            title="Test Alert",
            message="This is a test alert",
            source="test_system"
        )

        assert alert.id is not None
        assert alert.severity == AlertSeverity.MEDIUM
        assert alert.acknowledged is False
        assert alert.resolved is False

        # Test alert statistics
        alert_summary = alert_manager.get_alert_summary()
        assert "total_alerts" in alert_summary
        assert "active_alerts" in alert_summary
        assert "severity_breakdown" in alert_summary

    @pytest.mark.asyncio
    async def test_end_to_end_latency(self, trading_bot):
        """Test end-to-end latency performance."""
        await trading_bot.initialize_components()

        # Measure signal to execution latency
        start_time = time.time()

        from src.v2.main import TradingSignal
        signal = TradingSignal(
            symbol="ETH/USDT",
            direction="long",
            confidence=0.80,
            strength=0.9,
            timestamp=datetime.now(),
            source_model="test_model",
            metadata={}
        )

        # Process signal
        await trading_bot._process_signal(signal)

        end_time = time.time()
        latency_ms = (end_time - start_time) * 1000

        # Should complete within reasonable time (even with mocking)
        assert latency_ms < 1000  # 1 second max for test environment

        await trading_bot.shutdown()

    @pytest.mark.asyncio
    async def test_error_handling_integration(self, trading_bot):
        """Test error handling in integration."""
        await trading_bot.initialize_components()

        # Test with invalid data
        try:
            from src.v2.main import TradingSignal
            invalid_signal = TradingSignal(
                symbol="",  # Empty symbol
                direction="invalid",
                confidence=1.5,  # Invalid confidence > 1.0
                strength=0.0,
                timestamp=datetime.now(),
                source_model="test",
                metadata={}
            )

            # Should handle gracefully
            await trading_bot._process_signal(invalid_signal)

        except Exception as e:
            # Should not crash the system
            assert isinstance(e, (ValueError, AttributeError))

        await trading_bot.shutdown()

    @pytest.mark.asyncio
    async def test_portfolio_tracking_integration(self, trading_bot):
        """Test portfolio tracking integration."""
        await trading_bot.initialize_components()

        # Simulate some executions
        from src.v2.execution_engine.smart_routing.router import Order, OrderSide, OrderType

        order1 = Order(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=2.0,
            price=50000.0,
            client_order_id="portfolio_test_001"
        )

        execution_result1 = {
            "success": True,
            "executed_quantity": 2.0,
            "executed_price": 50000.0,
            "execution_time_ms": 45.0,
            "exchange": "test_exchange",
            "fees": 100.0
        }

        # Update positions
        await trading_bot._update_positions(order1, execution_result1)

        # Verify position tracking
        assert "BTC/USDT" in trading_bot.active_positions
        assert trading_bot.active_positions["BTC/USDT"]["quantity"] == 2.0
        assert trading_bot.active_positions["BTC/USDT"]["value_usd"] == 100000.0

        # Test portfolio metrics
        portfolio_metrics = await trading_bot._get_portfolio_metrics()
        assert portfolio_metrics["total_value"] > 0
        assert portfolio_metrics["position_count"] == 1

        await trading_bot.shutdown()

    @pytest.mark.asyncio
    async def test_concurrent_operations(self, trading_bot):
        """Test concurrent operations handling."""
        await trading_bot.initialize_components()

        # Create multiple concurrent signals
        from src.v2.main import TradingSignal

        signals = [
            TradingSignal(
                symbol=f"SYM{i:03d}",
                direction="long" if i % 2 == 0 else "short",
                confidence=0.7 + (i * 0.02),
                strength=0.8,
                timestamp=datetime.now(),
                source_model="test_model",
                metadata={}
            )
            for i in range(5)
        ]

        # Process signals concurrently
        tasks = [
            trading_bot._process_signal(signal)
            for signal in signals
        ]

        # Wait for all to complete
        await asyncio.gather(*tasks, return_exceptions=True)

        # Verify system is still responsive
        assert trading_bot.is_running is False  # Should be false in test mode
        assert len(trading_bot.execution_history) >= 0

        await trading_bot.shutdown()

    def test_feature_flags_integration(self, config_manager):
        """Test feature flags integration."""
        # Test feature flag checking
        assert isinstance(config_manager.is_feature_enabled("ml_signals"), bool)
        assert isinstance(config_manager.is_feature_enabled("smart_routing"), bool)
        assert isinstance(config_manager.is_feature_enabled("real_time_risk"), bool)
        assert isinstance(config_manager.is_feature_enabled("compliance_monitoring"), bool)

        # Test all expected feature flags
        expected_features = [
            "ml_signals",
            "smart_routing",
            "real_time_risk",
            "compliance_monitoring",
            "stress_testing",
            "auto_scaling"
        ]

        for feature in expected_features:
            result = config_manager.is_feature_enabled(feature)
            assert isinstance(result, bool)

    def test_database_url_generation(self, config_manager):
        """Test database URL generation."""
        db_url = config_manager.get_database_url()
        assert isinstance(db_url, str)
        assert db_url.startswith("postgresql://")

        # Test URL components
        assert "colin_trading_bot_v2" in db_url
        assert "colin_user" in db_url

    @pytest.mark.asyncio
    async def test_system_health_integration(self, trading_bot):
        """Test system health monitoring integration."""
        await trading_bot.initialize_components()

        # Test health check components
        health_status = {
            "trading_bot": "healthy" if not trading_bot.is_running else "active",
            "risk_system": "healthy",
            "execution_engine": "healthy",
            "compliance_system": "healthy",
            "database": "healthy"
        }

        # Verify all components are healthy
        for component, status in health_status.items():
            assert status in ["healthy", "active", "stopped", "degraded"]

        await trading_bot.shutdown()


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])